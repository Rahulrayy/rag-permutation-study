"""Provence (arXiv 2501.16214): sentence-level pruning folded into the reranker.

Checkpoint verified loadable 2026-08-29 -- plan Sec. 8 flags this as the
assumption most likely to blow up, with the gate "verify it loads in week 2,
not week 5". It loads: public, ungated, 435M params, 1.74 GB, needs
``trust_remote_code=True`` and ``nltk`` (plus its punkt/punkt_tab data).

**License: cc-by-nc-nd-4.0** -- non-commercial, no-derivatives. Fine for a
research artifact, and it must be stated in the write-up.

Two arms, not one
-----------------
``process()`` returns *both* a per-chunk ``reranking_score`` and a
sentence-pruned ``pruned_context``. Measured over 20 HotpotQA queries
(200 chunks, 40 gold) at threshold 0.1:

    as a reranker        top-1 chunk is gold 20/20; mean gold rank 2.02 of 10
    as a sentence pruner keeps 11.2% of text; prunes 28% of GOLD chunks to
                         empty; loses the answer string outright in 10% of
                         queries

Excellent reranking, brutal pruning. Collapsing those into a single arm would
report their sum and let neither be read, so they run as two:

``provence_rerank``  select top-k by score, keep the **original** chunk text.
                     Content stays matched against `placebo_pos` at equal k,
                     so this arm is comparable to every selection-only arm.
``provence_full``    select top-k by the same scores, then replace each kept
                     chunk's text with Provence's pruned version. This is the
                     method as published. Content is no longer matched at equal
                     k, so it is reported against input-token count (plan
                     Sec. 4.3), not against k.

Their difference isolates the contribution of sentence pruning with selection
held fixed, which is the decomposition nobody in this literature reports.

Expect ``provence_full`` to score *badly*. That is the finding, not a bug: a
method that deletes the answer in 10% of queries should show up as one.

Being used out of distribution relative to its training data, like every
off-the-shelf checkpoint here -- the study measures deployed-as-published
behaviour, not the method's ceiling.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..chunks import Chunk
from .base import Pruner, validate_rewrite, validate_selection

DEFAULT_CHECKPOINT = "naver/provence-reranker-debertav3-v1"

# Both arms want the same 1.74 GB of weights. Keyed by (checkpoint, device) so
# the two of them share one load instead of paying for it twice on a 6 GB card.
_MODELS: dict[tuple[str, str], Any] = {}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load(checkpoint: str, device: str) -> Any:
    device = _resolve_device(device)
    key = (checkpoint, device)
    if key not in _MODELS:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        _MODELS[key] = model.to(device).eval()
    return _MODELS[key]


def unload_all() -> None:
    """Drop every cached Provence model and give the VRAM back."""
    if not _MODELS:
        return
    _MODELS.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class _ProvenceBase(Pruner):
    """Shared machinery. Neither this nor its `name` is registered."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        threshold: float = 0.1,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.device = device
        self.batch_size = batch_size
        # process() output per query. The result does not depend on the budget,
        # but run.py visits each query once per budget, so without this the
        # model would run three times over for identical output.
        self._cache: dict[tuple[str, tuple[int, ...]], tuple[list[float], dict[int, str]]] = {}

    def _run(self, query: str, chunks: Sequence[Chunk]) -> tuple[list[float], dict[int, str]]:
        key = (query, tuple(c.idx for c in chunks))
        if key in self._cache:
            return self._cache[key]

        model = _load(self.checkpoint, self.device)
        out = model.process(
            question=query,
            context=[[c.text for c in chunks]],
            title=[[c.title for c in chunks]],
            threshold=self.threshold,
            batch_size=self.batch_size,
            # Titles are rendered by generate.build_prompt for *every* arm, so
            # letting Provence prepend them here too would duplicate them and
            # make this the only arm with a different context format. Keeping
            # the renderer uniform means the sole difference between arms is
            # the body text, which is the thing being measured.
            always_select_title=False,
            # `reorder` would return chunks in Provence's preferred order.
            # Ordering is the study's independent variable and is applied
            # afterwards by chunks.permute -- never by a pruner.
            reorder=False,
            enable_warnings=False,
        )
        scores = [float(x) for x in out["reranking_score"][0]]
        pruned = {c.idx: str(t) for c, t in zip(chunks, out["pruned_context"][0])}
        self._cache[key] = (scores, pruned)
        return scores, pruned

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        scores, _ = self._run(query, chunks)
        ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i].rank))
        kept = [chunks[i].idx for i in ranked[: min(budget, len(chunks))]]
        # Returned in score order, which carries no meaning: chunks.permute
        # overwrites it. Sorted here only so the cache key is stable.
        return validate_selection(sorted(kept), chunks, budget)

    def close(self) -> None:
        self._cache.clear()
        unload_all()


class ProvenceRerank(_ProvenceBase):
    """Provence's reranker only: top-k by score, original text preserved."""

    name = "provence_rerank"


class ProvenceFull(_ProvenceBase):
    """Provence as published: top-k by score, sentence-pruned text."""

    name = "provence_full"

    def rewrite(
        self, query: str, chunks: Sequence[Chunk], budget: int
    ) -> list[Chunk]:
        # `chunks` is the kept subset; the pruned text was computed over the
        # full context during select() and is looked up by idx. Provence scores
        # each passage independently given the question, so a chunk's pruned
        # text does not depend on which others survived.
        if not chunks:
            return []
        cached = [v for (q, _), v in self._cache.items() if q == query]
        if not cached:
            raise RuntimeError(
                "rewrite() called before select() for this query; the pruned "
                "text is produced by select()'s model pass"
            )
        _, pruned = cached[-1]
        out = [
            Chunk(
                idx=c.idx,
                title=c.title,
                # A chunk pruned to nothing keeps its slot and its title, and
                # contributes an empty body. Dropping it instead would break
                # the keep-count match against `placebo_pos`, and an empty
                # passage is genuinely what Provence returns.
                text=pruned.get(c.idx, c.text).strip(),
                rank=c.rank,
                is_gold=c.is_gold,
                meta={**c.meta, "provence_pruned": True},
            )
            for c in chunks
        ]
        return validate_rewrite(chunks, out)
