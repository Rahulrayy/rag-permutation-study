"""Cross-encoder rerank, keep top-k. The baseline everyone should beat.

This is the **default denominator arm for OAE** (configs/main.yaml ->
metrics.baseline_arm), which makes it load-bearing for the headline number:

    OAE(m) = mean_q[ mean_pi Q(m,q,pi) - mean_pi Q(b,q,pi) ]
             / mean_q[ SD_pi Q(b,q,pi) ]

Both the numerator's reference point and the denominator's noise scale come
from this arm, so nothing else in the study is interpretable without it.

Implemented on plain `transformers` rather than `sentence-transformers`
-----------------------------------------------------------------------
A cross-encoder is an `AutoModelForSequenceClassification` with a single output
logit, which is exactly what `sentence_transformers.CrossEncoder` wraps. Taking
the dependency would risk pip resolving a *PyPI* torch over the CUDA build this
environment needs (see HANDOFF Sec. 6 -- torch must come from the cu128 index),
and it buys nothing here. Verified equivalent on the target checkpoint: gold
chunks rank 1 and 3 of 10 on the first HotpotQA example.

Scoring input
-------------
Each chunk is scored as ``"{title}: {text}"``, the same unit
`generate.build_prompt` renders. Scoring the body alone would rank passages on
evidence the generator does not see in isolation, and titles carry real signal
on HotpotQA, where the answer is frequently the title of a gold paragraph.

Truncation at 512 tokens is the checkpoint's own limit and is left in place:
this arm measures the reranker as deployed, not a lifted version of it. Median
HotpotQA paragraphs are well inside it.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..chunks import Chunk
from .base import Pruner, validate_selection

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Shared across instances, like Provence's: one load, however many arms use it.
_MODELS: dict[tuple[str, str], Any] = {}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load(model_name: str, device: str) -> tuple[Any, Any]:
    device = _resolve_device(device)
    key = (model_name, device)
    if key not in _MODELS:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _MODELS[key] = (tok, model.to(device).eval())
    return _MODELS[key]


def unload_all() -> None:
    """Drop cached rerankers and give the VRAM back."""
    if not _MODELS:
        return
    _MODELS.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class RerankTopK(Pruner):
    name = "rerank_topk"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.device = device
        self.max_length = max_length
        # Scores do not depend on the budget, but run.py visits each query once
        # per budget; without this the reranker would run three times over.
        self._cache: dict[tuple[str, tuple[int, ...]], list[float]] = {}

    def _scores(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        key = (query, tuple(c.idx for c in chunks))
        if key in self._cache:
            return self._cache[key]

        import torch

        tok, model = _load(self.model, self.device)
        enc = tok(
            [query] * len(chunks),
            [f"{c.title}: {c.text}" for c in chunks],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits
        scores = logits.squeeze(-1).float().tolist()
        if isinstance(scores, float):  # a single chunk squeezes to a scalar
            scores = [scores]

        self._cache[key] = scores
        return scores

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        scores = self._scores(query, chunks)
        # Ties broken by original rank, so the selection is deterministic rather
        # than dependent on sort stability over floats that happen to collide.
        ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i].rank))
        kept = [chunks[i].idx for i in ranked[: min(budget, len(chunks))]]
        # Sorted before returning: the order a pruner picks in carries no
        # meaning and chunks.permute overwrites it. Returning score order would
        # invite someone downstream to read it as the reranker's ranking, which
        # is exactly the selection/ordering conflation this study is about.
        return validate_selection(sorted(kept), chunks, budget)

    def close(self) -> None:
        self._cache.clear()
        unload_all()
