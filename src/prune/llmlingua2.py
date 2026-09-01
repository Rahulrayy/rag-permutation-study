"""LLMLingua-2: token-level compression. Published method, different family.

Unlike every other arm here, this one does not *select* passages at all. It is a
rate-based compressor: it classifies each token keep/drop and hands back a
shorter string. That single fact drives all three design decisions below, each
of which was made and recorded before the arm ran.

1. Compression is per-chunk, never joint
----------------------------------------
LLMLingua-2 is normally applied to a whole concatenated context. **Doing that
here would break the study.** The compression would depend on the order the
chunks were concatenated in, so each of the P=5 permutations of a cell would
receive *different content* -- and content held fixed while order varies is the
one invariant the entire design rests on. The arm meant to test ordering would
be the arm that confounded it.

This is not hypothetical. Measured over 10 queries / 100 chunks / 3 orderings,
asking whether a given chunk's surviving text is identical across orderings:

    joint compression        0/100   (0%)
    per-chunk compression  100/100  (100%)

Not one chunk survived joint compression the same way twice. So each chunk is
compressed on its own, with ``use_context_level_filter=False``.

That LLMLingua-2 is order-sensitive at all is worth reporting in its own right:
it is a deterministic token classifier, so order-dependence in *it* is a good
deal more surprising than in an LLM asked to pick passages (see
`llm_pruner`). Both are the study's thesis reaching inside a method rather than
just its output.

2. The budget is spent as a rate, not a keep-count
--------------------------------------------------
There is no chunk ranking to take a top-k from, and inventing one -- say, mean
token keep-probability per chunk -- would be proposing a method, which plan
Sec. 9 rules out in as many words. So ``budget`` k becomes a per-chunk
compression rate of ``k / n_chunks``: all ten chunks survive, each compressed to
roughly k/10 of its tokens, so total input tokens land near what a keep-k arm
spends. The comparison this arm supports is *given the same token budget, is it
better to drop whole passages or compress all of them?* -- which is exactly the
comparison plan Sec. 4.3 asks for when it says to report against input-token
count rather than k.

Verified over 10 queries: total input characters against `rerank_topk` at the
same k came out at 0.86x (k=2), 0.94x (k=3) and 1.12x (k=5). Close enough to
call the budgets matched, and the residual is honest -- a reranker keeps the
chunks it judges relevant, which are not average-length.

``budget_is_keep_count`` is therefore False. This arm is **not** keep-k matched
and must never be pooled with the selection arms in a matched comparison,
including against `placebo_pos`.

3. What gets permuted
---------------------
The surviving chunk-level units, never individual tokens. Permuting tokens would
destroy the text and confound the ordering effect with scrambling, and nobody
deploys it that way.

A consequence to state rather than hide: this arm presents **n** permutable
slots where a keep-k arm presents **k**. More slots means more positional room,
so its raw permutation variance is not directly comparable in magnitude to
`rerank_topk` at k=3. That is inherent to the method, not an artifact, and it is
another reason the comparison is on tokens.
"""

from __future__ import annotations

import hashlib

from typing import Any, Sequence

from ..cache import artifact_key
from ..chunks import Chunk
from .base import Pruner, validate_rewrite, validate_selection

# The bert-base checkpoint rather than xlm-roberta-large: 0.71 GB against 2.2,
# on a 6 GB card that also drives the display. Both are released by the authors
# as LLMLingua-2; the choice is recorded here because it is a deviation from the
# paper's headline model and belongs in the write-up.
DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Tokens the compressor is told to preserve. This is the upstream default.
#
# DEVIATION FROM THE REGISTERED SETTING, 2026-08-31. This list originally read
# ``["\n", "?", ".", ","]`` -- the two sentence-punctuation marks were added on
# the reasoning that "without it the output loses sentence boundaries entirely
# and the generator sees a bag of words." That reasoning was wrong, and the
# first full-scale run is what exposed it: forced tokens are charged against the
# compression budget, so at aggressive rates the punctuation crowds out the
# content it was meant to punctuate. Same chunk (51 content words), same rates,
# only this list changed:
#
#     rate    with '.' and ','    upstream default
#     0.20     2 content words     8 content words
#     0.30     8                  11
#     0.50    16                  20
#
# At k=2 the arm was handing the generator ',., 3. 59 kilometres. Heritage.,,.'
# -- two content words out of fifty-one, and mostly punctuation.
#
# So the original setting traded a bag of words for a bag of punctuation. The
# upstream default is not free of the problem it was meant to solve -- its
# output is still unpunctuated -- but it spends the whole budget on content.
# Measured effect of the change, same 274 queries, same budgets:
# EM 0.0550 -> 0.0608, F1 0.0903 -> 0.0970. Real, and small.
#
# WHAT PROMPTED THE CHECK WAS A MISREADING, recorded here because the first
# version of this comment asserted it as fact. The claim was "the arm scores
# below the nocontext floor." It does not, and never did. `nocontext` is
# reported over all 300 sampled queries; every other arm is reported over the
# 274 that survive the memorization filter. Those are different populations,
# and the filter is *defined* as dropping the queries the generator can answer
# with no context -- so on the 274 actually studied, nocontext EM is 0.0000 by
# construction. The comparison that looked alarming (0.055 against 0.087) was
# reading one arm's mean against another arm's population.
#
# Anyone aggregating these arms should take that as the standing warning: the
# nocontext row is not commensurable with the rest of the grid as written, and
# any table putting them in the same column needs the floor recomputed on the
# studied qids.
#
# The change below is therefore justified on the content-word measurement
# alone, which is reproducible and stands on its own. It is not justified by
# the anomaly it was originally credited with fixing, because there was no
# anomaly. See commit df8ab6b for the version of this reasoning that was wrong.
#
# Aggregates from before this change are tracked in
# results/main_hotpotqa/_pre_llmlingua2_fix/ (arm_summary.csv and
# llmlingua2_deviation.json), so the deviation is auditable from the repo
# alone. The raw generations.csv beside them is gitignored per plan Sec. 6 and
# is reproducible from the generation cache. Do not pool pre-change results
# with post-change ones.
FORCE_TOKENS = ["\n", "?"]

_COMPRESSORS: dict[tuple[str, str], Any] = {}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load(model_name: str, device: str) -> Any:
    device = _resolve_device(device)
    key = (model_name, device)
    if key not in _COMPRESSORS:
        from llmlingua import PromptCompressor

        _COMPRESSORS[key] = PromptCompressor(
            model_name=model_name, use_llmlingua2=True, device_map=device
        )
    return _COMPRESSORS[key]


def unload_all() -> None:
    if not _COMPRESSORS:
        return
    _COMPRESSORS.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class LLMLingua2(Pruner):
    name = "llmlingua2"
    # Spends the budget as a compression rate. See the module docstring.
    budget_is_keep_count = False

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        min_rate: float = 0.05,
    ) -> None:
        self.model = model
        self.device = device
        self.min_rate = min_rate
        # Keyed on (hash of the chunk's text, rate). Compression depends on the
        # chunk's own text and the rate and on nothing else, which is exactly
        # why the key must BE the text.
        #
        # This was previously keyed on (chunk idx, rate), which is wrong in a way
        # that is invisible on a single query and silently corrupts every run
        # longer than one: `run.py` is arm-major, so one instance serves every
        # query, and idx 0..9 with rate k/10 gives only ~30 distinct keys for the
        # whole arm. Query 2 onwards therefore received query 1's compressed
        # passages. See ANALYSIS_PLAN Sec. 9, 2026-09-01.
        self._cache: dict[tuple[str, float], str] = {}

    def rate_for(self, budget: int, n_chunks: int) -> float:
        """Budget k over n chunks -> per-chunk rate k/n, so total tokens land
        near what a keep-k arm spends."""
        if n_chunks <= 0:
            return 1.0
        return max(self.min_rate, min(1.0, budget / n_chunks))

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        # Every chunk survives; the budget is spent on compression instead.
        # validate_selection is called against the full count, not the budget,
        # because this arm is explicitly not keep-k matched.
        kept = [c.idx for c in sorted(chunks, key=lambda c: c.rank)]
        return validate_selection(kept, chunks, len(chunks))

    #: Runs its own compressor rather than the study generator, so this work is
    #: in no cache and is re-paid on every restart (~24 min on the 2Wiki config).
    wants_cache = True

    def _compress(self, text: str, rate: float) -> str:
        # The chunk index is deliberately NOT part of this key: it does not
        # affect the output, and using it made two different chunks that happen
        # to share a slot collide.
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = (text_hash, round(rate, 4))
        if key in self._cache:
            return self._cache[key]
        if not text.strip():
            self._cache[key] = text
            return text

        # Disk cache. Same key material as the in-process one plus the model and
        # the forced tokens, both of which change the output.
        disk = getattr(self, "_disk_cache", None)
        dkey = None
        if disk is not None:
            dkey = artifact_key("llmlingua2", {"model": self.model, "rate": round(rate, 4),
                                               "force_tokens": FORCE_TOKENS, "text": text_hash})
            hit = disk.get(dkey)
            if hit is not None:
                self._cache[key] = str(hit)
                return self._cache[key]

        compressor = _load(self.model, self.device)
        out = compressor.compress_prompt(
            text,
            rate=rate,
            force_tokens=FORCE_TOKENS,
            # One chunk at a time, so there is no cross-chunk filtering to be
            # order-dependent about. This is the whole point.
            use_context_level_filter=False,
        )
        self._cache[key] = str(out["compressed_prompt"])
        if disk is not None and dkey is not None:
            disk.put(dkey, "llmlingua2", self._cache[key])
        return self._cache[key]

    def rewrite(
        self, query: str, chunks: Sequence[Chunk], budget: int
    ) -> list[Chunk]:
        if not chunks:
            return []
        rate = self.rate_for(budget, len(chunks))
        out = [
            Chunk(
                idx=c.idx,
                title=c.title,
                text=self._compress(c.text, rate),
                rank=c.rank,
                is_gold=c.is_gold,
                meta={**c.meta, "llmlingua2_rate": rate},
            )
            for c in chunks
        ]
        return validate_rewrite(chunks, out)

    def close(self) -> None:
        self._cache.clear()
        unload_all()
