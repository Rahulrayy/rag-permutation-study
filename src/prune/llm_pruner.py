"""Ask the generator which chunks to keep. Strong, expensive, common in practice.

The one arm where this study's thesis applies to the pruner itself
------------------------------------------------------------------
Every other arm scores chunks independently: Provence and the cross-encoder see
one passage at a time, so the order they are shown in cannot change what they
pick. This arm sees **all ten passages in one prompt**, which means its
*selection* -- not merely the answer that follows -- may be a function of the
ordering it was shown.

If that is true, an LLM pruner does not have "a" selection for a query at all;
it has a distribution over selections indexed by presentation order, and every
published evaluation of one has silently sampled a single draw from it.

So the presentation order is an explicit, recorded parameter rather than an
accident of implementation. ``selection_order`` defaults to ``rank`` (as-given),
which is what a normal implementation does implicitly. Varying it and measuring
how much the selection moves is a robustness check the study is uniquely set up
to run -- see ``selection_stability`` below.

**Measured, Qwen2.5-3B-Instruct, 20 HotpotQA queries, k=3, greedy**, selection
Jaccard across three presentations (as-given / reverse / random):

    perfectly order-invariant selector       1.000   (rerank_topk, provence:
                                                      they score independently)
    observed: llm_pruner                     0.263
    chance: three random 3-subsets of 10     0.048

**The selection changed in 19 of 20 queries.** 0.263 is well above chance, but
it is only ~23% of the way from re-drawing at random to being a function of the
content. Gold recall was 72% at k=3, against 90% for `rerank_topk` and 85% for
`provence_rerank`; the model also failed to name three passages in 10% of cells.

This is the study's own thesis landing on the pruner. A published LLM-pruner
number is one draw from a distribution over selections that its paper does not
mention -- and here the draw dominates the content.

Failure modes, both real
------------------------
**Over-selection.** The model returns more than ``budget`` indices.
``validate_selection`` would raise; instead the first ``budget`` valid ones are
taken, in the order the model gave them, and the event is counted.

**Under-selection.** It returns fewer than ``budget``, or nothing parseable.
This one matters more than it looks: the matched-keep-count comparison against
`placebo_pos` is the study's centerpiece, and an arm that quietly returns k-1
chunks is no longer matched. The deficit is filled deterministically from the
as-given order, and counted. A run where this fires often is reporting a
different arm than it claims to.

Both counters are exposed on ``.stats`` and should be reported, not buried:
"the LLM pruner failed to name k passages in X% of cells" is a finding about
deployed practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..chunks import Chunk, permute
from .base import Pruner, validate_selection

SELECTION_TEMPLATE = (
    "You are selecting which passages are needed to answer a question.\n\n"
    "{context}\n"
    "Question: {question}\n"
    "Select the {k} passages most useful for answering the question.\n"
    "Reply with only the passage numbers, separated by commas. "
    "Do not explain.\n"
    "Passages:"
)


@dataclass
class PrunerStats:
    """Counted, not swallowed. These belong in the write-up."""

    cells: int = 0
    over_selected: int = 0
    under_selected: int = 0
    unparseable: int = 0
    out_of_range: int = 0

    def as_dict(self) -> dict[str, Any]:
        d = {
            "cells": self.cells,
            "over_selected": self.over_selected,
            "under_selected": self.under_selected,
            "unparseable": self.unparseable,
            "out_of_range": self.out_of_range,
        }
        if self.cells:
            d["under_selected_rate"] = self.under_selected / self.cells
        return d


class LLMPruner(Pruner):
    name = "llm_pruner"
    #: run.py hands this arm the same cached generator the study runs on, so
    #: selection calls are cached and reruns are free like everything else.
    needs_generator = True

    def __init__(
        self,
        generator: Any = None,
        max_new_tokens: int = 64,
        selection_order: str = "rank",
        seed: int = 20260828,
    ) -> None:
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.selection_order = selection_order
        self.seed = seed
        self._params: Any = None
        self.stats = PrunerStats()
        self._cache: dict[tuple[str, int, tuple[int, ...]], list[int]] = {}

    def attach(self, generator: Any, params: Any, **run_state: Any) -> None:
        """Receive the run's generator and decode params (called by run.py).

        ``run_state`` carries the run's prompt template and permutation
        protocol. This arm uses neither: it builds its own SELECTION_TEMPLATE,
        and its presentation order is ``selection_order``, an explicit parameter
        of the arm rather than an inheritance from the generation grid. See the
        module docstring on why that order is a recorded variable here.
        """
        self.generator = generator
        # Selection needs more room than an answer does, but must stay greedy:
        # a sampled selection would put sampling noise inside the independent
        # variable, which is the one thing the design cannot tolerate.
        self._params = replace_decode(params, self.max_new_tokens)

    def _presented(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """The order the model is shown. Recorded, never incidental."""
        return permute(chunks, self.selection_order, seed=self.seed, key="selection")

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        if self.generator is None:
            raise RuntimeError(
                "llm_pruner has no generator; run.py attaches one via attach(). "
                "Constructing this arm standalone requires passing generator=."
            )
        key = (query, budget, tuple(c.idx for c in chunks))
        if key in self._cache:
            return self._cache[key]

        self.stats.cells += 1
        shown = self._presented(chunks)
        context = "\n\n".join(
            f"[{i + 1}] {c.title}: {c.text}" for i, c in enumerate(shown)
        )
        prompt = SELECTION_TEMPLATE.format(context=context, question=query, k=budget)
        text = self.generator.generate(prompt, self._params).text

        kept = self._parse(text, shown, budget)
        result = validate_selection(sorted(kept), chunks, budget)
        self._cache[key] = result
        return result

    def _parse(self, text: str, shown: Sequence[Chunk], budget: int) -> list[int]:
        """Map the model's reply onto chunk ids, repairing deterministically.

        The numbers refer to positions in the *presented* order, which is not
        chunk.idx order unless selection_order is "rank" -- mapping through
        `shown` is what keeps that straight.
        """
        raw = [int(n) for n in re.findall(r"\d+", text)]
        if not raw:
            self.stats.unparseable += 1

        picked: list[int] = []
        for n in raw:
            if not 1 <= n <= len(shown):
                self.stats.out_of_range += 1
                continue
            idx = shown[n - 1].idx
            if idx not in picked:
                picked.append(idx)

        if len(picked) > budget:
            self.stats.over_selected += 1
            picked = picked[:budget]

        if len(picked) < budget:
            self.stats.under_selected += 1
            # Fill from the as-given order, skipping what is already chosen.
            # Deterministic and content-blind, so the repair cannot smuggle in
            # selection signal the model did not provide.
            for c in sorted(shown, key=lambda c: c.rank):
                if len(picked) == budget:
                    break
                if c.idx not in picked:
                    picked.append(c.idx)

        return picked

    def close(self) -> None:
        # The generator is the run's, shared with every other arm. Clearing the
        # selection cache is this arm's business; unloading the model is not.
        self._cache.clear()


def replace_decode(params: Any, max_new_tokens: int) -> Any:
    """Copy decode params with a larger token budget, greedy preserved."""
    from dataclasses import replace

    return replace(params, max_new_tokens=max_new_tokens)


def selection_stability(
    pruner: "LLMPruner",
    query: str,
    chunks: Sequence[Chunk],
    budget: int,
    orders: Sequence[str] = ("rank", "reverse", "random"),
) -> dict[str, Any]:
    """How much does the *selection* move when the same chunks are reordered?

    The study's thesis, turned on the pruner. Returns the selection under each
    presentation order plus the size of their intersection: a Jaccard of 1.0
    means selection is order-invariant and this arm behaves like the
    independent scorers; anything lower means a published LLM-pruner result is
    one draw from a distribution its paper does not mention.

    Not called during a run -- this is the robustness check, run separately.
    """
    original = pruner.selection_order
    sels: dict[str, list[int]] = {}
    try:
        for order in orders:
            pruner.selection_order = order
            pruner._cache.clear()
            sels[order] = pruner.select(query, chunks, budget)
    finally:
        pruner.selection_order = original
        pruner._cache.clear()

    sets = [set(v) for v in sels.values()]
    inter = set.intersection(*sets)
    union = set.union(*sets)
    return {
        "selections": sels,
        "jaccard": len(inter) / len(union) if union else 1.0,
        "stable": len(sets[0]) == len(inter) == len(union),
    }
