"""The positional placebo. RQ4, and the centerpiece of the study.

Drop the same *number* of chunks as a real pruner, chosen by **position** rather
than by content. If a published pruner's gain does not survive this control, the
gain was positional promotion, not evidence selection.

Random-drop baselines exist in the literature (REFRAG, RAP, OPRM). A
position-matched placebo does not, which is the whole opening. Note OPRM
(arXiv 2505.07793) found random chunk selection beating full-context inference on
HotpotQA long-context, so a dumb control outperforming expectations here would
not be unprecedented.

Three variants, run as separate cells and reported separately: which one wins is
itself informative about where this generator's position bias lives.
"""

from __future__ import annotations

from typing import Literal, Sequence

from ..chunks import Chunk
from .base import Pruner, validate_selection

Strategy = Literal["middle_first", "edges_first", "tail_first"]


def _drop_order(n: int, strategy: Strategy) -> list[int]:
    """Positions (0-indexed, in rank order) ranked by how early they get dropped."""
    positions = list(range(n))
    center = (n - 1) / 2.0
    if strategy == "middle_first":
        # Sacrifice the low-visibility middle: the shape a lost-in-the-middle-aware
        # pruner would produce by accident.
        return sorted(positions, key=lambda p: (abs(p - center), p))
    if strategy == "edges_first":
        # The adversarial mirror: keep the middle, throw away the visible slots.
        return sorted(positions, key=lambda p: (-abs(p - center), p))
    if strategy == "tail_first":
        # Plain truncation, i.e. what a retriever cutoff already does.
        return sorted(positions, key=lambda p: -p)
    raise ValueError(f"unknown placebo strategy: {strategy!r}")


class PlaceboPositional(Pruner):
    name = "placebo_pos"
    # Config writes `placebo_pos`, the runner expands it to three separate arms
    # named `placebo_pos:middle_first` and so on. See prune.base.expand_arms.
    variant_param = "strategy"
    variants = ("middle_first", "edges_first", "tail_first")

    def __init__(self, strategy: Strategy = "middle_first") -> None:
        # Validate here, not in `select`: a mistyped strategy in a config should
        # fail before the model loads, not partway through an overnight grid.
        if strategy not in self.variants:
            raise ValueError(
                f"unknown placebo strategy: {strategy!r}; "
                f"expected one of {self.variants}"
            )
        self.strategy: Strategy = strategy

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        # Deliberately ignores `query` and chunk content. That is the point.
        ordered = sorted(chunks, key=lambda c: c.rank)
        n = len(ordered)
        n_drop = max(0, n - budget)
        dropped = set(_drop_order(n, self.strategy)[:n_drop])
        kept = [c.idx for p, c in enumerate(ordered) if p not in dropped]
        return validate_selection(kept, chunks, budget)
