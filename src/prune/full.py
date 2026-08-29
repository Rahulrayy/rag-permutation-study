"""Upper reference: all chunks, nothing dropped.

Still runs under the full permutation protocol -- this arm is where RQ1 lives,
and where the week-1 kill criterion is measured.
"""

from __future__ import annotations

from typing import Sequence

from ..chunks import Chunk
from .base import Pruner


class Full(Pruner):
    name = "full"
    # Ignores the budget by construction, so it is not keep-k matched and must
    # not be pooled with the selection arms in a matched comparison.
    budget_is_keep_count = False

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        # Budget is ignored by construction; `full` is the no-pruning reference.
        return [c.idx for c in sorted(chunks, key=lambda c: c.rank)]
