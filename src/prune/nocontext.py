"""Memorization control: question only, no passages.

Drives the filter in `data.memorization_filter`. Wikipedia-derived multi-hop
benchmarks leak into pretraining; without this arm you are measuring parametric
recall and calling it retrieval (plan Sec. 4.1).

Only one ordering exists for an empty context, so the runner collapses this arm
to P=1 rather than paying for five identical generations.
"""

from __future__ import annotations

from typing import Sequence

from ..chunks import Chunk
from .base import Pruner


class NoContext(Pruner):
    name = "nocontext"

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        return []
