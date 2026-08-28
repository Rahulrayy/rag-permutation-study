"""Provence (arXiv 2501.16214): sentence-level pruning folded into the reranker.

Verify the checkpoint actually loads in **week 2, not week 5** (plan Sec. 8).
This is the assumption most likely to blow up late.

Being used out of distribution relative to its training data, like every
off-the-shelf checkpoint here. That is a real limitation and must be stated in
the write-up: the study measures deployed-as-published behaviour, not each
method ceiling.

STATUS: stubbed (week 2).
"""

from __future__ import annotations

from typing import Sequence

from ..chunks import Chunk
from .base import Pruner


class Provence(Pruner):
    name = "provence"

    def __init__(self, checkpoint: str = "naver/provence-reranker-debertav3-v1") -> None:
        self.checkpoint = checkpoint

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        raise NotImplementedError("Provence not implemented (week 2)")
