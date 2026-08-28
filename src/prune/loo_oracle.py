"""Causal ceiling: keep the k chunks with the largest leave-one-out log-prob drop.

Needs `Generator.score` -- the log-probability of the reference answer sequence,
not a string match. This is the reason the primary generator is local (plan
Sec. 4.2); hosted APIs generally will not give you this.

Cost: 300 queries x 10 chunks x 5 permutations = 15,000 scored forward passes, no
decoding. Cheaper than the main generation grid.

Known limitation, worth stating in the write-up: LOO is additive by construction
and will misvalue complementary evidence. CUE-R (arXiv 2604.05467) found
non-additive interaction in ~20% of the HotpotQA examples it tested, with ~14%
fully complementary -- neither single removal hurt, joint removal broke the
answer. So this is a strong ceiling, not a true optimum.

STATUS: stubbed (week 3).
"""

from __future__ import annotations

from typing import Sequence

from ..chunks import Chunk
from .base import Pruner


class LOOOracle(Pruner):
    name = "loo_oracle"

    def __init__(self, generator=None) -> None:
        self.generator = generator

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        raise NotImplementedError("LOO oracle not implemented (week 3)")
