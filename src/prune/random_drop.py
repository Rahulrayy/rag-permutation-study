"""Noise floor: drop k uniformly at random.

Seeded per query so the draw is reproducible and the cache stays warm across
reruns. Contrast with `placebo_pos`, which drops by position rather than at
random -- the difference between the two arms is the positional signal itself.
"""

from __future__ import annotations

import random
from typing import Sequence

from ..chunks import Chunk
from .base import Pruner, validate_selection


class RandomDrop(Pruner):
    name = "random_drop"

    def __init__(self, seed: int = 20260828) -> None:
        self.seed = seed

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        ordered = sorted(chunks, key=lambda c: c.rank)
        rng = random.Random(f"{self.seed}:{query}:{budget}")
        kept = rng.sample([c.idx for c in ordered], k=min(budget, len(ordered)))
        return validate_selection(kept, chunks, budget)
