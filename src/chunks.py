"""Chunk representation and seeded permutation.

Sec. 4.4 is the whole trick: for every (query, arm, budget) cell, generate under
P orderings of the *kept* chunks. Permutations must be reproducible from a seed
and stable across reruns, or the cache is worthless and the paired statistics
are unpaired.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

PermStrategy = Literal["rank", "reverse", "random"]


@dataclass(frozen=True)
class Chunk:
    """One retrieved passage. ``rank`` is its original retriever/dataset position."""

    idx: int
    title: str
    text: str
    rank: int
    is_gold: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def permute(
    chunks: Sequence[Chunk],
    strategy: PermStrategy,
    seed: int,
    replicate: int = 0,
) -> list[Chunk]:
    """Return chunks in the requested order. Pure; does not mutate the input.

    ``replicate`` distinguishes the three independent random permutations so
    they don't collapse to the same ordering under a shared seed.
    """
    items = list(chunks)
    if strategy == "rank":
        return sorted(items, key=lambda c: c.rank)
    if strategy == "reverse":
        return sorted(items, key=lambda c: c.rank, reverse=True)
    if strategy == "random":
        # str seeding goes through sha512: deterministic across runs and
        # unaffected by PYTHONHASHSEED, unlike hashing a tuple.
        rng = random.Random(f"{seed}:{replicate}:{len(items)}")
        shuffled = sorted(items, key=lambda c: c.rank)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"unknown permutation strategy: {strategy!r}")


def permutation_set(
    chunks: Sequence[Chunk],
    strategies: Sequence[PermStrategy],
    seed: int,
) -> list[list[Chunk]]:
    """The P orderings for one cell. Random replicates are numbered in order."""
    out: list[list[Chunk]] = []
    replicate = 0
    for strategy in strategies:
        out.append(permute(chunks, strategy, seed=seed, replicate=replicate))
        if strategy == "random":
            replicate += 1
    return out


def keep(chunks: Sequence[Chunk], indices: Sequence[int]) -> list[Chunk]:
    """Subset by ``Chunk.idx``, preserving the given order of ``indices``.

    Pruners return indices; this is where selection stops and ordering begins.
    Keeping the two separate is the point of the whole study (plan Sec. 3,
    premise 3): pruning changes positions as well as content.
    """
    by_idx = {c.idx: c for c in chunks}
    missing = [i for i in indices if i not in by_idx]
    if missing:
        raise KeyError(f"indices not present in chunk set: {missing}")
    return [by_idx[i] for i in indices]


def positional_bucket(position: int, n: int) -> Literal["begin", "middle", "end"]:
    """Coarse bucket. Sec. 3 premise 2: the effect lives here, not in fine distance."""
    if n <= 0:
        raise ValueError("n must be positive")
    if position == 0:
        return "begin"
    if position == n - 1:
        return "end"
    return "middle"
