"""The pruner interface.

One interface for every arm. That is what makes adding a ninth arm a 40-line file
instead of a refactor (plan Sec. 6).

``select`` returns **indices**, never reordered chunks. Selection and ordering are
kept strictly separate everywhere in this codebase, because conflating them is the
error the whole study is about: removing chunks 3, 5 and 7 from a 10-chunk context
does not just remove content, it promotes 8, 9 and 10 into higher-visibility slots
(plan Sec. 3, premise 3). Ordering is applied afterwards by ``chunks.permute``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..chunks import Chunk

_REGISTRY: dict[str, type["Pruner"]] = {}


class Pruner(ABC):
    """Base class for every arm."""

    name: str = "base"

    @abstractmethod
    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        """Return the ``Chunk.idx`` values to keep.

        Contract, enforced by ``validate_selection``:
          - at most ``budget`` indices;
          - all drawn from ``chunks``;
          - no duplicates.

        The returned *order* carries no meaning and must not be relied on. The
        permutation protocol overwrites it.
        """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", "base") != "base":
            _REGISTRY[cls.name] = cls


def validate_selection(
    selected: Sequence[int],
    chunks: Sequence[Chunk],
    budget: int,
) -> list[int]:
    """Fail loudly at selection time rather than silently at analysis time.

    A pruner that quietly returns k+1 chunks breaks the matched-keep-count
    comparison against the placebo, which is the study's centerpiece.
    """
    available = {c.idx for c in chunks}
    selected = list(selected)
    if len(selected) > budget:
        raise ValueError(f"selected {len(selected)} chunks, budget is {budget}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"duplicate indices in selection: {selected}")
    unknown = [i for i in selected if i not in available]
    if unknown:
        raise ValueError(f"selected indices not in chunk set: {unknown}")
    return selected


def get_pruner(name: str, **kwargs: object) -> Pruner:
    """Look up an arm by config name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown arm {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)  # type: ignore[arg-type]


def registered_arms() -> list[str]:
    return sorted(_REGISTRY)
