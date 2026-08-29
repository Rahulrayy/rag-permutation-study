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


VARIANT_SEP = ":"


class Pruner(ABC):
    """Base class for every arm."""

    name: str = "base"

    #: Constructor keyword an ``arm:variant`` suffix binds to, or None if the
    #: arm has no variants. See ``get_pruner``.
    variant_param: str | None = None

    #: Allowed values for that keyword, checked at construction time so a typo
    #: in a config fails in milliseconds rather than hours into a grid.
    variants: tuple[str, ...] = ()

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

    def rewrite(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Optionally rewrite the *content* of the kept chunks. Identity by default.

        Selection, rewriting and ordering are three separate steps, and keeping
        them separate is the same discipline that keeps selection apart from
        ordering (see the module docstring). ``select`` answers *which* chunks,
        this answers *what text*, and ``chunks.permute`` answers *in what order*.

        Only compression methods override this -- Provence prunes sentences
        within a chunk, LLMLingua-2 compresses tokens. For every other arm the
        content is exactly what the dataset shipped, which is what makes a
        matched-keep-count comparison against `placebo_pos` a comparison of
        selection alone.

        Implementations must return the same chunks, in the same order, with
        the same ``idx`` values. Only ``text`` may change. ``validate_rewrite``
        enforces it.
        """
        return list(chunks)

    def close(self) -> None:
        """Release any model this pruner holds. Called once after selection.

        Provence is 1.74 GB and the generator is another 2.06 GB. On a 6 GB
        card that is the difference between fitting and not, and the pruner is
        finished with long before the first generation runs.
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


def parse_arm(name: str) -> tuple[str, str | None]:
    """Split ``"placebo_pos:middle_first"`` into ``("placebo_pos", "middle_first")``.

    Variants are separate arms in the grid, in the CSV, and in every downstream
    comparison -- which is what `placebo_pos`'s "run as separate cells and
    reported separately" requires. Collapsing them into one arm would average
    three different positional hypotheses into a single meaningless number.
    """
    base, sep, variant = name.partition(VARIANT_SEP)
    return base, (variant if sep else None)


def validate_rewrite(
    original: Sequence[Chunk],
    rewritten: Sequence[Chunk],
) -> list[Chunk]:
    """A rewrite may change text and nothing else.

    Silently dropping or reordering a chunk here would break the keep-count
    match against `placebo_pos` and quietly re-introduce the ordering effect
    the permutation protocol exists to control.
    """
    if len(rewritten) != len(original):
        raise ValueError(
            f"rewrite changed the chunk count: {len(original)} -> {len(rewritten)}"
        )
    before = [c.idx for c in original]
    after = [c.idx for c in rewritten]
    if before != after:
        raise ValueError(f"rewrite reordered or renumbered chunks: {before} -> {after}")
    return list(rewritten)


def get_pruner(name: str, **kwargs: object) -> Pruner:
    """Look up an arm by config name, resolving any ``arm:variant`` suffix."""
    base, variant = parse_arm(name)
    if base not in _REGISTRY:
        raise KeyError(f"unknown arm {base!r}; registered: {sorted(_REGISTRY)}")
    cls = _REGISTRY[base]
    if variant is not None:
        if not cls.variant_param:
            raise ValueError(
                f"arm {base!r} takes no {VARIANT_SEP}variant suffix, got {name!r}"
            )
        kwargs = {**kwargs, cls.variant_param: variant}
    return cls(**kwargs)  # type: ignore[arg-type]


def expand_arms(names: Sequence[str]) -> list[str]:
    """Expand a bare arm name into its variants; leave everything else alone.

    ``["full", "placebo_pos"]`` -> ``["full", "placebo_pos:middle_first", ...]``.
    An explicitly written ``placebo_pos:tail_first`` passes through untouched, so
    a config can pin one variant instead of taking all of them.
    """
    out: list[str] = []
    for name in names:
        base, variant = parse_arm(name)
        cls = _REGISTRY.get(base)
        if variant is None and cls is not None and cls.variants:
            out.extend(f"{base}{VARIANT_SEP}{v}" for v in cls.variants)
        else:
            out.append(name)
    return out


def registered_arms() -> list[str]:
    return sorted(_REGISTRY)
