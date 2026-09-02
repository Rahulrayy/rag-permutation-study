"""Two-level bootstrap and multiplicity correction.

Sec. 4.6, and the single easiest place to get a fake result. Permutations are
nested within queries. Resampling the P x N cells independently inflates n by 5x
and manufactures significance; the resampling unit is the **query**, and all P of
its permutations travel with it.

Every statistic in metrics.py is bootstrapped through the same machinery, so the
nesting is handled in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, ClassVar, Mapping, Sequence

import numpy as np

from .metrics import PerArmScores

Statistic = Callable[[PerArmScores], float]


@dataclass
class BootstrapResult:
    point: float
    lo: float
    hi: float
    ci: float
    replicates: np.ndarray

    def __repr__(self) -> str:  # readable in notebooks
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}] ({self.ci:.0%} CI)"

    #: Distances below this are floating-point noise, not evidence. A bound of
    #: 1.6e-17 is how a percentile interval reports "touches zero" when every
    #: replicate happens to land on the same side of it by cancellation error,
    #: and without a tolerance `excludes_zero` calls that significant. It found
    #: its way into a published artifact once -- the 27B replication's RQ1 for
    #: `rerank_topk` at k=5, lo = 1.56e-17 -- which is what this guards against.
    #: Every real interval in this study clears the tolerance by ten orders of
    #: magnitude: the next-closest of the 193 currently reported as excluding
    #: zero comes no nearer than 0.0078.
    ZERO_TOL: ClassVar[float] = 1e-12

    @property
    def excludes_zero(self) -> bool:
        return (self.lo > self.ZERO_TOL) or (self.hi < -self.ZERO_TOL)

    def p_two_sided(self) -> float:
        """Bootstrap p-value. Secondary presentation only -- CIs are primary.

        Computed on the finite replicates, matching the CI. A nan compares False
        against both <= 0 and >= 0, so counting it in the denominator would drag
        the p-value down without it ever having voted.
        """
        finite = self.replicates[np.isfinite(self.replicates)]
        n = len(finite)
        if n == 0:
            return 1.0
        prop = min((finite <= 0).sum() / n, (finite >= 0).sum() / n)
        return min(1.0, 2 * max(prop, 1.0 / n))


def _resample(
    scores: PerArmScores,
    query_ids: Sequence[str],
    rng: np.random.Generator,
) -> dict[str, dict[str, Sequence[float]]]:
    """One two-level draw: queries with replacement, permutations carried along.

    Repeated draws get distinct synthetic ids so a query sampled twice counts
    twice instead of silently collapsing to one dict entry.
    """
    picked = rng.integers(0, len(query_ids), size=len(query_ids))
    out: dict[str, dict[str, Sequence[float]]] = {arm: {} for arm in scores}
    for slot, i in enumerate(picked):
        qid = query_ids[i]
        new_id = f"{qid}#{slot}"
        for arm in scores:
            if qid in scores[arm]:
                out[arm][new_id] = scores[arm][qid]
    return out


def two_level_bootstrap(
    scores: PerArmScores,
    statistic: Statistic,
    n_replicates: int = 10_000,
    ci: float = 0.95,
    seed: int = 20260828,
) -> BootstrapResult:
    """Percentile CI for any statistic over the score container.

    ``statistic`` must accept a PerArmScores and return a scalar, so paired
    structure across arms is preserved automatically: a resampled query brings
    its rows in *every* arm.
    """
    if not scores:
        raise ValueError("no arms in scores")

    # Paired throughout: only queries observed in all arms are resampled.
    shared = sorted(set.intersection(*(set(scores[a]) for a in scores)))
    if not shared:
        raise ValueError("no queries shared across all arms")

    # The point estimate is computed on the same restricted population as the
    # replicates. Taking it from the unrestricted `scores` instead lets the two
    # be drawn from different query sets, and the point can then land outside
    # its own confidence interval -- which reads as a bug in the statistic
    # rather than in the plumbing.
    paired: PerArmScores = {
        arm: {q: scores[arm][q] for q in shared} for arm in scores
    }

    rng = np.random.default_rng(seed)
    reps = np.empty(n_replicates, dtype=float)
    for r in range(n_replicates):
        reps[r] = statistic(_resample(paired, shared, rng))

    alpha = (1.0 - ci) / 2.0
    finite = reps[np.isfinite(reps)]
    if finite.size == 0:
        raise ValueError("all bootstrap replicates were non-finite")
    return BootstrapResult(
        point=float(statistic(paired)),
        lo=float(np.quantile(finite, alpha)),
        hi=float(np.quantile(finite, 1.0 - alpha)),
        ci=ci,
        replicates=reps,
    )


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down, applied across the method-pair family.

    Returns adjusted p-values (monotone-enforced, capped at 1.0), keyed as input.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adjusted[key] = min(1.0, running)
    return adjusted


def pairwise_family(
    scores: PerArmScores,
    statistic_factory: Callable[[str, str], Statistic],
    arms: Sequence[str] | None = None,
    n_replicates: int = 10_000,
    ci: float = 0.95,
    seed: int = 20260828,
) -> dict[tuple[str, str], BootstrapResult]:
    """Bootstrap every method pair, for feeding into ``holm``."""
    arms = list(arms) if arms is not None else sorted(scores)
    return {
        (a, b): two_level_bootstrap(
            scores, statistic_factory(a, b), n_replicates, ci, seed
        )
        for a, b in combinations(arms, 2)
    }
