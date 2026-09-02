"""The two-level bootstrap.

Sec. 4.6 calls this out as the single easiest place to manufacture a fake result:
permutations are nested within queries, and resampling the P x N cells
independently inflates n by 5x. The first test here is the one that matters.
"""

import numpy as np
import pytest

from src.metrics import placebo_gap
from src.stats import BootstrapResult, holm, two_level_bootstrap


def _nested_scores(n_queries=40, n_perms=5, seed=0):
    """Strong between-query variation, small within-query variation.

    This is the structure that punishes naive flattening: most of the spread
    lives between queries, so treating permutations as independent observations
    massively understates the true uncertainty.
    """
    rng = np.random.default_rng(seed)
    a, b = {}, {}
    for i in range(n_queries):
        level_a = rng.normal(0.6, 0.25)
        level_b = rng.normal(0.5, 0.25)
        a[f"q{i}"] = list(level_a + rng.normal(0, 0.01, n_perms))
        b[f"q{i}"] = list(level_b + rng.normal(0, 0.01, n_perms))
    return {"m": a, "placebo_pos": b}


def test_bootstrap_recovers_the_point_estimate():
    scores = _nested_scores()
    res = two_level_bootstrap(
        scores, lambda s: placebo_gap(s, "m"), n_replicates=400, seed=1
    )
    assert res.point == pytest.approx(placebo_gap(scores, "m"))
    assert res.lo < res.point < res.hi


def test_nesting_is_not_flattened():
    """The regression guard for the 5x-n inflation warned about in Sec. 4.6.

    A bootstrap that resampled permutations independently would produce a CI
    roughly sqrt(P) times too narrow. Compare against that wrong answer directly.
    """
    scores = _nested_scores(n_queries=40, n_perms=5)
    correct = two_level_bootstrap(
        scores, lambda s: placebo_gap(s, "m"), n_replicates=600, seed=1
    )
    correct_width = correct.hi - correct.lo

    # The mistake, made explicit: every (query, permutation) cell as its own unit.
    flat = {
        arm: {f"{q}_{i}": [v] for q, vals in per_q.items() for i, v in enumerate(vals)}
        for arm, per_q in scores.items()
    }
    wrong = two_level_bootstrap(
        flat, lambda s: placebo_gap(s, "m"), n_replicates=600, seed=1
    )
    wrong_width = wrong.hi - wrong.lo

    assert correct_width > wrong_width * 1.5, (
        f"two-level CI ({correct_width:.4f}) is not meaningfully wider than the "
        f"flattened one ({wrong_width:.4f}); nesting is being ignored"
    )


def test_repeated_queries_are_counted_twice():
    """A query drawn twice must contribute twice, not collapse to one dict key."""
    scores = {"m": {"q1": [1.0], "q2": [0.0]}, "placebo_pos": {"q1": [0.0], "q2": [0.0]}}
    res = two_level_bootstrap(
        scores, lambda s: placebo_gap(s, "m"), n_replicates=500, seed=3
    )
    # Draws are {q1,q1}, {q1,q2}, {q2,q1}, {q2,q2} -> gaps 1.0, 0.5, 0.5, 0.0.
    assert set(np.unique(np.round(res.replicates, 6))) <= {0.0, 0.5, 1.0}
    assert res.replicates.max() == pytest.approx(1.0)


def test_bootstrap_is_deterministic_under_seed():
    scores = _nested_scores()
    kwargs = dict(n_replicates=200, seed=42)
    a = two_level_bootstrap(scores, lambda s: placebo_gap(s, "m"), **kwargs)
    b = two_level_bootstrap(scores, lambda s: placebo_gap(s, "m"), **kwargs)
    assert np.array_equal(a.replicates, b.replicates)


def test_excludes_zero_flags_a_real_effect():
    scores = _nested_scores()
    res = two_level_bootstrap(
        scores, lambda s: placebo_gap(s, "m"), n_replicates=400, seed=1
    )
    assert res.excludes_zero


def test_holm_is_step_down_and_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.08)
    assert adj["c"] == pytest.approx(0.2)
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_holm_caps_at_one():
    assert all(v <= 1.0 for v in holm({"a": 0.6, "b": 0.7, "c": 0.9}).values())


def _mean_of(arm):
    """Deliberately *not* internally paired, unlike the metrics in metrics.py:
    it is the shape of statistic that exposes a mismatched population."""
    def stat(scores):
        return float(np.mean([np.mean(v) for v in scores[arm].values()]))

    return stat


def test_point_estimate_uses_the_same_population_as_the_replicates():
    """Drawing the point from the unrestricted scores while resampling only the
    shared queries lets the point land outside its own CI."""
    scores = {
        "m": {f"q{i}": [1.0] * 3 for i in range(10)},
        "b": {f"q{i}": [1.0] * 3 for i in range(10)},
    }
    scores["m"]["only_in_m"] = [0.0] * 3  # unshared: never resampled

    res = two_level_bootstrap(scores, _mean_of("m"), n_replicates=200, seed=1)
    assert res.point == pytest.approx(1.0)
    assert res.lo <= res.point <= res.hi


def test_bootstrap_rejects_empty_scores():
    with pytest.raises(ValueError, match="no arms"):
        two_level_bootstrap({}, _mean_of("m"), n_replicates=10)


def test_p_value_ignores_non_finite_replicates():
    """nan compares False against both <= 0 and >= 0, so counting it in the
    denominator would let it drag the p-value down without ever voting."""
    from src.stats import BootstrapResult

    reps = np.array([1.0, 1.0, 1.0, np.nan])
    res = BootstrapResult(point=1.0, lo=1.0, hi=1.0, ci=0.95, replicates=reps)
    assert res.p_two_sided() == pytest.approx(2 / 3)


def test_excludes_zero_ignores_floating_point_noise():
    """A bound of 1.6e-17 is cancellation error, not evidence.

    Regression for a flag that reached a published artifact: the 27B
    replication's RQ1 for `rerank_topk` at k=5 came back lo = 1.5618e-17 and was
    reported as excluding zero. Every real interval in the study clears the
    tolerance by ten orders of magnitude, so the guard cannot suppress a finding.
    """
    noise = BootstrapResult(point=0.0161, lo=1.5618634624107305e-17, hi=0.03997,
                            ci=0.95, replicates=np.array([0.01]))
    assert not noise.excludes_zero

    real = BootstrapResult(point=0.2760, lo=0.2223, hi=0.3297,
                           ci=0.95, replicates=np.array([0.27]))
    assert real.excludes_zero

    # The nearest real interval the study reports as excluding zero.
    nearest = BootstrapResult(point=0.02, lo=0.0078, hi=0.05,
                              ci=0.95, replicates=np.array([0.02]))
    assert nearest.excludes_zero

    # Symmetric on the negative side.
    neg = BootstrapResult(point=-0.02, lo=-0.05, hi=-1e-17,
                          ci=0.95, replicates=np.array([-0.02]))
    assert not neg.excludes_zero
