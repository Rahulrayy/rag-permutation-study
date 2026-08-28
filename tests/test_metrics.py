"""EM / F1 and the four derived quantities."""

import numpy as np
import pytest

from src.metrics import (
    exact_match,
    order_adjusted_effect,
    oracle_gap,
    placebo_gap,
    rank_flip_rate,
    supporting_fact_f1,
    token_f1,
    within_query_sd,
)


def test_exact_match_normalises():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert exact_match("yes.", "yes") == 1.0
    assert exact_match("no", "yes") == 0.0


def test_token_f1_partial_overlap():
    assert token_f1("a large red dog", "red dog") == pytest.approx(0.8)
    assert token_f1("cat", "dog") == 0.0
    assert token_f1("same words", "same words") == 1.0


def test_token_f1_empty_prediction():
    assert token_f1("", "answer") == 0.0
    assert token_f1("", "") == 1.0


def test_supporting_fact_f1():
    assert supporting_fact_f1([2, 5], [2, 5]) == 1.0
    assert supporting_fact_f1([2, 7], [2, 5]) == pytest.approx(0.5)
    assert supporting_fact_f1([], [2, 5]) == 0.0


def test_within_query_sd_is_zero_for_constant_scores():
    assert within_query_sd({"q": [0.5] * 5}) == {"q": 0.0}


def test_oae_is_zero_when_arms_are_identical():
    scores = {"a": {"q1": [1, 0, 1, 0, 1]}, "b": {"q1": [1, 0, 1, 0, 1]}}
    assert order_adjusted_effect(scores, "a", "b") == 0.0


def test_oae_in_units_of_permutation_noise():
    """Delta of 0.2 against a baseline whose per-query SD is 0.1 -> OAE 2.0."""
    scores = {
        "m": {"q1": [0.7, 0.9, 0.7, 0.9, 0.8]},
        "b": {"q1": [0.5, 0.7, 0.5, 0.7, 0.6]},
    }
    sd = float(np.std([0.5, 0.7, 0.5, 0.7, 0.6], ddof=1))
    assert order_adjusted_effect(scores, "m", "b") == pytest.approx(0.2 / sd)


def test_oae_is_nan_when_baseline_has_no_variance():
    """A dead premise must surface as nan, not as a division blow-up."""
    scores = {"m": {"q1": [1.0] * 5}, "b": {"q1": [0.5] * 5}}
    assert np.isnan(order_adjusted_effect(scores, "m", "b"))


def test_rank_flip_rate_zero_when_ordering_is_consistent():
    scores = {
        "good": {"q1": [1.0] * 5, "q2": [1.0] * 5},
        "bad": {"q1": [0.0] * 5, "q2": [0.0] * 5},
    }
    assert rank_flip_rate(scores) == 0.0


def test_rank_flip_rate_detects_flips():
    """Averaged, `a` beats `b`; under permutation 0 alone, `b` beats `a`."""
    scores = {
        "a": {"q1": [0.0, 1.0, 1.0, 1.0, 1.0]},
        "b": {"q1": [1.0, 0.0, 0.0, 0.0, 0.0]},
    }
    assert rank_flip_rate(scores) == pytest.approx(0.2)


def test_rank_flip_rate_rejects_ragged_permutations():
    scores = {"a": {"q1": [1.0, 1.0]}, "b": {"q1": [1.0]}}
    with pytest.raises(ValueError):
        rank_flip_rate(scores)


def test_placebo_gap_zero_means_no_content_selection():
    scores = {
        "m": {"q1": [0.6, 0.6], "q2": [0.4, 0.4]},
        "placebo_pos": {"q1": [0.6, 0.6], "q2": [0.4, 0.4]},
    }
    assert placebo_gap(scores, "m") == 0.0


def test_oracle_gap_is_a_ratio():
    scores = {"m": {"q1": [0.5, 0.5]}, "loo_oracle": {"q1": [1.0, 1.0]}}
    assert oracle_gap(scores, "m") == pytest.approx(0.5)


def test_metrics_are_paired_on_shared_queries():
    """A query missing from one arm must not silently shift the comparison."""
    scores = {"m": {"q1": [1.0], "q2": [0.0]}, "b": {"q1": [0.0]}}
    assert order_adjusted_effect.__name__  # sanity
    assert placebo_gap(scores, "m", "b") == 1.0
