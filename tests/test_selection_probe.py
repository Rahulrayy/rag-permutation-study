"""Tests for the selection-stability probe.

The probe's expensive half needs a GPU and a checkpoint, so what is tested here
is everything around it: the Jaccard arithmetic, the two reference quantities the
observed number is read against, and the summary that turns per-query records
into the numbers the figure prints.
"""

from __future__ import annotations

from src.chunks import Chunk
from src.selection_probe import (
    _jaccard,
    _summarise,
    chance_jaccard,
    order_invariance,
)


def _chunks(n=10):
    return [Chunk(idx=i, title=f"t{i}", text=f"text {i}", rank=i) for i in range(n)]


class _IndependentScorer:
    """Scores each chunk on its own content, so presentation order cannot matter.

    Stands in for `rerank_topk` without loading a cross-encoder. The "score" is
    the chunk's own idx, which travels with the chunk through a permutation --
    the property that makes an independent scorer order-invariant.
    """

    def select(self, query, chunks, budget):
        ranked = sorted(chunks, key=lambda c: -c.idx)
        return sorted(c.idx for c in ranked[:budget])


class _PositionalScorer:
    """Keeps whatever it is shown first, so its selection is pure presentation."""

    def select(self, query, chunks, budget):
        return sorted(c.idx for c in chunks[:budget])


def test_jaccard_of_identical_sets_is_one():
    assert _jaccard([{1, 2, 3}, {1, 2, 3}, {1, 2, 3}]) == 1.0


def test_jaccard_of_disjoint_sets_is_zero():
    assert _jaccard([{1, 2}, {3, 4}, {5, 6}]) == 0.0


def test_jaccard_is_intersection_over_union():
    # intersection {2}, union {1,2,3,4}
    assert _jaccard([{1, 2}, {2, 3}, {2, 4}]) == 0.25


def test_chance_baseline_matches_the_documented_value():
    """Three random 3-subsets of 10 -- the number the observed Jaccard is read against."""
    chance = chance_jaccard(n_chunks=10, budget=3, orders=3, seed=20260828)
    assert 0.03 < chance["mean"] < 0.07, chance


def test_chance_rises_with_the_budget():
    """Sanity: bigger subsets of the same pool overlap more often."""
    small = chance_jaccard(10, 2, 3, seed=1)["mean"]
    large = chance_jaccard(10, 5, 3, seed=1)["mean"]
    assert large > small


def test_order_invariant_scorer_measures_as_invariant():
    """The upper reference has to be earned, so the probe must actually detect it."""
    result = order_invariance(
        _IndependentScorer(), "q", _chunks(), budget=3,
        orders=("rank", "reverse", "random"), seed=20260828,
    )
    assert result["jaccard"] == 1.0


def test_positional_scorer_is_caught():
    """And the probe must fail a selector whose choice is only presentation."""
    result = order_invariance(
        _PositionalScorer(), "q", _chunks(), budget=3,
        orders=("rank", "reverse", "random"), seed=20260828,
    )
    assert result["jaccard"] < 1.0


def test_summarise_counts_a_query_as_changed_below_one():
    records = [{"jaccard": 1.0}, {"jaccard": 0.5}, {"jaccard": 0.0}, {"jaccard": 1.0}]
    summary = _summarise(records)
    assert summary["n"] == 4
    assert summary["n_changed"] == 2
    assert summary["changed_fraction"] == 0.5
    assert summary["mean_jaccard"] == 0.625
