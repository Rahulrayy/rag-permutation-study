"""Arm contracts. Every pruner obeys the same budget rules or the
matched-keep-count comparison against the placebo is meaningless.
"""

import pytest

from src.prune import expand_arms, get_pruner, registered_arms
from src.prune.base import validate_selection

IMPLEMENTED = ["full", "nocontext", "random_drop", "placebo_pos"]


def test_all_nine_arms_are_registered():
    assert set(registered_arms()) == {
        "full", "nocontext", "rerank_topk", "provence", "llmlingua2",
        "llm_pruner", "random_drop", "placebo_pos", "loo_oracle",
    }


@pytest.mark.parametrize("arm", ["random_drop", "placebo_pos"])
@pytest.mark.parametrize("budget", [2, 3, 5])
def test_selection_respects_budget(arm, budget, chunks):
    selected = get_pruner(arm).select("q", chunks, budget)
    assert len(selected) == budget
    assert len(set(selected)) == budget
    assert set(selected) <= {c.idx for c in chunks}


@pytest.mark.parametrize("strategy", ["middle_first", "edges_first", "tail_first"])
def test_placebo_ignores_content(strategy, chunks):
    """The placebo must not look at the query or the text. That is the point:
    it isolates positional promotion from content selection."""
    pruner = get_pruner("placebo_pos", strategy=strategy)
    a = pruner.select("who founded Rome?", chunks, 3)
    b = pruner.select("what colour is the sky?", chunks, 3)
    assert a == b


def test_placebo_strategies_differ(chunks):
    picks = {
        s: tuple(get_pruner("placebo_pos", strategy=s).select("q", chunks, 3))
        for s in ("middle_first", "edges_first", "tail_first")
    }
    assert len(set(picks.values())) == 3


def test_middle_first_keeps_the_edges(chunks):
    kept = get_pruner("placebo_pos", strategy="middle_first").select("q", chunks, 4)
    assert 0 in kept and 9 in kept


def test_edges_first_keeps_the_middle(chunks):
    kept = get_pruner("placebo_pos", strategy="edges_first").select("q", chunks, 4)
    assert 0 not in kept and 9 not in kept


def test_tail_first_is_truncation(chunks):
    assert get_pruner("placebo_pos", strategy="tail_first").select("q", chunks, 4) == [0, 1, 2, 3]


def test_random_drop_is_seeded(chunks):
    a = get_pruner("random_drop", seed=1).select("q", chunks, 3)
    b = get_pruner("random_drop", seed=1).select("q", chunks, 3)
    assert a == b


def test_random_drop_varies_by_query(chunks):
    p = get_pruner("random_drop", seed=1)
    assert p.select("q1", chunks, 3) != p.select("q2", chunks, 3)


def test_full_keeps_everything(chunks):
    assert get_pruner("full").select("q", chunks, 3) == list(range(10))


def test_nocontext_keeps_nothing(chunks):
    assert get_pruner("nocontext").select("q", chunks, 3) == []


def test_validate_rejects_over_budget(chunks):
    with pytest.raises(ValueError, match="budget"):
        validate_selection([0, 1, 2, 3], chunks, 3)


def test_validate_rejects_duplicates(chunks):
    with pytest.raises(ValueError, match="duplicate"):
        validate_selection([0, 0, 1], chunks, 3)


def test_unimplemented_arms_raise_not_implemented(chunks):
    for arm in set(registered_arms()) - set(IMPLEMENTED):
        with pytest.raises(NotImplementedError):
            get_pruner(arm).select("q", chunks, 3)


def test_unknown_arm_raises():
    with pytest.raises(KeyError):
        get_pruner("magic_pruner")


def test_bare_arm_expands_to_its_variants():
    """A config writing `placebo_pos` means all three positional strategies,
    each as its own arm -- they test three different hypotheses."""
    assert expand_arms(["full", "placebo_pos"]) == [
        "full",
        "placebo_pos:middle_first",
        "placebo_pos:edges_first",
        "placebo_pos:tail_first",
    ]


def test_explicit_variant_is_not_re_expanded():
    assert expand_arms(["placebo_pos:tail_first"]) == ["placebo_pos:tail_first"]


def test_variant_suffix_selects_the_strategy(chunks):
    suffixed = get_pruner("placebo_pos:tail_first").select("q", chunks, 4)
    kwarg = get_pruner("placebo_pos", strategy="tail_first").select("q", chunks, 4)
    assert suffixed == kwarg


def test_unknown_variant_fails_at_construction(chunks):
    """A mistyped strategy must fail before the model loads, not hours in."""
    with pytest.raises(ValueError, match="unknown placebo strategy"):
        get_pruner("placebo_pos:sideways_first")


def test_variant_suffix_rejected_on_arms_without_variants():
    with pytest.raises(ValueError, match="variant suffix"):
        get_pruner("full:something")
