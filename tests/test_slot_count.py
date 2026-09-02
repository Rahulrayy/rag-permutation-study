"""Tests for the slot-count analysis.

The analysis exists to separate two things that move together in the raw grid:
the number of permutable slots and the amount of evidence retained. What matters
here is that the stratification actually does that -- that `full` and
`llmlingua2` are credited with ten slots rather than their budget, that the
resampling unit is the question, and that a stratum too thin to estimate is
dropped rather than reported.
"""

from __future__ import annotations

import csv

import pytest

from src.slot_count import MIN_CELLS, _cells, _stratum_means, analyse

FIELDS = ["qid", "arm", "budget", "perm", "perm_strategy", "hop_type", "kept",
          "order", "gold_positions", "n_gold_kept", "context_chars",
          "prediction", "gold", "em", "f1"]


def _rows(spec):
    """spec: list of (qid, arm, budget, gold, [f1 per perm])."""
    out = []
    for qid, arm, budget, gold, f1s in spec:
        for p, f1 in enumerate(f1s):
            out.append({
                "qid": qid, "arm": arm, "budget": budget, "perm": str(p),
                "perm_strategy": "s", "hop_type": "bridge", "kept": "[]",
                "order": "[]", "gold_positions": "[]", "n_gold_kept": str(gold),
                "context_chars": "100", "prediction": "x", "gold": "x",
                "em": "0", "f1": str(f1),
            })
    return out


def test_rate_based_arms_are_credited_with_ten_slots_not_their_budget():
    """`full` and `llmlingua2` retain every chunk, so their budget is not a slot
    count. Reading the budget instead would put a ten-slot arm at k=2 and invert
    the very relationship being measured."""
    cells = _cells(_rows([
        ("q0", "full", "2", 2, [0.1, 0.5]),
        ("q0", "llmlingua2", "5", 2, [0.1, 0.5]),
        ("q0", "rerank_topk", "5", 2, [0.1, 0.5]),
    ]))
    slots = {c["arm"]: c["slots"] for c in cells}
    assert slots["full"] == 10
    assert slots["llmlingua2"] == 10
    assert slots["rerank_topk"] == 5


def test_nocontext_is_excluded():
    cells = _cells(_rows([
        ("q0", "nocontext", "0", 0, [0.1, 0.2]),
        ("q0", "full", "3", 2, [0.1, 0.2]),
    ]))
    assert {c["arm"] for c in cells} == {"full"}


def test_single_permutation_cells_are_dropped():
    """A cell with one permutation has no within-question SD to contribute."""
    cells = _cells(_rows([("q0", "full", "3", 2, [0.4])]))
    assert cells == []


def test_a_resampled_question_contributes_its_cells_twice():
    """Questions are the resampling unit, so a question drawn twice must weigh
    twice -- collapsing it to one would silently shrink the bootstrap's spread."""
    cells = _cells(_rows([
        ("q0", "rerank_topk", "3", 2, [0.0, 1.0]),   # sd ~ 0.707
        ("q1", "rerank_topk", "3", 2, [0.5, 0.5]),   # sd 0
    ]))
    both = _stratum_means(cells, ["q0", "q1"])[(3, 2)]
    twice = _stratum_means(cells, ["q0", "q0"])[(3, 2)]
    assert both == pytest.approx(0.35355, abs=1e-4)
    assert twice == pytest.approx(0.70711, abs=1e-4)


def test_thin_strata_are_dropped_rather_than_reported():
    spec = [(f"q{i}", "rerank_topk", "3", 2, [0.0, 1.0]) for i in range(MIN_CELLS + 5)]
    spec += [("qX", "rerank_topk", "2", 1, [0.0, 1.0])]        # a stratum of one
    out = analyse(_cells(_rows(spec)), n_replicates=50)
    assert "3|2" in out["table"]
    assert "2|1" not in out["table"]


def test_slot_contrast_is_reported_within_each_stratum(tmp_path):
    """Two slot counts at a fixed gold level, with the higher genuinely more
    variable: the contrast must be positive and exclude zero."""
    spec = []
    for i in range(40):
        spec.append((f"q{i}", "rerank_topk", "2", 2, [0.5, 0.5]))       # sd 0
        spec.append((f"q{i}", "rerank_topk", "5", 2, [0.0, 1.0]))       # sd 0.707
    out = analyse(_cells(_rows(spec)), n_replicates=200)
    c = out["slot_contrast_within_gold_stratum"]["gold=2"]
    assert c["from_slots"] == 2 and c["to_slots"] == 5
    assert c["point"] == pytest.approx(0.70711, abs=1e-4)
    assert c["excludes_zero"]
