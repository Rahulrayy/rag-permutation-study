"""The week-1 gate must fire correctly in both directions."""

import pytest

from src.gate import KILL_THRESHOLD, format_report, gate_report


def _rows(per_query_scores, arm="full", budget=10):
    rows = []
    for qid, scores in per_query_scores.items():
        for i, s in enumerate(scores):
            rows.append({
                "qid": qid, "arm": arm, "budget": str(budget),
                "f1": str(s), "em": str(float(s > 0.5)),
            })
    return rows


def test_gate_fails_on_zero_variance():
    """Identical scores across permutations = the premise is dead."""
    stats = gate_report(_rows({f"q{i}": [0.5] * 5 for i in range(20)}))
    assert stats["median_within_query_sd"] == 0.0
    assert stats["frac_queries_zero_sd"] == 1.0
    assert "FAIL" in format_report(stats, "f1", "full")


def test_gate_passes_on_real_variance():
    stats = gate_report(_rows({f"q{i}": [0.0, 1.0, 0.0, 1.0, 0.5] for i in range(20)}))
    assert stats["median_within_query_sd"] >= KILL_THRESHOLD
    assert "PASS" in format_report(stats, "f1", "full")


def test_gate_reports_escalation_ladder_on_failure():
    report = format_report(gate_report(_rows({f"q{i}": [0.4] * 5 for i in range(5)})), "f1", "full")
    assert "MuSiQue" in report and "20 instead of 10" in report


def test_gate_warns_at_ceiling():
    """Near-perfect scores usually mean memorization, not stability."""
    report = format_report(gate_report(_rows({f"q{i}": [0.95] * 5 for i in range(5)})), "f1", "full")
    assert "parametric recall" in report


def test_gate_rejects_single_permutation():
    with pytest.raises(ValueError, match="one permutation"):
        gate_report(_rows({"q0": [0.5]}))


def test_gate_rejects_ragged_permutations():
    with pytest.raises(ValueError, match="ragged"):
        gate_report(_rows({"q0": [0.5, 0.6], "q1": [0.5]}))


def test_gate_rejects_unknown_arm():
    with pytest.raises(ValueError, match="no rows"):
        gate_report(_rows({"q0": [0.1, 0.2]}), arm="nonexistent")


def test_gate_refuses_to_pool_budgets():
    """Grouping by (arm, query) alone turns the `full` arm's three keep-k cells
    into fifteen values per query that are not fifteen permutations."""
    rows = (
        _rows({f"q{i}": [0.0, 1.0] for i in range(5)}, budget=3)
        + _rows({f"q{i}": [0.0, 1.0] for i in range(5)}, budget=5)
    )
    with pytest.raises(ValueError, match="span budgets"):
        gate_report(rows)


def test_gate_measures_one_budget_at_a_time():
    rows = (
        _rows({f"q{i}": [0.5] * 5 for i in range(5)}, budget=3)
        + _rows({f"q{i}": [0.0, 1.0, 0.0, 1.0, 0.5] for i in range(5)}, budget=5)
    )
    assert gate_report(rows, budget=3)["median_within_query_sd"] == 0.0
    assert gate_report(rows, budget=5)["median_within_query_sd"] > 0.0
    assert "budget=5" in format_report(gate_report(rows, budget=5), "f1", "full", 5)
