"""The week-1 gate must fire correctly in both directions."""

import pytest

from src.gate import KILL_THRESHOLD, format_report, gate_report


def _rows(per_query_scores, arm="full"):
    rows = []
    for qid, scores in per_query_scores.items():
        for i, s in enumerate(scores):
            rows.append({"qid": qid, "arm": arm, "f1": str(s), "em": str(float(s > 0.5))})
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
