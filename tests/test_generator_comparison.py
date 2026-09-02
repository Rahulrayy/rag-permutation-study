"""Tests for the matched 3B-vs-27B comparison.

The comparison's whole claim is that only the generator differs between the two
runs. That rests on one invariant -- the two CSVs present identical passage
orders for every shared cell -- and on the restriction to shared permutations
and shared queries actually happening. Those are what is tested here; the
bootstrap underneath is covered by test_stats.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src import generator_comparison as mgc

FIELDS = ["qid", "arm", "budget", "perm", "perm_strategy", "hop_type", "kept",
          "order", "gold_positions", "n_gold_kept", "context_chars",
          "prediction", "gold", "em", "f1"]

ARMS = ("full", "rerank_topk", "llm_pruner", "placebo_pos:middle_first")


def _write(path: Path, qids, n_perms, order_for=lambda q, p: f"[{p},0]", offset=0.0):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for q in qids:
            for arm in ARMS:
                for p in range(n_perms):
                    score = 0.4 + 0.1 * p + offset
                    w.writerow({
                        "qid": q, "arm": arm, "budget": "3", "perm": p,
                        "perm_strategy": "seeded", "hop_type": "bridge",
                        "kept": "[]", "order": order_for(q, p),
                        "gold_positions": "[]", "n_gold_kept": 1,
                        "context_chars": 100, "prediction": f"a{p}", "gold": "a0",
                        "em": 0, "f1": round(score, 4),
                    })
    return path


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """Two runs sharing queries q0-q4; the main run has 5 perms and 2 extra queries."""
    main = _write(tmp_path / "main.csv", [f"q{i}" for i in range(7)], 5)
    repl = _write(tmp_path / "repl.csv", [f"q{i}" for i in range(5)], 3, offset=0.05)
    monkeypatch.setattr(mgc, "MAIN", main)
    monkeypatch.setattr(mgc, "REPL", repl)
    monkeypatch.setattr(mgc, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(mgc, "FIG", tmp_path / "figures" / "f.png")
    return main, repl


def test_restricts_to_shared_queries_and_shared_perms(runs):
    out = mgc.main(n_replicates=50)
    assert out["n_shared_queries"] == 5
    assert out["n_3B_population"] == 7
    assert out["n_27B_population"] == 5
    assert out["perms"] == ["0", "1", "2"]
    # 5 queries x 3 perms, checked on the `full` arm.
    assert out["order_cells_verified_identical"] == 15


def test_mismatched_orders_raise_rather_than_report(runs):
    """The one failure that must never be reported as a result.

    If the two runs present different passage orders, the difference between
    them is no longer attributable to the generator, and a number computed
    anyway would look exactly like a real finding.
    """
    main, _ = runs
    rows = list(csv.DictReader(open(main, encoding="utf-8")))
    for r in rows:
        if r["arm"] == "full" and r["perm"] == "1":
            r["order"] = "[9,9]"
    with open(main, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(AssertionError, match="not order-matched"):
        mgc.main(n_replicates=50)


def test_paired_difference_is_reported_for_every_arm(runs):
    out = mgc.main(n_replicates=200)
    for arm in ARMS:
        d = out["arms"][arm]["paired_difference"]
        assert set(d) == {"point", "lo", "hi", "excludes_zero", "p"}
        assert d["lo"] <= d["point"] <= d["hi"]
    # Both runs were built with the same per-perm score steps, so their SDs
    # match and the paired difference is zero by construction -- a guard that
    # the restriction is comparing like with like rather than manufacturing a
    # gap out of the constant offset between the two runs' score levels.
    assert out["arms"]["full"]["paired_difference"]["point"] == pytest.approx(0.0, abs=1e-9)


def test_writes_its_artifacts(runs):
    mgc.main(n_replicates=50)
    assert mgc.OUT.exists()
    assert mgc.FIG.exists()
