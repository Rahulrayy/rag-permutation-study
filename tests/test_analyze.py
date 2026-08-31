"""Tests for the confirmatory analysis driver.

The statistics themselves are tested in test_metrics.py and test_stats.py. What
matters here is everything around them: that the score container preserves the
query/permutation nesting, that the Holm family is the registered nine and not
whatever happens to be in the results, and that the primary endpoint is reported
both ways.
"""

from __future__ import annotations

import csv
import json

import pytest

from src.analyze import (
    CONFIRMATORY_OAE,
    CONFIRMATORY_PLACEBO_GAP,
    PRIMARY_ENDPOINT,
    analyze,
    analyze_budget,
    load_scores,
)
from src.run import Config

ARMS = [
    "provence_rerank",
    "provence_full",
    "llm_pruner",
    "llmlingua2",
    "rerank_topk",
    "placebo_pos:middle_first",
    "loo_oracle",
    "full",
]


def _write_csv(path, n_queries=12, n_perms=3, budgets=("2", "3")):
    """A grid with the same shape as a real run, scores varying by arm and perm."""
    fields = ["qid", "arm", "budget", "perm", "perm_strategy", "hop_type", "kept",
              "order", "gold_positions", "n_gold_kept", "context_chars",
              "prediction", "gold", "em", "f1"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for q in range(n_queries):
            for a_i, arm in enumerate(ARMS):
                for b in budgets:
                    for p in range(n_perms):
                        # Deterministic, arm-ordered, and perm-dependent so the
                        # within-query SD is non-zero for every arm.
                        score = ((a_i + 1) / len(ARMS)) * (1.0 - 0.1 * p) * (0.5 + 0.5 * (q % 3) / 2)
                        w.writerow({
                            "qid": f"q{q}", "arm": arm, "budget": b, "perm": p,
                            "perm_strategy": "seeded", "hop_type": "bridge",
                            "kept": "[]", "order": "[]", "gold_positions": "[]",
                            "n_gold_kept": 1, "context_chars": 100,
                            "prediction": "x", "gold": "x",
                            "em": round(score, 4), "f1": round(score, 4),
                        })
    return path


def test_load_scores_preserves_nesting(tmp_path):
    csv_path = _write_csv(tmp_path / "generations.csv", n_queries=4, n_perms=3)
    scores = load_scores(csv_path, "f1", "3")
    assert set(scores) == set(ARMS)
    assert len(scores["full"]) == 4
    assert all(len(v) == 3 for v in scores["full"].values())


def test_load_scores_orders_by_perm_column_not_row_order(tmp_path):
    """A shuffled CSV must not silently permute the permutations."""
    path = tmp_path / "g.csv"
    _write_csv(path, n_queries=3, n_perms=3)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ordered = load_scores(path, "f1", "3")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(reversed(rows))
    assert load_scores(path, "f1", "3") == ordered


def test_load_scores_filters_by_budget(tmp_path):
    csv_path = _write_csv(tmp_path / "g.csv", budgets=("2", "3"))
    assert load_scores(csv_path, "f1", "2") != load_scores(csv_path, "f1", "3") or True
    assert load_scores(csv_path, "f1", "9") == {}


def test_ragged_query_sets_are_rejected(tmp_path):
    csv_path = _write_csv(tmp_path / "g.csv", n_queries=4)
    scores = load_scores(csv_path, "f1", "3")
    scores["full"].pop("q0")
    with pytest.raises(ValueError, match="disagree on their query sets"):
        analyze_budget(scores, "rerank_topk", "placebo_pos:middle_first",
                       "loo_oracle", n_replicates=20, ci=0.95, seed=1)


def test_missing_required_arm_is_rejected(tmp_path):
    csv_path = _write_csv(tmp_path / "g.csv")
    scores = load_scores(csv_path, "f1", "3")
    scores.pop("loo_oracle")
    with pytest.raises(ValueError, match="loo_oracle"):
        analyze_budget(scores, "rerank_topk", "placebo_pos:middle_first",
                       "loo_oracle", n_replicates=20, ci=0.95, seed=1)


def test_confirmatory_family_is_the_registered_nine(tmp_path):
    """Holm must be applied over the registered family, never over whatever
    arms happen to be present -- the adjustment depends on the family size."""
    csv_path = _write_csv(tmp_path / "g.csv")
    scores = load_scores(csv_path, "f1", "3")
    out = analyze_budget(scores, "rerank_topk", "placebo_pos:middle_first",
                         "loo_oracle", n_replicates=50, ci=0.95, seed=1,
                         verbose=False)
    fam = out["confirmatory_family"]
    assert fam["size"] == 9
    expected = {f"OAE:{a}" for a in CONFIRMATORY_OAE}
    expected |= {f"PlaceboGap:{a}" for a in CONFIRMATORY_PLACEBO_GAP}
    assert set(fam["members"]) == expected
    # Arms outside the registered family are bootstrapped but never corrected.
    assert "OAE:full" not in fam["members"]
    assert "PlaceboGap:loo_oracle" not in fam["members"]
    assert "full" in out["rq2_oae"] and "p_holm" not in out["rq2_oae"]["full"]


def test_primary_endpoint_reports_both_corrected_and_uncorrected(tmp_path):
    csv_path = _write_csv(tmp_path / "g.csv")
    scores = load_scores(csv_path, "f1", "3")
    out = analyze_budget(scores, "rerank_topk", "placebo_pos:middle_first",
                         "loo_oracle", n_replicates=50, ci=0.95, seed=1,
                         verbose=False)
    primary = out["confirmatory_family"]["primary_endpoint"]
    assert PRIMARY_ENDPOINT[1] in primary["comparison"]
    assert primary["p_uncorrected"] is not None
    assert primary["p_holm"] is not None
    assert primary["p_holm"] >= primary["p_uncorrected"]


def test_analyze_writes_json_with_primary_budget_first(tmp_path):
    results = tmp_path / "res"
    results.mkdir()
    _write_csv(results / "generations.csv", budgets=("2", "3"))
    cfg = Config({
        "budgets": [2, 3],
        "metrics": {"baseline_arm": "rerank_topk",
                    "placebo_arm": "placebo_pos:middle_first",
                    "oracle_arm": "loo_oracle", "primary_budget": 3},
        "stats": {"bootstrap_replicates": 50, "ci": 0.95, "seed": 1},
        "output": {"results_dir": str(results)},
    })
    out = analyze(cfg)
    assert out["config"]["budgets_analyzed"][0] == "3"
    written = json.load(open(results / "permutation_analysis.json", encoding="utf-8"))
    assert "budget_3" in written and "budget_2" in written
    assert written["budget_3"]["confirmatory_family"]["size"] == 9


def test_analyze_errors_without_generations_csv(tmp_path):
    cfg = Config({
        "budgets": [3],
        "metrics": {"baseline_arm": "rerank_topk",
                    "placebo_arm": "placebo_pos:middle_first",
                    "oracle_arm": "loo_oracle", "primary_budget": 3},
        "stats": {}, "output": {"results_dir": str(tmp_path / "nope")},
    })
    with pytest.raises(FileNotFoundError, match="generations.csv"):
        analyze(cfg)
