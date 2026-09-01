"""Tests for the figure module.

A figure test cannot assert that a plot is *good*, so it asserts the things that
have actually gone wrong here before: that the file gets written at all, that the
driver fails loudly rather than drawing an empty axis when an artifact is
missing, and -- the substantive one -- that an axis claiming a matched-keep-count
comparison marks the two arms for which that claim is false.
"""

from __future__ import annotations

import csv
import json

import pytest

from src.figures import (
    NOT_KEEP_COUNT_MARK,
    _budget_keys,
    _keep_count_matched,
    _ordered,
    _tick_labels,
    fig_rq1,
    make_figures,
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
    "random_drop",
]
BUDGETS = ("2", "3")


def _write_generations(path, n_queries=10, n_perms=3):
    fields = ["qid", "arm", "budget", "perm", "perm_strategy", "hop_type", "kept",
              "order", "gold_positions", "n_gold_kept", "context_chars",
              "prediction", "gold", "em", "f1"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for q in range(n_queries):
            for a_i, arm in enumerate(ARMS):
                for b in BUDGETS:
                    for p in range(n_perms):
                        # Half the queries are perfectly stable, which is the
                        # shape RQ1 exists to draw -- and it exercises the
                        # separate zero bar rather than only the histogram.
                        score = (a_i + 1) / len(ARMS)
                        if q % 2 == 0:
                            score *= 1.0 - 0.1 * p
                        w.writerow({
                            "qid": f"q{q}", "arm": arm, "budget": b, "perm": p,
                            "perm_strategy": "seeded", "hop_type": "bridge",
                            "kept": "[]", "order": "[]", "gold_positions": "[]",
                            "n_gold_kept": 1, "context_chars": 100,
                            "prediction": "x", "gold": "x",
                            "em": round(score, 4), "f1": round(score, 4),
                        })
    return path


def _entry(point=0.2):
    return {"point": point, "lo": point - 0.05, "hi": point + 0.05, "excludes_zero": True}


def _write_analysis(path):
    """A minimal analysis JSON with the blocks every figure reads."""
    out = {"config": {"metric": "f1", "primary_budget": "3"}}
    for b in BUDGETS:
        out[f"budget_{b}"] = {
            "rq1_mean_within_query_sd": {a: _entry(0.1) for a in ARMS},
            "rq2_oae": {a: _entry(0.3) for a in ARMS if a != "rerank_topk"},
            "rq3_rank_flip_rate": {"arms": ARMS[:6], **_entry(0.04)},
            "rq4_placebo_gap": {a: _entry(0.25) for a in ARMS
                                if not a.startswith("placebo_pos")},
            "oracle_gap": {a: 0.8 for a in ARMS if a != "loo_oracle"},
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return path


def _cfg(results_dir):
    return Config({
        "budgets": [2, 3],
        "metrics": {"baseline_arm": "rerank_topk",
                    "placebo_arm": "placebo_pos:middle_first",
                    "oracle_arm": "loo_oracle", "primary_budget": 3},
        "stats": {}, "output": {"results_dir": str(results_dir)},
    })


def test_make_figures_writes_every_figure(tmp_path):
    results = tmp_path / "res"
    results.mkdir()
    _write_generations(results / "generations.csv")
    _write_analysis(results / "permutation_analysis.json")

    written = make_figures(_cfg(results))

    assert len(written) == 4
    for path in written:
        assert path.exists()
        # A blank or truncated PNG is a few hundred bytes; a real one is tens of
        # kilobytes. This catches a figure that "wrote" but drew nothing.
        assert path.stat().st_size > 10_000, f"{path.name} is suspiciously small"
    assert {p.parent.name for p in written} == {"figures"}


def test_only_selects_a_subset(tmp_path):
    results = tmp_path / "res"
    results.mkdir()
    _write_generations(results / "generations.csv")
    _write_analysis(results / "permutation_analysis.json")

    written = make_figures(_cfg(results), only=["rq3"])

    assert [p.name for p in written] == ["rq3_rank_flip_rate.png"]


def test_unknown_figure_name_is_rejected(tmp_path):
    results = tmp_path / "res"
    results.mkdir()
    _write_analysis(results / "permutation_analysis.json")
    with pytest.raises(ValueError, match="unknown figure"):
        make_figures(_cfg(results), only=["rq9"])


def test_missing_analysis_points_at_the_command_that_makes_it(tmp_path):
    results = tmp_path / "res"
    results.mkdir()
    with pytest.raises(FileNotFoundError, match="src.analyze"):
        make_figures(_cfg(results))


def test_missing_generations_is_only_fatal_for_rq1(tmp_path):
    """RQ1 is the one figure that needs raw per-query scores; the rest do not."""
    results = tmp_path / "res"
    results.mkdir()
    _write_analysis(results / "permutation_analysis.json")

    assert len(make_figures(_cfg(results), only=["rq2", "rq3", "rq4"])) == 3
    with pytest.raises(FileNotFoundError, match="generations.csv"):
        make_figures(_cfg(results), only=["rq1"])


def test_arms_without_a_keep_count_are_marked():
    """The claim "at equal keep-count" is false for `full` and `llmlingua2`.

    Both declare `budget_is_keep_count = False` -- `full` prunes nothing and
    `llmlingua2` spends the budget as a compression rate -- and the plan compares
    them on input-token count instead. An axis that says "matched keep-count"
    while carrying their rows unmarked is making a claim the design forbids.
    """
    assert _keep_count_matched("provence_rerank")
    assert not _keep_count_matched("full")
    assert not _keep_count_matched("llmlingua2")

    labels, any_unmatched = _tick_labels(["provence_rerank", "full", "llmlingua2"])
    assert any_unmatched
    assert labels[0] == "provence_rerank"
    assert labels[1].endswith(NOT_KEEP_COUNT_MARK)
    assert labels[2].endswith(NOT_KEEP_COUNT_MARK)


def test_unregistered_arm_is_marked_rather_than_assumed_matched():
    """Fail toward flagging: an arm the registry does not know is not "matched"."""
    assert not _keep_count_matched("some_arm_that_does_not_exist")


def test_budget_keys_are_numerically_ordered():
    """The JSON stores the primary budget first; figures want budget order."""
    analysis = {"budget_3": {}, "budget_2": {}, "budget_10": {}, "config": {}}
    assert _budget_keys(analysis) == ["budget_2", "budget_3", "budget_10"]


def test_arm_order_is_by_role_then_unknown_alphabetically():
    ordered = _ordered(["random_drop", "zzz_new_arm", "full", "provence_rerank"])
    assert ordered == ["full", "provence_rerank", "random_drop", "zzz_new_arm"]


def test_rq1_refuses_an_arm_with_no_permutations():
    scores = {"full": {"q0": [0.5]}}
    analysis = {"rq1_mean_within_query_sd": {"full": _entry(0.1)}}
    with pytest.raises(ValueError, match="more than one permutation"):
        fig_rq1(scores, analysis, budget="3")


def _write_probe(path, n=40):
    """A selection-stability record shaped like the real probe's output."""
    per_query = [
        {"qid": f"q{i}", "jaccard": [0.0, 0.2, 0.25, 0.5, 1.0][i % 5], "stable": i % 5 == 4,
         "selections": {"rank": [0, 1, 2], "reverse": [1, 2, 3], "random": [2, 3, 4]}}
        for i in range(n)
    ]
    changed = sum(1 for r in per_query if r["jaccard"] < 1.0)
    probe = {
        "config": {"arm": "llm_pruner", "reference_arm": "rerank_topk",
                   "model": "Qwen/Qwen2.5-3B-Instruct", "n_queries": n, "budget": 3,
                   "orders": ["rank", "reverse", "random"], "seed": 1, "n_chunks": 10},
        "summary": {"n": n, "mean_jaccard": 0.39, "median_jaccard": 0.25,
                    "changed_fraction": changed / n, "n_changed": changed},
        "reference_order_invariant": {"n": n, "mean_jaccard": 1.0, "median_jaccard": 1.0,
                                      "changed_fraction": 0.0, "n_changed": 0},
        "chance": {"mean": 0.0467, "trials": 20000, "n_chunks": 10},
        "per_query": per_query,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(probe, fh)
    return path


def test_selection_figure_is_drawn_when_the_probe_artifact_exists(tmp_path):
    results = tmp_path / "res"
    results.mkdir()
    _write_analysis(results / "permutation_analysis.json")
    _write_probe(results / "selection_stability.json")

    written = make_figures(_cfg(results), only=["selection"])

    assert [p.name for p in written] == ["selection_stability.png"]
    assert written[0].stat().st_size > 10_000


def test_asking_for_the_selection_figure_without_the_probe_is_an_error(tmp_path):
    """Explicitly requested and missing: say which command produces it."""
    results = tmp_path / "res"
    results.mkdir()
    _write_analysis(results / "permutation_analysis.json")
    with pytest.raises(FileNotFoundError, match="src.selection_probe"):
        make_figures(_cfg(results), only=["selection"])


def test_drawing_everything_skips_the_probe_figure_rather_than_failing(tmp_path):
    """It is an optional artifact -- the main run does not produce it.

    Skipping is visible on stdout rather than silent, but it must not take the
    other four figures down with it.
    """
    results = tmp_path / "res"
    results.mkdir()
    _write_generations(results / "generations.csv")
    _write_analysis(results / "permutation_analysis.json")

    written = make_figures(_cfg(results))

    assert len(written) == 4
    assert not any("selection" in p.name for p in written)
