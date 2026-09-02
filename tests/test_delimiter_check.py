"""Tests for the delimiter robustness check's wiring.

The comparison machinery itself is covered by test_generator_comparison.py,
which exercises the same `matched_comparison.compare`. What matters here is that
this check is pointed at the right thing: all five permutations rather than the
replication's three, the un-pruned arm, and the alt template actually differing
from the default in the delimiters alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src import delimiter_check as dc
from src.generate import ALT_TEMPLATE, DEFAULT_TEMPLATE, build_prompt
from src.matched_comparison import compare
from src.run import Config

FIELDS = ["qid", "arm", "budget", "perm", "perm_strategy", "hop_type", "kept",
          "order", "gold_positions", "n_gold_kept", "context_chars",
          "prediction", "gold", "em", "f1"]


def _write(path: Path, qids, n_perms=5, scale=1.0):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for q in qids:
            for p in range(n_perms):
                w.writerow({
                    "qid": q, "arm": "full", "budget": "3", "perm": p,
                    "perm_strategy": "s", "hop_type": "bridge", "kept": "[]",
                    "order": f"[{p},0]", "gold_positions": "[]", "n_gold_kept": 1,
                    "context_chars": 100, "prediction": f"a{p}", "gold": "a0",
                    "em": 0, "f1": round(0.3 + 0.1 * p * scale, 4),
                })
    return path


def test_uses_all_five_permutations_and_the_unpruned_arm():
    """The 27B replication restricts to three perms because it only ran three.
    This check ran the full five, so restricting would throw away real data."""
    assert dc.PERMS == ("0", "1", "2", "3", "4")
    assert dc.ARMS == ("full",)


def test_config_differs_from_main_in_the_template_alone():
    """A drifted seed or population would confound the delimiter effect with a
    sampling change, which is the one thing this check must not do."""
    main = Config.load("configs/main.yaml")
    alt = Config.load("configs/robustness_delimiter.yaml")
    assert alt.get("prompt_template") == "alt"
    assert main.get("prompt_template", "default") == "default"
    for field in ("dataset", "split", "n_queries", "seed", "stratify_by",
                  "memorization_filter"):
        assert alt["data"][field] == main["data"][field], field
    for field in ("model", "max_new_tokens", "temperature", "do_sample", "seed"):
        assert alt["generator"][field] == main["generator"][field], field
    assert alt["permutation"] == main["permutation"]


def test_alt_template_differs_in_delimiters_only():
    """Enforced at import in generate.py; asserted here so the check's premise
    is covered by the suite rather than only by a module-level raise."""
    assert ALT_TEMPLATE == DEFAULT_TEMPLATE.replace(
        "{context}", "<context>\n{context}\n</context>"
    )


def test_both_templates_render_the_same_passage_numbering():
    """The variant must not change `[i] Title: text`, or it would vary the
    numbering as well as the fencing and stop isolating the delimiters."""
    from src.chunks import Chunk

    chunks = [Chunk(idx=i, title=f"T{i}", text=f"body {i}", rank=i) for i in range(3)]
    default = build_prompt("q?", chunks, DEFAULT_TEMPLATE)
    alt = build_prompt("q?", chunks, ALT_TEMPLATE)
    for i in range(1, 4):
        assert f"[{i}] T{i - 1}: body {i - 1}" in default
        assert f"[{i}] T{i - 1}: body {i - 1}" in alt
    assert "<context>" in alt and "<context>" not in default


def test_comparison_runs_over_five_perms(tmp_path):
    a = _write(tmp_path / "a.csv", [f"q{i}" for i in range(6)], scale=1.0)
    b = _write(tmp_path / "b.csv", [f"q{i}" for i in range(6)], scale=1.0)
    out = compare(
        list(csv.DictReader(open(a, encoding="utf-8"))),
        list(csv.DictReader(open(b, encoding="utf-8"))),
        "default", "alt", arms=dc.ARMS, budget="3", perms=dc.PERMS,
        n_replicates=100, verbose=False,
    )
    assert out["n_shared_queries"] == 6
    assert out["order_cells_verified_identical"] == 30      # 6 queries x 5 perms
    # Identical inputs: the paired difference must be zero and unstarred.
    d = out["arms"]["full"]["paired_difference"]
    assert d["point"] == pytest.approx(0.0, abs=1e-9)
    assert not d["excludes_zero"]
