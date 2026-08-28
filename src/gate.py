"""The week-1 gate. This is the real decision point; everything after it is execution.

Plan Sec. 7 and Sec. 9. Kill criterion, stated before any data was collected:

    if median within-query SD of token-F1 across 5 permutations is < 0.02
    at k=10, the premise fails at this scale.

The escalation ladder, in order, before declaring it dead:
    more chunks (20 instead of 10) -> longer chunks -> a smaller/weaker
    generator -> MuSiQue, which shows stronger position effects in published work.
If none of that produces variance, the fallback project in Sec. 9 applies.

Run:  python -m src.gate results/pilot_w1/generations.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

KILL_THRESHOLD = 0.02


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def by_query(
    rows: Sequence[dict[str, str]],
    metric: str = "f1",
    arm: str | None = None,
) -> dict[str, list[float]]:
    """Group per-permutation scores by query, for one arm."""
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if arm is not None and row["arm"] != arm:
            continue
        out[row["qid"]].append(float(row[metric]))
    return dict(out)


def gate_report(
    rows: Sequence[dict[str, str]],
    arm: str = "full",
    metric: str = "f1",
) -> dict[str, float]:
    scores = by_query(rows, metric=metric, arm=arm)
    if not scores:
        raise ValueError(f"no rows for arm {arm!r}")

    n_perms = {len(v) for v in scores.values()}
    if len(n_perms) != 1:
        raise ValueError(f"ragged permutation counts: {sorted(n_perms)}")
    if n_perms == {1}:
        raise ValueError("only one permutation per query; nothing to measure")

    sds = np.array([np.std(v, ddof=1) for v in scores.values()])
    ranges = np.array([max(v) - min(v) for v in scores.values()])
    per_query_mean = np.array([np.mean(v) for v in scores.values()])

    return {
        "n_queries": float(len(scores)),
        "n_permutations": float(n_perms.pop()),
        "mean_score": float(per_query_mean.mean()),
        "median_within_query_sd": float(np.median(sds)),
        "mean_within_query_sd": float(sds.mean()),
        "q25_sd": float(np.quantile(sds, 0.25)),
        "q75_sd": float(np.quantile(sds, 0.75)),
        "frac_queries_zero_sd": float((sds == 0).mean()),
        "frac_queries_flipping": float((ranges > 0).mean()),
        "median_range": float(np.median(ranges)),
        # Between-query SD of the per-query mean. The contrast that matters:
        # if within-query SD is an appreciable fraction of this, ordering moves
        # answers about as much as the choice of question does.
        "between_query_sd": float(per_query_mean.std(ddof=1)),
    }


def format_report(stats: dict[str, float], metric: str, arm: str) -> str:
    median_sd = stats["median_within_query_sd"]
    passed = median_sd >= KILL_THRESHOLD

    lines = [
        "=" * 62,
        f"WEEK-1 GATE   arm={arm}  metric={metric}",
        "=" * 62,
        f"  queries                      {int(stats['n_queries'])}",
        f"  permutations per query       {int(stats['n_permutations'])}",
        f"  mean {metric:<24}{stats['mean_score']:.4f}",
        "",
        f"  median within-query SD       {median_sd:.4f}   <-- kill criterion",
        f"  mean within-query SD         {stats['mean_within_query_sd']:.4f}",
        f"  IQR of within-query SD       [{stats['q25_sd']:.4f}, {stats['q75_sd']:.4f}]",
        f"  median within-query range    {stats['median_range']:.4f}",
        "",
        f"  queries with zero variance   {stats['frac_queries_zero_sd']:.1%}",
        f"  queries that move at all     {stats['frac_queries_flipping']:.1%}",
        f"  between-query SD             {stats['between_query_sd']:.4f}",
        "-" * 62,
    ]

    if passed:
        lines += [
            f"  VERDICT: PASS  ({median_sd:.4f} >= {KILL_THRESHOLD})",
            "  Permutation variance exists at this scale. Premise holds;",
            "  proceed to week 2 (implement the arms).",
        ]
    else:
        lines += [
            f"  VERDICT: FAIL  ({median_sd:.4f} < {KILL_THRESHOLD})",
            "  The premise fails at this scale. Escalate in order (plan Sec. 9):",
            "    1. more chunks (20 instead of 10)",
            "    2. longer chunks",
            "    3. a smaller / weaker generator",
            "    4. switch to MuSiQue",
            "  If none of that produces variance, take the Sec. 9 fallback:",
            "  the consolidation study, which reuses ~80% of this code.",
        ]

    # A high mean score is its own warning: the memorization filter does not run
    # in the week-1 pilot, so a model answering from parametric memory will look
    # stable for reasons that have nothing to do with position.
    if stats["mean_score"] > 0.75:
        lines += [
            "",
            f"  NOTE: mean {metric} is {stats['mean_score']:.2f}. Before trusting a",
            "  low-variance result, run the nocontext arm -- near-ceiling scores",
            "  usually mean parametric recall, not stability (plan Sec. 4.1).",
        ]

    lines.append("=" * 62)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="generations.csv from a pilot run")
    parser.add_argument("--arm", default="full")
    parser.add_argument("--metric", default="f1", choices=["f1", "em"])
    args = parser.parse_args()

    rows = load_rows(args.csv)
    stats = gate_report(rows, arm=args.arm, metric=args.metric)
    print(format_report(stats, args.metric, args.arm))


if __name__ == "__main__":
    main()
