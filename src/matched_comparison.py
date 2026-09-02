"""One matched within-question-SD comparison, shared by two robustness checks.

Both checks ask the same question with one thing varied: hold the questions and
the orderings fixed, change exactly one factor, and see whether the RQ1
permutation SD moves.

    src.generator_comparison   3B against 27B         -- does the effect survive
                                                         a change of model scale?
    src.delimiter_check        default against alt    -- does it survive a change
                               prompt delimiters         of context fencing?

Keeping the machinery here rather than duplicating it means both inherit the two
properties that make either trustworthy. First, the passage orders are
**asserted identical** cell by cell, and the comparison raises rather than
reports if they are not: a number computed across mismatched orders would look
exactly like a real finding. Second, the difference is bootstrapped **paired**
over the shared questions rather than read off the overlap of two marginal
intervals, because non-overlap implies a difference but overlap does not imply
its absence.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analyze import _boot  # noqa: E402
from .figures import COLORS, _errorbars, _style  # noqa: E402
from .metrics import within_query_sd  # noqa: E402


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _orders(
    rows, arm: str, budget: str, perms: Sequence[str]
) -> dict[tuple[str, str], str]:
    return {
        (r["qid"], r["perm"]): r["order"]
        for r in rows
        if r["arm"] == arm and r["budget"] == budget and r["perm"] in perms
    }


def _scores(
    rows, arm: str, budget: str, qids: set[str], perms: Sequence[str]
) -> dict[str, list[float]]:
    by_qid: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in rows:
        if (
            r["arm"] == arm
            and r["budget"] == budget
            and r["perm"] in perms
            and r["qid"] in qids
        ):
            by_qid[r["qid"]][r["perm"]] = float(r["f1"])
    return {q: [p[i] for i in sorted(p)] for q, p in by_qid.items()}


def compare(
    rows_a,
    rows_b,
    label_a: str,
    label_b: str,
    arms: Sequence[str],
    budget: str,
    perms: Sequence[str],
    n_replicates: int = 10_000,
    seed: int = 20260828,
    population_arm: str = "full",
    verbose: bool = True,
) -> dict[str, Any]:
    """Mean within-question SD under A and under B, and their paired difference."""
    q_a = {r["qid"] for r in rows_a if r["arm"] == population_arm}
    q_b = {r["qid"] for r in rows_b if r["arm"] == population_arm}
    shared = q_a & q_b
    if not shared:
        raise ValueError("the two runs share no queries on the population arm")

    o_a = _orders(rows_a, population_arm, budget, perms)
    o_b = _orders(rows_b, population_arm, budget, perms)
    common = {k for k in set(o_a) & set(o_b) if k[0] in shared}
    mismatched = [k for k in common if o_a[k] != o_b[k]]
    if mismatched:
        raise AssertionError(
            f"{len(mismatched)} of {len(common)} shared cells present different "
            "passage orders; the runs are not order-matched"
        )

    out: dict[str, Any] = {
        "labels": [label_a, label_b],
        "budget": budget,
        "perms": list(perms),
        "n_shared_queries": len(shared),
        f"n_{label_a}_population": len(q_a),
        f"n_{label_b}_population": len(q_b),
        "order_cells_verified_identical": len(common),
        "n_replicates": n_replicates,
        "seed": seed,
        "arms": {},
    }

    def say(line: str) -> None:
        if verbose:
            print(line, flush=True)

    say(f"budget k={budget}, perms {tuple(perms)}, {len(shared)} shared queries")
    say(f"  ({len(q_a)} in {label_a}, {len(q_b)} in {label_b})")
    say(f"  {len(common)} shared cells verified order-identical\n")
    say(
        f"  {'arm':22s} {label_a:>22s} {label_b:>22s}   ratio   "
        f"{'paired difference':>27s}"
    )

    for arm in arms:
        row: dict[str, Any] = {}
        for label, rows in ((label_a, rows_a), (label_b, rows_b)):
            scores = {arm: _scores(rows, arm, budget, shared, perms)}
            res = _boot(
                scores,
                [arm],
                lambda s, a=arm: float(np.mean(list(within_query_sd(s[a]).values()))),
                n_replicates,
                0.95,
                seed,
            )
            row[label] = {"point": res.point, "lo": res.lo, "hi": res.hi}

        denom = row[label_b]["point"]
        row["ratio"] = row[label_a]["point"] / denom if denom else float("nan")

        paired = {
            label_a: _scores(rows_a, arm, budget, shared, perms),
            label_b: _scores(rows_b, arm, budget, shared, perms),
        }
        diff = _boot(
            paired,
            [label_a, label_b],
            lambda s: float(np.mean(list(within_query_sd(s[label_a]).values())))
            - float(np.mean(list(within_query_sd(s[label_b]).values()))),
            n_replicates,
            0.95,
            seed,
        )
        row["paired_difference"] = {
            "point": diff.point,
            "lo": diff.lo,
            "hi": diff.hi,
            "excludes_zero": bool(diff.excludes_zero),
            "p": diff.p_two_sided(),
        }
        out["arms"][arm] = row

        def fmt(d: Mapping[str, float]) -> str:
            return f"{d['point']:.4f} [{d['lo']:.4f},{d['hi']:.4f}]"

        star = "*" if diff.excludes_zero else " "
        say(
            f"  {arm:22s} {fmt(row[label_a]):>22s} {fmt(row[label_b]):>22s}   "
            f"{row['ratio']:5.2f}x   diff {fmt(row['paired_difference']):>22s} "
            f"p={diff.p_two_sided():.4f} {star}"
        )
    return out


def write(out: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {path}")


def figure(
    out: Mapping[str, Any],
    path: Path,
    title: str,
    subtitle: str,
    series_labels: tuple[str, str],
    xlabel: str,
    footnote: str,
) -> None:
    """Both marginals drawn, significance taken from the paired difference.

    The marginals are here because a reader wants the magnitudes; the star is
    not, because overlapping intervals do not mean no difference.
    """
    label_a, label_b = out["labels"]
    arms = list(out["arms"])
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(arms) + 2.0))
    ys = np.arange(len(arms))[::-1]

    for label, colour, offset, series in (
        (label_a, COLORS["reference"], +0.15, series_labels[0]),
        (label_b, COLORS["method"], -0.15, series_labels[1]),
    ):
        for i, arm in enumerate(arms):
            e = out["arms"][arm][label]
            ax.errorbar(
                e["point"], ys[i] + offset, xerr=_errorbars(e), fmt="o",
                markersize=5, color=colour, ecolor=colour, elinewidth=1.4,
                capsize=3, label=series if i == 0 else None,
            )

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [
            f"{a}  {'*' if out['arms'][a]['paired_difference']['excludes_zero'] else ''}"
            for a in arms
        ],
        fontsize=9,
    )
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.text(
        0.005, -0.02, footnote, fontsize=7.5, color="#555555", ha="left", va="top"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
