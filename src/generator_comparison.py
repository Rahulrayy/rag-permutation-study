"""Matched 3B-vs-27B comparison of the RQ1 permutation SD.

The replication ran P=3 on its own memorization-filtered population, and the
main run ran P=5 on a larger one, so their headline SDs are not directly
comparable: fewer orderings give a score fewer chances to move, and the two
populations are filtered by different generators. ANALYSIS_PLAN Sec. 9 records
the P mismatch as a removable confound and says to remove it.

It is removable exactly, not approximately. Both configs seed the permutation
draw identically and the replication's three strategies are the prefix of the
main run's five, so for every shared query the two runs present *byte-identical*
passage orders in perms 0-2 -- asserted below rather than assumed. Restricting
the main run to those three perms and to the queries both populations retain
leaves the generator as the only difference.

    python -m src.generator_comparison
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analyze import _boot  # noqa: E402
from .figures import COLORS, _errorbars, _style  # noqa: E402
from .metrics import within_query_sd  # noqa: E402

MAIN = Path("results/main_hotpotqa/generations.csv")
REPL = Path("results/replication_groq/generations.csv")
OUT = Path("results/replication_groq/matched_generator_comparison.json")
FIG = Path("results/replication_groq/figures/matched_generator_sd.png")

#: The replication's permutations, as indices into the main run's five.
SHARED_PERMS = ("0", "1", "2")
ARMS = ("full", "rerank_topk", "llm_pruner", "placebo_pos:middle_first")


def _load(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _orders(rows, arm: str, budget: str) -> dict[tuple[str, str], str]:
    return {
        (r["qid"], r["perm"]): r["order"]
        for r in rows
        if r["arm"] == arm and r["budget"] == budget and r["perm"] in SHARED_PERMS
    }


def _scores(rows, arm: str, budget: str, qids: set[str]) -> dict[str, list[float]]:
    by_qid: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in rows:
        if r["arm"] == arm and r["budget"] == budget and r["perm"] in SHARED_PERMS:
            if r["qid"] in qids:
                by_qid[r["qid"]][r["perm"]] = float(r["f1"])
    return {q: [p[i] for i in sorted(p)] for q, p in by_qid.items()}


def main(budget: str = "3", n_replicates: int = 10_000, seed: int = 20260828) -> dict:
    main_rows, repl_rows = _load(MAIN), _load(REPL)

    # The population: queries both runs retained, after each generator's own
    # memorization filter. Neither is a subset of the other by construction.
    main_q = {r["qid"] for r in main_rows if r["arm"] == "full"}
    repl_q = {r["qid"] for r in repl_rows if r["arm"] == "full"}
    shared = main_q & repl_q

    # The orders must be identical, or this is not a matched comparison and
    # every number below is meaningless. Fail loudly rather than report it.
    mo, ro = _orders(main_rows, "full", budget), _orders(repl_rows, "full", budget)
    common = {k for k in set(mo) & set(ro) if k[0] in shared}
    mismatched = [k for k in common if mo[k] != ro[k]]
    if mismatched:
        raise AssertionError(
            f"{len(mismatched)} of {len(common)} shared cells present different "
            "passage orders; the runs are not order-matched"
        )

    out: dict = {
        "budget": budget,
        "n_shared_queries": len(shared),
        "n_main_population": len(main_q),
        "n_replication_population": len(repl_q),
        "perms": list(SHARED_PERMS),
        "order_cells_verified_identical": len(common),
        "n_replicates": n_replicates,
        "seed": seed,
        "arms": {},
    }

    print(f"budget k={budget}, perms {SHARED_PERMS}, {len(shared)} shared queries")
    print(f"  ({len(main_q)} in the 3B population, {len(repl_q)} in the 27B's)")
    print(f"  {len(common)} shared cells verified order-identical\n")
    print(f"  {'arm':26s} {'3B (P=3)':>22s} {'27B (P=3)':>22s}   ratio   "
          f"{'paired difference':>27s}")

    for arm in ARMS:
        row = {}
        for tag, rows in (("3B", main_rows), ("27B", repl_rows)):
            scores = {arm: _scores(rows, arm, budget, shared)}
            res = _boot(
                scores,
                [arm],
                lambda s, a=arm: float(np.mean(list(within_query_sd(s[a]).values()))),
                n_replicates,
                0.95,
                seed,
            )
            row[tag] = {"point": res.point, "lo": res.lo, "hi": res.hi}
        ratio = row["3B"]["point"] / row["27B"]["point"] if row["27B"]["point"] else float("nan")
        row["ratio_3B_over_27B"] = ratio

        # The two arms above are on the same queries under the same orders, so
        # the comparison is paired and the difference should be bootstrapped as
        # one quantity. Reading it off the overlap of two marginal CIs instead
        # would be the less powerful and less correct test -- non-overlap
        # implies a difference, but overlap does not imply its absence.
        paired = {
            "3B": _scores(main_rows, arm, budget, shared),
            "27B": _scores(repl_rows, arm, budget, shared),
        }
        diff = _boot(
            paired,
            ["3B", "27B"],
            lambda s: float(np.mean(list(within_query_sd(s["3B"]).values())))
            - float(np.mean(list(within_query_sd(s["27B"]).values()))),
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
        f = lambda d: f'{d["point"]:.4f} [{d["lo"]:.4f},{d["hi"]:.4f}]'
        star = "*" if diff.excludes_zero else " "
        print(f"  {arm:26s} {f(row['3B']):>22s} {f(row['27B']):>22s}   {ratio:5.2f}x   "
              f"diff {f(row['paired_difference']):>22s} p={diff.p_two_sided():.4f} {star}")

    _figure(out)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {OUT}")
    return out


def _figure(out: Mapping[str, Any]) -> None:
    """Paired SDs, 3B against 27B, one row per arm.

    Both marginals are drawn because a reader wants to see the magnitudes, but
    the significance mark comes from the paired difference, not from whether the
    two intervals overlap.
    """
    arms = list(out["arms"])
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(arms) + 2.0))
    ys = np.arange(len(arms))[::-1]

    for tag, colour, offset, label in (
        ("3B", COLORS["reference"], +0.15, "Qwen2.5-3B, local"),
        ("27B", COLORS["method"], -0.15, "Qwen3.8-27B, hosted"),
    ):
        for i, arm in enumerate(arms):
            e = out["arms"][arm][tag]
            ax.errorbar(
                e["point"], ys[i] + offset, xerr=_errorbars(e), fmt="o",
                markersize=5, color=colour, ecolor=colour, elinewidth=1.4,
                capsize=3, label=label if i == 0 else None,
            )

    labels = []
    for arm in arms:
        d = out["arms"][arm]["paired_difference"]
        labels.append(f"{arm}  {'*' if d['excludes_zero'] else ''}")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_xlabel("mean within-question SD of token-F1 across the three orderings")
    ax.set_title(
        "Order sensitivity survives a 9x scale jump, at a smaller size\n"
        f"same {out['n_shared_queries']} questions, same three orderings, "
        f"k={out['budget']}; only the generator differs",
        fontsize=10.5, loc="left",
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.text(
        0.005, -0.02,
        "*  the paired 3B-27B difference excludes zero. Exploratory: these four "
        "comparisons are not in the registered confirmatory family and carry no "
        "multiplicity correction.",
        fontsize=7.5, color="#555555", ha="left", va="top",
    )
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
