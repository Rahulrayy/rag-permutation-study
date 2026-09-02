"""Does the permutation effect track the number of permutable slots?

WRITEUP Sec. 4.9 reads the 3B-27B gap as largest on the un-pruned context and
absent on the keep-3 arms, and proposes that the effect scales with the number of
permutable slots rather than with the model alone. Sec. 7 filed that as future
work "testable within the existing protocol by varying the slot count on a fixed
generator". The main run already varied it -- k = 2, 3, 5 across the keep-k arms
and 10 for `full` and `llmlingua2` -- so the first half of the claim is
answerable from committed artifacts and this module answers it.

**The confound, which is the whole difficulty.** In every keep-k arm the slot
count and the amount of evidence retained rise together: at k = 5 a pruner keeps
more passages *and* more gold than at k = 2. A raw "SD rises with k" is therefore
uninterpretable -- it is equally consistent with "more slots to permute" and with
"more evidence, so more room for the score to move".

**The fix is to stratify on the evidence.** Every cell records how many of the
two gold passages survived, so conditioning on that holds the evidence fixed and
leaves the slot count as the only thing varying down a column. That is what the
table below reports, and it is why the answer is a two-way table rather than a
single number.

The `full` arm is the internal control: 10 slots and identical content at every
budget, so if the estimator were picking up budget rather than slots it would
move there, and it does not -- its SD is 0.1795 at all three.

**Scope.** This is the *within-generator* half of the Sec. 4.9 claim: does SD
track slots on a fixed model. The other half -- whether the 3B-27B *gap* tracks
slots -- needs both generators at more budgets than the replication ran, and
stays future work.

Exploratory. Not in the registered confirmatory family, and reported without
multiplicity correction.

    python -m src.slot_count
"""

from __future__ import annotations

import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .figures import COLORS, _style  # noqa: E402

MAIN = Path("results/main_hotpotqa/generations.csv")
OUT = Path("results/main_hotpotqa/slot_count.json")
FIG = Path("results/main_hotpotqa/figures/slot_count.png")

#: Arms whose budget is a compression rate rather than a keep-count: they retain
#: every chunk, so they present ten permutable slots at every budget. Mirrors
#: `Pruner.budget_is_keep_count`, read from the arm registry rather than
#: hardcoded where possible -- but `llmlingua2` and `full` are the only two and
#: the CSV carries no slot column, so the mapping lives here.
ALL_SLOT_ARMS = {"full": 10, "llmlingua2": 10}

#: Strata are exact gold counts, because HotpotQA distractor ships exactly two
#: gold paragraphs per question. A cell is dropped from a stratum with too few
#: observations to estimate a mean from.
MIN_CELLS = 20


def _cells(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """One record per (arm, budget, question): its within-question SD, slots, gold."""
    scores: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    tags: dict[tuple[str, str, str], tuple[int, int]] = {}
    for r in rows:
        if r["arm"] == "nocontext":
            continue
        key = (r["arm"], r["budget"], r["qid"])
        scores[key].append(float(r["f1"]))
        tags[key] = (
            ALL_SLOT_ARMS.get(r["arm"], int(r["budget"])),
            int(float(r["n_gold_kept"])),
        )
    out = []
    for key, vals in scores.items():
        if len(vals) < 2:
            continue
        slots, gold = tags[key]
        out.append(
            {
                "qid": key[2],
                "arm": key[0],
                "budget": key[1],
                "slots": slots,
                "gold": gold,
                "sd": float(np.std(vals, ddof=1)),
            }
        )
    return out


def _stratum_means(
    cells: Sequence[Mapping[str, Any]], qids: Sequence[str] | None = None
) -> dict[tuple[int, int], float]:
    """Mean cell SD per (slots, gold), optionally over a resampled question set.

    Questions are the resampling unit, as everywhere else in this study, and a
    question sampled twice contributes its cells twice -- so the counter is over
    (question occurrence, arm, budget) rather than over distinct cells.
    """
    by_qid: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for c in cells:
        by_qid[c["qid"]].append(c)
    chosen = list(by_qid) if qids is None else list(qids)

    sums: dict[tuple[int, int], float] = collections.defaultdict(float)
    counts: dict[tuple[int, int], int] = collections.defaultdict(int)
    for q in chosen:
        for c in by_qid.get(q, ()):
            k = (c["slots"], c["gold"])
            sums[k] += c["sd"]
            counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums if counts[k]}


def analyse(
    cells: Sequence[Mapping[str, Any]],
    n_replicates: int = 10_000,
    ci: float = 0.95,
    seed: int = 20260828,
) -> dict[str, Any]:
    """The stratified table, with question-level bootstrap intervals."""
    qids = sorted({c["qid"] for c in cells})
    point = _stratum_means(cells)
    counts = collections.Counter((c["slots"], c["gold"]) for c in cells)
    keep = {k for k, n in counts.items() if n >= MIN_CELLS}

    rng = np.random.default_rng(seed)
    reps: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for _ in range(n_replicates):
        draw = [qids[i] for i in rng.integers(0, len(qids), size=len(qids))]
        means = _stratum_means(cells, draw)
        for k in keep:
            if k in means:
                reps[k].append(means[k])

    alpha = (1.0 - ci) / 2.0
    table: dict[str, Any] = {}
    for (slots, gold) in sorted(keep):
        r = np.asarray(reps[(slots, gold)], dtype=float)
        table[f"{slots}|{gold}"] = {
            "slots": slots,
            "gold": gold,
            "n_cells": counts[(slots, gold)],
            "point": point[(slots, gold)],
            "lo": float(np.quantile(r, alpha)),
            "hi": float(np.quantile(r, 1.0 - alpha)),
        }

    # The contrast the section turns on: within a fixed evidence stratum, does
    # going from the fewest slots to the most move the SD? Bootstrapped as one
    # paired quantity over the same resampled questions, not read off two
    # marginal intervals.
    contrasts: dict[str, Any] = {}
    for gold in sorted({g for _, g in keep}):
        slots_here = sorted(s for s, g in keep if g == gold)
        if len(slots_here) < 2:
            continue
        lo_s, hi_s = slots_here[0], slots_here[-1]
        diffs = [
            m[(hi_s, gold)] - m[(lo_s, gold)]
            for m in (
                _stratum_means(cells, [qids[i] for i in rng.integers(0, len(qids), size=len(qids))])
                for _ in range(n_replicates)
            )
            if (hi_s, gold) in m and (lo_s, gold) in m
        ]
        d = np.asarray(diffs, dtype=float)
        contrasts[f"gold={gold}"] = {
            "gold": gold,
            "from_slots": lo_s,
            "to_slots": hi_s,
            "point": point[(hi_s, gold)] - point[(lo_s, gold)],
            "lo": float(np.quantile(d, alpha)),
            "hi": float(np.quantile(d, 1.0 - alpha)),
            "excludes_zero": bool(
                np.quantile(d, alpha) > 0 or np.quantile(d, 1.0 - alpha) < 0
            ),
        }

    return {
        "n_questions": len(qids),
        "n_cells": len(cells),
        "n_replicates": n_replicates,
        "ci": ci,
        "seed": seed,
        "min_cells_per_stratum": MIN_CELLS,
        "table": table,
        "slot_contrast_within_gold_stratum": contrasts,
    }


def figure(out: Mapping[str, Any], path: Path) -> None:
    """SD against slot count, one line per evidence stratum.

    Drawn as separate lines rather than pooled, because pooling is exactly the
    mistake this analysis exists to avoid: it would show a steep rise that is
    partly the evidence coming with the slots.
    """
    rows = list(out["table"].values())
    golds = sorted({r["gold"] for r in rows})
    colours = {0: COLORS["control"], 1: COLORS["muted"], 2: COLORS["method"]}

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for g in golds:
        pts = sorted((r for r in rows if r["gold"] == g), key=lambda r: r["slots"])
        xs = [r["slots"] for r in pts]
        ys = [r["point"] for r in pts]
        err = np.array([[r["point"] - r["lo"] for r in pts],
                        [r["hi"] - r["point"] for r in pts]])
        ax.errorbar(
            xs, ys, yerr=err, marker="o", markersize=5, linewidth=1.6,
            capsize=3, color=colours.get(g, COLORS["muted"]),
            label=f"{g} of 2 gold passages retained",
        )

    ax.set_xticks([2, 3, 5, 10])
    ax.set_xlabel("permutable slots in the context")
    ax.set_ylabel("mean within-question SD of token-F1")
    ax.set_title(
        "Order sensitivity tracks the number of slots, not just the evidence\n"
        f"{out['n_questions']} questions, {out['n_cells']} arm-budget cells; "
        "reading down a line holds the evidence fixed",
        fontsize=10.5, loc="left",
    )
    _style(ax)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(
        0.005, -0.04,
        "Exploratory, and within-generator only. Slot count and evidence "
        "retained are confounded across budgets in every keep-k arm; "
        "stratifying on the gold count is what separates them.",
        fontsize=7.5, color="#555555", ha="left", va="top", wrap=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main(n_replicates: int = 10_000, seed: int = 20260828) -> dict[str, Any]:
    with open(MAIN, encoding="utf-8") as fh:
        cells = _cells(list(csv.DictReader(fh)))
    out = analyse(cells, n_replicates=n_replicates, seed=seed)

    print(f"{out['n_cells']} arm-budget cells over {out['n_questions']} questions\n")
    print(f"  {'slots':>6s}  " + "  ".join(f"gold={g}".rjust(22) for g in (0, 1, 2)))
    for slots in sorted({r["slots"] for r in out["table"].values()}):
        cells_row = []
        for g in (0, 1, 2):
            e = out["table"].get(f"{slots}|{g}")
            cells_row.append(
                f"{e['point']:.4f} [{e['lo']:.4f},{e['hi']:.4f}]".rjust(22)
                if e else "-".rjust(22)
            )
        print(f"  {slots:6d}  " + "  ".join(cells_row))

    print("\n  slot contrast within a fixed evidence stratum:")
    for key, c in out["slot_contrast_within_gold_stratum"].items():
        star = "*" if c["excludes_zero"] else " "
        print(f"    {key}, {c['from_slots']} -> {c['to_slots']} slots: "
              f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] {star}")

    figure(out, FIG)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
