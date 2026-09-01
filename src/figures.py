"""Figures for the write-up: one per research question.

    python -m src.figures --config configs/main.yaml
    python -m src.figures --config configs/main.yaml --only rq1

Everything here reads artifacts and draws them. It computes no statistic that
`metrics.py` and `stats.py` do not already own -- the CIs come from
`permutation_analysis.json` exactly as `src.analyze` wrote them, and the one
quantity derived from raw generations (RQ1's per-query SD) goes through
`metrics.within_query_sd` on a container built by `analyze.load_scores`. A figure
that recomputed its own numbers could disagree with the text and nobody would
notice.

Two design rules inherited from the analysis, both of which are silent
correctness traps rather than style:

  * **One arm, one budget.** `load_scores` takes a budget and filters to it. A
    figure grouping by (arm, query) across budgets would draw fifteen values per
    query that are not fifteen permutations of one context, and the spread it
    showed would mostly be the budget contrast.

  * **`nocontext` is not on these axes.** It runs at one permutation, so it has
    no within-query SD, and it is reported over all sampled queries while every
    other arm is reported over the memorization-filtered subset. Putting it in
    the same column as the rest would be comparing one arm's mean against another
    arm's population. `analyze` already drops it; this module never adds it back.

RQ1 is drawn as a distribution rather than as its median, which is the
presentation registered in ANALYSIS_PLAN Sec. 9 (2026-08-29) after the pilot's
median landed on the seam of a bimodal distribution and was shown to be unstable
to a one-query change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

# Before pyplot, and not negotiable: these figures are written to disk from
# scripts and tests, never shown. Without this matplotlib picks an interactive
# backend and a headless run dies on a missing display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402

from .analyze import load_scores  # noqa: E402
from .metrics import within_query_sd  # noqa: E402
from .prune import arm_budget_is_keep_count  # noqa: E402
from .run import Config  # noqa: E402

#: Marks an arm whose budget is not a keep-count, wherever a figure's axis
#: describes the comparison as matched at equal k.
NOT_KEEP_COUNT_MARK = " †"

#: Arms grouped by the role they play in the design, which is also the order
#: they are drawn in. Roles carry the colour, so a reader can see at a glance
#: whether a bar is a reference, a method under test, or a control -- the
#: distinction the whole study turns on.
REFERENCE_ARMS = ("full", "loo_oracle")
METHOD_ARMS = ("rerank_topk", "provence_rerank", "provence_full", "llm_pruner", "llmlingua2")
CONTROL_ARMS = (
    "placebo_pos:middle_first",
    "placebo_pos:edges_first",
    "placebo_pos:tail_first",
    "random_drop",
)
ARM_ORDER = REFERENCE_ARMS + METHOD_ARMS + CONTROL_ARMS

#: Okabe-Ito, which stays distinguishable in the common colour-vision
#: deficiencies and in greyscale print.
COLORS = {
    "reference": "#333333",
    "method": "#0072B2",
    "control": "#E69F00",
    "primary": "#009E73",
    "muted": "#999999",
}

#: The registered primary endpoint's arm, drawn with its own colour wherever it
#: appears so the confirmatory comparison is not just one row among many.
PRIMARY_ARM = "provence_rerank"

#: OAE is a method-against-baseline quantity. The placebo and random arms have
#: OAEs in the -2 to -5 range, which is a floor effect in an arm nobody would
#: deploy, and putting them on the axis compresses every comparison a reader
#: cares about into the right-hand tenth of it. Omitted by default and named
#: here rather than silently dropped.
OAE_ARMS = REFERENCE_ARMS + METHOD_ARMS


def _role(arm: str) -> str:
    if arm in REFERENCE_ARMS:
        return "reference"
    if arm in METHOD_ARMS:
        return "method"
    return "control"


def _color(arm: str) -> str:
    return COLORS["primary"] if arm == PRIMARY_ARM else COLORS[_role(arm)]


def _ordered(arms: Sequence[str]) -> list[str]:
    """Design order first, then anything unrecognised, alphabetically."""
    known = [a for a in ARM_ORDER if a in arms]
    return known + sorted(a for a in arms if a not in ARM_ORDER)


def _keep_count_matched(arm: str) -> bool:
    """Whether an axis may honestly call this arm's comparison matched at equal k.

    `full` keeps everything by construction and `llmlingua2` spends the budget as
    a compression *rate*, so neither has a keep-count to match on; the plan
    compares them on input-token count instead (Sec. 4.3). An arm the registry
    does not recognise is treated as unmatched too, so a figure fails toward
    flagging the caveat rather than toward silently claiming a matching it has
    not checked.
    """
    try:
        return arm_budget_is_keep_count(arm)
    except KeyError:
        return False


def _tick_labels(arms: Sequence[str]) -> tuple[list[str], bool]:
    """Arm names for the y-axis, daggering the ones that are not keep-k matched."""
    labels = [a if _keep_count_matched(a) else a + NOT_KEEP_COUNT_MARK for a in arms]
    return labels, any(not _keep_count_matched(a) for a in arms)


def _keep_count_footnote(fig: Any) -> None:
    fig.text(
        0.005, -0.012,
        "†  budget is not a keep-count. `full` prunes nothing, `llmlingua2` spends it as a "
        "compression rate. These two are matched on input-token count, not on k "
        "(ANALYSIS_PLAN Sec. 4.3), so their row is not a matched-k comparison.",
        fontsize=7.5, color="#555555", ha="left", va="top",
    )


def _budget_keys(analysis: Mapping[str, Any]) -> list[str]:
    """Budgets present in an analysis JSON, numerically ordered.

    The JSON stores the primary budget first, because `analyze` runs it first so
    an interrupted run still leaves the confirmatory result on disk. Figures want
    them in budget order instead.
    """
    keys = [k for k in analysis if k.startswith("budget_")]
    return sorted(keys, key=lambda k: float(k.split("_", 1)[1]))


def _budget_label(key: str) -> str:
    return f"k={key.split('_', 1)[1]}"


def _style(ax: Any) -> None:
    """One house style, applied in one place."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def _errorbars(entry: Mapping[str, float]) -> np.ndarray:
    """A bootstrap entry's CI as matplotlib's (lower, upper) *distances*."""
    return np.array([[entry["point"] - entry["lo"]], [entry["hi"] - entry["point"]]])


# --------------------------------------------------------------------------- #
# RQ1 -- does the order matter at all, and for which queries
# --------------------------------------------------------------------------- #

def fig_rq1(
    scores: Mapping[str, Mapping[str, Sequence[float]]],
    budget_analysis: Mapping[str, Any],
    budget: str,
    arm: str = "full",
) -> Any:
    """Left: the SD distribution that the median hides. Right: every arm's mean.

    The left panel is the figure this study most needs to publish. "Median
    within-query SD = 0.026" reads as a tiny effect; the distribution behind it
    is bimodal, with half the queries perfectly stable under reordering and the
    other half swinging by roughly 0.42 F1 on identical content. The median falls
    on the seam between the two modes, which is why it is reported but never
    quoted alone.
    """
    sds = within_query_sd(scores[arm])
    values = np.array(sorted(sds.values()))
    if values.size == 0:
        raise ValueError(f"arm {arm!r} has no query with more than one permutation")

    movers = values[values > 0]
    zero_share = float((values == 0).mean())

    fig, (ax_dist, ax_arms) = plt.subplots(
        1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # --- left: the distribution ------------------------------------------- #
    # The exactly-zero queries get their own bar, drawn left of the axis origin,
    # rather than sharing the histogram's first bin with the smallest movers.
    # Sharing it merges "never moves, under any ordering" with "moves a little",
    # which are the two claims this panel exists to separate.
    top = float(max(0.6, values.max()))
    bar_w = top / 30.0
    ax_dist.bar(
        -1.6 * bar_w, (values == 0).sum(), width=bar_w * 1.4, color=COLORS["control"],
        edgecolor="white", label=f"never move ({(values == 0).sum()} queries)",
    )
    if movers.size:
        ax_dist.hist(
            movers, bins=np.linspace(0, top, 31), color=COLORS["method"],
            edgecolor="white", label=f"move at all ({movers.size} queries)",
        )
    ax_dist.axvline(0, color="#BBBBBB", linewidth=0.9)
    ax_dist.axvline(float(np.median(values)), color=COLORS["primary"], linewidth=1.6,
                    label=f"median of all  {np.median(values):.4f}")
    if movers.size:
        ax_dist.axvline(float(np.median(movers)), color=COLORS["primary"], linewidth=1.6,
                        linestyle="--", label=f"median of movers  {np.median(movers):.4f}")
    ax_dist.set_xlim(-3.0 * bar_w, top)
    ax_dist.set_xlabel(
        f"within-query SD of token-F1 across permutations  ({arm}, k={budget})"
    )
    ax_dist.set_ylabel("queries")
    ax_dist.set_title("A.  The median sits between two modes", loc="left", fontsize=11)
    ax_dist.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax_dist)
    ax_dist.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)

    ax_dist.text(
        0.97, 0.60,
        f"n = {values.size} queries\n"
        f"{zero_share:.0%} never move\n"
        f"{1 - zero_share:.0%} move, by {np.median(movers) if movers.size else 0:.2f} F1 (median)\n\n"
        "the median of all falls on the seam\nbetween the two modes, so it is\nreported but never quoted alone",
        transform=ax_dist.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#CCCCCC"),
    )

    # --- right: every arm's mean SD, with its bootstrap CI ------------------ #
    rq1 = budget_analysis["rq1_mean_within_query_sd"]
    arms = _ordered(list(rq1))
    ypos = np.arange(len(arms))[::-1]
    for y, a in zip(ypos, arms):
        ax_arms.errorbar(
            rq1[a]["point"], y, xerr=_errorbars(rq1[a]), fmt="o", markersize=5,
            color=_color(a), ecolor=_color(a), elinewidth=1.4, capsize=3,
        )
    ax_arms.set_yticks(ypos)
    ax_arms.set_yticklabels(arms, fontsize=9)
    ax_arms.set_xlabel("mean within-query SD  (95% CI)")
    ax_arms.set_title("B.  Pruning reduces order sensitivity", loc="left", fontsize=11)
    ax_arms.set_xlim(left=0)
    _style(ax_arms)

    fig.suptitle(
        "RQ1  Reordering a fixed context moves the answer on about half of queries",
        x=0.012, ha="left", fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------- #
# A forest shared by RQ2 and RQ4
# --------------------------------------------------------------------------- #

def _forest(
    ax: Any,
    analysis: Mapping[str, Any],
    key: str,
    arms: Sequence[str],
    budgets: Sequence[str],
    reference_lines: Sequence[tuple[float, str, str, float]] = (),
) -> None:
    """One row per arm, one marker per budget, bootstrap CI as the whisker.

    Budgets are offset within the row rather than drawn as separate panels: the
    question a reader asks of these plots is how an effect moves with the budget,
    and that is far easier to see when the three estimates share a baseline.
    """
    ypos = np.arange(len(arms))[::-1].astype(float)
    offsets = np.linspace(0.24, -0.24, len(budgets))
    markers = ["o", "s", "^", "D", "v"]

    # Reference-line labels carry their own height, because three of them at a
    # common height overlap into an unreadable smear once the lines are close
    # together (0, 0.5 and 1.0 on the OAE axis were doing exactly that).
    for x, label, style, y_frac in reference_lines:
        ax.axvline(x, color=COLORS["muted"], linewidth=1.0, linestyle=style, zorder=0)
        ax.text(
            x, y_frac, f" {label}", color="#777777", fontsize=8, va="top",
            transform=ax.get_xaxis_transform(),
        )

    for b_i, budget in enumerate(budgets):
        block = analysis[budget][key]
        xs, ys, err, colors = [], [], [], []
        for y, arm in zip(ypos, arms):
            if arm not in block:
                continue
            xs.append(block[arm]["point"])
            ys.append(y + offsets[b_i])
            err.append(_errorbars(block[arm]).ravel())
            colors.append(_color(arm))
        if not xs:
            continue
        # One errorbar call per (budget, arm) so each keeps its role colour;
        # matplotlib will not take a colour sequence for error bars.
        for x, y, e, c in zip(xs, ys, err, colors):
            ax.errorbar(
                x, y, xerr=e.reshape(2, 1), fmt=markers[b_i % len(markers)],
                markersize=4.5, color=c, ecolor=c, elinewidth=1.2, capsize=2.5,
                alpha=1.0 - 0.22 * b_i,
            )

    # Budget legend, drawn in neutral grey: the marker shape carries the budget,
    # the colour carries the arm's role, and mixing the two would be unreadable.
    handles = [
        plt.Line2D([], [], color=COLORS["muted"], marker=markers[i % len(markers)],
                   linestyle="none", markersize=4.5, label=_budget_label(b))
        for i, b in enumerate(budgets)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")

    labels, _ = _tick_labels(arms)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    _style(ax)


def fig_rq4_placebo_gap(analysis: Mapping[str, Any], budgets: Sequence[str]) -> Any:
    """The primary endpoint: does a pruner beat dropping the same count by position?

    A CI containing zero means the arm is not doing content selection at that
    budget -- its apparent gain is positional promotion, which the placebo
    reproduces without reading the passages. `random_drop` is the arm that should
    land there, and it does; that is what makes the rest of the column mean
    something.
    """
    arms = _ordered([a for a in analysis[budgets[0]]["rq4_placebo_gap"]])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    _forest(ax, analysis, "rq4_placebo_gap", arms, budgets,
            reference_lines=[(0.0, "no content selection", "-", 0.99)])
    ax.set_xlabel(
        "Placebo Gap: token-F1 against placebo_pos:middle_first at equal keep-count  (95% CI)"
    )
    ax.set_title(
        "RQ4  Published pruners are doing content selection, not positional promotion\n"
        f"primary endpoint: {PRIMARY_ARM} (green), pre-registered at k=3",
        loc="left", fontsize=11.5,
    )
    fig.tight_layout()
    if _tick_labels(arms)[1]:
        _keep_count_footnote(fig)
    return fig


def fig_rq2_oae(analysis: Mapping[str, Any], budgets: Sequence[str]) -> Any:
    """The same gains measured in orderings-worth of noise, where they shrink.

    OAE divides an arm's gain over the baseline by the baseline's within-query
    permutation SD: how many orderings-worth of noise does choosing this method
    actually buy. H2 registered < 0.5 as the threshold; 1.0 is the interpretive
    reference where reordering the same context moves the score as much as
    changing the method does.
    """
    arms = [a for a in OAE_ARMS if a in analysis[budgets[0]]["rq2_oae"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    _forest(ax, analysis, "rq2_oae", arms, budgets, reference_lines=[
        (0.0, "no gain", "-", 0.99),
        (0.5, "H2 threshold", ":", 0.90),
        (1.0, "one ordering's noise", "--", 0.81),
    ])
    ax.set_xlabel("Order-Adjusted Effect vs rerank_topk, in baseline permutation SDs  (95% CI)")
    ax.set_title(
        "RQ2  The method you choose matters less than the order you feed it in\n"
        "controls omitted: their OAE is a floor effect, not a method result",
        loc="left", fontsize=11.5,
    )
    fig.tight_layout()
    if _tick_labels(arms)[1]:
        _keep_count_footnote(fig)
    return fig


def fig_rq3_rank_flip(analysis: Mapping[str, Any], budgets: Sequence[str]) -> Any:
    """How often a method-pair comparison reverses sign under some single ordering.

    The tension with RQ1 is the discussion section, so the figure states both:
    order destabilises *answers* (half of queries) far more than it destabilises
    *conclusions* (a few percent of rankings). Per-query volatility largely
    averages out at the method level -- but it rises with the budget, which is
    the regime real systems operate in.
    """
    entries = [analysis[b]["rq3_rank_flip_rate"] for b in budgets]
    # Three discrete budgets, drawn as three discrete estimates. An earlier
    # version joined them with a shaded CI band, which draws a continuum across
    # k values that were never measured and reads as far stronger evidence of a
    # trend than three overlapping intervals actually are.
    xs = np.arange(len(budgets), dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axhline(0.10, color=COLORS["control"], linestyle=":", linewidth=1.2)
    # Parked in the gap between two budgets: at the right-hand edge it sits on
    # top of the widest error bar.
    ax.text((len(budgets) - 1) / 2 + 0.5, 0.103, "H3 threshold, 0.10",
            color=COLORS["control"], fontsize=8.5, va="bottom", ha="center")
    ax.axhline(0.0, color=COLORS["muted"], linewidth=0.8)

    for x, e in zip(xs, entries):
        ax.errorbar(
            x, e["point"], yerr=_errorbars(e), fmt="o", markersize=6,
            color=COLORS["method"], ecolor=COLORS["method"], elinewidth=1.6, capsize=4,
        )
        ax.annotate(f"{e['point']:.3f}", (x, e["point"]), textcoords="offset points",
                    xytext=(11, -4), ha="left", fontsize=9)

    n_arms = len(entries[0].get("arms", []))
    ax.set_xticks(xs)
    ax.set_xticklabels([_budget_label(b) for b in budgets])
    ax.set_xlim(-0.5, len(budgets) - 0.5)
    ax.set_xlabel("keep-k budget")
    ax.set_ylabel("fraction of method pairs whose sign reverses")
    ax.set_ylim(bottom=min(0.0, min(e["lo"] for e in entries)) - 0.01)
    ax.set_title(
        "RQ3  Order destabilises answers far more than it destabilises conclusions\n"
        f"H3 (rank flip rate > 0.10) is NOT supported: the estimate is below the "
        f"threshold at every budget",
        loc="left", fontsize=11.5,
    )
    ax.text(
        0.02, 0.98,
        f"over the {n_arms} deployable arms\n\n"
        "the point estimate rises with the budget, but the\n"
        "intervals overlap: this shows the level, it does\n"
        "not establish a trend",
        transform=ax.transAxes, fontsize=8, color="#555555", ha="left", va="top",
    )
    _style(ax)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# The selection-stability probe -- the thesis applied to the pruner
# --------------------------------------------------------------------------- #

def fig_selection_stability(probe: Mapping[str, Any]) -> Any:
    """Where an LLM pruner's selection sits between chance and order-invariance.

    Both reference points are earned rather than assumed: the upper one is
    `rerank_topk` measured through the same permutations, the lower one is
    simulated random k-subsets. Without them a Jaccard of ~0.26 is unreadable --
    the finding is that it is far above chance and nowhere near invariant, which
    is a statement about two numbers this figure has to show.

    Drawn as bars on the observed values rather than as a smooth histogram: with
    k of n, the Jaccard of three sets can only take a handful of rational values,
    and a smooth density would invent a continuum that does not exist.
    """
    per_query = probe["per_query"]
    summary = probe["summary"]
    chance = probe["chance"]["mean"]
    reference = probe["reference_order_invariant"]["mean_jaccard"]
    cfg = probe["config"]
    values = np.array([r["jaccard"] for r in per_query], dtype=float)

    fig, (ax_dist, ax_scale) = plt.subplots(
        2, 1, figsize=(8.6, 6.4), gridspec_kw={"height_ratios": [2.6, 1.0]}
    )

    # --- top: the distribution over queries -------------------------------- #
    uniq, counts = np.unique(np.round(values, 6), return_counts=True)
    ax_dist.bar(uniq, counts, width=0.022, color=COLORS["method"], edgecolor="white")
    ax_dist.axvline(chance, color=COLORS["control"], linewidth=1.6, linestyle=":",
                    label=f"chance, {len(cfg['orders'])} random {cfg['budget']}-subsets  {chance:.3f}")
    ax_dist.axvline(summary["mean_jaccard"], color=COLORS["primary"], linewidth=1.8,
                    label=f"observed mean  {summary['mean_jaccard']:.3f}")
    ax_dist.axvline(reference, color=COLORS["reference"], linewidth=1.6, linestyle="--",
                    label=f"{cfg['reference_arm']}, measured  {reference:.3f}")
    # The sharpest fact in the distribution, and the one a reader skims past:
    # a Jaccard of exactly 0 means the three presentations selected three sets
    # with no passage in common at all.
    n_disjoint = int((values == 0).sum())
    if n_disjoint:
        ax_dist.annotate(
            f"{n_disjoint} queries: the three\nselections share no\npassage at all",
            xy=(0.0, n_disjoint), xycoords="data",
            xytext=(0.42, 0.60), textcoords="axes fraction",
            fontsize=9, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#999999", linewidth=1.0,
                            connectionstyle="arc3,rad=-0.2"),
        )

    ax_dist.set_xlim(-0.04, 1.06)
    ax_dist.set_xlabel("selection Jaccard across the three presentation orders, per query")
    ax_dist.set_ylabel("queries")
    ax_dist.legend(frameon=False, fontsize=8.5, loc="upper center")
    ax_dist.set_title(
        f"The pruner picks different passages when shown the same passages in a different order\n"
        f"{cfg['model']}, n={summary['n']} queries, k={cfg['budget']}, greedy. "
        f"the selection changed in {summary['n_changed']} of {summary['n']}",
        loc="left", fontsize=10.5,
    )
    _style(ax_dist)
    ax_dist.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)

    # --- bottom: how far it travelled from chance to content-determined ----- #
    travelled = (summary["mean_jaccard"] - chance) / (reference - chance) if reference > chance else float("nan")
    ax_scale.hlines(0, chance, reference, color="#CCCCCC", linewidth=6, zorder=0)
    ax_scale.hlines(0, chance, summary["mean_jaccard"], color=COLORS["primary"], linewidth=6, zorder=1)
    for x, label, color in (
        (chance, f"chance\n{chance:.3f}", COLORS["control"]),
        (summary["mean_jaccard"], f"observed\n{summary['mean_jaccard']:.3f}", COLORS["primary"]),
        (reference, f"order-invariant\n{reference:.3f}", COLORS["reference"]),
    ):
        ax_scale.plot([x], [0], marker="o", markersize=8, color=color, zorder=2)
        ax_scale.annotate(label, (x, 0), textcoords="offset points", xytext=(0, -30),
                          ha="center", fontsize=9, color=color)
    ax_scale.annotate(
        f"{travelled:.0%} of the way from redrawing at random\nto being determined by the content",
        (summary["mean_jaccard"], 0), textcoords="offset points", xytext=(0, 22),
        ha="center", fontsize=9.5, color=COLORS["primary"],
    )
    ax_scale.set_xlim(-0.04, 1.06)
    ax_scale.set_ylim(-0.85, 0.75)
    ax_scale.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax_scale.spines[side].set_visible(False)
    ax_scale.set_xticks([])

    fig.text(
        0.005, 0.005,
        "A robustness check, outside the registered confirmatory family. LLM order-sensitivity is "
        "well established in the in-context-learning and\nmultiple-choice literature; the claim here is "
        "narrower: that the known effect reaches into a pruner's selection, and that no published "
        "LLM-pruner\nevaluation controls for it.",
        fontsize=7.5, color="#555555", ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

#: Figure name -> the RQ it answers. `--only` takes these names.
#:
#: `selection` is drawn from `selection_stability.json`, which the main run does
#: not produce -- it needs `python -m src.selection_probe` and a GPU. Asking for
#: it explicitly and not having it is an error; drawing "everything" without it
#: is a visible skip, because it is an optional artifact rather than a broken one.
FIGURE_NAMES = ("rq1", "rq2", "rq3", "rq4", "selection")


def make_figures(
    cfg: Config,
    only: Sequence[str] | None = None,
    metric: str = "f1",
    dpi: int = 200,
) -> list[Path]:
    """Draw every figure from a finished run's artifacts. Returns what it wrote."""
    results_dir = Path(cfg["output"]["results_dir"])
    analysis_path = results_dir / "permutation_analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"{analysis_path} not found -- run "
            f"`python -m src.analyze --config ...` first; the figures draw its CIs "
            f"rather than recomputing them"
        )
    with open(analysis_path, "r", encoding="utf-8") as fh:
        analysis = json.load(fh)

    budgets = _budget_keys(analysis)
    if not budgets:
        raise ValueError(f"{analysis_path} holds no budget_* block")

    wanted = tuple(only) if only else FIGURE_NAMES
    unknown = set(wanted) - set(FIGURE_NAMES)
    if unknown:
        raise ValueError(f"unknown figure(s) {sorted(unknown)}; known: {list(FIGURE_NAMES)}")

    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path}", flush=True)

    primary = str(cfg["metrics"].get("primary_budget", budgets[0].split("_", 1)[1]))
    primary_key = f"budget_{primary}"
    if primary_key not in analysis:
        primary_key = budgets[0]
        primary = primary_key.split("_", 1)[1]

    if "rq1" in wanted:
        # The only figure that needs raw generations: the per-query SD
        # distribution is not in the analysis JSON, which stores its mean.
        csv_path = results_dir / "generations.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found -- rq1 draws per-query SDs from it")
        scores = load_scores(csv_path, metric, primary)
        scores.pop("nocontext", None)
        save(fig_rq1(scores, analysis[primary_key], primary), "rq1_permutation_sd")

    if "rq2" in wanted:
        save(fig_rq2_oae(analysis, budgets), "rq2_order_adjusted_effect")
    if "rq3" in wanted:
        save(fig_rq3_rank_flip(analysis, budgets), "rq3_rank_flip_rate")
    if "rq4" in wanted:
        save(fig_rq4_placebo_gap(analysis, budgets), "rq4_placebo_gap")

    if "selection" in wanted:
        probe_path = results_dir / "selection_stability.json"
        if probe_path.exists():
            with open(probe_path, "r", encoding="utf-8") as fh:
                save(fig_selection_stability(json.load(fh)), "selection_stability")
        elif only:
            raise FileNotFoundError(
                f"{probe_path} not found -- run "
                f"`python -m src.selection_probe --config ...` first (needs the GPU); "
                f"this one is not produced by the main run"
            )
        else:
            print(
                f"  skipped selection_stability: no {probe_path.name} "
                f"(run `python -m src.selection_probe --config ...`)",
                flush=True,
            )

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--only", nargs="*", choices=FIGURE_NAMES,
        help="draw only these figures (default: all)",
    )
    parser.add_argument("--metric", default="f1", choices=("f1", "em"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    print(f"figures from {args.config}")
    written = make_figures(
        Config.load(args.config), only=args.only, metric=args.metric, dpi=args.dpi
    )
    print(f"{len(written)} figure(s) written")


if __name__ == "__main__":
    main()
