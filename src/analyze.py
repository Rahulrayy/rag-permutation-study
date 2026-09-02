"""Confirmatory analysis: RQ1-RQ4 over a completed run's generations.csv.

    python -m src.analyze --config configs/main.yaml
    python -m src.analyze --config configs/main.yaml --budgets 3   # primary only

`run.py` stops once generations.csv is written; everything downstream of that
lives here. The four quantities are `metrics.py`'s and their formulas are frozen
by ANALYSIS_PLAN.md -- this module only decides *what to feed them* and *how to
resample*, and both of those are the easy places to fake a result:

  * The resampling unit is the query, never the cell. `stats.two_level_bootstrap`
    enforces that; this module's job is to hand it a container whose nesting is
    intact (arm -> qid -> [score per permutation], permutation order consistent).

  * Scores are built per budget and never pooled across budgets. Pooling would
    mix cells that differ in content, and the within-query SD -- the study's
    primary endpoint -- would then be measuring the budget, not the ordering.

`nocontext` is dropped from every table here, for two independent reasons. It
runs at one permutation, so it has no within-query SD to contribute; and it is
reported over all sampled queries while every other arm is reported over the
subset surviving the memorization filter, so its mean is not commensurable with
theirs (see results/main_hotpotqa/arm_summary.csv, row `nocontext@studied`,
which recomputes the floor on the studied qids).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .metrics import (
    PerArmScores,
    order_adjusted_effect,
    oracle_gap,
    placebo_gap,
    rank_flip_rate,
    within_query_sd,
)
from .run import Config
from .stats import BootstrapResult, holm, two_level_bootstrap

#: Arms compared for the rank-flip rate. The placebo and random arms are left
#: out on purpose: RFR asks whether *method* rankings survive a change of
#: ordering, and a family that includes arms nobody would deploy inflates the
#: denominator with comparisons no reader cares about.
RFR_ARMS = (
    "full",
    "provence_rerank",
    "rerank_topk",
    "provence_full",
    "llm_pruner",
    "llmlingua2",
)

#: The Holm family, exactly as registered (ANALYSIS_PLAN Sec. 5): nine
#: comparisons at the primary budget, on the filtered population, in token-F1.
#: "One family of nine, not two families of four and five -- splitting them
#: would buy power at the cost of a reviewer reasonably calling it
#: family-splitting."
#:
#: Every other arm is bootstrapped too, because the CIs are worth having, but
#: those are exploratory and must never be added to this list: Holm's adjustment
#: depends on the family's size, so quietly widening it changes the confirmatory
#: numbers. `full`, `loo_oracle`, `random_drop` and the two exploratory placebo
#: variants are outside it by registration, not by oversight.
CONFIRMATORY_OAE = ("provence_rerank", "provence_full", "llmlingua2", "llm_pruner")
CONFIRMATORY_PLACEBO_GAP = CONFIRMATORY_OAE + ("rerank_topk",)

#: Reported uncorrected as the single pre-specified comparison *and* corrected
#: within the family, always both, so the choice cannot be made after the fact.
PRIMARY_ENDPOINT = ("placebo_gap", "provence_rerank")


def load_scores(csv_path: str | Path, metric: str, budget: str) -> dict[str, dict[str, list[float]]]:
    """Build ``arm -> qid -> [score per permutation]`` for one budget.

    Permutations are ordered by the `perm` column rather than by row order, so a
    reordered CSV cannot silently permute the permutations.
    """
    nested: dict[str, dict[str, dict[int, float]]] = collections.defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["budget"] != budget:
                continue
            nested[row["arm"]].setdefault(row["qid"], {})[int(row["perm"])] = float(row[metric])
    return {
        arm: {qid: [perms[i] for i in sorted(perms)] for qid, perms in by_qid.items()}
        for arm, by_qid in nested.items()
    }


def answer_stability(
    csv_path: str | Path, budget: str, arm: str = "full"
) -> dict[str, Any]:
    """The three-way split behind RQ1: stable, same-score-different-answer, moved.

    A within-question SD of zero means the *score* did not move, not that the
    answer did not. Separating the two is the point of the RQ1 presentation, and
    the counts were previously computed by hand -- which put a number in the
    write-up that no committed code produced, against Sec. 8's claim that every
    quantity is reproducible. It lives here now.

    Two notions of "the same answer" are reported rather than one, because they
    disagree by two questions out of 274 and the write-up called the looser one
    "byte-identical":

      ``identical``          the raw generated strings agree exactly.
      ``identical_casefold`` they agree after stripping and lowercasing.

    Case-only differences are arguably not different answers, so the looser count
    is the more meaningful one; it is simply not the byte-identical one.
    """
    by_qid: dict[str, dict[int, tuple[float, str]]] = collections.defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] == arm and row["budget"] == budget:
                by_qid[row["qid"]][int(row["perm"])] = (
                    float(row["f1"]),
                    row["prediction"],
                )
    if not by_qid:
        raise ValueError(f"no rows for arm {arm!r} at budget {budget!r} in {csv_path}")

    n = len(by_qid)
    moved = zero_sd = zero_sd_diff_answer = identical = identical_cf = 0
    for cells in by_qid.values():
        f1s = [cells[i][0] for i in sorted(cells)]
        preds = [cells[i][1] for i in sorted(cells)]
        sd = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0
        same_raw = len(set(preds)) == 1
        same_cf = len({p.strip().lower() for p in preds}) == 1
        identical += same_raw
        identical_cf += same_cf
        if sd > 0:
            moved += 1
        else:
            zero_sd += 1
            zero_sd_diff_answer += not same_raw
    return {
        "arm": arm,
        "budget": budget,
        "n_queries": n,
        "n_permutations": max(len(c) for c in by_qid.values()),
        "score_moved": moved,
        "score_moved_share": moved / n,
        "zero_sd": zero_sd,
        "zero_sd_but_different_answer": zero_sd_diff_answer,
        "identical": identical,
        "identical_share": identical / n,
        "identical_casefold": identical_cf,
        "identical_casefold_share": identical_cf / n,
    }


def _boot(
    scores: PerArmScores,
    arms: Sequence[str],
    statistic: Callable[[PerArmScores], float],
    n_replicates: int,
    ci: float,
    seed: int,
) -> BootstrapResult:
    """Bootstrap restricted to the arms the statistic actually reads.

    `two_level_bootstrap` derives its resampling population from the qids shared
    by the arms in `scores`, and draws indices into that population from `seed`
    alone. Every arm in a run carries the same qids (asserted in `analyze`), so
    narrowing the container leaves the population -- and therefore every draw --
    identical, while doing a fraction of the dict-building work per replicate.
    Verified equal to the all-arm call before being relied on.
    """
    return two_level_bootstrap(
        {arm: scores[arm] for arm in arms}, statistic, n_replicates, ci, seed
    )


def _as_dict(result: BootstrapResult, with_p: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "point": result.point,
        "lo": result.lo,
        "hi": result.hi,
        "excludes_zero": bool(result.excludes_zero),
    }
    if with_p:
        out["p"] = result.p_two_sided()
    return out


def analyze_budget(
    scores: dict[str, dict[str, list[float]]],
    baseline: str,
    placebo: str,
    oracle: str | None,
    n_replicates: int,
    ci: float,
    seed: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run RQ1-RQ4 for one budget's score container."""
    qid_sets = {frozenset(by_qid) for by_qid in scores.values()}
    if len(qid_sets) != 1:
        raise ValueError(
            "arms disagree on their query sets; every comparison here is paired "
            "and _boot's arm-narrowing assumes one shared population"
        )
    arms = sorted(scores)
    # `oracle` is optional: a hosted-backend config has no loo_oracle arm,
    # because chat APIs do not return the reference answer's log-prob.
    for required in filter(None, (baseline, placebo, oracle)):
        if required not in scores:
            raise ValueError(f"arm {required!r} is not in the results")

    out: dict[str, Any] = {}

    def say(line: str) -> None:
        if verbose:
            print(line, flush=True)

    # RQ1 -- is the context order worth measuring at all.
    say("\nRQ1  mean within-query SD across permutations")
    rq1 = {}
    for arm in arms:
        res = _boot(
            scores,
            [arm],
            lambda s, a=arm: float(np.mean(list(within_query_sd(s[a]).values()))),
            n_replicates,
            ci,
            seed,
        )
        rq1[arm] = _as_dict(res)
        say(f"  {arm:26s} {res}")
    out["rq1_mean_within_query_sd"] = rq1

    # RQ2 -- the headline effect, in orderings-worth of noise.
    say(f"\nRQ2  OAE vs {baseline}")
    rq2 = {}
    for arm in arms:
        if arm == baseline:
            continue
        res = _boot(
            scores,
            [arm, baseline],
            lambda s, a=arm: order_adjusted_effect(s, a, baseline),
            n_replicates,
            ci,
            seed,
        )
        rq2[arm] = _as_dict(res, with_p=True)
        say(f"  {arm:26s} {res}  {'*' if res.excludes_zero else ''}")
    out["rq2_oae"] = rq2

    # RQ3 -- does single-order evaluation reach the permutation-averaged answer.
    rfr_family = [a for a in RFR_ARMS if a in scores]
    res = _boot(scores, rfr_family, lambda s: rank_flip_rate(s, rfr_family), n_replicates, ci, seed)
    out["rq3_rank_flip_rate"] = {"arms": rfr_family, **_as_dict(res)}
    say(f"\nRQ3  rank flip rate over {len(rfr_family)} arms: {res}")

    # RQ4 -- the primary endpoint: is the gain content selection or position.
    say(f"\nRQ4  placebo gap vs {placebo}")
    rq4 = {}
    for arm in arms:
        if arm.startswith("placebo_pos"):
            continue
        res = _boot(
            scores,
            [arm, placebo],
            lambda s, a=arm: placebo_gap(s, a, placebo),
            n_replicates,
            ci,
            seed,
        )
        rq4[arm] = _as_dict(res, with_p=True)
        say(f"  {arm:26s} {res}  {'*' if res.excludes_zero else ''}")
    out["rq4_placebo_gap"] = rq4

    # Descriptive, not tested: a ratio of means with no null worth correcting for.
    # Absent entirely when there is no oracle arm, rather than present and null:
    # a reader of the JSON should not have to tell "no oracle was run" apart
    # from "the oracle gap came out empty".
    if oracle is not None:
        out["oracle_gap"] = {a: oracle_gap(scores, a, oracle) for a in arms if a != oracle}

    # The confirmatory family. Holm is applied here and nowhere else -- the
    # per-arm blocks above carry raw p-values only, so nothing in them can be
    # mistaken for a corrected confirmatory result.
    family: dict[str, float] = {}
    for arm in CONFIRMATORY_OAE:
        if arm in rq2:
            family[f"OAE:{arm}"] = rq2[arm]["p"]
    for arm in CONFIRMATORY_PLACEBO_GAP:
        if arm in rq4:
            family[f"PlaceboGap:{arm}"] = rq4[arm]["p"]
    adjusted = holm(family)
    kind, primary_arm = PRIMARY_ENDPOINT
    primary_key = f"PlaceboGap:{primary_arm}"
    out["confirmatory_family"] = {
        "size": len(family),
        "members": {
            key: {
                "point": (rq2 if key.startswith("OAE:") else rq4)[key.split(":", 1)[1]]["point"],
                "p_raw": family[key],
                "p_holm": adjusted[key],
                # bool(), because `adjusted[key]` is a numpy float
                # wherever the p-value came through numpy, and np.bool_
                # is NOT a bool subclass the way np.float64 is a float
                # subclass -- json.dump serialises the floats happily and
                # then dies on this one field.
                "significant_holm": bool(adjusted[key] < 0.05),
            }
            for key in family
        },
        "primary_endpoint": {
            "comparison": f"{kind} of {primary_arm} vs {placebo}",
            **{k: v for k, v in rq4.get(primary_arm, {}).items()},
            "p_uncorrected": rq4.get(primary_arm, {}).get("p"),
            "p_holm": adjusted.get(primary_key),
        },
    }
    say(f"\nConfirmatory family ({len(family)} comparisons, Holm)")
    for key in sorted(family, key=lambda k: family[k]):
        mark = "*" if adjusted[key] < 0.05 else ""
        say(f"  {key:30s} raw {family[key]:.4f}  holm {adjusted[key]:.4f} {mark}")
    say(f"  primary endpoint -> {primary_key}: uncorrected {family.get(primary_key)}, "
        f"Holm {adjusted.get(primary_key)}")
    return out


def analyze(cfg: Config, budgets: Sequence[str] | None = None, metric: str = "f1") -> dict[str, Any]:
    results_dir = Path(cfg["output"]["results_dir"])
    csv_path = results_dir / "generations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run `python -m src.run --config ...` first"
        )

    met = cfg["metrics"]
    stats_cfg = cfg.get("stats", {})
    n_replicates = stats_cfg.get("bootstrap_replicates", 10_000)
    ci = stats_cfg.get("ci", 0.95)
    seed = stats_cfg.get("seed", 20260828)

    # The registered primary budget first, so a run interrupted partway through
    # the secondary budgets still leaves the confirmatory result on disk.
    all_budgets = [str(b) for b in cfg["budgets"]]
    primary = str(met.get("primary_budget", all_budgets[len(all_budgets) // 2]))
    ordered = [primary] + [b for b in all_budgets if b != primary]
    wanted = [b for b in ordered if budgets is None or b in set(budgets)]

    out: dict[str, Any] = {
        "config": {
            "baseline": met["baseline_arm"],
            "placebo": met["placebo_arm"],
            "oracle": met.get("oracle_arm"),
            "metric": metric,
            "n_replicates": n_replicates,
            "ci": ci,
            "seed": seed,
            "primary_budget": primary,
            "budgets_analyzed": wanted,
            "excluded_arms": ["nocontext"],
        }
    }
    out_path = results_dir / "permutation_analysis.json"

    for budget in wanted:
        scores = load_scores(csv_path, metric, budget)
        scores.pop("nocontext", None)
        if not scores:
            raise ValueError(f"no rows for budget {budget!r} in {csv_path}")
        print(f"\n{'=' * 74}")
        print(f"BUDGET k={budget}  ({metric.upper()}, {n_replicates} replicates, seed {seed})")
        print(f"{'=' * 74}", flush=True)
        out[f"budget_{budget}"] = analyze_budget(
            scores,
            baseline=met["baseline_arm"],
            placebo=met["placebo_arm"],
            oracle=met.get("oracle_arm"),
            n_replicates=n_replicates,
            ci=ci,
            seed=seed,
        )
        out[f"budget_{budget}"]["answer_stability"] = answer_stability(
            csv_path, budget
        )

        # Checkpoint after every budget: the whole grid is ~70 minutes and the
        # primary result is finished within the first third of it.
        #
        # Serialise to a temporary file and rename, rather than opening the real
        # path in "w" mode. Opening it truncates immediately, so a dump that
        # raises partway -- as one did on 2026-09-01, on a numpy scalar --
        # loses the previous analysis as well as failing to write the new one.
        # os.replace is atomic on Windows and POSIX alike.
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, out_path)
        print(f"\n[checkpoint] budget {budget} -> {out_path}", flush=True)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--budgets",
        nargs="*",
        help="restrict to these budgets (default: all, primary first)",
    )
    parser.add_argument(
        "--metric",
        default="f1",
        choices=("f1", "em"),
        help="score column to analyse (default: f1)",
    )
    args = parser.parse_args()
    analyze(Config.load(args.config), budgets=args.budgets, metric=args.metric)


if __name__ == "__main__":
    main()
