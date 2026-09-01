"""Does an LLM pruner pick the same passages when shown them in a different order?

    python -m src.selection_probe --config configs/main.yaml --n 100

This is the study's thesis turned on the pruner rather than on the answer, and
it is a *robustness check*, not a confirmatory endpoint -- it is outside the
registered Holm family (ANALYSIS_PLAN Sec. 5) and nothing here may be reported as
though it were inside it.

Why it needs its own probe rather than falling out of the main run
-----------------------------------------------------------------
It cannot be recovered from `generations.csv`. In the grid, a pruner selects
once from the as-given order and the P permutations then reorder *what it
selected* -- selection and ordering are separate steps, which is the design
invariant the whole study rests on. So `kept` is identical across all five
permutations of a cell by construction, and the CSV has nothing to say about
what the pruner would have picked had it been *shown* a different order. That
question needs the passages re-presented, which is what `selection_stability`
does and what this module drives.

Three quantities, so the observed number has something to sit between
--------------------------------------------------------------------
* **The arm itself.** Selection Jaccard across presentations, per query.
* **An order-invariant reference, measured rather than asserted.** `rerank_topk`
  scores each passage independently against the query, so reordering the input
  cannot move its selection -- but "cannot" is a claim about an implementation,
  and this runs it through the same permutations and checks. A reference that is
  assumed rather than measured is exactly the kind of thing that is quietly
  wrong.
* **Chance.** Three random k-subsets of n, simulated. Without it, a Jaccard of
  0.263 is unreadable: the interesting fact is that it is well above chance and
  still nowhere near order-invariant.

The subsample is nested (`data._stratified_order`), so the first 20 queries at
this seed are exactly the 20 the 2026-08-29 probe used. The output therefore
carries a `nested_prefix` block that recomputes the summary on that prefix,
which turns the original figure into something checkable rather than something
remembered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .chunks import Chunk, permute
from .data import load_dataset
from .prune import get_pruner, parse_arm
from .prune.llm_pruner import selection_stability
from .run import Config, build_generator, decode_params

DEFAULT_ORDERS = ("rank", "reverse", "random")

#: The arm whose selection is order-dependent, and the independent scorer it is
#: measured against. Both are read from the same config the run used, so the
#: probe cannot drift from the grid it is describing.
PROBE_ARM = "llm_pruner"
REFERENCE_ARM = "rerank_topk"


def _jaccard(sets: Sequence[set[int]]) -> float:
    union = set().union(*sets)
    return len(set.intersection(*sets)) / len(union) if union else 1.0


def chance_jaccard(
    n_chunks: int, budget: int, orders: int, seed: int, trials: int = 20_000
) -> dict[str, float]:
    """Jaccard of `orders` independent k-subsets, by simulation.

    Closed form exists but is fiddly for three-way intersections, and the
    simulation is instant and obviously correct.
    """
    rng = np.random.default_rng(seed)
    values = [
        _jaccard([set(rng.choice(n_chunks, size=budget, replace=False)) for _ in range(orders)])
        for _ in range(trials)
    ]
    return {"mean": float(np.mean(values)), "trials": trials, "n_chunks": n_chunks}


def order_invariance(
    pruner: Any, query: str, chunks: Sequence[Chunk], budget: int, orders: Sequence[str], seed: int
) -> dict[str, Any]:
    """Re-present the passages to an independent scorer and compare selections.

    Unlike `selection_stability`, which drives `llm_pruner`'s own recorded
    `selection_order`, this permutes the chunk *sequence* handed to `select`.
    That is the right probe for an arm with no presentation-order parameter:
    the chunks arrive in a different order, and since `select` returns indices,
    an order-invariant scorer must return the same set regardless.
    """
    sels = {
        order: set(pruner.select(query, permute(chunks, order, seed=seed, key="selection"), budget))
        for order in orders
    }
    return {
        "selections": {k: sorted(v) for k, v in sels.items()},
        "jaccard": _jaccard(list(sels.values())),
    }


def _summarise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    js = np.array([r["jaccard"] for r in records], dtype=float)
    changed = float(np.mean(js < 1.0)) if js.size else 0.0
    return {
        "n": int(js.size),
        "mean_jaccard": float(js.mean()) if js.size else 0.0,
        "median_jaccard": float(np.median(js)) if js.size else 0.0,
        "changed_fraction": changed,
        "n_changed": int((js < 1.0).sum()),
    }


def probe(
    cfg: Config,
    n_queries: int = 100,
    budget: int | None = None,
    orders: Sequence[str] = DEFAULT_ORDERS,
    prefix_n: int = 20,
) -> dict[str, Any]:
    """Run the probe and return its record. Writes nothing; `main` saves it."""
    data_cfg = cfg["data"]
    seed = data_cfg.get("seed", 20260828)
    examples = load_dataset(
        name=data_cfg["dataset"],
        split=data_cfg.get("split", "validation"),
        n_queries=n_queries,
        seed=seed,
        stratify_by=data_cfg.get("stratify_by"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    if budget is None:
        budget = int(cfg["metrics"].get("primary_budget", cfg["budgets"][0]))

    arm_params = cfg.get("arm_params", {}) or {}
    generator = build_generator(cfg)
    params = decode_params(cfg)

    pruner = get_pruner(PROBE_ARM, **(arm_params.get(parse_arm(PROBE_ARM)[0], {}) or {}))
    pruner.attach(generator, params)

    reference = get_pruner(REFERENCE_ARM, **(arm_params.get(parse_arm(REFERENCE_ARM)[0], {}) or {}))
    if getattr(reference, "needs_generator", False):
        reference.attach(generator, params)

    per_query: list[dict[str, Any]] = []
    reference_records: list[dict[str, Any]] = []
    for i, ex in enumerate(examples, start=1):
        result = selection_stability(pruner, ex.question, ex.chunks, budget, orders)
        per_query.append({
            "qid": ex.qid,
            "jaccard": result["jaccard"],
            "stable": result["stable"],
            "selections": {k: sorted(v) for k, v in result["selections"].items()},
        })
        reference_records.append(
            order_invariance(reference, ex.question, ex.chunks, budget, orders, seed)
        )
        if i % 10 == 0 or i == len(examples):
            running = _summarise(per_query)["mean_jaccard"]
            print(f"  {i}/{len(examples)} queries, running mean Jaccard {running:.4f}", flush=True)

    n_chunks = len(examples[0].chunks) if examples else 10
    out: dict[str, Any] = {
        "config": {
            "arm": PROBE_ARM,
            "reference_arm": REFERENCE_ARM,
            "model": cfg["generator"]["model"],
            "n_queries": len(examples),
            "budget": budget,
            "orders": list(orders),
            "seed": seed,
            "n_chunks": n_chunks,
            "note": "robustness check, outside the registered confirmatory family",
        },
        "summary": _summarise(per_query),
        "reference_order_invariant": _summarise(reference_records),
        "chance": chance_jaccard(n_chunks, budget, len(orders), seed),
        "per_query": per_query,
    }
    # The 2026-08-29 probe's population, recomputed: nested subsampling makes
    # these the same queries, so this either reproduces 0.263 or says it does not.
    if len(per_query) > prefix_n:
        out["nested_prefix"] = {"prefix_n": prefix_n, **_summarise(per_query[:prefix_n])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=100, help="queries to probe (default 100)")
    parser.add_argument("--budget", type=int, default=None, help="default: the primary budget")
    parser.add_argument("--orders", nargs="*", default=list(DEFAULT_ORDERS))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    out = probe(cfg, n_queries=args.n, budget=args.budget, orders=args.orders)

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "selection_stability.json"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)

    s, ref, chance = out["summary"], out["reference_order_invariant"], out["chance"]
    print(f"\n{PROBE_ARM}: mean Jaccard {s['mean_jaccard']:.4f} over {s['n']} queries; "
          f"selection changed in {s['n_changed']}/{s['n']}")
    print(f"{REFERENCE_ARM} (order-invariant reference): {ref['mean_jaccard']:.4f}")
    print(f"chance ({out['config']['orders'].__len__()} random {out['config']['budget']}-subsets "
          f"of {out['config']['n_chunks']}): {chance['mean']:.4f}")
    if "nested_prefix" in out:
        p = out["nested_prefix"]
        print(f"first {p['prefix_n']} queries (the 2026-08-29 population): "
              f"{p['mean_jaccard']:.4f}, changed {p['n_changed']}/{p['n']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
