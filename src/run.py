"""Experiment driver. Entry point for every Makefile target.

Not in the plan's Sec. 6 tree, but the Makefile needs something to call. Keeps
the orchestration in one place so `analysis/` notebooks stay free of analysis
logic, per the repo rule.

    python -m src.run --config configs/pilot.yaml
    python -m src.run --config configs/pilot.yaml --backend dummy   # plumbing only
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .cache import GenerationCache
from .chunks import keep, permutation_set
from .data import load_dataset, memorization_filter
from .generate import (
    ALT_TEMPLATE,
    DEFAULT_TEMPLATE,
    CachedGenerator,
    DecodeParams,
    DummyGenerator,
    GroqGenerator,
    LocalGenerator,
    build_prompt,
)
from .metrics import exact_match, token_f1
from .prune import expand_arms, get_pruner, parse_arm

TEMPLATES = {"default": DEFAULT_TEMPLATE, "alt": ALT_TEMPLATE}


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def build_generator(cfg: Config) -> CachedGenerator:
    gen_cfg = cfg["generator"]
    backend_name = gen_cfg["backend"]
    if backend_name == "local":
        backend = LocalGenerator(
            model=gen_cfg["model"],
            quantization=gen_cfg.get("quantization", "4bit"),
            batch_size=gen_cfg.get("batch_size", 8),
            max_vram_fraction=gen_cfg.get("max_vram_fraction"),
            batch_pause_s=gen_cfg.get("batch_pause_s", 0.0),
        )
    elif backend_name == "groq":
        backend = GroqGenerator(model=gen_cfg["model"])
    elif backend_name == "dummy":
        backend = DummyGenerator()
    else:
        raise ValueError(f"unknown generator backend: {backend_name!r}")

    return CachedGenerator(
        backend=backend,
        cache=GenerationCache(cfg["cache"]["path"]),
        # Flush a batch at a time so an interrupted overnight run keeps
        # everything it has already paid for, and prints as it goes.
        flush_every=max(1, gen_cfg.get("batch_size", 8)) * 8,
        progress=True,
    )


def decode_params(cfg: Config) -> DecodeParams:
    gen_cfg = cfg["generator"]
    params = DecodeParams(
        max_new_tokens=gen_cfg.get("max_new_tokens", 32),
        temperature=gen_cfg.get("temperature", 0.0),
        do_sample=gen_cfg.get("do_sample", False),
        seed=gen_cfg.get("seed", 20260828),
    )
    # Guard the one assumption the whole design rests on. Sampling noise and
    # permutation noise would be confounded and nothing downstream would mean
    # anything (plan Sec. 4.2).
    if params.do_sample or params.temperature != 0.0:
        raise ValueError(
            "greedy decoding is required: set do_sample=false and temperature=0.0"
        )
    return params


def run(cfg: Config) -> Path:
    """Execute the full (query, arm, budget, permutation) grid and write rows."""
    data_cfg = cfg["data"]
    examples = load_dataset(
        name=data_cfg["dataset"],
        split=data_cfg.get("split", "validation"),
        n_queries=data_cfg.get("n_queries"),
        seed=data_cfg.get("seed", 20260828),
        stratify_by=data_cfg.get("stratify_by"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    print(f"loaded {len(examples)} queries")

    generator = build_generator(cfg)
    params = decode_params(cfg)
    perm_cfg = cfg["permutation"]
    arm_params = cfg.get("arm_params", {}) or {}
    template_name = cfg.get("prompt_template", "default")
    if template_name not in TEMPLATES:
        raise ValueError(
            f"unknown prompt_template {template_name!r}; "
            f"expected one of {sorted(TEMPLATES)}"
        )
    template = TEMPLATES[template_name]
    started = time.time()

    # The nocontext arm runs first: it defines the filtered analysis population
    # that every other arm is scored on (plan Sec. 4.1). One ordering exists for
    # an empty context, so it collapses to P=1 rather than paying for five
    # identical generations.
    rows: list[dict[str, Any]] = []
    nocontext_preds: dict[str, str] = {}
    if "nocontext" in cfg["arms"]:
        prompts = [build_prompt(ex.question, [], template) for ex in examples]
        for ex, gen in zip(examples, generator.generate_batch(prompts, params)):
            nocontext_preds[ex.qid] = gen.text
            rows.append(_row(ex, "nocontext", 0, 0, "none", [], [], 0, gen.text))
        print(f"nocontext arm: {len(nocontext_preds)} generations")

    if data_cfg.get("memorization_filter"):
        if not nocontext_preds:
            raise ValueError(
                "memorization_filter is on but the nocontext arm is not in "
                "cfg['arms']; the filter needs its predictions"
            )
        before = len(examples)
        examples = memorization_filter(examples, nocontext_preds)
        print(f"memorization filter: {before} -> {len(examples)} queries")

    # `placebo_pos` in a config means all three positional strategies, each as
    # its own arm (`placebo_pos:middle_first`, ...). They test three different
    # hypotheses about where this generator's position bias lives and are
    # reported separately; averaging them would answer none of the three.
    arms = expand_arms([a for a in cfg["arms"] if a != "nocontext"])
    budgets = list(cfg["budgets"])
    n_cells = len(examples) * len(arms) * len(budgets)
    print(f"{n_cells} cells x {len(perm_cfg['strategies'])} permutations")

    # Batch across the whole grid, not per cell: an 8-wide batch of 5-permutation
    # cells would otherwise waste most of every batch.
    pending_prompts: list[str] = []
    pending_meta: list[tuple[Any, str, int, int, str, list[int], list[int]]] = []

    # Arm-major, not query-major. Each pruner is constructed, used for every
    # query, then closed before the next one is built, so at most one pruner
    # model is resident at a time -- Provence alone is 1.74 GB against the
    # generator's 2.06 GB on a 6 GB card. (Arms that share the run's generator
    # are listed after the ones holding their own weights in the shipped
    # configs; if that is ever reordered, the VRAM cap turns the overlap into a
    # clean OOM rather than a display crash.) Constructing once per arm also
    # matters on its own: rebuilding per cell would reload a checkpoint
    # thousands of times.
    arm_stats: dict[str, Any] = {}

    for arm in arms:
        # arm_params are keyed by base name, so all three placebo variants and
        # both provence variants share one entry.
        pruner = get_pruner(arm, **(arm_params.get(parse_arm(arm)[0], {}) or {}))
        if getattr(pruner, "needs_generator", False):
            # llm_pruner asks the study's own generator which chunks to keep,
            # so its selection calls go through the same cache as everything
            # else and replay for free on a rerun.
            pruner.attach(generator, params)

        for ex in examples:
          for budget in budgets:
            selected = pruner.select(ex.question, ex.chunks, budget)
            kept_chunks = keep(ex.chunks, selected)
            # Third step, separate from selection and ordering: compression
            # methods rewrite chunk *content* (Provence prunes sentences).
            # Identity for every other arm. See prune.base.Pruner.rewrite.
            kept_chunks = pruner.rewrite(ex.question, kept_chunks)

            orderings = permutation_set(
                kept_chunks,
                perm_cfg["strategies"],
                seed=perm_cfg.get("seed", 20260828),
                # Per-query, so the three random draws are not one shared trio
                # reused for the whole dataset. See chunks.permute.
                key=ex.qid,
            )
            for perm_idx, ordering in enumerate(orderings):
                pending_prompts.append(build_prompt(ex.question, ordering, template))
                pending_meta.append(
                    (
                        ex,
                        arm,
                        budget,
                        perm_idx,
                        perm_cfg["strategies"][perm_idx],
                        selected,
                        [c.idx for c in ordering],
                        # Arms compress at different granularities, so keep-k is
                        # not a common currency across them (plan Sec. 4.3).
                        # Recorded at generation time because it cannot be
                        # reconstructed later once the rewritten text is gone.
                        sum(len(c.text) for c in ordering),
                    )
                )

        # Repair counters, where an arm keeps them. llm_pruner records how often
        # the model named too many passages or too few; an arm that failed to
        # name k in a large fraction of cells is not the arm it claims to be, so
        # this is reported rather than buried.
        stats = getattr(pruner, "stats", None)
        if stats is not None:
            arm_stats[arm] = stats.as_dict()
            print(f"  {arm}: {arm_stats[arm]}")
        pruner.close()

    generations = generator.generate_batch(pending_prompts, params)
    for meta, gen in zip(pending_meta, generations):
        rows.append(_row(*meta, gen.text))

    elapsed = time.time() - started
    print(
        f"cache: {generator.hits} hits, {generator.misses} misses "
        f"| {elapsed:.1f}s ({elapsed / max(1, generator.misses):.2f}s per new generation)"
    )

    if not rows:
        raise ValueError(
            "no rows produced: check that cfg['arms'] and cfg['budgets'] are "
            "non-empty and that the dataset subsample is not empty"
        )

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")

    if arm_stats:
        stats_path = out_dir / "arm_stats.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(arm_stats, fh, indent=2, sort_keys=True)
            fh.write(chr(10))
        print(f"wrote arm repair counters -> {stats_path}")

    return out_path


def _row(
    ex: Any,
    arm: str,
    budget: int,
    perm_idx: int,
    perm_strategy: str,
    selected: list[int],
    order: list[int],
    context_chars: int,
    prediction: str,
) -> dict[str, Any]:
    return {
        "qid": ex.qid,
        "arm": arm,
        "budget": budget,
        "perm": perm_idx,
        "perm_strategy": perm_strategy,
        "hop_type": ex.hop_type,
        "kept": json.dumps(selected),
        "order": json.dumps(order),
        # Where the gold paragraphs ended up *after* pruning and permutation.
        # This is the column that separates content selection from positional
        # promotion, so it has to be recorded at generation time -- it cannot be
        # reconstructed later.
        "gold_positions": json.dumps(
            [order.index(g) for g in ex.gold_chunk_ids if g in order]
        ),
        "n_gold_kept": sum(1 for g in ex.gold_chunk_ids if g in order),
        "context_chars": context_chars,
        "prediction": prediction,
        "gold": ex.answer,
        "em": exact_match(prediction, ex.answer),
        "f1": token_f1(prediction, ex.answer),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arms", nargs="*", help="override the config arm list")
    parser.add_argument("--backend", help="override the generator backend")
    parser.add_argument("--n", type=int, help="override n_queries")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.arms is not None:
        cfg.raw["arms"] = args.arms
    if args.backend:
        cfg.raw["generator"]["backend"] = args.backend
        if args.backend == "dummy":
            cfg.raw["cache"]["path"] = "cache/dummy.sqlite"
            cfg.raw["output"]["results_dir"] += "_dummy"
    if args.n is not None:
        cfg.raw["data"]["n_queries"] = args.n
    if args.figures_only:
        raise NotImplementedError("figure generation not implemented (week 6)")

    run(cfg)


if __name__ == "__main__":
    main()
