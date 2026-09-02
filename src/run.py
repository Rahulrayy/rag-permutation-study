"""Experiment driver. Entry point for every Makefile target.

Not in the plan's Sec. 6 tree, but the Makefile needs something to call. Keeps
the orchestration in one place so `analysis/` notebooks stay free of analysis
logic, per the repo rule.

    python -m src.run --config configs/pilot.yaml
    python -m src.run --config configs/pilot.yaml --backend dummy   # plumbing only
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from .cache import ArtifactCache, GenerationCache, cache_key
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

#: Seconds between selection-progress lines. Long enough that arms deciding
#: instantly print nothing at all, short enough that a multi-hour hosted run
#: never looks hung.
SELECTION_PROGRESS_S = 30.0

#: The cache every shipped config points at. `quarantine_dummy` steers dummy
#: runs away from this one and leaves any other path untouched.
SHARED_CACHE = "cache/generations.sqlite"


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
        backend = GroqGenerator(
            model=gen_cfg["model"],
            concurrency=gen_cfg.get("concurrency", 4),
        )
    elif backend_name == "dummy":
        backend = DummyGenerator()
    else:
        raise ValueError(f"unknown generator backend: {backend_name!r}")

    return CachedGenerator(
        backend=backend,
        cache=GenerationCache(cfg["cache"]["path"]),
        # Flush a batch at a time so an interrupted overnight run keeps
        # everything it has already paid for, and prints as it goes.
        #
        # Overridable because the right block size is a property of the backend,
        # not of the grid: nothing is written until a whole block completes, so
        # the default 64 is ~1 minute of exposure on the local GPU but well over
        # an hour on a rate-limited hosted run, where the calls are the thing
        # you cannot afford to repeat. See configs/replication.yaml.
        flush_every=gen_cfg.get("flush_every", max(1, gen_cfg.get("batch_size", 8)) * 8),
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


def quarantine_dummy(cfg: Config) -> None:
    """Redirect dummy-backend output away from the real cache and results dir.

    `DummyGenerator` returns a hash of the prompt. Its numbers are meaningless by
    construction, so they must never land where a real result is expected.

    This used to live in `main()`, which meant the guard only covered the CLI.
    Calling `run()` directly with a dummy backend -- which the tests do, and
    which is the natural way to drive it from a notebook -- wrote hash noise into
    the genuine results directory under a name indistinguishable from real
    output. That happened: `results/replication_groq/` was populated with dummy
    rows whose `llm_pruner` stats read `under_selected_rate: 1.0`, because the
    dummy answers parse as no passage numbers at all.

    Idempotent, so a config that has already been redirected is not suffixed
    twice.
    """
    if cfg["generator"].get("backend") != "dummy":
        return

    out = cfg["output"]["results_dir"]
    if not out.endswith("_dummy"):
        cfg.raw["output"]["results_dir"] = out + "_dummy"

    # The cache redirect is hygiene rather than safety, and so is conditional.
    # Dummy rows cannot corrupt a real result -- cache_key hashes the model name
    # and DummyGenerator's is "dummy" -- they would just be clutter in the
    # shared file. So redirect only the shared default, and leave an explicitly
    # chosen path alone: a caller that named its own cache (a test using tmp_path,
    # say) meant it, and silently overriding that would break isolation.
    if cfg["cache"]["path"] == SHARED_CACHE:
        cfg.raw["cache"]["path"] = "cache/dummy.sqlite"


def run(cfg: Config) -> Path:
    """Execute the full (query, arm, budget, permutation) grid and write rows."""
    # Before anything reads the cache path or the results dir.
    quarantine_dummy(cfg)
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
    # Shares the generation cache's file, in a separate table. Arms that run
    # their own encoder attach to it below, so their selections survive a
    # restart the way generations already do.
    artifact_cache = ArtifactCache(cfg["cache"]["path"])
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
            # llm_pruner asks the study's own generator which chunks to keep and
            # loo_oracle asks it to score the reference answer, so both sets of
            # calls go through the same cache as everything else and replay for
            # free on a rerun. The rest of `run_state` is what an attached arm
            # needs to build prompts the way this run builds them; each arm takes
            # the parts it uses and ignores the rest.
            pruner.attach(
                generator,
                params,
                template=template,
                orders=perm_cfg["strategies"],
                seed=perm_cfg.get("seed", 20260828),
            )
        if getattr(pruner, "wants_cache", False):
            # Arms that run their own encoder. Their work lands in no cache
            # otherwise, so an interrupted run re-pays every selection pass on
            # restart -- ~90 minutes on the 2Wiki config, where these arms are on
            # CPU. Generations have always survived a restart; this gives
            # selections the same property.
            pruner.attach_cache(artifact_cache)
        if getattr(pruner, "needs_answers", False):
            # loo_oracle scores logP(gold answer | context), which `select` has
            # no room for in its signature. It is a ceiling, not a method: see
            # prune.loo_oracle, and ANALYSIS_PLAN Sec. 8 on why its Oracle Gap
            # stays out of the confirmatory family.
            pruner.attach_answers(examples)

        # Selection progress, printed on a timer rather than a counter.
        #
        # Most arms decide instantly and would only add noise; `llm_pruner` and
        # `loo_oracle` call the generator once (or n+1 times) per cell, which on
        # a rate-limited hosted backend is ~12s each. That is upwards of half an
        # hour during which nothing was printed, in the one phase where the
        # per-cell work does NOT go through `CachedGenerator.generate_batch` and
        # so gets none of its block reporting. An idle-looking log is exactly
        # what the flush_every blocks exist to prevent (see cache.py); this
        # closes the same gap on the path that bypasses them.
        #
        # Time-based, so an arm that finishes inside one interval stays silent
        # and no threshold has to be tuned per arm.
        arm_cells = len(examples) * len(budgets)
        cell_i = 0
        arm_started = last_print = time.time()

        for ex in examples:
          for budget in budgets:
            selected = pruner.select(ex.question, ex.chunks, budget)
            kept_chunks = keep(ex.chunks, selected)
            # Third step, separate from selection and ordering: compression
            # methods rewrite chunk *content* (Provence prunes sentences).
            # Identity for every other arm. See prune.base.Pruner.rewrite.
            kept_chunks = pruner.rewrite(ex.question, kept_chunks, budget)

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

            cell_i += 1
            now = time.time()
            if now - last_print >= SELECTION_PROGRESS_S:
                done = now - arm_started
                # Remaining is a flat extrapolation of the mean rate so far. On
                # a rate-limited backend that is honest enough to plan an
                # evening around, which is all it is for.
                eta = done / cell_i * (arm_cells - cell_i)
                print(
                    f"    {arm}: selected {cell_i}/{arm_cells} cells "
                    f"({done / 60:.0f}m elapsed, ~{eta / 60:.0f}m left)",
                    flush=True,
                )
                last_print = now

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

    audit_n = cfg.get("audit_determinism") or 0
    if audit_n:
        audit = audit_determinism(
            generator, pending_prompts, params, audit_n, perm_cfg.get("seed", 20260828)
        )
        audit_path = out_dir / "determinism_audit.json"
        with open(audit_path, "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2, sort_keys=True)
            fh.write(chr(10))
        print(f"wrote determinism audit -> {audit_path}")

    return out_path


def audit_determinism(
    generator: CachedGenerator,
    prompts: Sequence[str],
    params: DecodeParams,
    n: int,
    seed: int,
) -> dict[str, Any]:
    """Re-issue already-cached prompts and report how many come back identical.

    Turns the hosted-backend caveat into a number. `temperature=0` on a hosted
    API is best-effort, not greedy: serving batches requests across tenants, and
    for an MoE model the batch composition changes the arithmetic. The study's
    primary endpoint IS a within-query variance, so drift would land inside the
    quantity being reported with nothing to separate it from position bias.

    Two structural defences already exist and this measures what is left. First,
    the cache pays for each distinct prompt once, so a cell is never *itself*
    re-sampled. Second, `run` emits the P permutations of a cell consecutively,
    so the calls the primary SD is computed from are issued seconds apart --
    day-scale drift therefore lands between cells, not inside the endpoint.
    Neither argument covers a run that spans a rate-limit day boundary, which
    this grid does, hence this check.

    Run it on the SECOND day, when the cache is warm and the routing has had a
    chance to change: that is the comparison with teeth. Same-session repeats
    measure very little.

    Deliberately bypasses the cache wrapper -- going through it would return the
    stored answer and report a perfect score, which is the one result this
    cannot be allowed to produce.
    """
    distinct = sorted(set(prompts))
    if not distinct:
        return {"checked": 0, "identical": 0, "divergences": []}

    sample = random.Random(seed).sample(distinct, min(n, len(distinct)))
    checked = 0
    divergences: list[dict[str, str]] = []
    # Day composition of what was actually re-issued. The identical rate is only
    # the strong version of this test if the answers it compares against were
    # generated on earlier days; recording the spread means a reader does not
    # have to take that on trust.
    sampled_days: collections.Counter[str] = collections.Counter()

    for prompt in sample:
        key = cache_key(generator.model, prompt, params.as_key())
        cached = generator.cache.get(key)
        if cached is None:
            # Not generated yet -- a partial run. Nothing to compare against.
            continue
        checked += 1
        created = generator.cache.created_at(key)
        sampled_days[created[:10] if created else "unknown"] += 1
        fresh = generator.backend.generate(prompt, params)
        if fresh.text != cached.text:
            divergences.append(
                {
                    "prompt_sha256": key,
                    "cached": cached.text,
                    "fresh": fresh.text,
                }
            )

    identical = checked - len(divergences)
    rate = identical / checked if checked else float("nan")
    print(
        f"determinism audit: {identical}/{checked} identical on re-issue "
        f"({rate:.1%})"
    )
    spread = ", ".join(f"{d} x{c}" for d, c in sorted(sampled_days.items()))
    print(f"    re-issued answers were first generated on: {spread}")
    for d in divergences[:5]:
        # ascii(), not !r: this is the only place a *generation* reaches stdout,
        # and Windows consoles default to cp1252. A 2WikiMultihopQA answer such
        # as "Malgorzata Braunek" (with the Polish l-stroke) would otherwise
        # raise UnicodeEncodeError and take the audit down at the point where it
        # is reporting a divergence, which is exactly when the output matters.
        # The JSON written beside it is utf-8 and keeps the real characters.
        print(f"    cached {ascii(d['cached'])} -> fresh {ascii(d['fresh'])}")
    return {
        "checked": checked,
        "identical": identical,
        "identical_rate": rate,
        "model": generator.model,
        "sampled_first_generated_on": dict(sorted(sampled_days.items())),
        "divergences": divergences,
    }


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
    parser.add_argument(
        "--audit",
        type=int,
        metavar="N",
        help="re-issue N already-cached prompts and report how many come "
        "back identical. Run this on the second day of a multi-day hosted "
        "run: same-session repeats measure very little.",
    )
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.arms is not None:
        cfg.raw["arms"] = args.arms
    if args.backend:
        cfg.raw["generator"]["backend"] = args.backend
        # The dummy redirect is not done here any more -- `run()` does it, so it
        # covers every caller and not just this one. See quarantine_dummy.
    if args.n is not None:
        cfg.raw["data"]["n_queries"] = args.n
    if args.audit is not None:
        cfg.raw["audit_determinism"] = args.audit
    if args.figures_only:
        # Imported here, not at module scope: matplotlib is a figure-time
        # dependency and a GPU run should not pay for importing it.
        from .figures import make_figures

        make_figures(cfg)
        return

    run(cfg)


if __name__ == "__main__":
    main()
