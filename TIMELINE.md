# Timeline

What has been done, and what is left. Status as of **2026-08-29**, branch
`master`.

This file is chronological and deliberately thin on rationale. For the *why* of
any decision, read the other three:

| Document | Answers |
|---|---|
| [`rag-permutation-project-plan.md`](rag-permutation-project-plan.md) | Why this project exists; the design |
| [`HANDOFF.md`](HANDOFF.md) | Current state, locked decisions, design invariants |
| [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) | What will be measured, fixed before the data |

---

## Status in one line

**Week 1 is complete. The gate PASSED at 0.0263 (threshold 0.02).** The premise
holds at this scale: answer quality does vary across permutations of a fixed
context. Week 2 is cleared to start.

Read the caveat in the gate entry below before quoting that number — the median
is a knife-edge statistic on this distribution, and the real result is stronger
and stranger than 0.0263 suggests.

```
week 1  #####################  DONE, gate PASSED 0.0263
week 2  #####...............  provence done; 3 arms + registration left
week 3  ....................  LOO oracle, memorization filter, lit re-check
week 4  ....................  main run n=300, bootstrap
week 5  ....................  2WikiMultihopQA + Groq replication
week 6  ....................  figures, write-up, polish
```

---

## Done

### 2026-08-28 — Week 1 build (`0ce0287`)

Environment verified end to end: Python 3.11.9, torch 2.11.0+cu128, RTX 3060
Laptop (6.4 GB), Qwen2.5-3B-Instruct in 4-bit nf4 loading at 2.06 GB.
`requirements.lock` committed.

Infrastructure, all with tests:

- `chunks.py` — chunk representation, seeded permutation protocol (Sec. 4.4)
- `data.py` — HotpotQA distractor loader, memorization filter, seeded /
  stratified / nested subsampling
- `cache.py` — SQLite content-hash cache on `sha256(model, prompt, decode_params)`
- `generate.py` — `LocalGenerator` (4-bit, greedy, answer log-probs), prompt
  templates, caching wrapper
- `metrics.py` — EM, token-F1, OAE, Rank Flip Rate, placebo gap, oracle gap
- `stats.py` — two-level bootstrap, Holm correction
- `gate.py` — week-1 kill criterion and the Sec. 9 escalation ladder
- `run.py` — experiment driver; `smoke.py` — GPU check

Arms implemented: `full`, `nocontext`, `random_drop`, `placebo_pos` — including
the study's novel positional-placebo control.

Measurements made along the way, none of them the gate:

- The dataset has **no positional confound** — gold paragraphs are spread
  near-uniformly across all ten slots (1,427–1,507 occurrences per slot)
- Hop split 5,899 bridge / 1,446 comparison (80.3% / 19.7%); median context
  ~1,381 tokens
- 60 of 7,405 validation rows (0.81%) do not ship exactly 10 paragraphs —
  excluded as a comparability exclusion. Working population **7,345**
- Prompt template chosen on a 12-query x 5-permutation probe, on **accuracy and
  answer-format match**, explicitly not on permutation SD
- The LOO oracle has signal, though **less than first recorded**: a 12.4-nat drop
  was written down here, but it does not reproduce. The committed code gives
  1.79 nats on the same example — see `HANDOFF.md` Sec. 5
- A 12-query probe showed median within-query SD of 0.067. **Twelve queries is
  not one hundred — this is not the gate.**

### 2026-08-28 — Handoff written (`0b53e43`)

`HANDOFF.md`: state, locked decisions, design invariants, next steps.

### 2026-08-29 — Correctness audit (`98cba17`)

Full read-through of every module. 22 files changed, +652/-80, **77 to 100
tests**. Every fix has a regression test that fails without it.

Bugs that would have produced wrong numbers silently:

- `gate.by_query` pooled budgets — on a main-run CSV the `full` arm's three
  keep-k cells became "15 permutations", and the kill criterion would have
  reported the budget contrast
- `two_level_bootstrap` drew its point estimate from a different population than
  its CI, so the point could land outside its own interval
- `rank_flip_rate` was the only unpaired metric
- `CachedGenerator.generate_batch` could silently shift every row against its
  metadata; `order_adjusted_effect` returned a silent nan

Bugs that would have cost a night:

- `main.yaml`'s `placebo_pos` block crashed the main grid, and no mechanism
  existed to run its three variants as separate arms — added `arm:variant`
- Nothing was cached until the whole run finished: a crash at hour 9 cached zero
- No dedup, so the `full` arm was generated three times over

Protocol change, free now and not later: **random permutations are seeded per
query**. They were seeded on `(seed, replicate, n)` alone, so every query saw the
identical trio of random orderings. Recorded in `ANALYSIS_PLAN.md` Sec. 4.

### 2026-08-29 — VRAM cap (`8bd02a9`)

Batch 4 ran but starved the laptop compositor and cut the monitor out. Added
`max_vram_fraction` (0.65) so an overrun raises a clean CUDA OOM instead, and
dropped `batch_size` to 2. The stopped pilot lost nothing: 224 of 500
generations were already banked and replayed.

### 2026-08-29 — **Week-1 gate: PASS**

```
median within-query SD   0.0263   >=  0.02   PASS
mean within-query SD     0.1778
mean f1                  0.4867     (100 queries x 5 permutations, `full` arm)
between-query SD         0.3815
```

Artifacts in `results/pilot_w1/`: `gate_report.txt`, `per_query_summary.csv`,
`gate_stats.json`. Raw generations stay gitignored per plan Sec. 6 and are
reproducible from the cache.

**Read this before quoting 0.0263.** The distribution is bimodal, and the median
lands on the seam:

- **Exactly 50 of 100 queries have SD = 0.** They never move, whatever the
  ordering.
- **The other 50 move enormously** — median SD among movers **0.4177**, max
  **0.5477**.
- So the median is the midpoint of the largest zero and the smallest mover:
  `(0.0000 + 0.0526) / 2 = 0.0263`. It is literally half the smallest non-zero
  SD. **With 51 static queries instead of 50 the gate would have printed FAIL**
  while the moving half looked exactly the same.

The pre-registered criterion passed on its own terms and that stands. But the
honest reading of RQ1 is not "SD is 0.026, just over the line" — it is "half the
queries are perfectly stable and half swing by ~0.42 F1 on identical content
under greedy decoding". That is a stronger result than the headline, and it is
the instance-level variance hiding behind stable averages that "Lost in the
Evidence?" (arXiv 2605.27105) describes.

Of the 50 static queries: 22 always right, 20 always wrong, 8 stuck in between.
Movement does not depend on hop type — bridge and comparison both move at
exactly 50%.

---

## Remaining

### Week 1 — DONE

- [x] Smoke test — all six sections pass
- [x] Pilot — 100 queries x 5 permutations, 500 rows
- [x] Gate — **PASS at 0.0263**, committed

### Week 2 — the remaining arms, and registration

Ordered by risk, not convenience:

- [x] **`provence`** — checkpoint verified loadable (the assumption plan Sec. 8
      calls most likely to blow up), and implemented as **two** arms:
      `provence_rerank` (selection only, content-matched) and `provence_full`
      (as published, sentence-pruned). License is `cc-by-nc-nd-4.0`
- [ ] `rerank_topk` — the OAE denominator arm; nothing else is interpretable
      without it
- [ ] `llm_pruner` — watch for it returning more than `budget` indices, and for
      the *selection prompt itself* being order-sensitive. Log the ordering used
- [ ] `llmlingua2` last — it has an unresolved design question (below)
- [ ] Fill the `ANALYSIS_PLAN.md` TODOs: primary endpoint, `nocontext`
      correctness definition, Holm family definition, H2/H3 thresholds,
      remaining exclusions, and confirm the placebo comparator
- [ ] **Register** — record the commit SHA and date in `ANALYSIS_PLAN.md` Sec. 8

Gate: all arms produce sane output on 20 queries.

### Week 3 — oracle, filter, literature

- [ ] **Re-check the literature.** A scheduled task, not an optional one:
      targeted search on *"permutation-controlled evaluation context pruning"*
      and *"positional placebo baseline RAG"*. If someone published exactly this
      in the interim, the Sec. 9 fallback applies
- [ ] `loo_oracle` — depends on `Generator.score`, which is verified working
- [ ] Memorization filter on real generations — wired, but never run
- [ ] Full pilot at n=100

Gate: OAE and RFR computable end to end.

### Week 4 — the main run

- [ ] `python -m src.run --config configs/main.yaml`, n=300
- [ ] Two-level bootstrap analysis, Holm across the method-pair family

**Compute is larger than the plan's arithmetic.** The plan says 300 x 8 arms x 3
budgets x 5 permutations = 36,000. `placebo_pos` is three arms, not one, so the
grid is 10 non-`nocontext` arms:

```
300 queries x 10 arms x 3 budgets x 5 permutations    45,000
+ nocontext (P=1, no budget)                             300
- `full` dedup (identical at every budget)            -3,000
                                                      ------
                                                      ~42,300 generations (~12 h)
```

The memorization filter shrinks this further, since it runs before the grid is
built. `HANDOFF.md` Sec. 8 asks whether all three placebo variants need to be
confirmatory — that decision is worth roughly three hours of GPU time.

Gate: headline numbers exist.

### Week 5 — replication

- [ ] `data._load_2wikimultihop` — currently a stub; same shape as HotpotQA, so
      reuse the existing `Chunk` construction rather than duplicating it
- [ ] `GroqGenerator` — currently a stub. Check
      `console.groq.com/settings/limits` before sizing; tokens-per-minute is the
      likelier binding constraint, not requests-per-day
- [ ] Groq cross-generator check, n=100

Gate: the finding holds, or does not, on a second dataset and a second model.

### Week 6 — ship

- [ ] Figures — `--figures-only` currently raises `NotImplementedError`
- [ ] Write-up, README, repo polish

Two things that must appear in the write-up: off-the-shelf checkpoints are used
out of distribution, so this measures deployed-as-published behaviour rather than
each method's ceiling; and `rank` is **as-given order**, not retriever rank —
reserve that term for the NQ-open arm, where a real retriever produces it.

---

## Open decisions

| Question | Status |
|---|---|
| How is a token-compressed context permuted? | Blocks `llmlingua2`. Honest default: permute the surviving chunk-level units, not tokens. **Decide before the arm runs, not after** |
| Which `placebo_pos` variant is confirmatory for H4? | `middle_first` proposed in `ANALYSIS_PLAN.md`, the other two exploratory. Confirm before registering |
| `nocontext` correctness: EM, or token-F1 above a cutoff? | `memorization_filter` takes a `threshold`, so both are expressible. Must be registered either way |
| Is memorization going to bite? | The 12-query probe showed mean EM 0.517, high for a 3B on multi-hop. Watch for the dangerous pattern: **low permutation SD arriving with a high mean** — that is parametric recall, not stability |
| Delete the 0-byte `main.py`? | PyCharm placeholder, committed as-is. `src/` is the real entry point |
| NQ-open single-hop contrast | Needs a Pyserini prebuilt index. The plan says add only if time allows |

---

## Things that waste time if forgotten

- **The cache makes reruns free.** 500 generations replay in under a second.
  Never hand-edit results to avoid a rerun; just rerun
- **`--backend dummy`** exercises the whole pipeline with no GPU. Its numbers are
  meaningless by construction — never report a gate result from it
- **`python -m src.smoke`** before any long job. It catches a bad checkpoint, a
  broken CUDA install or non-deterministic decoding in about a minute
- **`--n 20`** shrinks any config for a fast sanity pass
- **VRAM is the binding constraint.** `batch_size` is 4. Batch 5 reached 5.9 of
  6.4 GB in the smoke test. On an OOM, drop to 2 before suspecting anything else
- **Keep the HuggingFace cache out of the project directory** — this repo lives
  under OneDrive, and weights will sync to the cloud in the background
