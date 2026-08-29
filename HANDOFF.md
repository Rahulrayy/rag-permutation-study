# Handoff

State of `rag-order-audit` as of **2026-08-28**, commit `0ce0287` (initial commit,
branch `master`).

Read [`rag-permutation-project-plan.md`](rag-permutation-project-plan.md) first —
it is the design document and this file does not repeat it. This file covers what
exists, what was decided and why, what was learned that the plan could not have
known, and what to do next.

---

## 1. Where the project actually is

**Week 1 of 6 is complete except for the one number it exists to produce.**

The plan's week-1 gate (Sec. 7) asks a single question: *does answer quality vary
across permutations of a fixed context, at this scale, with this model?* All the
machinery to answer it is built, tested and committed. The GPU pass that produces
the number has not been run.

```bash
python -m src.run  --config configs/pilot.yaml       # 500 generations, ~5-15 min
python -m src.gate results/pilot_w1/generations.csv  # prints PASS or FAIL
```

**Nothing downstream should be built until that prints PASS.** The kill criterion
(median within-query SD of token-F1 < 0.02) was fixed before any data was
collected and is enforced in `src/gate.py`. On FAIL, the tool prints the Sec. 9
escalation ladder — more chunks → longer chunks → weaker generator → MuSiQue —
and the fallback consolidation study reuses roughly 80% of this code.

Early indications are encouraging but are **not** the gate: a 12-query probe
(§5 below) showed median within-query SD of 0.067, and a 3-query smoke test showed
token-F1 spreads of 0.750 and 0.895 on two of three questions. Twelve queries is
not one hundred.

---

## 2. What is real and what is a stub

Everything under `src/` imports and compiles. Stubs raise `NotImplementedError`
with the week they are due — they never return fake data.

| File | State | Notes |
|---|---|---|
| `src/chunks.py` | **done** | Seeded permutation protocol, Sec. 4.4 |
| `src/data.py` | **done** | HotpotQA loader, memorization filter, nested subsampling |
| `src/cache.py` | **done** | SQLite, keyed `sha256(model, prompt, decode_params)` |
| `src/metrics.py` | **done** | EM, token-F1, OAE, Rank Flip Rate, placebo gap, oracle gap |
| `src/stats.py` | **done** | Two-level bootstrap, Holm correction |
| `src/gate.py` | **done** | Week-1 kill criterion + escalation ladder |
| `src/run.py` | **done** | Experiment driver; batches across the whole grid |
| `src/smoke.py` | **done** | Five-point GPU check; run before any long job |
| `src/generate.py` | **mostly** | `LocalGenerator` written and verified on GPU; `GroqGenerator` is a week-5 stub |
| `src/prune/full,nocontext,random_drop,placebo_pos` | **done** | Includes the study's novel control |
| `src/prune/rerank_topk` | stub | Week 2. Default OAE baseline arm |
| `src/prune/provence` | stub | Week 2. **Verify the checkpoint loads early** — see §7 |
| `src/prune/llmlingua2` | stub | Week 2. Has an unresolved design question, see §7 |
| `src/prune/llm_pruner` | stub | Week 2 |
| `src/prune/loo_oracle` | stub | Week 3. Depends on `Generator.score`, which is verified working |
| `src/data.py::_load_2wikimultihop` | stub | Week 5 |

`analysis/` and `results/` are empty. Figures (`--figures-only`) are week 6.

**100 tests pass** (`python -m pytest -q`, ~3s). One test downloads HotpotQA on
first run and carries the `network` marker; `pytest -m "not network"` skips it.

---

## 3. Design invariants — do not break these

These are not style preferences. Each one, if violated, silently invalidates
results rather than throwing an error.

**Selection and ordering are separate, always.** `Pruner.select()` returns
*indices*, never reordered chunks, and its return order carries no meaning.
Ordering is applied afterwards by `chunks.permute`. This separation is the entire
point of the study — plan Sec. 3, premise 3: removing chunks 3, 5 and 7 from a
10-chunk context does not just remove content, it promotes 8, 9 and 10 into
higher-visibility slots. A pruner that returns chunks in its own preferred order
would confound the two and there would be no way to detect it after the fact.

**Greedy decoding, everywhere, no exceptions.** `run.decode_params` raises if
`do_sample` or a non-zero temperature appears in a config, and
`LocalGenerator.generate_batch` raises again before it imports torch. Sampling
noise and permutation noise would be confounded and every number in the study
would be meaningless. Verified empirically: identical output across repeated runs
of the same prompt.

**Permutations are nested within queries and the bootstrap must respect it.**
`stats.two_level_bootstrap` resamples *queries* and carries all P permutations
along. Resampling the P×N cells independently inflates n by 5× and manufactures
significance. `tests/test_stats.py::test_nesting_is_not_flattened` builds data
with strong between-query and weak within-query variation and asserts the correct
CI is >1.5× wider than the flattened one. If that test ever starts failing,
something has gone badly wrong — do not "fix" it by loosening the threshold.

**The cache key is a pure function of what determines the output.** Never add
query id, arm, or permutation index to it — many grid cells legitimately resolve
to the same prompt, and they should share a cache entry.

**Every arm honours its budget exactly.** `validate_selection` enforces it. A
pruner quietly returning k+1 chunks breaks the matched-keep-count comparison
against `placebo_pos`, which is the study's centerpiece.

---

## 4. Decisions already locked

Recorded in `ANALYSIS_PLAN.md`. Changing any of these after the main run starts is
a protocol deviation and belongs in its §9, not in a silent edit.

**Exclusion: rows without exactly 10 paragraphs.** HotpotQA distractor validation
is not rectangular — 60 of 7,405 rows (0.81%) ship between 2 and 9 paragraphs.
All 60 still contain both gold paragraphs, so this is a *comparability* exclusion,
not a coverage one: a fixed context size is what makes a position, a positional
bucket and a keep-k budget mean the same thing across queries. A 2-paragraph row
at k=5 is not pruned at all and would enter the analysis as a free win for every
arm. Applied in `data._require_fixed_context`. Working population: **7,345**.

**Prompt template, frozen.** See §5 for the measurement behind it. The verbatim
text is in `ANALYSIS_PLAN.md` §4 and in `generate.DEFAULT_TEMPLATE`.

**Subsampling is nested.** `data._stratified_order` produces one deterministic
ordering whose every prefix is proportionally stratified, so n=100 is an exact
prefix of n=300 at the same seed. The obvious alternative — shuffle, then allocate
per-stratum quotas — breaks this, because quota rounding at 100 and at 300 can
select different examples, and then pilot numbers are not comparable to main-run
numbers. Tested.

**Seed is 20260828 everywhere.**

**Random permutations are keyed on the query id.** The three random orderings
differ from query to query; `rank` and `reverse` do not, since they stand for
single fixed ordering rules. Seeding on `(seed, replicate, n)` alone would reuse
one trio of orderings across the whole dataset, making the random draws a sample
of size three from n! whose sampling error never averages out over queries.
Recorded in `ANALYSIS_PLAN.md` Sec. 4.

---

## 5. What was learned that the plan did not know

**The dataset has no positional confound.** Gold paragraphs are spread near-
uniformly across all ten positions (1,427–1,507 occurrences per slot). Position
effects found later are properties of the model, not of how HotpotQA was built.

**Hop-type split:** 5,899 bridge / 1,446 comparison (80.3% / 19.7%). Stratified
sampling reproduces this at every n.

**Median context is ~1,381 tokens**, inside the plan's predicted 1,200–1,800 band.

**`rank` means "as-given", not "retriever rank".** HotpotQA distractor has no
retriever. The `rank` permutation strategy reproduces the dataset's own paragraph
order. It is still the right reference ordering — it is the one any evaluation on
this dataset implicitly uses — but **do not call it "retriever rank" in the
write-up.** Reserve that term for the NQ-open arm, where a real retriever produces
it. Noted in the `data.py` module docstring.

**The prompt template mattered more than expected, and in the opposite direction
to the obvious guess.** A 12-query × 5-permutation comparison, Qwen2.5-3B-Instruct
4-bit:

| template | mean F1 | mean EM | median within-query SD | answer words |
|---|---|---|---|---|
| loose ("Reply with the short answer only") | 0.524 | 0.367 | 0.0434 | 9.0 |
| terse (frozen) | 0.611 | 0.517 | 0.0673 | 3.5 |
| gold answers | — | — | — | 2.2 |

The prediction was that verbosity *inflated* the SD through answer-length churn.
The reverse is true: a model that answers in a full sentence every time scores a
uniformly mediocre ~0.25 regardless of ordering, which compresses token-F1 toward
the middle and **damps** the variance. Verbosity was masking position sensitivity.

**The template was chosen on accuracy and answer-format match, explicitly not on
the SD.** This distinction matters and should survive into the write-up: EM and
token-F1 only measure what they claim to when the model emits an answer rather
than a sentence about the answer. Choosing a prompt because it maximises the
study's own headline quantity would be a garden-of-forking-paths error and would
hand a reviewer an easy shot at the result. If the terse template had scored worse
on accuracy, the loose one would have been correct despite its lower SD.

**`ALT_TEMPLATE` now differs from the default in delimiters only.** It originally
varied instruction wording too, which would have confounded the Sec. 8 delimiter
robustness check with the verbosity effect above. An `assert` at import time
enforces the isolation.

**The LOO oracle has real signal.** `logP(answer | all 10 paragraphs) = -1.271`
versus `-13.647` with the gold paragraphs removed — a 12.4-nat drop, on one
example. This is the only exercise of `Generator.score` before week 3, and it is
the reason the primary generator is local rather than hosted.

---

## 6. Environment

Verified working, `requirements.lock` committed.

| | |
|---|---|
| Python | 3.11.9, venv at `.venv/` |
| GPU | RTX 3060 Laptop, **6.4 GB**, compute 8.6, CUDA 12.8 |
| torch | 2.11.0+cu128 — **from the CUDA index, not PyPI** |
| transformers / bitsandbytes / accelerate | 5.16.1 / 0.50.2 / 1.14.0 |
| datasets | 5.0.1 |
| Model | `Qwen/Qwen2.5-3B-Instruct`, 4-bit nf4, double-quant, fp16 compute |
| VRAM at load | 2.06 GB |
| Load time | ~6s warm (~490s first time, mostly download) |

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**VRAM is the binding constraint.** `batch_size` is **4** in both configs. Batch 5
reached 5.9 GB of 6.4 GB during the smoke test; 8 will OOM. If you hit an OOM
anyway, drop to 2 before suspecting anything else.

**Keep the HuggingFace cache out of the project directory.** This repo lives under
`OneDrive\Desktop\...`. An early default put the HF cache at `./hf_cache`, and 1.5
GB of HotpotQA began syncing to the cloud; a 3B checkpoint would have added ~6 GB
more. Both `data.py` and `generate.py` now default to `None`, meaning the standard
`~/.cache/huggingface`. **Do not point `cache_dir` back inside the project.**

`make` is not installed. The Makefile is correct but you will need
`pacman -S make` in Git Bash, or just use the `python -m ...` commands — the
README has the mapping.

---

## 7. Next steps, in order

**1. Run the gate.** Two commands, §1 above. Commit the result either way: a PASS
is the green light for week 2; a FAIL is a real finding, with the threshold
already on record, and sends you to the Sec. 9 ladder.

**2. Before week 3, re-check the literature.** The plan (Sec. 2) commits to a
targeted search on *"permutation-controlled evaluation context pruning"* and
*"positional placebo baseline RAG"*. If someone published exactly this in the
interim, the Sec. 9 fallback applies. This is a scheduled task, not an optional
one.

**3. Week 2: implement the four remaining pruner arms.** Each is one file behind
`Pruner.select(query, chunks, budget) -> list[int]`. Order them by risk:

- **`provence` first.** Plan Sec. 8 flags the checkpoint availability as the
  assumption most likely to blow up, and the gate is "verify it loads in week 2,
  **not** week 5." Do this before anything else in the week.
- **`rerank_topk`** — the default OAE denominator arm, so nothing else is
  interpretable without it.
- **`llm_pruner`** — watch for two failure modes: the model returning more than
  `budget` indices (`validate_selection` catches it), and the *selection prompt
  itself* being order-sensitive, which would make this arm's selection, not just
  its answer, a function of the ordering it was shown. Log the ordering used for
  selection; it is a confound worth reporting.
- **`llmlingua2` last** — it has an open design question, below.

**4. Week 2: finish `ANALYSIS_PLAN.md`.** Remaining `TODO`s: the primary endpoint,
the `nocontext` correctness definition, the Holm family definition, and the H2/H3
thresholds. Then register: record the commit SHA and date in its §8. The plan
calls this "the single most credible thing in the whole project" and it costs an
hour.

**5. Week 3: LOO oracle and the memorization filter.** The filter is wired
(`run.py` runs `nocontext` first and unconditionally) but has never run on real
generations.

---

## 8. Open questions

**How is a token-compressed context permuted?** `llmlingua2` compresses *within*
chunks rather than selecting whole ones. Two consequences: a keep-k budget is not
comparable across arms (plan Sec. 4.3 handles this — report against input-token
count, not k), and it is genuinely unclear what a permutation *means* for a
token-compressed context. The honest default is to permute the surviving
chunk-level units rather than individual tokens, but this needs deciding
explicitly and recording before the arm runs, not after.

**Is memorization going to bite?** The 12-query probe showed mean EM of 0.517,
which is high for a 3B model on multi-hop questions. That is consistent with
Wikipedia leakage into pretraining, and it is exactly what the `nocontext` filter
exists for. Watch for the specific dangerous pattern: **low permutation SD arriving
together with a high mean.** That is parametric recall, not stability, and it
would produce a misleading FAIL at the gate. `gate.py` prints a warning above 0.75
mean F1, but the real check is the `nocontext` arm.

**Does `placebo_pos` need all three variants in the main grid?** All three run:
a bare `placebo_pos` in a config expands to `placebo_pos:middle_first`,
`:edges_first` and `:tail_first`, one arm each, each under its own name in
`generations.csv`. `ANALYSIS_PLAN.md` now names `middle_first` as the
confirmatory comparator for H4 and the other two as exploratory — still a TODO
to confirm before registering. To run only one, write it out in `arms` instead of
the bare name.

**Is `main.py` wanted?** It is a 0-byte PyCharm placeholder, committed as-is
because it was pre-existing. `src/` is the real entry point. Probably delete it.

---

## 9. Things that will waste your time if you forget them

- **The cache makes reruns free.** 500 generations replay in under a second. Never
  hand-edit results to avoid a rerun; just rerun.
- **`--backend dummy`** exercises the entire pipeline with no GPU. Its numbers are
  meaningless by construction — the backend returns a hash-derived word from a
  five-word vocabulary — so never report a gate result from it. It is for plumbing
  only.
- **`python -m src.smoke`** before any long job. It catches a bad checkpoint, a
  broken CUDA install, or non-deterministic decoding in about a minute, rather
  than ten hours into an overnight run.
- **`--n 20`** shrinks any config for a fast sanity pass.
- Background pip and model downloads buffer their output; an empty log file does
  not mean a hung process. Check `nvidia-smi` instead.
