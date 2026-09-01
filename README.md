# rag-permutation-study

**Is it the pruning, or the ordering?** A permutation-controlled re-evaluation of
context selection in RAG.

Reorder the passages in a RAG context without changing a word of them, and the
answer changes. Every published context-pruning method is evaluated in one fixed
order, and pruning itself *moves* passages between positions. This study
measures how much of a reported pruning gain is actually that.

---

## Findings

**Answer quality is bimodal under reordering, not uniformly noisy.**
Across 100 questions × 5 permutations of a fixed ten-paragraph context, greedy
decoding, content identical throughout:

| | |
|---|---|
| queries whose answer never changes | **50%** |
| queries that do change | **50%**, swinging ~**0.42** token-F1 |
| median within-query SD (pre-registered kill threshold was 0.02) | **0.0263** |
| the same, on the memorization-filtered set | **0.1209** |

Both medians understate it. The distribution has two modes and the median falls
on the seam between them, so it is unstable to a one-query change in either
direction. The result that does not depend on the median: **half these questions
are perfectly stable under reordering and half swing by roughly 0.42 F1 on
identical content.**

**Two pruning methods are order-dependent in their internals, not just their
output.** This is the sharper version of the thesis: not "the score moves" but
"the method does something different depending on what order you showed it".

- **An LLM asked which passages to keep picks different ones when shown the same
  passages in a different order.** Selection Jaccard across three presentations is
  **0.213** over 100 questions, against **1.000** for a cross-encoder scored
  through the same permutations, and **0.047** for chance. **The selection changed
  in 98 of 100 questions, and in 23 of them the three presentations shared no
  passage at all.** So a published LLM-pruner result is one draw from a
  distribution over selections that its paper does not mention.
- **LLMLingua-2's compression is order-dependent too**, when applied to a
  concatenated context the normal way: **0 of 100** passages compressed identically
  across orderings. It is a deterministic token classifier, which makes this the
  more surprising of the two. The arm here therefore compresses each passage
  independently, which restores stability to 100/100.

**Memorization is not the explanation.** Only **10/100** questions are answerable
with no passages at all (mean no-context token-F1 0.115), so the effect is not a
3B model reciting Wikipedia.

*The leave-one-out oracle has since been built and run at full scale. The
second dataset and the hosted cross-generator replication are still to come; see
Status.*

---

## Related work, and what is genuinely new here

Order sensitivity in LLMs is well established. In-context example order moves
few-shot accuracy by tens of points, multiple-choice option order by up to 75%,
lost-in-the-middle describes the positional shape of it, and benchmarks such as
RGB have shuffled retrieved documents and confirmed order matters. Context
pruning is equally well covered, from Provence and LLMLingua-2 through to
budget-constrained selection under token limits.

What is missing is the **join**. A literature re-check in August 2026 found no
work that evaluates pruning methods under multiple permutations with content held
fixed, none that runs a position-matched placebo, and none that reports how often
method rankings flip across orderings. The nearest neighbours, conformal
coverage-controlled filtering (arXiv 2511.17908) and answer-survival diagnostics
for budgeted packing (arXiv 2607.00725), address adjacent questions and use a single
fixed ordering throughout.

A note on the LLM-pruner result above: that an LLM's selection is order-sensitive
is *not* a surprising phenomenon given the in-context-learning literature, and it
is not claimed as one. It is also a robustness check rather than a confirmatory
endpoint, and sits outside the pre-registered multiplicity family. The claim is narrower: this known effect reaches into the
pruner's selection, and no published evaluation of an LLM pruner controls for it.
The LLMLingua-2 result is less exposed to that objection, since it is a
deterministic token classifier rather than a prompted model.

## Why this is a gap

Context pruning is a crowded subfield: a dozen methods claim they can discard
60 to 90% of retrieved context with little quality loss. Every one is evaluated with
the passages in one fixed order, usually retriever rank. Separately, other work
has established that RAG answers are unstable under permutation of that same set.

Nobody has put those two facts together. Pruning does not only remove content, it
**changes positions**: dropping passages 3, 5 and 7 from a ten-passage context
promotes 8, 9 and 10 into higher-visibility slots. So part of what looks like
better evidence selection may be a lucky interaction with position bias, measured
against a reference point that moves.

This is not a new pruning method. It is a protocol, two controls nobody runs, and
a number.

## Design

- **Data.** HotpotQA distractor: ten paragraphs per question, two gold, no
  retrieval needed. Rows not shipping exactly ten paragraphs are excluded
  (60/7,405, 0.81%) so a position, a positional bucket and a keep-k budget mean
  the same thing across questions.
- **Generator.** Qwen2.5-3B-Instruct, 4-bit, local, because leave-one-out
  attribution needs the log-probability of the answer sequence, which hosted APIs
  generally do not expose.
- **Greedy decoding everywhere.** Sampling noise and permutation noise would be
  confounded and every number would be meaningless. Guarded in three places.
- **The permutation protocol.** For every (question, arm, budget) cell, generate
  under P=5 orderings of the *kept* passages: as-given, reverse, and three seeded
  random. Random orderings are seeded per question, so the three draws are not one
  shared trio reused across the dataset.
- **Selection, rewriting and ordering are three separate steps.** `select()`
  returns indices, `rewrite()` returns text, `permute()` returns order. Conflating
  any two of them is the error this study is about.

### Arms

| Arm | What it is |
|---|---|
| `full` | all ten passages, upper reference |
| `nocontext` | question only, the memorization control |
| `rerank_topk` | cross-encoder rerank, keep top-k. The OAE denominator |
| `provence_rerank` | Provence's reranker only, original text |
| `provence_full` | Provence as published, sentence-pruned text |
| `llmlingua2` | token-level compression, budget spent as a rate |
| `llm_pruner` | ask the generator which passages to keep |
| `random_drop` | noise floor |
| **`placebo_pos`** | **drop k by position, not content, the novel control** |
| `loo_oracle` | keep the k passages with the largest LOO log-prob drop |

### Derived quantities

**Order-Adjusted Effect** is a method's gain over baseline, divided by the
baseline's within-question permutation SD. How many orderings-worth of noise does
this method actually buy you. **Rank Flip Rate** is the fraction of method-pair
comparisons whose sign reverses under some single ordering. **Placebo Gap** is
quality against the position-matched placebo at equal keep-count; near zero means
the method is not doing content selection. **Oracle Gap** is the headroom against the
leave-one-out ceiling.

## Statistics

Permutations are nested within questions, so the P×N cells are **not**
independent, and treating them as such inflates n by 5× and manufactures
significance. The resampling unit is the question, and all P of its permutations
travel with it. The test suite contains a regression guard that builds data with
strong between-question and weak within-question variation and asserts the correct
CI is >1.5× wider than the flattened one.

Paired comparisons throughout. Holm correction across one family of nine
confirmatory comparisons. CIs, not p-values, as the primary presentation.

The analysis was **pre-registered before any main-run data existed**: hypotheses,
thresholds, primary endpoint, analysis population and multiplicity family all
fixed and committed in advance. Retrievable from git history at commit `2f24548`.

## Setup

```bash
python -m pip install -r requirements.txt      # torch must come from the CUDA index
```

`requirements.lock` pins the exact environment a run was made in.

## Running

| Task | Command |
|---|---|
| GPU smoke test | `python -m src.smoke` |
| Pilot | `python -m src.run --config configs/pilot.yaml` |
| The week-1 gate | `python -m src.gate results/pilot_w1/generations.csv` |
| Main run | `python -m src.run --config configs/main.yaml` |
| Figures | `python -m src.figures --config configs/main.yaml` |
| Selection-stability probe | `python -m src.selection_probe --config configs/main.yaml --n 100` |
| Tests | `python -m pytest -q` |

Every generation is content-hash cached on `sha256(model, prompt, decode_params)`
and flushed per block, so reruns are free and interrupting a run loses nothing.

Useful flags: `--backend dummy` exercises the whole pipeline with no GPU (its
numbers are meaningless by construction), `--n 20` shrinks the question set,
`--arms full` restricts the grid, `--budget` pins a keep-k.

## Status

| Piece | State |
|---|---|
| `data.py`, `chunks.py`, `cache.py`, `metrics.py`, `stats.py`, `gate.py`, `run.py`, `smoke.py` | done |
| `generate.py`, the local 4-bit generator, greedy, answer log-probs | done |
| arms `full` / `nocontext` / `random_drop` / `placebo_pos` (3 variants) | done |
| arms `rerank_topk` / `provence_rerank` / `provence_full` / `llm_pruner` / `llmlingua2` | done |
| arm `loo_oracle` | done, ran at full scale, n=274 |
| main run + confirmatory analysis | done, 45,510 rows |
| figures, one per research question, plus the selection probe, `python -m src.figures` | done |
| 2WikiMultihopQA + hosted cross-generator replication | to come |

**222 tests** pass (`python -m pytest -q -m "not network"`, ~10s). Three are
marked `network` and download HotpotQA on first run; `pytest -m "not network"`
deselects them.

## Limitations

Off-the-shelf pruner checkpoints are used out of distribution relative to their
training data, so this measures deployed-as-published behaviour, not each method's
ceiling. The Provence checkpoint is `cc-by-nc-nd-4.0`, non-commercial. `rank` is
the dataset's **as-given** order, not a retriever ranking; HotpotQA distractor has
no retriever, and the term is reserved for the NQ-open arm. `llmlingua2` presents
n permutable slots where a keep-k arm presents k, so its raw permutation variance
is not comparable in magnitude and it is compared on input-token count instead.
