# rag-permutation-study

**Is it the pruning, or the ordering?** A permutation-controlled re-evaluation of
context selection in RAG.

## The gap

Context pruning is a crowded subfield: a dozen methods claim they can discard
60–90% of retrieved context with little quality loss. Every one of them is
evaluated with the retrieved passages in **one fixed order**. Separately, other
work has established that RAG answers are unstable under permutation of that same
set — reorder the passages, keep the content identical, and the answer changes.

Nobody has put those two facts together. Pruning does not only remove content, it
**changes positions**: dropping chunks 3, 5 and 7 from a ten-chunk context
promotes 8, 9 and 10 into higher-visibility slots. So part of what looks like
better evidence selection may be a lucky interaction with position bias.

This is not a new pruning method. It is a protocol, two controls nobody runs, and
a number.

## Design

- **Data.** HotpotQA distractor: ten paragraphs per question, two gold, no
  retrieval needed. Rows not shipping exactly ten paragraphs are excluded
  (60/7,405, 0.81%) so that a position, a positional bucket and a keep-k budget
  mean the same thing across queries.
- **Generator.** Qwen2.5-3B-Instruct, 4-bit, local. Local because leave-one-out
  attribution needs the log-probability of the answer sequence, which hosted APIs
  generally do not expose.
- **Greedy decoding everywhere.** Sampling noise and permutation noise would be
  confounded and every number would be meaningless. Guarded in three places.
- **The permutation protocol.** For every (query, arm, budget) cell, generate
  under P=5 orderings of the *kept* chunks — as-given, reverse, and three seeded
  random. Random orderings are seeded per query, so the three draws are not one
  shared trio reused across the dataset.
- **Selection, rewriting and ordering are three separate steps.** `select()`
  returns indices, `rewrite()` returns text, `permute()` returns order. Conflating
  any two of them is the error the whole study is about.

### Arms

| Arm | What it is |
|---|---|
| `full` | all ten chunks, upper reference |
| `nocontext` | question only — the memorization control |
| `rerank_topk` | cross-encoder rerank, keep top-k. The OAE denominator |
| `provence_rerank` | Provence's reranker only, original text |
| `provence_full` | Provence as published, sentence-pruned text |
| `llmlingua2` | token-level compression, budget spent as a rate |
| `llm_pruner` | ask the generator which chunks to keep |
| `random_drop` | noise floor |
| **`placebo_pos`** | **drop k by position, not content — the novel control** |
| `loo_oracle` | keep the k chunks with the largest LOO log-prob drop |

### Derived quantities

**Order-Adjusted Effect** — a method's gain over baseline, divided by the
baseline's within-query permutation SD. How many orderings-worth of noise does
this method actually buy you. **Rank Flip Rate** — the fraction of method-pair
comparisons whose sign reverses under some single ordering. **Placebo Gap** —
quality against the position-matched placebo at equal keep-count; near zero means
the method is not doing content selection. **Oracle Gap** — headroom against the
leave-one-out ceiling.

## Findings so far

**The premise holds.** Week-1 gate: median within-query SD of token-F1 across
five permutations is **0.0263** against a pre-registered kill threshold of 0.02
(n=100, `full` arm). On the filtered analysis population it is **0.1209**.

Both numbers understate the result, because the distribution is bimodal and the
median lands on the seam between its two halves. The finding that does not depend
on the median: **half the queries never move at all, and the half that do swing by
roughly 0.42 F1 on identical content under greedy decoding.**

**Two methods are order-dependent in their internals, not just their output.**

- `llm_pruner`'s *selection* changes with the order it was shown the passages.
  Jaccard across three presentations is **0.263**, against 1.000 for a selector
  that scores chunks independently and 0.048 for chance. The selection changed in
  **19 of 20** queries.
- LLMLingua-2's compression is order-dependent when applied to a concatenated
  context: **0 of 100** chunks compressed identically across orderings. It is a
  deterministic token classifier, which makes this the more surprising of the two.
  The arm therefore compresses each chunk independently.

**Memorization is low.** Only 10/100 queries are answerable with no passages at
all (mean no-context token-F1 0.115), so the memorization filter keeps 90% rather
than gutting the analysis population.

## Statistics

Permutations are nested within queries, so the P×N cells are **not** independent —
treating them as such inflates n by 5× and manufactures significance. The
resampling unit is the query, and all P of its permutations travel with it. The
test suite contains a regression guard that builds data with strong between-query
and weak within-query variation and asserts the correct CI is >1.5× wider than the
flattened one.

Paired comparisons throughout. Holm correction across one family of nine
confirmatory comparisons. CIs, not p-values, as the primary presentation.

The analysis was **pre-registered before any main-run data existed** — hypotheses,
thresholds, primary endpoint, analysis population and multiplicity family all
fixed and committed in advance. The registration is retrievable from git history
at commit `2f24548`.

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
| Tests | `python -m pytest -q` |

Every generation is content-hash cached on `sha256(model, prompt, decode_params)`
and flushed per block, so reruns are free and interrupting a run loses nothing.

Useful flags: `--backend dummy` exercises the whole pipeline with no GPU (its
numbers are meaningless by construction), `--n 20` shrinks the query set,
`--arms full` restricts the grid, `--budget` pins a keep-k.

## Status

Weeks 1–2 complete; analysis plan registered.

| Piece | State |
|---|---|
| `data.py`, `chunks.py`, `cache.py`, `metrics.py`, `stats.py`, `gate.py`, `run.py`, `smoke.py` | done |
| `generate.py` — local 4-bit generator, greedy, answer log-probs | done |
| arms `full` / `nocontext` / `random_drop` / `placebo_pos` (3 variants) | done |
| arms `rerank_topk` / `provence_rerank` / `provence_full` / `llm_pruner` / `llmlingua2` | done |
| arm `loo_oracle` | week 3 |
| 2WikiMultihopQA + hosted cross-generator replication | week 5 |
| figures | week 6 |

**129 tests** pass (`python -m pytest -q`, ~9s). One is marked `network` and
downloads HotpotQA on first run; `pytest -m "not network"` skips it.

## Limitations

Off-the-shelf pruner checkpoints are used out of distribution relative to their
training data, so this measures deployed-as-published behaviour, not each method's
ceiling. The Provence checkpoint is `cc-by-nc-nd-4.0` — non-commercial. `rank` is
the dataset's **as-given** order, not a retriever ranking; HotpotQA distractor has
no retriever. `llmlingua2` presents n permutable slots where a keep-k arm presents
k, so its raw permutation variance is not comparable in magnitude and it is
compared on input-token count instead.
