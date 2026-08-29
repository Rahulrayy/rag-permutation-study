# Pre-registered analysis plan

**Status: TEMPLATE — NOT YET PRE-REGISTERED.**

Fill every `TODO` and commit this file *before* the first main run
(plan Sec. 4.6, Sec. 7 week 2). Record the commit SHA in Sec. 8 below.
Anything decided after seeing main-run results goes in Sec. 9 as exploratory,
never silently into the confirmatory sections.

---

## 1. Hypotheses

- **H1 (RQ1).** Median within-query SD of token-F1 across P=5 permutations at
  fixed content and fixed budget is > 0.02. *Directional. This is also the
  week-1 kill criterion (plan Sec. 9).*
  **Settled in the week-1 pilot, 2026-08-29: 0.0263, PASS** (n=100, `full`
  arm, k=10). See the caveat in Sec. 9 — the median is fragile on this
  distribution and RQ1's primary presentation should not rest on it.
- **H2 (RQ2).** OAE of published pruners against `rerank_topk` is < TODO
  orderings-worth of noise.
- **H3 (RQ3).** Rank Flip Rate across single orderings is > TODO.
- **H4 (RQ4).** Placebo Gap — `Q(m) - Q(placebo_pos)` at matched keep-count —
  does not exclude zero for at least one published pruner. All three placebo
  variants run as separate arms; the confirmatory comparator is
  **`placebo_pos:middle_first`**, the shape a lost-in-the-middle-aware pruner
  produces by accident and so the specific confound RQ4 targets. `edges_first`
  and `tail_first` are reported as exploratory. TODO: confirm before registering.

## 2. Primary endpoint

TODO — one metric, one arm pair, one budget. Everything else is secondary.
Proposed: OAE of `provence_rerank` vs `rerank_topk` at k=3, token-F1, filtered
set. (Provence runs as two arms: `provence_rerank` is selection-only and so is
content-matched at equal k; `provence_full` is the published method and is
reported against input-token count. See `src/prune/provence.py`.)

## 3. Analysis population

- Primary: **filtered** — queries the generator answers incorrectly under
  `nocontext` (memorization control, plan Sec. 4.1).
- Correctness under `nocontext` defined as: TODO (EM, or token-F1 >= TODO).
- Unfiltered numbers reported alongside, always, never instead.
- **Exclusion, already applied in `data._require_fixed_context`:** rows not
  shipping exactly 10 paragraphs. On HotpotQA distractor validation this drops
  60 of 7,405 rows (0.81%), leaving 7,345. All 60 excluded rows do contain both
  gold paragraphs, so this is a comparability exclusion, not a gold-coverage one:
  a fixed context size is what makes a position, a positional bucket and a
  keep-k budget mean the same thing across queries. Decided before any
  generation was run.
- Further exclusions: TODO (malformed records, empty gold, context over token budget).

## 4. Fixed parameters

| Parameter | Value |
|---|---|
| Generator (primary) | Qwen/Qwen2.5-3B-Instruct, 4-bit nf4, double-quant, fp16 compute |
| Decoding | greedy, `do_sample=False`, no temperature/top_p passed |
| Seed | 20260828 |
| Permutations P | 5 (as-given, reverse, 3 seeded random) |
| Random permutation seeding | per query: `seed:qid:replicate:n` |
| Budgets k | 2, 3, 5 |
| n (main) | 300 |
| Prompt template | `generate.DEFAULT_TEMPLATE`, frozen below |

Frozen prompt template (verbatim):

```
Answer the question using only the passages below.
Reply with the shortest possible answer: a name, a phrase, a date, or yes/no. Do not write a sentence. Do not explain.

{context}
Question: {question}
Short answer:
```

Context rendering: `[i] {title}: {text}`, one-indexed, blank-line separated, in
permutation order. The `nocontext` and `alt` (delimiter robustness) templates
carry identical instruction wording; see `src/generate.py`.

Template selection was made on a 12-query x 5-permutation comparison before the
pilot, on the basis of **accuracy and answer-format match**, not permutation SD:

| template | mean F1 | mean EM | median within-query SD | answer words |
|---|---|---|---|---|
| loose ("Reply with the short answer only") | 0.524 | 0.367 | 0.0434 | 9.0 |
| terse (frozen) | 0.611 | 0.517 | 0.0673 | 3.5 |
| gold answers | — | — | — | 2.2 |

That the frozen template also shows the higher SD is recorded as a finding, not
as the reason for the choice. Selecting a prompt to maximise the study's headline
quantity would be a forking-paths error; the stated criterion is that EM and
token-F1 only measure what they claim to when the model emits an answer rather
than a sentence about the answer.

**Random permutations are drawn per query, not once for the dataset.** The three
random orderings are seeded on the query id as well as the run seed, so two
queries see different random arrangements. The alternative — one trio of
orderings reused for every query, which is what seeding on `(seed, replicate, n)`
alone produces — makes the study's random draws a single sample of size three
from n!, and the sampling error in that one draw does not average out over
queries however many queries are run. Since the week-1 gate is a directional test
against a fixed threshold, an unlucky trio could bias the median within-query SD
in either direction with nothing in the data to reveal it.

`rank` and `reverse` are deliberately *not* keyed on the query: they are single
fixed ordering rules, which is what they are meant to represent, and they are the
two strategies standing in for how a published evaluation actually fixes an
order.

Note on the `rank` permutation strategy: HotpotQA distractor has no retriever, so
"rank" is the **as-given** dataset paragraph order, not a retriever ranking. Use
the term "as-given order" in the write-up for this dataset and reserve "retriever
rank" for the NQ-open arm.

## 5. Statistical procedure

- Two-level bootstrap: resample **queries** with replacement, carrying all P
  permutations of each sampled query. 10,000 replicates. Permutations are
  nested within queries and are **not** resampled independently.
- Paired comparisons throughout (same queries across arms).
- Holm correction across the method-pair family. Family defined as: TODO.
- CIs (95%, percentile) are the primary presentation; p-values secondary.

## 6. Derived quantities

Defined in `src/metrics.py`, formulas in plan Sec. 4.5: OAE, Rank Flip Rate,
Placebo Gap, Oracle Gap. Any change to a formula after registration is a
protocol deviation and goes in Sec. 9.

## 7. Robustness checks (planned, not exploratory)

- Prompt-template variant, one alternative delimiter style (plan Sec. 8).
- **LLM-pruner selection stability.** `llm_pruner` is the only arm that sees all
  ten passages at once, so its *selection* can depend on the order it was shown,
  where `rerank_topk` and `provence` score chunks independently and cannot.
  Report selection Jaccard across as-given / reverse / random presentations,
  against two reference points: 1.000 for an order-invariant selector and the
  chance value for k random subsets. Pilot at n=20, k=3 gave **0.263** with the
  selection changing in 19/20 queries. The confirmatory run uses the registered
  n and the same three presentations. `prune.llm_pruner.selection_stability`.
- 2WikiMultihopQA replication.
- Groq cross-generator replication, n=100.

## 8. Registration

- Registered commit: TODO
- Date: TODO
- Environment lockfile: TODO (`pip freeze > requirements.lock`)

## 9. Protocol deviations and exploratory analyses

*(Append-only. Date each entry.)*

- **2026-08-29 — pilot finding, not a deviation.** The week-1 gate passed at a
  median within-query SD of 0.0263 against a threshold of 0.02. The criterion was
  fixed before any data was collected and was not altered after seeing the
  result. However, the distribution is bimodal: exactly 50 of 100 queries have
  SD = 0 and the other 50 have a median SD of 0.4177 (max 0.5477). The median
  therefore falls on the seam between the two groups and equals half the smallest
  non-zero SD; a single additional static query would have produced 0.0000 and a
  FAIL, with the moving half unchanged.

  **Implication for the main run, to be decided before it starts, not after:**
  RQ1's primary presentation should be distributional — the fraction of queries
  with non-zero SD, and the SD distribution among those that move — rather than a
  median that is unstable under a one-query change. Choosing that presentation
  *now*, from pilot data, and fixing it before the n=300 run is the point of
  registering; changing it after seeing main-run results would not be. The
  pre-registered median is still reported, unchanged, alongside whatever
  distributional summary is registered.
