# Pre-registered analysis plan

**Status: REGISTERED 2026-08-29, before any main-run generation.**

Every confirmatory choice below — hypotheses and their thresholds, the primary
endpoint, the analysis population, the multiplicity family — was fixed on this
date. The main run (`configs/main.yaml`, n=300) had not been executed and no
data existed for any arm beyond the week-1 pilot, which used the `full` arm
only. Anything decided after seeing main-run results goes in Sec. 9 as
exploratory, never silently into the confirmatory sections above.

Decisions informed by the week-1 pilot are marked as such and are legitimate:
a pilot exists to inform design. What matters is that they were fixed *before*
the confirmatory data existed, and the git history shows when.

---

## 1. Hypotheses

- **H1 (RQ1).** Median within-query SD of token-F1 across P=5 permutations at
  fixed content and fixed budget is > 0.02. *Directional. This is also the
  week-1 kill criterion (plan Sec. 9).*
  **Settled in the week-1 pilot, 2026-08-29: 0.0263, PASS** (n=100, `full`
  arm, k=10). See the caveat in Sec. 9 — the median is fragile on this
  distribution and RQ1's primary presentation should not rest on it.
- **H2 (RQ2).** OAE of published pruners against `rerank_topk` is **< 0.5**
  orderings-worth of noise.

  *Interpretation, fixed in advance.* H2 fails if OAE >= 0.5. The project's
  argument is supported by any OAE below about 1.0 — one ordering's worth of
  noise, the point where reordering the same context moves the score as much as
  changing method does. So a result in **[0.5, 1.0) fails H2 while still
  supporting the thesis**. If that happens it is to be reported as "H2 not
  supported", in those words, with the observed value and CI, and must not be
  re-described as a success. The 1.0 line is recorded here as the interpretive
  reference only; no test is performed against it, and it is not a fallback
  threshold.
- **H3 (RQ3).** Rank Flip Rate across single orderings is **> 0.10** — more
  than one in ten method-pair comparisons reverses sign depending on which
  single arbitrary ordering the comparison was made at. Note RFR is computed on
  per-arm means across queries, which are far more stable than the per-query
  swings seen in the pilot, so this is a real risk of failing even though the
  instance-level effect is large.
- **H4 (RQ4).** Placebo Gap — `Q(m) - Q(placebo_pos)` at matched keep-count —
  does not exclude zero for at least one published pruner. All three placebo
  variants run as separate arms; the confirmatory comparator is
  **`placebo_pos:middle_first`**, the shape a lost-in-the-middle-aware pruner
  produces by accident and so the specific confound RQ4 targets. `edges_first`
  and `tail_first` are reported as exploratory. **Confirmed at registration.**

## 2. Primary endpoint

**Placebo Gap of `provence_rerank` against `placebo_pos:middle_first`, at k=3,
token-F1, on the filtered set, reported as a point estimate with a 95%
percentile CI from the two-level bootstrap.**

    Q(provence_rerank) - Q(placebo_pos:middle_first)

Read as: does a published pruner beat dropping the *same number* of chunks by
position alone? A CI containing zero means the method is not doing content
selection at this budget — its apparent gain is positional promotion, which the
placebo reproduces without reading the passages.

Chosen over OAE because plan Sec. 3 names RQ4 the centerpiece and the
position-matched placebo the control nobody runs. OAE remains the headline
*descriptive* quantity (plan Sec. 4.5) and is reported prominently, but it is
secondary to this.

`provence_rerank` rather than `provence_full` because the primary endpoint must
be a **matched-keep-count** comparison: `provence_rerank` is selection-only and
holds content fixed at equal k, which is exactly what makes the placebo
contrast interpretable. `provence_full` changes chunk content and is compared on
input-token count instead (plan Sec. 4.3), as a secondary endpoint.

Everything else in this document is secondary or exploratory.

## 3. Analysis population

- Primary: **filtered** — queries the generator answers incorrectly under
  `nocontext` (memorization control, plan Sec. 4.1).
- Correctness under `nocontext` defined as **token-F1 >= 0.8**. The filter's
  purpose is to remove queries answerable from parametric memory, and a reply of
  "Vilnius Old Town, Lithuania" against a gold of "Vilnius Old Town" is recall
  even though exact match scores it wrong. **EM is reported alongside as a
  sensitivity check.** Measured on the week-1 pilot (n=100) before registering,
  the choice moves one query:

  | rule | excluded | kept | median within-query SD of the kept set |
  |---|---|---|---|
  | EM == 1.0 | 10 | 90 | 0.1209 |
  | token-F1 >= 0.9 | 10 | 90 | 0.1209 |
  | token-F1 >= 0.8 | 11 | 89 | 0.1095 |
  | token-F1 >= 0.5 | 11 | 89 | 0.1095 |

  Nothing rests on the definition; it is registered so the choice is not made
  after seeing main-run numbers.
- Unfiltered numbers reported alongside, always, never instead.
- **Exclusion, already applied in `data._require_fixed_context`:** rows not
  shipping exactly 10 paragraphs. On HotpotQA distractor validation this drops
  60 of 7,405 rows (0.81%), leaving 7,345. All 60 excluded rows do contain both
  gold paragraphs, so this is a comparability exclusion, not a gold-coverage one:
  a fixed context size is what makes a position, a positional bucket and a
  keep-k budget mean the same thing across queries. Decided before any
  generation was run.
- **Further exclusions: none.** The candidates were checked against the full
  working population (7,345 rows) before registering, and none occur:

  | candidate exclusion | rows affected |
  |---|---|
  | empty gold answer | 0 |
  | empty question | 0 |
  | any empty paragraph | 0 |
  | not exactly 2 gold paragraphs | 0 |
  | context over the generator's window | 0 (max ~4,097 tokens vs a 32k window) |

  If any arise in a later dataset (2WikiMultihopQA, NQ-open) the rule is fixed
  here in advance: drop the row, report the count, and never drop a row on the
  basis of the answer it produced.

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
- Holm correction across the method-pair family. **The family is the nine
  confirmatory pairwise comparisons at the primary budget (k=3), on the primary
  population (filtered), with the primary metric (token-F1):**

  | | comparisons |
  |---|---|
  | H2, OAE vs `rerank_topk` | `provence_rerank`, `provence_full`, `llmlingua2`, `llm_pruner` (4) |
  | H4, Placebo Gap vs `placebo_pos:middle_first` | the four above plus `rerank_topk` (5) |

  One family of nine, not two families of four and five — splitting them would
  buy power at the cost of a reviewer reasonably calling it family-splitting.

  The primary endpoint is one of the nine. It is reported **both** uncorrected
  (as the single pre-specified primary comparison) **and** Holm-corrected within
  the family, and both numbers are reported always, so the choice between them
  cannot be made after seeing which is more favourable.

  Everything outside that set is exploratory and reported without family-wise
  correction, labelled as such: the other budgets (k=2, k=5), the other placebo
  variants, the unfiltered population, EM and supporting-fact F1, per-hop-type
  breakdowns, and the arms that are not keep-k matched (`full`, `llmlingua2`)
  wherever a matched comparison is implied.
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

- **Registered:** 2026-08-29.
- **Code registered:** `d4ba648` — the state of `src/` this plan applies to.
  All nine arms in the Holm family are implemented and tested at that commit;
  `loo_oracle` (week 3) is not, and its Oracle Gap is therefore a secondary
  quantity here rather than part of the confirmatory family.
- **Plan commit:** `2c14be0` "REGISTER the analysis plan, before any main-run
  generation". Recorded here in the following commit rather than in `2c14be0`
  itself, because a file cannot contain its own hash. `git show 2c14be0` is the
  registered text; the only change after it is this line. Neither commit touches
  `src/`.
- **Environment lockfile:** `requirements.lock`, regenerated at registration
  (66 packages; torch 2.11.0+cu128, transformers 5.16.1, llmlingua 0.2.2,
  nltk 3.10.3).
- **Data state at registration:** week-1 pilot only — n=100, `full` arm, k=10,
  5 permutations, plus its `nocontext` companion. Committed under
  `results/pilot_w1/`. No pruner arm has produced a generation.

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
