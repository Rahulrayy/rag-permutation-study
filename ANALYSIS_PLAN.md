# Pre-registered analysis plan

**Status: REGISTERED 2026-08-29, before any main-run generation.**

This is the pre-registration for **[rag-permutation-study](README.md)**, a
permutation-controlled re-evaluation of context selection in RAG. The findings
are in [`README.md`](README.md); the full account, including the design this
plan operationalises, is in [`WRITEUP.md`](WRITEUP.md). This document is the
protocol, and Section 9 is the record of every departure from it.

Every confirmatory choice below — hypotheses and their thresholds, the primary
endpoint, the analysis population, the multiplicity family — was fixed on this
date. The main run (`configs/main.yaml`, n=300) had not been executed and no
data existed for any arm beyond the week-1 pilot, which used the `full` arm
only. Anything decided after seeing main-run results goes in Section 9 as
exploratory, never silently into the confirmatory sections above.

Decisions informed by the week-1 pilot are marked as such and are legitimate:
a pilot exists to inform design. What matters is that they were fixed *before*
the confirmatory data existed, and the git history shows when.

**The registration is checkable rather than asserted**, and these commands are
the check:

```bash
git show 2f24548:ANALYSIS_PLAN.md                 # the registered text, 271 lines
git log -1 --format=%ad --date=short 2f24548      # 2026-08-29
git log --diff-filter=A --format='%ad %h' --date=short -- 'results/main_hotpotqa/*' | tail -1
                                                  # 2026-08-31 — two days later
git diff --name-only 822bb24 2f24548 -- src/      # empty: no code moved with the plan
```

## Reading this document

| | |
|---|---|
| [1. Hypotheses](#1-hypotheses) | H1–H4, with thresholds fixed in advance |
| [2. Primary endpoint](#2-primary-endpoint) | the single pre-specified comparison |
| [3. Analysis population](#3-analysis-population) | the memorization filter and every exclusion |
| [4. Fixed parameters](#4-fixed-parameters) | model, decoding, budgets, permutations |
| [5. Statistical procedure](#5-statistical-procedure) | two-level bootstrap, the Holm family of nine |
| [6. Derived quantities](#6-derived-quantities) | OAE, Rank Flip Rate, Placebo Gap, Oracle Gap |
| [7. Robustness checks](#7-robustness-checks-planned-not-exploratory) | planned in advance, not chosen afterwards |
| [8. Registration](#8-registration) | commits, lockfile, data state at registration |
| [9. Protocol deviations](#9-protocol-deviations-and-exploratory-analyses) | **the log — 19 entries, and the bulk of this file** |

**A convention worth knowing before Section 9.** References of the form "plan
Sec. 4.3" point at the project's *design* document, a separate working file that
is not published; the design it describes is covered in `WRITEUP.md` Section 3.
References of the form "Sec. 4" without the word "plan" are sections of *this*
document. The distinction matters most in Section 9, where both appear.

**Section 9 is append-only and is the point of the exercise.** It records every
departure from the protocol above, every exploratory addition, and every error
found after the fact — with the measured impact of each, whether or not that
impact was zero. Several entries document defects in this project's own analysis
that were caught and corrected; they are here because a pre-registration whose
deviation log is empty is not evidence of discipline, it is evidence that nobody
looked.

| date | entry |
|---|---|
| 2026-08-29 | pilot finding, not a deviation |
| 2026-08-30 | cross-generator replication model changed, forced |
| 2026-08-30 | repeat-call determinism on the hosted backend, measured |
| 2026-08-30 | replication analysis population, measured in advance |
| 2026-08-30 | two mitigations added for the limitations recorded above |
| 2026-08-30 | cross-family probe result. RQ1 is not Qwen-specific |
| 2026-08-30 | the P=3 / P=5 gap between replication and main run is removable |
| 2026-08-30 | cross-device determinism gate: PASSED, 50/50 |
| 2026-08-31 | `llmlingua2`'s `force_tokens` changed after the main run had started. A protocol deviation, on a confirmatory arm |
| 2026-08-31/09-01 | the Holm family was applied wrongly in the first analysis pass, and is corrected. No conclusion changed |
| 2026-09-01 | the `llm_pruner` selection-stability probe re-run at n=100. Exploratory, and it lowers the number this project has been quoting |
| 2026-09-01 | 2WikiMultihopQA exclusion count, under the rule registered in Sec. 3 |
| 2026-09-01 | `llmlingua2` results corrected: a caching defect was serving one question's compressed passages to every question |
| 2026-09-01 | the unfiltered population, reported at last. Plan Sec. 4.1 asked for both; only the filtered numbers existed |
| 2026-09-02 | the memorization filter applied `em >= 1.0`, not the registered `f1 >= 0.8`. Deviation, found late, immaterial to every result |
| 2026-09-02 | full-project audit. Four documentation defects, one unrun registered check, one control strengthened. No result changed |
| 2026-09-02 | a significance flag that was floating-point noise, found while running the delimiter check |
| 2026-09-02 | the registered delimiter variant, run at last. Effect unchanged |
| 2026-09-02 | the slot-count interaction, answered from committed artifacts |

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
position-matched placebo the primary control. OAE remains the headline
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
- **Code registered:** `822bb24` — the state of `src/` this plan applies to.
  All nine arms in the Holm family are implemented and tested at that commit;
  `loo_oracle` (week 3) is not, and its Oracle Gap is therefore a secondary
  quantity here rather than part of the confirmatory family.
- **Plan commit:** `2f24548` "REGISTER the analysis plan, before any main-run
  generation". Recorded in a following commit rather than in `2f24548` itself,
  because a file cannot contain its own hash. `git show 2f24548` is the
  registered text, and no `src/` file changes in it or after it.
- **Note on history.** Commit messages were rewritten once, before the
  repository was first published, to strip tooling trailers. No file content,
  author, or author date changed. That rewrite changed every SHA, so the two
  above are the post-rewrite values, and SHAs quoted inside older *commit
  messages* refer to the pre-rewrite history and will not resolve. Ordering,
  dates and tree contents — which is what the registration claim actually rests
  on — are unaffected: `git log --format='%ad %s' --date=iso` shows the plan was
  registered before any main-run generation existed, and `results/` contains
  only the week-1 pilot.
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

- **2026-08-30 — cross-generator replication model changed, forced.** Sec. 7
  registers "Groq cross-generator replication, n=100" without naming a model;
  the working plan named Llama 3.3 70B. That checkpoint has been retired from
  Groq's catalogue and returns 404. The replacement is **`qwen/qwen3.8-27b`**,
  chosen by measurement rather than preference, and the choice narrows what the
  replication can claim.

  Every other chat model on the account is a *reasoning* model whose chain of
  thought is billed against the completion budget. At the registered
  `max_new_tokens=32` they return an **empty** answer. Measured on 10 real
  HotpotQA prompts, greedy, `max_new_tokens=32`:

  | model | truncated | mean F1 | mean EM |
  |---|---|---|---|
  | `qwen/qwen3.8-27b` | 0/10 | 0.830 | 0.700 |
  | `openai/gpt-oss-120b` (`reasoning_effort=low`) | 6/10 | 0.380 | 0.300 |
  | `openai/gpt-oss-20b` | 10/10 | — | — |
  | `qwen/qwen3.6-27b` | leaks `<think>` into the answer | — | — |

  `gpt-oss-120b` becomes usable if `max_new_tokens` is raised to give the
  reasoning room, and that was considered. It was rejected because a reasoning
  model re-reads the context internally, so a *smaller* permutation effect there
  would be uninterpretable — scale, model family and reasoning would all be
  confounded in one comparison. Preserving the registered decode parameters was
  judged worth more than the wider model contrast.

  **Consequence, to be stated in the write-up rather than buried:** the
  replication is a **scale** check (3B → 27B, ~9x) and **not a family** check.
  Qwen2.5-3B-Instruct and Qwen3.8-27B share a training lineage and may share
  positional biases, so this run cannot rule out a Qwen-family artifact. The
  original claim — "not a 3B artifact" — still holds if the effect survives;
  the stronger claim "not a Qwen artifact" does not, and must not be made.

- **2026-08-30 — repeat-call determinism on the hosted backend, measured.**
  Registering a hosted generator raised the concern that serving
  nondeterminism at `temperature=0` would contaminate the within-query SD, which
  is the study's primary endpoint. Measured before the run: 4 HotpotQA queries x
  5 identical repeat calls to `qwen/qwen3.8-27b`, greedy, fixed seed —
  **20/20 byte-identical, 0 divergences.**

  This is evidence, not a guarantee. All 20 calls were one session against
  whatever server pool the account was routed to, and the answers are 3–5 tokens,
  which is the easiest case. It does not license treating the Groq SD as
  interchangeable with the local one. It does mean the contamination is small
  enough that a non-zero replicated SD is a real signal rather than an artifact
  of the transport, which is the question that mattered.

- **2026-08-30 — replication analysis population, measured in advance.** The
  memorization filter was run on the replication generator before the grid, so
  the population size is known rather than discovered afterwards: **100 → 74
  queries** survive on `qwen/qwen3.8-27b`. Sec. 3 defines the population as the
  nocontext-incorrect set *for the generator in question*, so this is the
  registered procedure applied to a second model, not a new decision.

  Recorded because it is the number that would otherwise be quietly compared
  against the main run's: the two populations are different sets, and any
  arm-level figure from this run is conditioned on these 74 queries. The
  filtered-out 26 are queries a 27B answers from memory with no context; if the
  main run's local filter removes a different fraction, that difference is
  itself reportable and must not be presented as a difference in arm behaviour.

- **2026-08-30 — two mitigations added for the limitations recorded above.**
  Both address weaknesses introduced by the forced model change, both are
  post-registration and therefore secondary, and neither enters the
  confirmatory Holm family.

  **(a) Cross-family probe, RQ1 only.** `configs/replication_xfamily.yaml`:
  `allam-2-7b`, the only non-Qwen generative model on the account that answers
  in the registered format at `max_new_tokens=32` (1/10 truncated, F1 0.445,
  EM 0.300 on the same 10-prompt probe). One arm — `full` — because the
  question is narrow: RQ1 lives there, and `full` is all that is needed to say
  whether a permutation effect exists outside the Qwen family. 400 calls.

  Powered for **existence, not magnitude.** ALLaM-2-7B is a weaker model on
  this task than either Qwen, so a different SD is expected and says nothing
  about scale; comparing SD magnitudes across the two would confound family
  with capability. The claim it supports is "the effect is not confined to the
  Qwen family", and no more.

  Also recorded so it is never undone: `groq/compound` and `groq/compound-mini`
  are excluded permanently. They are agentic systems with web search, and a
  generator that can look up the answer voids the closed-context premise the
  whole study rests on.

  **(b) Determinism audit.** `src.run --audit N` re-issues N already-cached
  grid prompts and reports how many return byte-identical, writing
  `determinism_audit.json` beside the generations. It deliberately bypasses the
  cache wrapper — going through it would replay the stored answer and report a
  perfect score.

  The 20/20 figure recorded above was measured in a single session, which is
  the weak version of the test. The live risk is different: this grid exceeds
  the daily request cap and therefore **spans two days**, so day-1 and day-2
  generations may come from different serving conditions inside one run. Run
  the audit on the second day; a same-session repeat measures very little.

  Two structural defences already limit the exposure and should be stated
  alongside whatever the audit returns. The cache pays for each distinct prompt
  once, so no cell is re-sampled. And `run` emits the P permutations of a cell
  consecutively, so the calls the within-query SD is computed from are issued
  seconds apart — day-scale drift lands *between* cells, not inside the primary
  endpoint. The audit measures what those two arguments do not cover.

- **2026-08-30 — cross-family probe result. RQ1 is not Qwen-specific.**
  `configs/replication_xfamily.yaml` completed on `allam-2-7b`: 79 queries after
  the filter, 3 permutations, `full` arm, 337 rows.
  `results/replication_xfamily/gate_report.txt`.

  | | local Qwen2.5-3B (pilot) | ALLaM-2-7B (this probe) |
  |---|---|---|
  | queries / permutations | 100 / 5 | 79 / 3 |
  | memorization filter | **off** | **on** |
  | mean F1 | 0.4867 | 0.3153 |
  | median within-query SD | 0.0263 | **0.2887** |
  | queries that move at all | 50.0% | **64.6%** |
  | queries with zero variance | 50.0% | 32.9% |

  **What this establishes:** a permutation effect exists outside the Qwen
  family. 64.6% of queries change answer under reordering alone, and the gate
  passes by a factor of ~14. The pre-registered claim — that the finding is not
  an artifact of one model lineage — holds.

  **What it does NOT establish, and must not be written up as if it did:** that
  the effect is *larger* outside the Qwen family. The 0.2887-against-0.0263
  comparison is confounded three ways and none is small.

  1. **Different populations.** The pilot ran with the filter OFF and this ran
     with it ON. The filter removes queries the generator answers from memory —
     which are precisely the queries that do not need the context and are
     therefore the most likely to be *stable* under permutation. Removing them
     mechanically raises the measured SD. This confound alone could account for
     much of the gap, and it applies to the main run too: **expect the n=300
     filtered SD to exceed the pilot's 0.0263 for this reason before any other.**
  2. **P=3 against P=5.** An SD over three draws is a noisier estimate, and the
     median-of-SDs statistic is not invariant to P. **This one is removable and
     should be removed** — see the 2026-08-30 entry on prefix-restriction below.
  3. **Capability.** ALLaM is the weaker model here (mean F1 0.3153 against
     0.4867). A model with more headroom to be wrong has more room to vary, and
     that is capability, not family.

  One thing the comparison *does* support cleanly: **the median is no longer
  knife-edge.** The 2026-08-29 entry recorded that the pilot's 0.0263 sat on the
  seam of a bimodal distribution and that one further static query would have
  produced 0.0000 and a FAIL. At 0.2887, with 64.6% moving, this probe's median
  sits far from that seam. The distribution is still bimodal (IQR
  [0.0000, 0.5774]) so the distributional presentation registered on 2026-08-29
  still applies, but the statistic is not fragile here.

- **2026-08-30 — the P=3 / P=5 gap between replication and main run is
  removable.** The replication configs use `strategies: [rank, reverse, random]`
  where `main.yaml` uses `[rank, reverse, random, random, random]`, which looked
  like a permanent comparability cost of the cheaper sizing. It is not.

  `chunks.permutation_set` numbers random replicates in order and seeds each on
  `(seed, replicate, qid)`, so the first random draw is the same draw in both
  lists. Verified on a 10-chunk cell: the P=3 orderings are byte-identical to
  the first three P=5 orderings, index for index.

  **Therefore the main run must be prefix-restricted to its first three
  permutations when compared against either replication.** Comparing the main
  run's 5-permutation SD against a replication's 3-permutation SD would be
  comparing two different estimators and attributing the difference to the
  generator. The full P=5 SD remains the primary within the main run itself;
  this restriction applies only to cross-generator comparisons.

  Confound (1) of the entry above — the memorization filter — is *not* removable
  the same way, because the populations genuinely differ per generator. Confound
  (3), capability, is not removable at all. Only report what survives all three.

- **2026-08-30 — cross-device determinism gate: PASSED, 50/50.** Generation for
  the main run moves from the laptop RTX 3060 to a Colab T4. `cache.py` keys on
  `sha256(model, prompt, decode_params)` and records neither GPU nor batch size,
  so rows from the two machines would collide under identical keys — and the
  ~1,049 Qwen2.5-3B rows already cached from the week-1 pilot *are* reused by the
  main run, because `full` ignores budget and its prompts at k=2/3/5 are
  byte-identical to the pilot's at k=10.

  Measured before relying on it, with `src.run --config configs/pilot.yaml
  --audit 50`: **50/50 byte-identical on re-issue (100.0%)**. The audit bypasses
  the cache wrapper, so these are real generations compared against the stored
  ones, not cache reads.

  Two supporting observations, both exact matches rather than approximations:
  Colab runs the same `torch 2.11.0+cu128` build as the development machine, and
  the 4-bit checkpoint allocates the same 2.06 GB. `src.smoke` check 3b also
  passed on the T4 (batch position does not change the answer) at `batch_size:
  4`, which is the value `configs/main_colab.yaml` uses — that check has *not*
  been run at any larger batch, so raising it would require re-running 3b first.

  **Consequence:** the pilot rows stand, the main run reuses them, and generations
  from the two machines may be pooled. Had this diverged, the registered response
  was to delete the `Qwen/Qwen2.5-3B-Instruct` rows and regenerate on the T4.
  Recorded so the pooling is a documented decision with a number behind it rather
  than an assumption nobody checked.

- **2026-08-31 — `llmlingua2`'s `force_tokens` changed after the main run had
  started. A protocol deviation, on a confirmatory arm.** Recorded here because
  Sec. 5 puts `llmlingua2` inside the nine-comparison Holm family, so this is not
  a change to an exploratory arm.

  The arm was built with `force_tokens = ["\n", "?", ".", ","]`. Forced tokens
  are charged against LLMLingua-2's compression budget, and at the k=2 rate the
  punctuation was consuming most of it: the old list kept **2 content words out
  of 51** where upstream's `["\n", "?"]` kept **8**. `.` and `,` were dropped to
  match upstream (commit `df8ab6b`) and the arm was regenerated 21:47-22:33 on
  2026-08-31, after the rest of the grid was already complete.

  **Measured effect on the arm: EM 0.0550 → 0.0608, token-F1 0.0903 → 0.0970.**
  Real, and small. It does not change any conclusion about the arm — see below.

  **The alarm that prompted the change was a misreading, and the original
  rationale is retracted** (commit `765e931`). The trigger was `llmlingua2`
  appearing to score below the `nocontext` floor. It never did: `nocontext` is
  reported over all 300 sampled queries while every other arm is reported over
  the 274 surviving the memorization filter, and the filter is *defined* as
  dropping the queries the generator can answer with no context — so **on the
  studied 274 the nocontext floor is 0.0000 by construction** (0.0089 token-F1;
  `arm_summary.csv`, row `nocontext@studied`). Comparing 0.055 against 0.087 was
  reading one arm's mean against another arm's population. The change stands on
  the content-word measurement alone, which is a separate and sufficient reason.

  **Auditability.** The pre-change aggregates are preserved rather than
  overwritten, in `results/main_hotpotqa/_pre_llmlingua2_fix/` (commit
  `52276b6`), with the per-arm deviation in `llmlingua2_deviation.json`. **Do
  not pool pre- and post-change `llmlingua2` numbers.** Every `llmlingua2`
  figure reported anywhere is post-change unless it says otherwise.

  **One thing was obtained for free.** The regeneration re-ran the whole grid
  through the cache, and **eleven of the twelve arms returned identical to four
  decimal places** — an unplanned determinism check across process restarts,
  independent of the one in Sec. 9's cross-device entry.

- **2026-08-31/09-01 — the Holm family was applied wrongly in the first analysis
  pass, and is corrected. No conclusion changed.** The first pass over the
  finished main run was made with a scratch script rather than committed code.
  It corrected RQ2 and RQ4 as **two separate families**, each spanning whichever
  arms happened to be present in the results — ten comparisons and eight — where
  Sec. 5 registers **one family of nine**: OAE vs `rerank_topk` for
  `provence_rerank`, `provence_full`, `llmlingua2` and `llm_pruner`, plus Placebo
  Gap vs `placebo_pos:middle_first` for those four and `rerank_topk`. `full`,
  `loo_oracle`, `random_drop` and the two exploratory placebo variants are
  outside the family **by registration**, not by oversight.

  The error was in the conservative direction — a larger family over-corrects —
  but it is still the wrong number. `provence_rerank`'s OAE was reported at Holm
  p = 0.0848 where the registered family gives **0.0636**. Both are above 0.05,
  so H2's outcome is unchanged and the primary endpoint (Holm p = 0.0018) is
  unaffected; the point is that 0.0848 is the number that would otherwise have
  gone into the write-up.

  **Fixed in code, not in prose** (commit `83d1759`). `src/analyze.py` now owns
  the confirmatory analysis, applies Holm in exactly one place over
  `CONFIRMATORY_OAE` / `CONFIRMATORY_PLACEBO_GAP`, and leaves raw p-values in the
  per-arm blocks so nothing there can be mistaken for a corrected confirmatory
  result. `tests/test_analyze.py` pins the family at nine regardless of which
  arms are present in a given run. **Arms must not be added to those tuples to
  "be thorough": Holm's adjustment depends on the size of the family, so
  widening it silently changes the registered numbers.**

  `results/main_hotpotqa/permutation_analysis.json` was regenerated from
  `python -m src.analyze --config configs/main.yaml` on **2026-09-01** and
  committed. The superseded scratch output was never committed, for this reason.

- **2026-09-01 — the `llm_pruner` selection-stability probe re-run at n=100.
  Exploratory, and it lowers the number this project has been quoting.** The
  2026-08-29 probe measured selection Jaccard across three presentation orders on
  **20** queries and got **0.263**, changed in 19 of 20. `src/selection_probe.py`
  now drives the same check as a committed artifact
  (`results/main_hotpotqa/selection_stability.json`) and it was run at **n=100**:

  | | n=20 (2026-08-29) | n=100 (this run) |
  |---|---|---|
  | mean selection Jaccard | 0.263 | **0.213** |
  | selection changed | 19/20 | **98/100** |
  | three presentations sharing no passage at all | — | **23/100** |
  | distance from chance to order-invariant | ~23% | **~17%** |

  **This is not a protocol deviation.** The quantity is a robustness check on an
  arm's internals, not one of the nine confirmatory comparisons in Sec. 5, and
  nothing here enters the Holm family or bears on the primary endpoint. It is
  recorded because the number changed and the old one is quoted in several
  places.

  Two things make the change auditable rather than merely a different answer.
  Subsampling is nested (Sec. 4), so the n=20 population is a strict *prefix* of
  the n=100 one: recomputed on exactly those 20 queries the probe returns
  **0.2625, changed 19/20**, reproducing the original. And the upper reference is
  now **measured rather than asserted** — `rerank_topk` is run through the same
  three permutations and returns **1.000**, so the claim "an independent scorer is
  order-invariant" is a result here rather than an argument about implementations.

  **Prefer the n=100 figure and say which is which.** The larger sample is the
  better estimate; the n=20 one is not wrong, it is small.

- **2026-09-01 — 2WikiMultihopQA exclusion count, under the rule registered in
  Sec. 3.** The secondary dataset is implemented and its population is now
  measured rather than assumed. Sec. 3 lists "not exactly 2 gold paragraphs"
  among the candidate exclusions, records it at **0 rows** on HotpotQA, and fixes
  the rule for later datasets in advance: *drop the row, report the count, and
  never drop a row on the basis of the answer it produced.* Applied to
  2WikiMultihopQA validation (`framolfese/2WikiMultihopQA`, 12,576 rows):

  | | rows | share |
  |---|---|---|
  | not exactly 10 paragraphs | **0** | 0% |
  | not exactly 2 gold paragraphs | **2,751** | **21.88%** |
  | working population | **9,825** | 78.12% |

  **The 2,751 excluded rows are exactly the `bridge_comparison` type**, every one
  of which carries four gold paragraphs. The exclusion is therefore not random
  with respect to difficulty: it removes one of the dataset's four question
  types entirely, and the surviving population is `compositional` (5,236),
  `comparison` (3,040) and `inference` (1,549). **The write-up must say that the
  2Wiki replication covers three of four question types, and which one is
  missing.**

  *Why the rule is right even though it costs a fifth of the data.* At k=2 a
  four-gold question cannot retain all its evidence even in principle, so gold
  recall, the Placebo Gap at matched keep-count and the Oracle Gap would each
  mean something different on those rows than on every other row in this study.
  Keeping them would not add generality, it would silently mix two populations
  inside one number.

  *Two things that make this auditable rather than convenient.* The rule was
  registered before the count was known, on 2026-08-29, when the answer on
  HotpotQA was zero and there was nothing to gain by writing it. And the filter
  is applied **uniformly to every dataset** rather than switched on for 2Wiki:
  re-measured on HotpotQA it drops **0 of 7,345** rows, so the completed main
  run's population is provably undisturbed by its introduction.

  *Also recorded:* 2Wiki has **four** hop types against HotpotQA's two, so
  stratification runs over four strata (three after the exclusion). Its contexts
  are **shorter**, median 3,162 characters against HotpotQA's 5,292 for the
  `full` arm. And 30 of 300 sampled answers contain non-ASCII characters, where
  HotpotQA's are effectively all ASCII; nothing in the analysis depends on this,
  but it did surface one latent defect in the determinism audit's console
  output, fixed rather than worked around.

- **2026-09-01 — `llmlingua2` results corrected: a caching defect was serving one
  question's compressed passages to every question.** Not a protocol deviation. A
  code defect, its effect on registered quantities, and the correction.

  The arm's per-passage compression cache was keyed on `(chunk idx, rate)`.
  Neither identifies a passage: `run.py` is arm-major so one instance serves every
  question, and `rate` is k/n, constant across questions. The arm had ~30 distinct
  keys for the whole dataset, so **every question after the first received the
  first question's compressed passages**. Fixed by keying on `sha256(text)` and
  the rate (commit `5807566`), with a regression test that fails on the old key.

  **Effect on the registered quantities**, all at k = 3, 10,000 replicates:

  | quantity | as reported | corrected |
  |---|---|---|
  | `llmlingua2` mean token-F1 | 0.0983 | **0.3218** |
  | Placebo Gap (confirmatory) | -0.0879 [-0.1309, -0.0452] | **+0.1356 [+0.0884, +0.1839]** |
  | OAE (confirmatory) | -2.8858 | **-0.9294** |
  | RQ1 within-question SD | 0.0881 | **0.1666** |

  Both of this arm's confirmatory comparisons moved, and the Placebo Gap **changed
  sign**: it was significantly worse than the positional placebo and is
  significantly better. At k = 2 and k = 5 the gap likewise goes -0.0698 to
  +0.1192 and -0.1648 to +0.0958.

  **What did not change, verified rather than assumed.** Every other arm
  reproduces to four decimal places. The **primary endpoint is identical**:
  +0.2760 [0.2223, 0.3297], uncorrected p = 0.0002, Holm p = 0.0018. Every Holm
  value in the family is unchanged, because the affected p-values were already at
  the bootstrap floor of 0.0002 both before and after, so the step-down ordering
  did not move. **RQ3 is identical at all three budgets** despite `llmlingua2`
  being one of its six arms: it was and remains the lowest-ranked of them, so no
  pairwise sign flipped.

  **Consequences for the write-up, since a reported finding was an artifact.** The
  claim that this arm loses to a positional placebo, and the interpretation built
  on it that "evidence in shredded form is worth less than the evidence intact",
  are **withdrawn**. So is the empirical status of "per-chunk compression preserved
  100/100 across orderings": identical cache keys returned identical entries, so
  that comparison could not have returned anything else. Per-chunk compression is
  order-invariant *by construction*, which is a design property and not a
  measurement. The joint-compression result, **0 of 100**, is unaffected: no
  per-chunk cache participates in it.

  **Also withdrawn:** the measured effect of the 2026-08-31 `force_tokens` change
  (EM 0.0550 to 0.0608, F1 0.0903 to 0.0970). Both sides of that comparison were
  computed on aliased data. The change itself stands on the content-word count,
  which was measured on text directly.

  Pre-correction aggregates are preserved in
  `results/main_hotpotqa/_pre_alias_fix/` with a README stating what in them is
  invalid.

- **2026-09-01 — the unfiltered population, reported at last. Plan Sec. 4.1 asked
  for both; only the filtered numbers existed.** Not a deviation: the filtered
  set remains the registered primary population and nothing here enters the
  confirmatory family. The whole grid was regenerated with
  `memorization_filter: false` (`configs/main_unfiltered.yaml`, 49,800 rows over
  300 questions) and the identical analysis run against it.

  **The question this answers** is narrower than "does the filter matter". It
  obviously shifts the level. The risk worth testing is whether it shifts the
  arms *differentially*, since the filter removes the questions answerable
  without context and those may be the ones where evidence quality matters least.
  That would bias every arm-versus-arm comparison in the study.

  | | filtered 274 | unfiltered 300 |
  |---|---|---|
  | primary endpoint, k=2 | +0.2895 | **+0.2793** |
  | primary endpoint, k=3 | +0.2760 | **+0.2562** |
  | primary endpoint, k=5 | +0.1780 | **+0.1623** |
  | Holm survivors, k=2 / k=3 / k=5 | 6/9, 6/9, 6/9 | 6/9, 6/9, **5/9** |
  | `random_drop` excludes zero | no, at all three | no, at all three |
  | RQ1 zero-SD share (`full`, k=3) | 50.0% | 51.7% |

  Holm p = 0.0018 for the primary endpoint in all six cells. All fifteen Placebo
  Gap comparisons survive on both populations and no arm ordering changes on any
  research question.

  **Two changes, recorded rather than rounded away.** `OAE:llmlingua2` at k=5
  goes from Holm 0.0056 to 0.0744 and loses significance, the only such change in
  eighteen comparisons; its point estimate moves -0.4586 to -0.3288, so the arm
  looks slightly *less* bad unfiltered. And the rank flip rate at k=5 rises from
  0.0667 to **0.1067**, crossing the registered H3 threshold of 0.10. H3 is
  specified at the primary budget, where the rate is 0.0400 on both populations,
  so its registered outcome is unchanged; but on the fuller population at the
  largest budget the threshold is nominally met and the write-up says so.

  **Stated limit.** The populations share 274 of 300 questions, an overlap of
  91%, so this tests for bias introduced by the exclusion and is not an
  independent replication. Differential movement between arms would have shown up
  even at that overlap, and did not.

- **2026-09-02 — the memorization filter applied `em >= 1.0`, not the registered
  `f1 >= 0.8`. Deviation, found late, immaterial to every result.** Sec. 3
  registers correctness under `nocontext` as **token-F1 >= 0.8**, with EM
  reported alongside as a sensitivity check. The code applied EM at 1.0: the
  call site in `run.py` passed no rule, and `data.memorization_filter` defaults
  to `exact_match` at 1.0. So the two roles were swapped — the registered
  sensitivity check became the primary population and the registered primary
  rule was never run. Found while updating the working documents after the
  replication landed, not by a check that was looking for it.

  **What it changes.** One query of 300, `5a8a40015542996c9b8d5e72`, which the
  generator answers with no context at EM 0 and token-F1 exactly 0.8000. The
  registered rule excludes it; the applied rule keeps it. Population 274 rather
  than the registered 273. Re-running the primary endpoint on the registered
  population, since the applied population is a strict superset and every
  generation already exists:

  | primary endpoint | applied (`em >= 1.0`, n=274) | registered (`f1 >= 0.8`, n=273) |
  |---|---|---|
  | k=2 | +0.2895 [0.2363, 0.3420] | +0.2906 [0.2371, 0.3435] |
  | k=3 | **+0.2760 [0.2223, 0.3297]** | **+0.2742 [0.2219, 0.3275]** |
  | k=5 | +0.1780 [0.1314, 0.2259] | +0.1799 [0.1337, 0.2269] |

  p = 0.0002 uncorrected in all six cells. The largest move in a point estimate
  is 0.002, in the third decimal, and no interval, ordering or conclusion
  changes. This is the outcome Sec. 3 predicted when it said "nothing rests on
  the definition" and registered it anyway so the choice could not be made after
  seeing the numbers — which is exactly why the deviation is recorded here
  rather than quietly corrected.

  **Not re-baselined, deliberately.** Switching the published population to the
  registered 273 now would churn every number in the write-up, the figures and
  the replication's matched comparison, to move a third decimal. The applied rule
  is what produced the published numbers and is kept, with the registered rule
  reported above as the sensitivity check Sec. 3 asked for in the other
  direction.

  **Fixed so it cannot recur silently.** `run.py` now reads the rule from the
  config and prints it (`memorization filter (em >= 1.0): 100 -> 74 queries`),
  and every config that filters states `metric` and `threshold` explicitly rather
  than inheriting a function default. A registered parameter reachable only
  through a default is the defect here; the wrong value was the symptom.

- **2026-09-02 — full-project audit. Four documentation defects, one unrun
  registered check, one control strengthened. No result changed.** A systematic
  re-derivation of every headline number from the committed artifacts. The
  arithmetic held: all 24 Placebo Gap cells, the three `random_drop` p-values,
  every RQ1 and RQ3 figure, all per-arm means, the selection-probe Jaccard
  (0.2134), its 98/100 and 23/100 counts, `llm_pruner`'s 200/822, and every
  `loo_oracle` statistic reproduce exactly from `generations.csv` and
  `arm_stats.json`. What follows is what did not.

  **1. "Byte-identical" was case-folded.** Sec. 4.1 reported 98 of 274 questions
  (35.8%) returning a byte-identical answer under all five orderings, and 39 of
  137 zero-SD questions producing more than one answer string. The byte-identical
  counts are **96 (35.0%)** and **41 (29.9%)**; 98/39 come from comparing
  `strip().lower()`. Two questions of 274, and the three-way split moves from
  36/14/50 to 35/15/50. Corrected, with both counts now reported.

  **2. Those counts were computed by hand.** No committed code produced them,
  against Sec. 8's claim that every quantity is reproducible from committed code.
  They are now `analyze.answer_stability`, emitted into
  `permutation_analysis.json`, with tests covering the case-only distinction.

  **3. "Fifteen confirmatory Placebo Gap comparisons" overstates the
  registration.** Sec. 5 registers nine comparisons **at the primary budget**,
  and names the other budgets exploratory. `analyze_budget` applies Holm within
  each budget, so k=2 and k=5 get their own families of nine. That is
  conservative — it corrects what the plan said needed no correction — but only
  the five k=3 rows are confirmatory. The write-up now says so.

  **4. A bootstrap p-value has a floor and several results sit on it.**
  `p_two_sided` returns `2 * max(prop, 1/B)`, so at B=10,000 the smallest
  attainable value is 0.0002 and the reported "p = 0.0002" means p < 0.0002. For
  the primary endpoint the floor is far from binding — 0 of 10,000 replicates
  fall at or below zero and the minimum is +0.1754 — but the reading is now
  stated rather than left to a reader who knows how the statistic is computed.
  The Holm 0.0018 is nine times the same floor.

  **5. A registered robustness check was never run and was not disclosed as
  unrun.** Sec. 7 lists a prompt-template variant with an alternative delimiter
  style. It is implemented (`ALT_TEMPLATE`, `prompt_template: alt`) and tested,
  but no config uses it and it has no results. It is now named as an omission in
  WRITEUP Sec. 7. It matters more than its cost: passages are numbered by
  presentation position, so a reordering changes the numeric labels as well as
  the semantic order, and this is the check that bounds how much of RQ1 is the
  numbering. Roughly 1,400 local generations.

  **6. The placebo is keep-count matched but not position matched, and the study
  already had the control.** Real pruners select near-uniformly across the ten
  slots; each placebo variant takes a fixed three-point set. The gap could
  therefore mix evidence starvation with positional handicap. The three
  registered variants span three maximally different configurations — 0-indexed,
  `middle_first` {0,8,9}, `edges_first` {4,5,6}, `tail_first` {0,1,2}; the
  write-up gives the same sets 1-indexed to match the prompt's `[1]`-based
  passage labels — and score
  0.1862 / 0.1753 / 0.1855, with `random_drop` at 0.1672. A spread of **0.019**
  against a primary endpoint of **0.2760** bounds the positional contribution at
  about 7%, with the gold column (1.686 against 0.650) carrying the rest. Added
  to WRITEUP Sec. 4.2. `edges_first` and `tail_first` were registered as
  exploratory and are used here after the fact, so the control is post-hoc in its
  use though not in its arms.

- **2026-09-02 — a significance flag that was floating-point noise, found while
  running the delimiter check.** `BootstrapResult.excludes_zero` tested
  `lo > 0 or hi < 0` with no tolerance. A percentile interval whose replicates
  all land on one side of zero by cancellation error then reports as excluding
  it. That happened once, in a published artifact: the 27B replication's RQ1 for
  `rerank_topk` at k=5 came back **lo = 1.5618e-17**, and
  `permutation_analysis.json` recorded `excludes_zero: true`.

  Nothing rests on it and nothing else is affected. Of the 193 intervals the
  study reports as excluding zero, that was the only one within 1e-12 of it; the
  next-closest comes no nearer than **0.0078**, ten orders of magnitude clear. A
  `ZERO_TOL` of 1e-12 therefore moves exactly one flag and cannot suppress a
  finding. Added, with a regression test naming the case.

  **It had propagated into prose**, which is the part worth recording. WRITEUP
  Sec. 4.9 and Sec. 5 both said "every 27B interval excludes zero" on the
  strength of it, written the same day and not checked against k=5. Corrected to
  scope the claim to the primary budget, with the k=5 exception now stated
  explicitly beside the OAE-denominator limitation it shares a cause with: at
  k=5 on this arm the 27B is close to order-insensitive, which is why both the
  interval touches zero and the ratio blows up.

  Same session, `src.matched_comparison` was extracted so the generator
  comparison and the delimiter check share one implementation of the
  order-identity assertion and the paired bootstrap. The generator comparison
  reproduces every value it published before the extraction.

- **2026-09-02 — the registered delimiter variant, run at last. Effect
  unchanged.** Sec. 7 has listed a prompt-template robustness check since
  registration; the audit earlier today found it implemented, tested, and never
  run. `configs/robustness_delimiter.yaml` runs it: the `full` arm, k=3, the same
  274 questions and the same five orderings as the main run, 1,370 generations at
  1.20 s each, with every field but `prompt_template` copied from
  `configs/main.yaml` and a test asserting that.

  | mean within-question SD, `full`, k=3, n=274 | |
  |---|---|
  | default | 0.1795 [0.1554, 0.2038] |
  | alt (`<context>` fencing) | 0.1748 [0.1501, 0.1988] |
  | paired difference | **0.0047 [-0.0140, 0.0239]**, p = 0.6298 |

  All 1,370 cells verified to present byte-identical passage orders, so the two
  runs differ in the template alone. The effect does not move: ratio 1.03, and
  the difference nowhere near excluding zero. RQ1 is not an artifact of one
  delimiter style. WRITEUP Sec. 4.10.

  **What it does not settle, stated because the check is easy to over-read.** It
  varies the fencing, not the `[i]` passage numbering, which changes under
  reordering because the indices mark presentation slots. A template without
  indices would be a different prompt rather than a delimiter variant. Sec. 6
  keeps that as an open limitation and Sec. 7 now names it as the follow-up.

- **2026-09-02 — the slot-count interaction, answered from committed artifacts.**
  Sec. 4.9 of the write-up read the 3B-27B gap as the effect scaling with
  permutable slots, and Sec. 7 filed the test as future work "by varying the slot
  count on a fixed generator". The main run already varied it — k = 2, 3, 5 for
  the keep-k arms and 10 for `full` and `llmlingua2` — so the within-generator
  half needed analysis, not generation. `src.slot_count`, WRITEUP Sec. 4.11.

  **The confound had to be handled first.** Slot count and evidence retained rise
  together in every keep-k arm, so a raw "SD rises with k" cannot distinguish
  them. Stratifying on the gold count each cell retained holds the evidence fixed
  and leaves the slots varying. Within every stratum the SD still rises, and all
  three contrasts exclude zero:

  | stratum | contrast | |
  |---|---|---|
  | 0 of 2 gold | 2 → 5 slots | +0.0372 [+0.0211, +0.0549] |
  | 1 of 2 gold | 2 → 5 slots | +0.0567 [+0.0416, +0.0724] |
  | 2 of 2 gold | 2 → 10 slots | +0.0809 [+0.0540, +0.1073] |

  Reading across a row, the evidence effect is real too. Both mechanisms
  contribute and neither explains the other away. `full` is the internal control:
  ten slots and identical content at all three budgets, SD 0.1795 at each, so the
  estimator does not respond to the budget label on its own.

  Two things worth keeping. The effect survives in the **0-gold stratum**, where
  the context holds neither gold passage — order moves the score even when the
  evidence is absent, which makes this partly a property of the context as an
  object. And it **saturates**: +0.0713 from 2 to 5 slots, +0.0095 from 5 to 10.

  Exploratory, uncorrected, and **within-generator only**. Whether the 3B-27B gap
  tracks slot count needs both generators at more budgets than the replication
  ran; Sec. 7 now says so instead of proposing the test that was already
  answerable.
