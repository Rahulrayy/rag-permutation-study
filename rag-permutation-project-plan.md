# Is it the pruning, or the ordering?

**A permutation-controlled re-evaluation of context selection in RAG**

Working title. Project plan, v1.

---

## 1. The pitch, in one paragraph

Context pruning is now a crowded subfield: a dozen methods claim they can throw away 60–90% of retrieved context with little or no quality loss. Every one of them is evaluated with the retrieved passages in one fixed order — usually retriever rank. Separately, a different line of work has established that RAG answers are unstable under permutation of that same retrieved set: reorder the passages, keep the content identical, and the model's answer changes. Nobody has put these two facts together. If answer quality moves when you shuffle the context, then a pruner's reported gain is measured against a noisy reference point, and part of what looks like "better evidence selection" may just be a lucky interaction with position bias. This project measures how much.

The output is not a new pruning method. It is a protocol, two controls nobody currently runs, and a number: how large is a typical pruning gain relative to the variance induced by ordering alone.

---

## 2. What already exists

I did the literature pass before committing. Summary of the landscape as of August 2026:

**Context attribution (which chunks caused the answer).** Saturated. ContextCite (Cohen-Wang et al., NeurIPS 2024) defined the problem and the surrogate-model approach. AttriBoT (arXiv 2411.15102) is a bag of efficiency tricks for approximating leave-one-out. ARC-JSD (arXiv 2505.16415) does it mechanistically via Jensen-Shannon divergence and reports 3x speedups over ContextCite. There's a multi-armed-bandit formulation (arXiv 2506.19977), an attention-based variant at ECIR 2026, and RISE (arXiv 2602.01378) which explicitly targets redundancy-insensitivity.

**Context pruning / compression.** Also saturated. Provence (arXiv 2501.16214) folds sentence-level pruning into the reranker so it costs nothing extra; XProvence (arXiv 2601.18886) extends it multilingually. Information Gain Pruning (arXiv 2601.17532) reports 76–79% input-token reduction with F1 *gains*. AdaGReS (arXiv 2512.25052) does redundancy-aware greedy selection under a token budget. AttentionRAG, LLMLingua-2, LooComp, CORE, REFRAG — all in this space. Shapley Context Pruning (arXiv 2607.16209, July 2026) does the cooperative-game version with Monte-Carlo Shapley and a permutation-invariant Deep Sets value function.

**Position and permutation effects.** Established but studied separately from pruning. Lost-in-the-Middle (Liu et al. 2024) is the origin. Stable-RAG (arXiv 2601.02993) shows answers vary substantially across permutations of a top-5 retrieved set even when the gold document is pinned first. "Lost in the Evidence?" (arXiv 2605.27105) reproduces position and context-size effects and notes that stable-looking averages can hide large instance-level swings. The Weakest Link Law paper (arXiv 2601.12499) shows multi-hop performance collapses to the visibility level of the least-visible evidence item.

**Interaction / non-additivity.** Barely touched. CUE-R (arXiv 2604.05467) ran a two-support ablation on 51 HotpotQA examples and found non-additive interaction in ~20% of them, with ~14% showing full complementarity (neither single removal hurt, joint removal broke the answer). n=51 is a footnote, not a result.

**The conclusion I draw.** The *method* space is closed — do not try to propose a better pruner, you will be scooped and you will not beat five well-funded groups on engineering. The *evaluation* space is wide open. This is the Ferrari Dacrema situation in a younger field: methods proliferating faster than the protocol used to compare them.

Assumption stated: I could have missed something. Before week 3, do a targeted search on "permutation-controlled evaluation context pruning" and "positional placebo baseline RAG" and re-check. If someone has published exactly this in the interim, the fallback in §9 applies.

---

## 3. The gap, stated sharply

Three premises, all supported above:

1. Pruners are evaluated under a single, arbitrary context ordering.
2. RAG answer quality is sensitive to that ordering, with the effect concentrated at coarse positional buckets (beginning/middle/end) rather than fine-grained distance.
3. Pruning *changes positions*. Removing chunks 3, 5, and 7 from a 10-chunk context does not just remove content — it promotes chunks 8, 9, 10 into higher-visibility slots.

Premise 3 is the one nobody says out loud. A pruner that happens to remove middle chunks is doing two things at once: content selection and positional promotion. Current evaluations cannot separate them.

**Research questions.**

- **RQ1.** How large is within-query answer variance across context permutations, at fixed content and fixed budget, under greedy decoding?
- **RQ2.** How large are published pruning gains relative to that variance? (Effect size in units of permutation noise.)
- **RQ3.** Does the *ranking* of pruning methods depend on which single ordering you evaluate at? If so, how often does the ranking flip?
- **RQ4.** How much of a pruner's gain survives a **positional placebo** — a control that removes the same number of chunks, chosen by position rather than by content?

RQ4 is the centerpiece. It is cheap, it is obvious in hindsight, and as far as I can find nobody runs it. Random-drop baselines exist in a few papers (REFRAG, RAP, OPRM), but a *position-matched* placebo does not. Worth noting: OPRM (arXiv 2505.07793) found random chunk selection beating full-context inference on HotpotQA long-context, which is precisely the kind of "the dumb control is stronger than expected" result that motivates this.

---

## 4. Design

### 4.1 Data

Primary: **HotpotQA distractor**. Ten paragraphs per question, two of them gold, packaged. No corpus, no index, no retrieval — which is the single decision that makes this fit on a laptop. Positions are meaningful and permutable by construction.

Secondary: **2WikiMultihopQA** (same shape, different construction, tests generality) and **NQ-open with retrieved top-10** (single-hop contrast; needs a Pyserini prebuilt index, add only if time allows).

n = 300 queries for the main run, 100 for the pilot. Stratify by hop type where the dataset labels it.

**Memorization control (non-negotiable).** Wikipedia-derived multi-hop benchmarks leak into pretraining. Run a no-context arm first and **restrict the main analysis to queries the generator answers incorrectly with no context**. Otherwise you are measuring parametric recall and calling it retrieval. Report both the filtered and unfiltered numbers.

### 4.2 Generators

- **Primary: a small local model.** Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct, 4-bit, on the 3060. This is not a compromise — it is required. Proper leave-one-out attribution needs the log-probability of the answer sequence, not a string match, and hosted APIs generally don't expose that. It also removes the rate limit from the critical path.
- **Cross-generator check: Groq.** A 100-query replication with Llama 3.3 70B or Llama 4 Scout, to show the finding isn't a 3B artifact.

**Greedy decoding, temperature 0, fixed seed, everywhere.** If you sample, sampling noise and permutation noise are confounded and the whole design collapses.

### 4.3 Arms

| Arm | What it is | Why |
|---|---|---|
| `full` | all 10 chunks | upper reference |
| `nocontext` | question only | memorization control |
| `rerank_topk` | cross-encoder rerank, keep top-k | the baseline everyone should beat |
| `provence` | off-the-shelf Provence checkpoint | published method |
| `llmlingua2` | token-level compression | published method, different family |
| `llm_pruner` | ask the generator which chunks to keep | strong, expensive, common in practice |
| `random_drop` | drop k uniformly at random | noise floor |
| **`placebo_pos`** | **drop k by position (middle-first, edges-first, tail-first)** | **the novel control** |
| `loo_oracle` | keep the k chunks with the largest LOO log-prob drop | causal ceiling |

Budgets: keep-k ∈ {2, 3, 5}. Report against input-token count, not k, so methods with different granularity are comparable.

### 4.4 The permutation protocol

For every (query, arm, budget) cell, generate under **P = 5** orderings of the *kept* chunks, sampled with a fixed seed: retriever-rank order, reverse order, and three random permutations. Record quality per permutation, not just the mean.

This is the whole trick. Everything else is standard.

### 4.5 Metrics

Answer quality: EM and token-F1. For HotpotQA, supporting-fact F1 as a secondary signal.

Then three derived quantities, all of which are the actual contribution:

**Order-Adjusted Effect (OAE).** For method *m* against baseline *b*:

```
OAE(m) = mean_q[ mean_π Q(m,q,π) - mean_π Q(b,q,π) ] / mean_q[ SD_π Q(b,q,π) ]
```

Read as: how many orderings-worth of noise does this method actually buy you. This is the headline number.

**Rank Flip Rate (RFR).** Take each single ordering π in turn, rank all methods by mean quality under that ordering alone, and count the fraction of method-pair comparisons whose sign disagrees with the permutation-averaged ranking. If this is high, single-order evaluation is unsound, full stop.

**Placebo Gap.** `Q(m) - Q(placebo_pos)` at matched keep-count. Near zero means the method is not doing content selection.

**Oracle Gap.** `Q(m) / Q(loo_oracle)` at matched budget. Headroom left on the table.

### 4.6 Statistics

Permutations are nested within queries, so do **not** treat the P×N cells as independent — that inflates n by 5x and will give you fake significance.

- Two-level bootstrap: resample queries with replacement, carry all P permutations of each sampled query along. 10,000 replicates.
- Paired comparisons throughout (same queries across arms).
- Holm correction across the method-pair family.
- Report CIs, not p-values, as the primary presentation.

Pre-register the analysis in the repo before the main run. It costs an hour and it is the single most credible thing in the whole project.

---

## 5. Compute arithmetic

Main run, local generator:

```
300 queries x 8 arms x 3 budgets x 5 permutations = 36,000 generations
```

Answers are short (10–30 tokens); prefill is ~1,200–1,800 tokens. On a 3060 mobile with a 4-bit 3B and batched inference, budget ~1s per generation → **~10 hours**, i.e. one overnight run, two with reruns. Fine.

LOO oracle: `300 x 10 chunks x 5 permutations = 15,000` scored forward passes, no decoding needed (score the reference answer). Cheaper than the above.

Groq replication: `100 x 4 arms x 2 budgets x 3 permutations = 2,400` calls. Free tier is roughly 30 RPM with a daily request cap — sources disagree on whether that cap is 1,000 or 14,400 per day, and it varies per model, so **check console.groq.com/settings/limits yourself before sizing this**. The binding constraint is more likely tokens-per-minute (6,000 TPM on several models): 2,400 calls × ~1,500 input tokens ≈ 3.6M tokens ≈ 10 hours of wall-clock at 6k TPM. Overnight, again.

Non-negotiable engineering: **cache every generation from day one**, keyed on `sha256(model, prompt, decode_params)`, in SQLite or JSONL. You will rerun the analysis twenty times and you should never pay for the same generation twice.

---

## 6. Repo shape

```
rag-order-audit/
  configs/            # YAML per experiment; no magic numbers in code
  src/
    data.py           # dataset loaders, memorization filter
    chunks.py         # chunk representation, permutation with seeded RNG
    prune/            # one module per arm, all behind the same interface
      base.py         # Pruner.select(query, chunks, budget) -> List[int]
    generate.py       # local + Groq backends behind one interface
    cache.py          # content-hash cache
    metrics.py        # EM, F1, OAE, RFR, placebo gap, oracle gap
    stats.py          # two-level bootstrap
  analysis/           # notebooks -> figures/, no analysis logic in notebooks
  results/            # committed CSVs of aggregates, not raw generations
  ANALYSIS_PLAN.md    # written and committed BEFORE the main run
  Makefile            # make pilot / make main / make figures
```

One interface for pruners, one for generators. That's what makes adding a ninth arm a 40-line file instead of a refactor.

---

## 7. Timeline (6 weeks, part-time)

| Week | Work | Gate |
|---|---|---|
| 1 | Data loaders, local generator, cache, EM/F1. Pilot: 100 queries, `full` arm only, 5 permutations. | **Measure permutation SD. If it's near zero, the premise is dead — see §9.** |
| 2 | Implement all arms except oracle. Verify Provence checkpoint actually loads. Write ANALYSIS_PLAN.md. | All arms produce sane output on 20 queries |
| 3 | LOO oracle + memorization filter. Full pilot at n=100. | OAE and RFR computable end to end |
| 4 | Main run, n=300, HotpotQA. Bootstrap analysis. | Headline numbers exist |
| 5 | 2WikiMultihopQA replication + Groq cross-generator check | Finding holds or doesn't, on a second dataset and a second model |
| 6 | Figures, README, write-up, repo polish | Shippable |

Week 1's gate is the real decision point. Everything after it is execution.

---

## 8. Assumptions I'm making explicit

- HotpotQA gold answers are clean enough for EM to be a usable signal at n=300. (Known to be imperfect; the memorization filter and F1 partially compensate.)
- A 3B model shows position sensitivity at ~1,500-token contexts. Plausible given published results at similar scales, but **unverified until week 1**.
- Permutation effects are not purely an artifact of the prompt template's chunk delimiters. Mitigation: run one template-variant robustness check.
- The Provence checkpoint is publicly available and loadable. Verify in week 2, not week 5.
- Off-the-shelf pruner checkpoints are being used out of distribution relative to their training data. This is a real limitation and must be stated in the write-up — it means the study measures deployed-as-published behaviour, not each method's ceiling.

---

## 9. Risks and kill criteria

**Risk: permutation variance is negligible.** Kill criterion: if median within-query SD of token-F1 across 5 permutations is < 0.02 at k=10 in the week-1 pilot, the premise fails at this scale. Escalation ladder, in order: more chunks (20 instead of 10) → longer chunks → a smaller/weaker generator → switch to MuSiQue, which shows stronger position effects in published work. If none of that produces variance, pivot to the fallback.

**Fallback project.** The same infrastructure supports a straight consolidation study: re-evaluate the published pruners under one protocol, one generator set, one budget definition, with the positional placebo and the LOO oracle as bookends. That is still a genuinely missing artifact in this literature and it reuses ~80% of the code. You would lose the novel framing and keep the rigor.

**Risk: scooped mid-project.** For a portfolio piece this matters far less than for a thesis. A well-executed, reproducible study with pre-registered analysis is a strong CV artifact even if a paper lands on the same idea in October. Do not let this one drive decisions.

**Risk: scope creep into building a better pruner.** Don't. The moment you propose a method you are competing with the five groups in §2 on their turf.

---

## 10. What this looks like on a CV

One line:

> Ran a permutation-controlled re-evaluation of RAG context-pruning methods, showing that reported gains are [X]x the variance induced by context ordering alone and that method rankings flip in [Y]% of single-order comparisons. Introduced a position-matched placebo control that isolates content selection from positional promotion.

What makes it hold up in an interview: pre-registered analysis, a two-level bootstrap that correctly handles nesting, a memorization control most published work skips, and an explicit kill criterion you defined before running anything. Those four things are what separate this from a repo full of LangChain calls.

---

## 11. Reading list, in order

1. Liu et al. 2024 — Lost in the Middle (the origin)
2. Stable-RAG, arXiv 2601.02993 (permutation sensitivity of answers)
3. Provence, arXiv 2501.16214 (the pruner you'll benchmark against)
4. Information Gain Pruning, arXiv 2601.17532 (relevance ≠ end-task utility)
5. ContextCite, NeurIPS 2024 (attribution framing)
6. AttriBoT, arXiv 2411.15102 (making LOO affordable)
7. Weakest Link Law, arXiv 2601.12499 (position effects in multi-hop)
8. CUE-R, arXiv 2604.05467 (non-additivity, small n)
9. Shapley Context Pruning, arXiv 2607.16209 (the current method frontier)
10. Ferrari Dacrema et al. 2019 — *Are we really making much progress?* (the methodological template for all of this)
