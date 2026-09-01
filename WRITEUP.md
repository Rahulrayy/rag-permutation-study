# Is it the pruning, or the ordering?

## A permutation-controlled re-evaluation of context selection in retrieval-augmented generation

**Draft, 2026-09-01.** Written against the completed HotpotQA main run. The
hosted cross-generator replication is not finished, and Section 7 says what
remains.

All numbers here are from the corrected run. An earlier version reported the
`llmlingua2` arm wrongly because of a caching defect; Section 4.8 documents it,
its effect, and why it survived a full analysis and a draft.

---

## Abstract

Context pruning methods for retrieval-augmented generation claim to discard most
of a retrieved context with little loss of answer quality. Every such method is
evaluated with the passages in one fixed order. Separately, it is well
established that language model outputs are sensitive to the order of the
material in their context. Pruning does not only remove passages, it moves the
survivors into new positions, so a reported pruning gain and a position effect
are confounded in the standard evaluation.

This study separates them. Holding passage content fixed, we generate answers
under five orderings of every retained context, and we introduce a
position-matched placebo that drops the same number of passages by position
alone. Across 45,510 generations covering eleven arms, 274 questions, three
budgets and five permutations, we find three things. First, reordering an
identical context changes the score for half of all questions, and for those
questions the swing is large, with a median within-question standard deviation of
0.39 token-F1. Only 36% of questions return a byte-identical answer under all
five orderings. Second, the pre-registered primary endpoint is
clearly positive: published pruners beat the position-matched placebo at equal
keep-count by 0.2760 token-F1 (95% CI [0.2223, 0.3297], Holm-adjusted p = 0.0018),
so their gains are genuine content selection rather than positional promotion.
Third, and in tension with the second, no practical pruner separates from a plain
cross-encoder top-k baseline once the gain is measured in units of the baseline's
own permutation noise. We also show that order dependence reaches inside two of
the methods themselves: an LLM asked which passages to keep returns selections
with a mean Jaccard of 0.213 across three presentation orders, and LLMLingua-2,
a deterministic classifier, compresses a concatenated context differently
depending on the order it is given, preserving 0 of 100 passages identically
across orderings when applied the way it normally is.

---

## 1. Introduction

A retrieval-augmented generation system retrieves a set of passages and places
them in a prompt. Because context windows and inference budgets are finite, a
large literature proposes ways to shorten that context: rerank and keep the top
few, prune sentences within passages, or compress the token sequence directly.
These methods report that most of a retrieved context can be discarded at little
cost in answer quality.

Every one of these evaluations shares a design decision that is rarely stated.
The passages are presented in one fixed order, usually retriever rank in work
that retrieves at all, and the
comparison is between a pruned context in that order and the full context in that
same order. A separate body of work has established that language models are
sensitive to the order of material in their prompts, and that this sensitivity is
large.

These two facts interact in a way that has not been measured. Pruning changes
positions. Dropping passages three, five and seven from a ten-passage context
promotes passages eight, nine and ten into earlier, more visible slots. If answer
quality depends on position, then part of what a pruning method appears to gain
may be an interaction with position bias rather than better evidence selection,
and it is measured against a reference point that moves when the method acts.

This study asks how much of a reported pruning gain survives when ordering is
controlled. It is not a new pruning method. It is an evaluation protocol, two
controls that the literature does not run, and a set of numbers.

The research questions, fixed before any data was collected, are:

- **RQ1.** Does answer quality vary across permutations of a fixed context, with
  content held constant?
- **RQ2.** How large is a pruning method's gain over a simple baseline when
  expressed in units of that baseline's own permutation noise?
- **RQ3.** How often does the ranking of two methods reverse depending on which
  single arbitrary ordering the comparison happens to be made at?
- **RQ4.** Does a published pruner beat a position-matched placebo that discards
  the same number of passages by position alone?

RQ4 is the centerpiece and provides the pre-registered primary endpoint.

---

## 2. Related work

**Order sensitivity in language models.** The order of in-context examples moves
few-shot accuracy between near state-of-the-art and near chance [1]. The order and
labelling of multiple-choice options moves answer selection, which [2] traces to a
token-level prior over option identifiers rather than to content. Within a long
context, the position of the relevant evidence has a characteristic shape:
accuracy is highest when it sits at the beginning or the end and degrades in the
middle [3]. Recent work locates part of the mechanism in causal attention itself,
since a causal mask prevents later tokens from attending to material that follows
them [4]. This literature establishes the phenomenon that the present study uses
as a control rather than as a finding.

**Context pruning and compression.** Reranking followed by top-k selection is the
standard baseline. Provence prunes at the sentence level while also producing a
reranking score, and is the method this study runs as two separate arms [5].
LLMLingua-2 compresses at the token level with a trained classifier, formulating
compression as token classification [6]. Further work addresses redundancy-aware
selection under a token budget [7].

**The gap.** A literature check in August 2026 found no work that evaluates
pruning methods under multiple permutations with content held fixed, none that
runs a position-matched placebo, and none that reports how often method rankings
flip across orderings. The nearest neighbours address adjacent questions and each
uses a single fixed ordering throughout. Conformal, coverage-controlled filtering
selects passages under a statistical guarantee on evidence recall [8]. An
answer-in-context diagnostic asks whether the gold answer survives budgeted
packing at all, which is adjacent to our observation about aggressive sentence
pruning [9]. Benchmark work evaluates robustness to retrieval noise [10]. Closest
in spirit is a reproduction of position and context-size effects in realistic RAG
pipelines, which identifies topic sampling as a major source of variance and warns
that small topic sets can mask or exaggerate ordering effects [11]; that is a
result about the stability of *conclusions* across samples, where the present
study asks about the stability of a method's measured gain across orderings of one
sample.

One framing point should be made explicitly, because it is the main risk to the
contribution. That an LLM's behaviour is order-sensitive is not a new phenomenon,
and this study does not claim it as one. The claim made here is narrower: this
known effect propagates into the selection step of pruning methods, and no
published evaluation of such a method controls for it.

---

## 3. Design

### 3.1 Data

HotpotQA in the distractor setting. Each question ships ten paragraphs, two of
which are gold and eight of which are distractors, so no retrieval step is
required and the passage set is fixed by the dataset. Of 7,405 validation rows,
60 (0.81%) do not contain exactly ten paragraphs and are excluded, leaving a
working population of 7,345. This is a comparability exclusion rather than a
coverage one: a fixed context size is what makes a position, a positional bucket
and a keep-k budget mean the same thing across questions. All 60 excluded rows
still contain both gold paragraphs.

Sampling is seeded and stratified by hop type, and it is nested, meaning that the
sample at n = 100 is a strict prefix of the sample at n = 300 under the same
seed. This makes a pilot directly comparable to the main run rather than merely
similar to it.

The main run samples 300 questions. A memorization filter then restricts the
analysis population to the questions the generator answers incorrectly with no
context at all, which leaves **274**. This matters for reading any table: the
no-context arm is reported over all 300 sampled questions, while every other arm
is reported over the 274, and on those 274 the no-context floor is by
construction near zero (mean token-F1 0.0089). The filter is defined as removing
the questions the model can already answer, so the two populations are not
interchangeable.

### 3.2 Generator and decoding

Qwen2.5-3B-Instruct, quantized to 4-bit, run locally. A local model is required
because the leave-one-out oracle arm needs the log-probability of the answer
sequence, which hosted chat APIs do not generally expose.

Decoding is greedy everywhere, with no exceptions. Sampling noise and permutation
noise would otherwise be confounded and no number in the study would be
interpretable. This is enforced in the configuration loader and again in the
generator before any model is loaded, and it was verified empirically as
reproducing identical output across repeated runs and across process restarts.

The prompt template was frozen before the main run, chosen on answer accuracy and
answer-format match rather than on permutation variance. Choosing a prompt because
it maximised the study's own headline quantity would be a garden-of-forking-paths
error. It is worth recording that the choice went against the initial expectation:
a template that produces full-sentence answers scores a uniformly mediocre result
regardless of ordering, which compresses token-F1 toward the middle and damps the
measured variance. Verbosity was masking position sensitivity rather than creating
it.

### 3.3 The permutation protocol

For every (question, arm, budget) cell, answers are generated under P = 5
orderings of the passages the arm retained: as-given, reversed, and three seeded
random draws. The random orderings are seeded per question, so the three draws
differ from question to question. Seeding them globally would reuse one trio of
orderings across the whole dataset, making the random draws a sample of size
three from n! whose sampling error never averages out.

**This paper calls the reference ordering the "as-given" order** and does not
call it rank, because HotpotQA distractor involves no retriever and there is no
ranking to speak of. It is the dataset's own paragraph order, and it remains the
right reference, since it is the ordering any evaluation on this dataset
implicitly uses. Readers of the code should note that the corresponding strategy
identifier there is `rank`, a name retained because it is recorded in the
completed run's data and in the registered configuration; "as-given" is the term
used throughout this document.

### 3.4 Separation of selection, rewriting and ordering

The design rests on keeping three operations apart. Selection returns indices,
rewriting returns text, and ordering returns a permutation. A pruner that
returned its passages in its own preferred order would confound selection with
ordering, and the confound could not be detected afterwards. A pruner that folded
rewriting into selection would make a matched-keep-count comparison impossible.
Every arm is implemented behind an interface that enforces this separation, and
budget compliance is checked rather than assumed.

Two arms genuinely have no keep-count. The full-context arm prunes nothing by
construction, and LLMLingua-2 is rate-based, spending the budget as a compression
rate of k/n because it produces no passage ranking from which a top-k could be
taken. Both declare this explicitly, and they are compared on input-token count
rather than on k. Any table or figure describing a comparison as matched at equal
keep-count must mark them, and ours do.

### 3.5 Arms

| Arm | What it is |
|---|---|
| `full` | all ten passages, upper reference |
| `nocontext` | question only, the memorization control |
| `rerank_topk` | cross-encoder rerank, keep top-k, the baseline |
| `provence_rerank` | Provence's reranker only, original passage text |
| `provence_full` | Provence as published, with sentence-pruned text |
| `llmlingua2` | token-level compression, budget spent as a rate |
| `llm_pruner` | ask the generator which passages to keep |
| `random_drop` | noise floor |
| `placebo_pos` | drop k by position, not content, in three variants |
| `loo_oracle` | keep the k passages with the largest leave-one-out log-probability drop |

The placebo is the novel control. It discards exactly as many passages as the
method under test, but chooses them by position alone and never reads their
content. If a method cannot beat it at equal keep-count, the method is not doing
content selection.

### 3.6 Derived quantities

**Order-Adjusted Effect (OAE)** is a method's gain over the baseline divided by
the baseline's within-question permutation standard deviation. It answers how
many orderings-worth of noise the choice of method actually buys.

Two properties of that denominator are worth stating, because both are easy to
misread. It is the **baseline's** spread, not the method's, so it is the same
constant for every arm at a given budget and a method is never penalised for its
own variance. Consequently OAE is a linear rescaling of the raw gain within a
budget, and the raw gains are reported alongside it in Section 4.3 so the reader
can judge the effect without the normalisation. Section 4.3 also reports the
result under two alternative denominators, because the mean of per-question SDs
is outlier-sensitive.

**Rank Flip Rate (RFR)** is the fraction of method-pair comparisons whose sign
reverses under some single ordering.

**Placebo Gap** is quality against the position-matched placebo at equal
keep-count. This provides the primary endpoint.

**Oracle Gap** is the headroom against the leave-one-out ceiling.

### 3.7 Statistics

Permutations are nested within questions, so the P by N cells are not
independent. Treating them as independent would inflate the effective sample size
fivefold and manufacture significance. All resampling is therefore two-level: the
unit is the question, and all five of its permutations travel with it. The test
suite contains a regression guard that constructs data with strong
between-question and weak within-question variation and asserts that the correct
interval is more than 1.5 times wider than the flattened one.

All comparisons are paired, restricted to the questions shared by the arms being
compared, with the point estimate drawn from the same restricted population that
is resampled. Bootstrap intervals use 10,000 replicates at 95%. Multiplicity is
controlled by Holm correction across one pre-registered family of nine
confirmatory comparisons: the OAE against the baseline for four methods, and the
Placebo Gap for those four plus the baseline. Arms outside that family are
reported with intervals but never corrected, since widening a Holm family changes
the adjusted values of everything inside it. Confidence intervals rather than
p-values are the primary presentation.

The analysis plan was registered before any main-run data existed, fixing the
hypotheses, the primary endpoint, the analysis population, the thresholds and the
multiplicity family. It remains retrievable from version control at the
registration commit.

**The registered analysis was executed twice**, and a reader is entitled to know
which numbers these are. The first execution ran on data in which one arm was
affected by a caching defect (Section 4.8). The defect was found afterwards, the
arm was regenerated, and the identical pre-specified procedure was re-run on the
corrected data; the numbers reported here are from that second execution. No part
of the specification changed in response to seeing results: the endpoint, the
nine-comparison family, the population definition, the thresholds and the code
path are the registered ones. We would resist calling the corrected analysis
post-hoc for that reason, since what changed was an input error and not an
analytic choice, but the sequence is stated here rather than left to be inferred
from the repository.

---

## 4. Results

The main run comprises 45,510 generations: eleven arms by 274 questions by three
budgets by five permutations, plus the no-context baseline.

### 4.1 RQ1: order sensitivity is real, and it is bimodal

Mean within-question standard deviation of token-F1 across permutations, at
k = 3, is 0.1795 (95% CI [0.1554, 0.2038]) for the full-context arm and between
0.057 and 0.167 for every other arm. Every interval excludes zero. The full
context is clearly separated from every keep-k pruning arm, which is the expected
direction: pruning leaves fewer passages to reorder, so it reduces the amount of
order sensitivity available.

`llmlingua2` is the instructive exception at **0.1666 [0.1426, 0.1912]**, second
only to the full context. It is not a keep-k arm: it retains all ten passages and
compresses each, so it presents ten permutable slots where a keep-3 arm presents
three. More positional room, more positional variance. That is the mechanism this
section describes, showing up in the one arm whose design isolates it.

Within the pruning arms, the standard deviation rises monotonically with the
budget in every case. The full-context arm is flat across budgets at 0.1795,
which is not an exception but a consequence of its definition: it ignores the
budget and always presents all ten passages.

The distribution matters more than its centre. At k = 3 on the analysis
population, exactly **137 of 274 questions (50.0%) have a within-question
standard deviation of exactly zero**. The other 137 move, and they move a great
deal: the median standard deviation among them is **0.3912** and the maximum is
0.5477.

**A zero standard deviation means the score did not move, not that the answer did
not move**, and the difference is substantial. Of those 137 questions, the
generator produced **more than one distinct answer string in 39 of them
(28.5%)**; only **98 questions of 274 (35.8%)** return a byte-identical answer
under all five orderings. The 39 are almost entirely cases where every ordering
is wrong in a different way and each scores zero, for instance one question
answered variously as "Elbridge Gerry", "Elbridge, New York" and "Hobart", all
scoring 0.000.

So the honest three-way split is: **36% of questions are genuinely stable, 14%
change their answer without changing their score, and 50% change their score.**
Only the first group is unaffected by ordering in any meaningful sense, and an
earlier draft of this section described all 50% that way.

This produces a statistical trap that is worth stating plainly, because the
study's own pre-registered kill criterion was a median. The median of the whole
distribution is **0.0186**, which is the midpoint of the largest zero and the
smallest non-zero value. It is not a measure of central tendency of anything; it
is an artifact of exactly half the questions sitting at zero. One additional
static question would move it to 0.0000 while leaving the moving half completely
unchanged. The pilot showed the same pathology at a different value, and the
analysis plan was amended before the main run to register a distributional
presentation of RQ1 for precisely this reason.

The finding to carry forward is therefore not a median. It is that **half of
these questions change score under reordering alone, by roughly 0.39 token-F1 on
identical content under greedy decoding, and only about a third are stable in the
stronger sense of returning the same answer every time**.

### 4.2 RQ4, the primary endpoint: pruners are doing content selection

The pre-registered primary endpoint is the Placebo Gap of `provence_rerank`
against `placebo_pos:middle_first` at k = 3 in token-F1 on the filtered
population.

```
+0.2760   95% CI [0.2223, 0.3297]   uncorrected p = 0.0002   Holm p = 0.0018
```

Both the uncorrected and the corrected value are reported, always, so that the
choice between them cannot be made after seeing which is more favourable.

The result holds at every budget, and it holds for every method in the
confirmatory family. All fifteen confirmatory Placebo Gap comparisons, five arms
by three budgets, survive Holm correction.

| Arm | k = 2 | k = 3 | k = 5 |
|---|---|---|---|
| `loo_oracle` | +0.3815 | +0.3536 | +0.2628 |
| `full` | +0.3308 | +0.2869 | +0.1882 |
| `provence_rerank` | **+0.2895** | **+0.2760** | **+0.1780** |
| `rerank_topk` | +0.2648 | +0.2418 | +0.1612 |
| `provence_full` | +0.2799 | +0.2075 | +0.1399 |
| `llm_pruner` | +0.2052 | +0.1964 | +0.1496 |
| `random_drop` | -0.0111 | -0.0190 | -0.0036 |
| `llmlingua2` | +0.1192 | +0.1356 | +0.0958 |

Five of these arms are inside the pre-registered confirmatory family
(`provence_rerank`, `provence_full`, `llmlingua2`, `llm_pruner` and
`rerank_topk`) and are Holm-corrected jointly. The remaining three (`full`,
`loo_oracle` and `random_drop`) are outside it by registration and are reported
with intervals only. They are shown here because a reference, a ceiling and a
noise floor make the corrected rows readable, not because they were tested.

Two rows carry most of the evidential weight.

**The control behaves.** `random_drop` is the one arm whose placebo gap fails to
exclude zero, and it fails to do so at all three budgets (p = 0.6264, 0.4794,
0.9044). This is three independent opportunities for a false positive, none of
them taken. An arm that discards passages at random should be indistinguishable
from an arm that discards them by position, and it is. Without this row the
positive results would be much weaker evidence, because a placebo gap that came
out positive for everything would suggest the comparison itself was biased.

**Every method passes, including the compressor.** `llmlingua2` clears the
placebo at all three budgets (+0.1192, +0.1356, +0.0958), which is worth stating
explicitly because an earlier version of this analysis reported the opposite. A
caching defect was feeding that arm one question's compressed passages for every
question in the run; it is described in Section 4.8, and the numbers here are
from the corrected run.

The direct answer to the motivating suspicion is therefore negative. Published
pruners are not merely exploiting positional promotion. At matched keep-count
they select better passages than position alone would, by a margin that is both
large and stable across budgets.

### 4.3 RQ2: the methods do not separate from a plain baseline

Expressed in units of the baseline's own permutation noise, the picture changes.

| Arm | k = 2 | k = 3 | k = 5 |
|---|---|---|---|
| `provence_rerank` | +0.3715 (p = 0.1780) | +0.2995 (p = 0.0212) | +0.1176 (p = 0.1822) |
| `provence_full` | +0.2270 (p = 0.5330) | -0.2995 (p = 0.1038) | -0.1492 (p = 0.2818) |
| `llm_pruner` | -0.8960 (p = 0.0194) | -0.3975 (p = 0.0570) | -0.0813 (p = 0.5196) |
| `loo_oracle` | +1.7539 | +0.9794 | +0.7126 |

The first three rows are confirmatory and Holm-corrected within the family of
nine; uncorrected p-values are shown in the table and the corrected values are
quoted in the text below. `loo_oracle` is outside the family and is reported as a
ceiling with an interval rather than as a test. `llmlingua2` is omitted from this
table for readability, since its OAE runs from -2.19 to -0.46 and would compress
every other row; it is significantly negative at all three budgets (p = 0.0002,
0.0002, 0.0014), so it is the one arm that is clearly *behind* the baseline in
orderings-worth of noise, by about one ordering's worth at the primary budget.

**The same comparisons unstandardised**, so the normalisation can be judged
rather than trusted. At k = 3, mean token-F1: `rerank_topk` 0.4279,
`provence_rerank` 0.4621, `provence_full` 0.3937, `llm_pruner` 0.3825,
`llmlingua2` 0.3218, `loo_oracle` 0.5398, `full` 0.4731. The best practical
method is therefore **+0.034 token-F1** over the baseline, against the **+0.276**
by which it beats the positional placebo. That contrast is the whole of RQ2: the
cross-encoder baseline is strong, and the remaining headroom above it is small in
absolute terms before any normalisation is applied.

**Is the null an artifact of the denominator?** It is not, and the check is worth
reporting because the denominator is genuinely outlier-sensitive: the baseline's
per-question SD is **exactly zero for 67.9%** of questions, its median is 0.0000,
and the top decile of questions contributes **45%** of the total. Recomputing
`provence_rerank`'s OAE at k = 3 under three denominators:

| denominator | value | OAE |
|---|---|---|
| mean of per-question SDs (as reported) | 0.1142 | **+0.2995** |
| 10% trimmed mean | 0.0975 | +0.3509 |
| median among questions that move at all | 0.4346 | +0.0787 |

The estimate moves by a factor of four, and **every version stays far below the
registered H2 threshold of 0.5 and further below the 1.0 interpretive line.** The
null is a property of the small raw gain, not of how the spread is summarised.

No practical pruner survives Holm correction against the baseline at any budget.
The best case is `provence_rerank` at the primary budget, at an uncorrected
p = 0.0212 and a Holm-adjusted p = 0.0636. Reading only that row would suggest a
near-miss that a larger sample might rescue. The budget sweep shows it is not:
the same comparison gives Holm-adjusted values of 0.3560 at k = 2 and 0.5466 at
k = 5. The effect is not marginal everywhere, it is absent at two budgets out of
three.

The registered hypothesis H2 predicted an OAE below 0.5 for published pruners,
and that is what is observed for every practical method. The only arm that clears
one full ordering's worth of noise is the leave-one-out oracle, at +0.9794 (95%
CI [0.5451, 1.5066]) at the primary budget, which puts the entire headroom from a
deployed method to a cheating upper bound at roughly one permutation's worth of
noise.

The summary is that **the choice of pruning method matters less than the ordering
it happens to be evaluated under**.

### 4.4 RQ3: rankings are far more stable than answers

| Budget | Rank Flip Rate | 95% CI |
|---|---|---|
| k = 2 | 0.0267 | [0.0000, 0.1067] |
| k = 3 | 0.0400 | [0.0133, 0.1067] |
| k = 5 | 0.0667 | [0.0533, 0.2267] |

The registered hypothesis H3 predicted a rank flip rate above 0.10. **H3 is not
supported.** The point estimate is below the threshold at all three budgets,
although the intervals do not exclude it. The point estimate rises with the
budget, which is the regime real systems operate in, but the intervals overlap
heavily and this should be read as establishing the level rather than the trend.

### 4.5 Order dependence inside the methods

The results above concern the answer. Two arms turn out to be order-dependent in
their internals, which is a sharper version of the same thesis: not that the
score moves, but that the method does something different depending on the order
it was shown.

**An LLM pruner's selection moves with presentation order, and sits closer to a
random redraw than to a stable choice.** Shown
the same ten passages in three different orders (as-given, reversed, random) and
asked which three to keep, the model returns selections whose mean Jaccard is
**0.213** over 100 questions. The reference points are measured rather than
assumed: a cross-encoder run through the same three permutations returns
**1.000**, since it scores each passage independently, and three random
3-subsets of ten give **0.047**. The observed value therefore sits only about
17% of the way from redrawing at random to being determined by the content. **The
selection changed in 98 of 100 questions, and in 23 of them the three
presentations produced sets with no passage in common at all.**

The consequence for the literature is that a published LLM-pruner result is one
draw from a distribution over selections that its paper does not report. As noted
in Section 2, the underlying order sensitivity is not itself novel; what is new
is that it reaches the selection step and that no evaluation controls for it.

**LLMLingua-2's compression is also order-dependent**, when applied in the usual
way to a concatenated context. Asking whether a given passage's surviving text is
identical across orderings, joint compression preserved **0 of 100** passages.
Not one passage survived joint compression the same way twice.

An earlier draft called this the more surprising of the two results on the
grounds that LLMLingua-2 is deterministic rather than prompted. That reasoning
does not hold and is withdrawn: determinism is not order-invariance. The
compressor is a function of a concatenated string, and concatenating the same
passages in a different order produces a different string, so a token classifier
reading local context *should* be order-dependent. Expecting otherwise is the
error.

The three arms in fact fall into three distinct categories, which the earlier
framing conflated:

| category | arms | why |
|---|---|---|
| **order-invariant by construction** | `rerank_topk`, `provence_*` | each passage is scored on its own, so the input to the model does not change when the order does |
| **order-dependent by construction** | `llmlingua2` as normally applied | its input *is* the concatenated context, and reordering changes that input |
| **order-dependent by fragility** | `llm_pruner` | its input is unchanged as a *set*, and its answer changes anyway |

Only the third is a defect in any interesting sense. The value of the 0 of 100
measurement is not that the direction is surprising, it is the **magnitude**: a
practitioner treats a compressor's output as a property of the passage set, and
not one passage in a hundred survives that assumption.

Per-passage compression is order-invariant by construction rather than by
measurement, and this should be stated as such: a passage compressed on its own
cannot depend on the order of passages it never saw. An earlier draft reported
"100 of 100" as an empirical counterpart to the 0 of 100, which it was not; the
comparison ran through a cache that returned the same entry, so it could not have
produced any other answer. It is also the reason this arm compresses each
passage independently: compressing jointly would give each of the five
permutations different content, breaking the one invariant the design rests on,
in the very arm intended to test ordering.

**Where LLMLingua-2 lands.** At k = 3 the arm retains both gold passages in
every case (mean gold retained 2.000, against 0.650 for the placebo) and scores
**0.3218** mean token-F1 against the placebo's 0.1862. So token-level compression
keeps enough of the evidence to beat positional dropping comfortably, while still
trailing every selection method: it sits below `llm_pruner` at 0.3825 and well
below `provence_rerank` at 0.4621.

The honest reading is that compression and selection are not equivalent uses of
the same token budget, but the gap is a matter of degree rather than the arm
being unusable. Against the baseline in orderings-worth of noise it is the one
arm clearly behind, at -0.93 [-1.39, -0.53] at k = 3, so about one ordering's
worth of noise *worse* than simply reranking and truncating.

One caution against over-reading the budget trend: because the budget is spent as
a compression rate, a higher k means *less* compression, so the arm's absolute
score improves with the budget while its margin over the placebo narrows
(+0.1192, +0.1356, +0.0958 at k = 2, 3, 5). The narrowing is a floor effect in
the comparator, which is still acquiring gold passages as k grows, rather than a
degradation of the method.

### 4.6 Arm-level behaviour worth reporting

**The LLM pruner frequently fails to comply with its budget.** Across 822 cells
it named fewer passages than requested in 200 of them (24.3%) and produced
unparseable output in a further 28.

**This rate reproduces across generators.** The partially complete hosted
replication uses a 27B model from a different size class, and over its 148
completed selection cells the under-selection rate is **0.2432**, against
**0.2433** locally. Agreement to four decimal places across a 3B local model and
a 27B hosted one, on different sample sizes, suggests the behaviour is a property
of asking a language model to name k items rather than a quirk of one small
model. It is the one cross-generator result already in hand; the permutation
numbers from that run are not yet complete. Deficits are filled deterministically from
the as-given order and counted rather than silently absorbed. This is a separate
defect from the order sensitivity above, and it is a finding about deployed
practice rather than about the method's ceiling.

**The leave-one-out oracle is weaker than expected, and its ranking is partly
arbitrary.** At full scale its gold recall is 0.694, and **45.8% of single-passage
removals actually raise the answer's log-probability**. The distribution of
log-probability drops is severely skewed, with a median of 0.008 against a mean of
1.285 and a maximum of 123.5, so a typical passage moves the answer essentially
not at all while a handful dominate every aggregate. Consistent with this, 23.6%
of cells have ties at the selection boundary, and its selection order-Jaccard is
0.171. Where passages are indistinguishable the selection is decided by arbitrary
tie-breaking rather than by position, and those are different claims that should
not be merged.

### 4.7 Robustness: the same analysis on the unfiltered population

The memorization filter removes the 26 questions of 300 that the generator can
answer with no context at all. Those are plausibly also the questions least
sensitive to ordering, and the concern this raises is not that the overall level
shifts, which it must, but that the filter might flatter some arms more than
others and so bias the arm-versus-arm comparisons the study is built on. The
project plan asked for both populations to be reported; this is that check. The
filtered population remains the registered primary one and nothing here enters
the confirmatory family.

The whole grid was regenerated with the filter off, 49,800 rows over 300
questions, and the identical analysis run against it.

**The primary endpoint holds at every budget:**

| | filtered, 274 | unfiltered, 300 |
|---|---|---|
| k = 2 | +0.2895 [0.2363, 0.3420] | **+0.2793 [0.2279, 0.3308]** |
| k = 3 | +0.2760 [0.2223, 0.3297] | **+0.2562 [0.2049, 0.3075]** |
| k = 5 | +0.1780 [0.1314, 0.2259] | **+0.1623 [0.1163, 0.2078]** |

Holm-adjusted p = 0.0018 in all six cells. All fifteen Placebo Gap comparisons
survive correction on both populations, `random_drop` fails to exclude zero in
all six cells, and no arm ordering changes on any research question. RQ1's
distribution is likewise stable: 51.7% of questions have zero within-question SD
against 50.0% filtered, 38.3% return an identical answer against 35.8%, and the
median among movers is 0.4117 against 0.3912.

**Two things did change, and neither is rounded away here.**

`OAE:llmlingua2` at k = 5 moves from Holm 0.0056 to 0.0744 and so drops out of
significance, taking that budget from six of nine survivors to five. Its point
estimate moves from -0.4586 to -0.3288, meaning the arm looks slightly less bad
against the baseline on the fuller population. It was the smallest-magnitude
comparison in that family and therefore the most likely to cross. It is the only
significance change in eighteen comparisons.

The rank flip rate at k = 5 rises from 0.0667 to **0.1067**, crossing the
registered H3 threshold of 0.10. H3 was specified at the primary budget, where
the rate is 0.0400 on both populations, so the registered conclusion that H3 is
not supported is unaffected. But it is worth stating that on the fuller
population at the largest budget the threshold is nominally met, which is
consistent with the reading that single-order evaluation degrades as more
context is kept.

**What this does and does not establish.** The two populations share 274 of 300
questions, an overlap of 91%, so this is a test for bias introduced by the
exclusion and not an independent replication. A filter that favoured particular
arms would show up as differential movement between them, and it does not: every
arm moves in the same direction by a similar small amount. Whether the findings
generalise to unseen questions or another dataset is a different question, and
one this study does not answer.

### 4.8 A corrected defect, and what it cost

One arm's results in an earlier version of this analysis were wrong, and the
correction is reported here rather than left in the commit history, because the
way the error survived is itself informative.

LLMLingua-2 compresses each passage separately, and the implementation memoised
that work. The cache was keyed on the passage's **index and rate** rather than on
its **text**. Neither component identifies a passage: the driver is arm-major, so
one instance serves every question in a run, and the rate is k/n, constant across
questions. The whole arm therefore had about thirty distinct keys, and every
question after the first received the first question's compressed passages.

The effect was large. At k = 3 the arm's mean token-F1 was 0.0983 and is 0.3218
corrected; its Placebo Gap was -0.0879 and is +0.1356; its Order-Adjusted Effect
was -2.89 and is -0.93. Every other arm reproduced to four decimal places, and
the primary endpoint is unchanged at +0.2760 [0.2223, 0.3297], Holm p = 0.0018.
The pre-fix aggregates are kept beside the results for audit.

**Why it hid.** The instrumentation reported `n_gold_kept` = 2.000 for this arm
throughout, because that quantity is computed from passage metadata rather than
passage text. The arm therefore looked like it was retaining all the evidence and
simply answering badly, which is a describable phenomenon rather than an obvious
fault. The project twice proposed a substantive explanation for it: first that
the arm scored below the no-context floor, which was correctly retracted as a
population-comparison error, and then that "evidence in shredded form is worth
less than the evidence intact", which reached a full draft of this document.

**The lesson worth taking.** An arm that retains all the gold evidence and scores
below random dropping is not a finding, it is a contradiction, and a contradiction
should be treated as a defect until it has been ruled out as one. The measurement
that would have settled it, comparing the compressed text of the same passage
across two different questions, takes a minute and was never run, because the
result already had an explanation. No test caught it either: every test exercised
a single question's passages, and the defect only appears from the second question
onward.

---

## 5. Discussion

**The two central results are in tension, and the tension is the point.** RQ1
finds that reordering identical content changes the score for half of all
questions, often dramatically, and changes the answer string for a further 14%
without moving the score. RQ3 finds that only about 4% of method-pair
rankings reverse across orderings. Both are true, and they are not contradictory.
Per-question volatility is large but largely uncorrelated across questions, so it
averages out when arms are compared at the level of a dataset mean. Order
destabilises answers far more than it destabilises conclusions. A reader
encountering both numbers will perceive a contradiction unless it is addressed
directly.

This has a practical reading in both directions. For a practitioner deploying a
system, the instance-level result is the relevant one: for half of user questions,
an arbitrary implementation detail about passage ordering determines whether the
answer is right, and for two thirds of them it determines what the answer says. For a researcher comparing methods on a benchmark, the aggregate
is more stable than the instance-level volatility would suggest, which is
reassuring about the existing literature's conclusions, though 4% is not zero and
it rises with the budget.

**The motivating suspicion was not confirmed, and this is a useful result.** The
study was designed around the possibility that pruning gains are substantially
positional artifacts. The primary endpoint rejects that cleanly, with a
well-behaved control. Published pruners select better passages than position
alone. Reporting this plainly matters more than the fact that it was not the
anticipated outcome, and the position-matched placebo remains the right control
to run precisely because it could have gone the other way and did not.

**A candidate explanation for RQ2, from the oracle.** Section 4.6 reports that
**45.8% of single-passage removals raise the answer's log-probability**, with a
median drop of 0.008 against a mean of 1.285. That is a statement about how
little most passages matter to this generator: for roughly half of them, removal
helps as often as it hurts, and the aggregate is carried by a handful. If the
evidence value of a typical passage is that close to noise, then two methods
selecting different passages will mostly be exchanging passages that make little
difference, and the space in which a better selector could distinguish itself is
correspondingly narrow. That is a mechanism which would produce exactly the RQ2
null, and it is testable: a selector's advantage should track the dispersion of
per-passage leave-one-out drops within a question. We have not run that test, and
flag it as the most promising follow-up in the data already collected.

**The null in RQ2 is the more uncomfortable finding.** Once a method's gain is
divided by the permutation noise of a plain cross-encoder baseline, none of the
practical methods separate from it, at any budget, after correction. The methods
are doing real content selection, as RQ4 shows, but they are not doing
detectably better content selection than reranking and truncating. Combined with
the oracle result, the entire distance from a deployed pruner to a cheating upper
bound is about one ordering's worth of noise. This suggests that reporting a
pruning result without a permutation-noise denominator can make a difference
visible that is smaller than the variation induced by an arbitrary presentation
choice.

**Order dependence inside a method is a distinct and stronger claim, but only
for one of the two arms.** The LLM pruner's selection instability is the real
result: its input as a *set* is unchanged, and its choice changes anyway, in 98
of 100 questions. LLMLingua-2 is a different case, and the two should not be
filed together. Its input genuinely changes when the order does, so
order-dependence there is expected rather than anomalous. What the measurement
adds is magnitude and consequence: not one passage in a hundred survives
reordering identically, and practitioners nonetheless treat the compressed output
as a property of the passage set. The finding is about an unexamined assumption
in deployment, not about a surprising property of the model.

---

## 6. Limitations

**Off-the-shelf checkpoints are used out of distribution.** Each pruner is run as
published on a dataset it was not tuned for. This measures deployed-as-published
behaviour, not each method's ceiling, and a method could perform better after
adaptation.

**One dataset, and one generator family.** The main run is HotpotQA distractor
with Qwen2.5-3B-Instruct, and this is the study's principal limitation. Whether
the result generalises to another multi-hop dataset is untested, and a
single-dataset finding should be read as such. A cross-family probe on a 7B model outside
the Qwen lineage confirms that a permutation effect exists there as well (64.6%
of questions move, against 50.0% here), but that probe used a different
population, three permutations rather than five and a weaker model, so it
supports the existence of the effect outside one model family and not any
comparison of magnitude. A hosted cross-generator replication at 27B is
partially complete, at 1,642 of roughly 1,655 calls, held up by a
tokens-per-day quota rather than by anything about the method.

**"Rank" is the dataset's as-given order.** HotpotQA distractor has no retriever,
so the reference ordering is the dataset's own paragraph order rather than a
retrieval ranking.

**Two arms are not keep-count matched.** The full-context arm and LLMLingua-2
have no keep-count by construction and are compared on input-token count instead.
LLMLingua-2 additionally presents n permutable units where a keep-k arm presents
k, so its raw permutation variance is not comparable in magnitude to the others.

**The analysis population excludes questions answerable without context.** That
is the registered primary population, and Section 4.7 reports the same analysis
on the unfiltered 300 as a robustness check: the primary endpoint holds at every
budget, the control still straddles zero, and no arm ordering changes. Two
comparisons of eighteen do move, and are named there. Because the populations
overlap by 91%, that check rules out bias from the exclusion rather than
demonstrating generality.

**The median is not a usable summary of RQ1 on this distribution**, as Section
4.1 shows. This limits comparison with any prior work that reports a median
instead of a distribution.

**Licensing.** The Provence checkpoint is released under a non-commercial,
no-derivatives licence, which restricts reuse of that arm outside research.

---

## 7. What remains

- The hosted cross-generator replication at 27B is **1,642 of roughly 1,655
  calls complete**, with 13 remaining. It is rate-limited by a tokens-per-day
  quota rather than blocked, and every completed call is cached, so it resumes
  rather than restarts. Sections 5 and 6 will need revision once it lands. Note
  what it can and cannot establish: it is a **scale** check, 3B to 27B, and not a
  family check, since both models share a training lineage. A separate
  cross-family probe covers that and is reported in Section 6.
- The related-work citations are described rather than formally cited, and need
  checking against the papers before they become a bibliography.
- **A second dataset is prepared but not run.** The loader and config for
  2WikiMultihopQA exist and are tested, so the protocol transfers without
  modification: the same ten paragraphs per question and the same column layout.
  Characterising it surfaced one thing a future run must handle. Its
  `bridge_comparison` questions carry **four** gold paragraphs rather than two,
  which is **2,751 of 12,576 validation rows (21.88%)**, and they are excluded
  under the rule fixed in advance for later datasets, because at k = 2 such a
  question cannot retain all its evidence even in principle and its gold recall
  and Placebo Gap would not mean what they mean elsewhere. A replication there
  would therefore cover three of the dataset's four question types, and would
  need to say which one is missing.

---

## 8. Reproduction

All quantities above are computed by committed code from committed artifacts.
The permutation analysis is regenerated with a single command over the run's
generation table, the figures are drawn from that analysis rather than
recomputing anything, and the selection-stability probe writes its own artifact.
The analysis plan was registered in version control before any main-run data
existed, and every departure from it, together with every exploratory addition,
is recorded and dated in the plan's protocol-deviation section.

---

## References

Every identifier below was checked against arXiv rather than carried over from
working notes.

[1] Y. Lu, M. Bartolo, A. Moore, S. Riedel, P. Stenetorp. *Fantastically Ordered
Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity.*
arXiv:2104.08786, 2021.

[2] C. Zheng, H. Zhou, F. Meng, J. Zhou, M. Huang. *Large Language Models Are Not
Robust Multiple Choice Selectors.* arXiv:2309.03882, 2023. ICLR 2024.

[3] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni,
P. Liang. *Lost in the Middle: How Language Models Use Long Contexts.*
arXiv:2307.03172, 2023. TACL.

[4] H. Ok, J. Lee. *Lost in the Prompt Order: Revealing the Limitations of Causal
Attention in Language Models.* arXiv:2601.14152, 2026. Findings of ACL 2026.

[5] N. Chirkova, T. Formal, V. Nikoulina, S. Clinchant. *Provence: Efficient and
Robust Context Pruning for Retrieval-Augmented Generation.* arXiv:2501.16214,
2025. ICLR 2025.

[6] Z. Pan, Q. Wu, H. Jiang, M. Xia, X. Luo, J. Zhang, Q. Lin, V. Ruhle, Y. Yang,
C.-Y. Lin, H. V. Zhao, L. Qiu, D. Zhang. *LLMLingua-2: Data Distillation for
Efficient and Faithful Task-Agnostic Prompt Compression.* arXiv:2403.12968, 2024.
Findings of ACL 2024.

[7] C. Peng, B. Wang, Z. Long, J. Sheng. *AdaGReS: Adaptive Greedy Context
Selection via Redundancy-Aware Scoring for Token-Budgeted RAG.* arXiv:2512.25052.

[8] D. Chakraborty, E. Yang, D. Khashabi, D. Lawrie, K. Duh. *Principled Context
Engineering for RAG: Statistical Guarantees via Conformal Prediction.*
arXiv:2511.17908.

[9] A. N. Bala. *Recall Is Not Enough: A Reader-Context Diagnostic for
Budget-Constrained Retrieval-Augmented Generation.* arXiv:2607.00725.

[10] J. Chen, H. Lin, X. Han, L. Sun. *Benchmarking Large Language Models in
Retrieval-Augmented Generation.* arXiv:2309.01431, 2023.

[11] J. Gabin, A. Perez, J. Parapar. *Lost in the Evidence? Reproducing Document
Position and Context Size Effects in RAG.* arXiv:2605.27105.
