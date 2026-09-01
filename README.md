# rag-permutation-study

**Is it the pruning, or the ordering?** A permutation-controlled re-evaluation of
context selection in RAG.

> **[Read the full technical write-up in `WRITEUP.md`.](WRITEUP.md)**
> This page is the short version. The write-up has the design, the statistics,
> the complete result tables and the limitations.

---

## The question, in one paragraph

A RAG system retrieves passages and puts them in a prompt. Because context is
expensive, many published methods shorten that context by throwing most of it
away. All of them are evaluated with the passages in one fixed order.

Separately, it is well known that language models are sensitive to the order of
what is in their prompt.

Nobody had put those two facts together, and they interact. Pruning does not only
remove passages, it **moves the survivors into new positions**. Drop passages 3,
5 and 7 from a ten-passage context and passages 8, 9 and 10 get promoted into
more visible slots. So part of what looks like better evidence selection could
just be a lucky interaction with position, measured against a reference point
that moves when the method acts.

This study separates the two. It is not a new pruning method. It is an evaluation
protocol, two controls nobody runs, and a set of numbers.

---

## What we found

**1. Reordering an identical context changes the answer about half the time.**
Same passages, same words, greedy decoding, only the order differs. Half of
questions are completely unaffected. The other half swing by about 0.39 token-F1,
which is enormous. There is no "typical" question: the distribution has two modes
and almost nothing in between.

![Within-question variation under reordering](results/main_hotpotqa/figures/rq1_permutation_sd.png)

**2. But pruning methods really are selecting on content, not position.** This
was the study's main hypothesis and it did not survive contact with the data,
which is the useful outcome. We built a placebo that drops the same number of
passages by position alone, without reading them. Real pruners beat it by
**+0.2760 token-F1** (95% CI [0.2223, 0.3297]), at every budget tested. The
control behaves too: an arm that drops passages at random is indistinguishable
from the placebo, exactly as it should be.

![Placebo gap by arm and budget](results/main_hotpotqa/figures/rq4_placebo_gap.png)

**3. Yet no method beats a plain baseline by more than the noise that ordering
alone creates.** Measure each method's gain in units of "how much does the score
move when you just reshuffle the passages", and every practical method fails to
separate from simple rerank-and-truncate. The one arm that clears that bar is a
cheating upper bound that peeks at the answer. The short version: **the method
you choose matters less than the order you happen to feed it in.**

**4. Two methods are order-dependent inside themselves, not just in their
scores.** This is the sharper version of the thesis.

- Ask an LLM which passages to keep, then show it the same passages in a
  different order, and it picks different ones. Agreement between its three
  selections is **0.213** (1.000 would mean order makes no difference, 0.047 is
  random guessing). **The selection changed in 98 of 100 questions, and in 23 of
  them the three attempts had no passage in common at all.**
- LLMLingua-2 compresses a concatenated context differently depending on the
  order it is given: **0 of 100** passages survive identically across orderings
  when it is applied the normal way, to the whole context at once. It is a
  deterministic classifier, not a prompted model, which makes this the more
  surprising of the two.

The LLM pruner also fails to name the requested number of passages in **24.3%**
of cases, and that rate reproduces to four decimal places on a 27B model from a
different size class (0.2432 against 0.2433), so it is a property of asking a
model to name k items rather than a quirk of one small model.

**Memorization is not the explanation.** Only 10% of questions can be answered
with no passages at all, and the analysis is restricted to the questions the
model gets wrong without context.

---

## How it works, briefly

- **Data.** HotpotQA distractor: ten paragraphs per question, two of them
  relevant, no retrieval needed.
- **Generator.** Qwen2.5-3B-Instruct, 4-bit, run locally, greedy decoding
  everywhere so that sampling noise cannot be confused with ordering noise.
- **The protocol.** For every (question, method, budget), generate under five
  different orderings of whatever the method kept.
- **The key control.** `placebo_pos` drops k passages by position and never reads
  them. If a method cannot beat that at equal keep-count, it is not selecting on
  content.
- **Statistics.** Permutations are nested inside questions, so resampling treats
  the question as the unit and carries all five permutations with it. Treating
  the cells as independent would inflate the sample fivefold and manufacture
  significance.
- **Pre-registered.** Hypotheses, primary endpoint, analysis population and
  multiplicity family were all fixed and committed before any main-run data
  existed, and are retrievable at commit `2f24548`.

The scale: **45,510 generations**, 11 arms, 274 questions, 3 budgets, 5
permutations.

`WRITEUP.md` covers all of this properly, including the parts that are easy to
get wrong.

---

## Running

```bash
python -m pip install -r requirements.txt      # torch must come from the CUDA index
```

| Task | Command |
|---|---|
| GPU smoke test | `python -m src.smoke` |
| Main run | `python -m src.run --config configs/main.yaml` |
| Analysis | `python -m src.analyze --config configs/main.yaml` |
| Figures | `python -m src.figures --config configs/main.yaml` |
| Selection-stability probe | `python -m src.selection_probe --config configs/main.yaml --n 100` |
| Tests | `python -m pytest -q -m "not network"` |

Every generation is cached on a hash of the model, prompt and decode parameters,
and flushed as it goes, so reruns are free and interrupting a long run loses
nothing.

Useful flags: `--backend dummy` exercises the whole pipeline with no GPU (its
numbers are meaningless by construction), `--n 20` shrinks the question set,
`--arms full` restricts the grid.

## Status

| Piece | State |
|---|---|
| pipeline, all 11 arms, caching, statistics | done |
| main run, 45,510 generations | done |
| confirmatory analysis and figures | done |
| hosted cross-generator replication at 27B | 1,610 of ~1,655 calls, rate-limited |

**231 tests** pass (`python -m pytest -q -m "not network"`, about 10s). Three
more are marked `network` and download the dataset on first run.

## Main limitations

Pruner checkpoints are used as published on a dataset they were not tuned for, so
this measures deployed behaviour rather than each method's ceiling. Results are
from one dataset and one model family, which is the main limitation, with a
cross-family probe confirming the effect exists elsewhere but not its size. `rank` here means the dataset's given
order, not a retriever ranking, since the distractor setting has no retriever. The
Provence checkpoint is non-commercial (`cc-by-nc-nd-4.0`).

The full list is in [`WRITEUP.md`](WRITEUP.md), Section 6.
