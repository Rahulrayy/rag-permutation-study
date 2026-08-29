# rag-order-audit

**Is it the pruning, or the ordering?** A permutation-controlled re-evaluation of
context selection in RAG.

Full rationale, research questions and design: [`rag-permutation-project-plan.md`](rag-permutation-project-plan.md).
Pre-registered analysis: [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) — **must be filled in and
committed before the main run.**
Picking this up cold, or after a break: [`HANDOFF.md`](HANDOFF.md) — current state,
locked decisions, design invariants, and what to do next.
What is done and what is left, week by week: [`TIMELINE.md`](TIMELINE.md).

## What this is

Not a new pruning method. A protocol, two controls nobody currently runs
(a position-matched placebo and a leave-one-out oracle), and a number: how large
a typical pruning gain is relative to the variance induced by context ordering alone.

## Setup

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Running

`make` is not installed on this machine. Either install it (Git Bash: `pacman -S make`)
or use the underlying commands directly:

| Make target      | Equivalent command                                   |
|------------------|------------------------------------------------------|
| `make pilot`     | `python -m src.run --config configs/pilot.yaml`      |
| `make main`      | `python -m src.run --config configs/main.yaml`       |
| `make test`      | `python -m pytest -q`                                |

Every generation is content-hash cached (`src/cache.py`) on
`sha256(model, prompt, decode_params)`, so reruns are free and interrupting a run
loses nothing.

Useful flags: `--backend dummy` exercises the whole pipeline with no GPU (its
numbers are meaningless), `--n 20` shrinks the query set, `--arms full` restricts
the grid.

## The week-1 gate

This is the real decision point; everything after it is execution.

```bash
python -m src.run  --config configs/pilot.yaml      # 100 queries x 5 permutations
python -m src.gate results/pilot_w1/generations.csv
```

The gate prints the median within-query SD of token-F1 across permutations and
compares it to the pre-registered kill threshold of 0.02. It measures one arm at
one budget: if the CSV holds more than one keep-k for that arm it refuses to run
until you pass `--budget`, because pooling budgets would treat different keep-k
cells as permutations of one another. On FAIL it prints the
escalation ladder from plan Sec. 9 (more chunks → longer chunks → weaker
generator → MuSiQue) before you fall back to the consolidation study.

## Status

**Week 1 complete** except the gate number itself, which needs a GPU run.

| Piece | State |
|---|---|
| `data.py` — HotpotQA loader, seeded/stratified/nested subsampling, memorization filter | done |
| `cache.py` — SQLite content-hash cache | done |
| `chunks.py` — seeded permutation protocol | done |
| `metrics.py` — EM, F1, OAE, RFR, placebo gap, oracle gap | done |
| `stats.py` — two-level bootstrap, Holm | done |
| `gate.py` — week-1 kill criterion | done |
| `generate.py` — `LocalGenerator` (4-bit, greedy, answer log-probs) | written, needs a GPU run |
| arms `full` / `nocontext` / `random_drop` / `placebo_pos` (3 variants) | done |
| arms `rerank_topk` / `provence` / `llmlingua2` / `llm_pruner` / `loo_oracle` | week 2–3 |

100 tests pass (`python -m pytest -q`; one downloads HotpotQA and is marked
`network`). The suite includes a regression guard for
the 5x-n inflation that flattening the permutation nesting would cause — the
single easiest way to manufacture a fake result in this design.

Data note: the loader drops the 60/7,405 HotpotQA validation rows (0.81%) that do
not ship exactly 10 paragraphs. Recorded as a pre-registered exclusion in
`ANALYSIS_PLAN.md` Sec. 3.
