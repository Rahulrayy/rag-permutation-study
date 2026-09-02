# NOTE: make is not installed on this machine; see README for raw equivalents.
PY := .venv/Scripts/python.exe

.PHONY: smoke pilot gate main oracle main-no-oracle replication replication-xfamily audit analyze generator-comparison figures test test-all dummy clean

# Verify CUDA, checkpoint load, greedy determinism and answer log-probs.
# Run this before burning an overnight on the main grid.
smoke:
	$(PY) -m src.smoke

# Week-1 pilot: 100 queries, `full` arm, 5 permutations.
pilot:
	$(PY) -m src.run --config configs/pilot.yaml

# The week-1 kill criterion. This is the real decision point.
gate:
	$(PY) -m src.gate results/pilot_w1/generations.csv

# Do not run before ANALYSIS_PLAN.md is filled in and committed.
# ~11h on this laptop: ~8h generating plus ~3h of oracle scoring, which all
# happens in the selection phase before the first generation. Fully resumable --
# both generations and the oracle's score() calls go through the same cache, so
# re-running after an interruption continues where it stopped.
#
# Close browsers first. max_vram_fraction is a fraction of TOTAL VRAM, and
# Chrome/Edge/PyCharm hold real GPU memory on a 6 GB card.
#
# expandable_segments stops allocator fragmentation from pushing a long run over
# the cap; it is what the OOM message itself recommends.
main:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	  $(PY) -m src.run --config configs/main.yaml

# The oracle, split out so it can run in its own session: it needs (n+1) x P
# scored passes per query, and every arm's selection completes before any
# generation starts, so bundling it delays the whole grid by hours.
#
# `nocontext` is in the list because it has to be. main.yaml sets
# memorization_filter, the filter needs nocontext's predictions, and run.py
# raises without them. It costs nothing -- those generations are already cached.
oracle:
	$(PY) -m src.run --config configs/main.yaml --arms nocontext loo_oracle

# The complement: everything except the oracle. Use these two when a session is
# too short for the whole grid; the cache makes the split free.
main-no-oracle:
	$(PY) -m src.run --config configs/main.yaml --arms nocontext full rerank_topk \
	  provence_rerank provence_full llmlingua2 llm_pruner random_drop placebo_pos

# Week-5 cross-generator check on Groq. Needs GROQ_API_KEY (repo-root .env);
# runs on CPU only, so it is the one long job that does not want the GPU.
# ~1,700 calls against a 1,000/day cap: expect to run it twice, on two days.
# The cache replays day 1 for free.
replication:
	$(PY) -m src.run --config configs/replication.yaml

# Cross-FAMILY probe, RQ1 only, on a non-Qwen model. 400 calls, one sitting.
# Answers the one question `replication` cannot: is the effect Qwen-specific?
replication-xfamily:
	$(PY) -m src.run --config configs/replication_xfamily.yaml

# Run this on the SECOND day of the replication, once the cache is warm and the
# routing has had a chance to change. A same-session repeat measures very little.
audit:
	$(PY) -m src.run --config configs/replication.yaml --audit 50

# RQ1-RQ4 over a finished run. CPU only, ~70 minutes at 10,000 replicates;
# checkpointed per budget, so an interruption keeps the budgets already done.
analyze:
	$(PY) -m src.analyze --config configs/main.yaml

# The matched 3B-vs-27B comparison behind WRITEUP Sec. 4.9. Needs both
# results/main_hotpotqa/ and results/replication_groq/ to exist.
generator-comparison:
	$(PY) -m src.generator_comparison

figures:
	$(PY) -m src.run --config configs/main.yaml --figures-only

# The documented invocation: three tests carry the `network` marker and download
# the dataset on first run, so the default excludes them. `test-all` includes
# them. A bare `pytest -q` here would not match what README and HANDOFF quote.
test:
	$(PY) -m pytest -q -m "not network"

test-all:
	$(PY) -m pytest -q

# Plumbing check with no GPU. Its numbers are meaningless by construction.
dummy:
	$(PY) -m src.run --config configs/pilot.yaml --backend dummy

clean:
	rm -rf __pycache__ src/__pycache__ src/prune/__pycache__ .pytest_cache
