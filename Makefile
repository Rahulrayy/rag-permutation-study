# NOTE: make is not installed on this machine; see README for raw equivalents.
PY := .venv/Scripts/python.exe

.PHONY: smoke pilot gate main oracle replication replication-xfamily audit figures test clean

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
main:
	$(PY) -m src.run --config configs/main.yaml

oracle:
	$(PY) -m src.run --config configs/main.yaml --arms loo_oracle

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

figures:
	$(PY) -m src.run --config configs/main.yaml --figures-only

test:
	$(PY) -m pytest -q

# Plumbing check with no GPU. Its numbers are meaningless by construction.
dummy:
	$(PY) -m src.run --config configs/pilot.yaml --backend dummy

clean:
	rm -rf __pycache__ src/__pycache__ src/prune/__pycache__ .pytest_cache
