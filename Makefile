# NOTE: make is not installed on this machine; see README for raw equivalents.
PY := .venv/Scripts/python.exe

.PHONY: smoke pilot gate main oracle figures test clean

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

figures:
	$(PY) -m src.run --config configs/main.yaml --figures-only

test:
	$(PY) -m pytest -q

# Plumbing check with no GPU. Its numbers are meaningless by construction.
dummy:
	$(PY) -m src.run --config configs/pilot.yaml --backend dummy

clean:
	rm -rf __pycache__ src/__pycache__ src/prune/__pycache__ .pytest_cache
