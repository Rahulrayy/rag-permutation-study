"""The determinism audit and the selection-progress reporting.

The audit exists to put a number on the one thing a hosted generator cannot
promise, so its own failure modes matter more than most."""

import pytest

import src.run
from src.cache import CachedGeneration, GenerationCache
from src.generate import CachedGenerator, DecodeParams, DummyGenerator
from src.run import audit_determinism


class _Drifting:
    """Answers differently the second time it sees a prompt."""

    model = "drifting"

    def __init__(self):
        self.seen = set()

    def generate(self, prompt, params):
        text = "second" if prompt in self.seen else "first"
        self.seen.add(prompt)
        return CachedGeneration(text=text)

    def generate_batch(self, prompts, params):
        return [self.generate(p, params) for p in prompts]

    def score(self, prompt, answer):
        return 0.0


def _warm(backend, tmp_path, prompts):
    gen = CachedGenerator(backend, GenerationCache(tmp_path / "c.sqlite"))
    gen.generate_batch(prompts, DecodeParams())
    return gen


def test_audit_reports_a_stable_backend_as_identical(tmp_path):
    prompts = [f"p{i}" for i in range(10)]
    gen = _warm(DummyGenerator(), tmp_path, prompts)

    audit = audit_determinism(gen, prompts, DecodeParams(), n=5, seed=1)
    assert audit["checked"] == 5
    assert audit["identical"] == 5
    assert audit["identical_rate"] == 1.0
    assert audit["divergences"] == []


def test_audit_catches_drift(tmp_path):
    """The whole point. A backend whose answer moves between calls must not
    come back clean."""
    prompts = [f"p{i}" for i in range(6)]
    gen = _warm(_Drifting(), tmp_path, prompts)

    audit = audit_determinism(gen, prompts, DecodeParams(), n=6, seed=1)
    assert audit["checked"] == 6
    assert audit["identical"] == 0
    assert len(audit["divergences"]) == 6
    assert audit["divergences"][0]["cached"] == "first"
    assert audit["divergences"][0]["fresh"] == "second"


def test_audit_goes_to_the_backend_not_the_cache(tmp_path):
    """Routing through the cache wrapper would replay the stored answer and
    report a perfect score -- the one result this must never be able to give."""
    backend = _Drifting()
    gen = _warm(backend, tmp_path, ["a", "b"])
    before = gen.hits

    audit_determinism(gen, ["a", "b"], DecodeParams(), n=2, seed=1)
    assert gen.hits == before  # the wrapper was bypassed entirely
    assert backend.seen == {"a", "b"}


def test_audit_skips_prompts_that_were_never_generated(tmp_path):
    """A partial run leaves uncached prompts. Counting those as identical would
    inflate the rate with comparisons that never happened."""
    prompts = [f"p{i}" for i in range(4)]
    gen = _warm(DummyGenerator(), tmp_path, prompts[:2])

    audit = audit_determinism(gen, prompts, DecodeParams(), n=4, seed=1)
    assert audit["checked"] == 2


def test_audit_deduplicates_before_sampling(tmp_path):
    """`full` resolves many cells to one prompt. Sampling the raw list would
    spend the budget re-checking the same handful of prompts."""
    gen = _warm(DummyGenerator(), tmp_path, ["a", "b", "c"])

    audit = audit_determinism(gen, ["a"] * 50 + ["b", "c"], DecodeParams(), n=3, seed=1)
    assert audit["checked"] == 3


def test_audit_on_an_empty_grid_is_not_an_error(tmp_path):
    gen = _warm(DummyGenerator(), tmp_path, [])
    assert audit_determinism(gen, [], DecodeParams(), n=5, seed=1)["checked"] == 0


# --------------------------------------------------------------------------- #
# Selection progress
# --------------------------------------------------------------------------- #
#
# `llm_pruner` and `loo_oracle` call the generator once (or n+1 times) per cell,
# on the one path that does NOT go through CachedGenerator.generate_batch and so
# gets none of its block reporting. On a rate-limited backend that was half an
# hour of silence in the middle of a multi-hour run.


# --------------------------------------------------------------------------- #
# Dummy quarantine
# --------------------------------------------------------------------------- #


def test_dummy_backend_is_quarantined_when_run_is_called_directly():
    """The regression this exists for. The redirect used to live in main(), so
    it only covered the CLI -- calling run() directly wrote hash noise into a
    real results directory under a name indistinguishable from real output."""
    cfg = src.run.Config.load("configs/replication.yaml")
    cfg.raw["generator"]["backend"] = "dummy"
    real_dir = cfg["output"]["results_dir"]

    src.run.quarantine_dummy(cfg)

    assert cfg["output"]["results_dir"] == real_dir + "_dummy"
    assert cfg["cache"]["path"] == "cache/dummy.sqlite"


def test_quarantine_is_idempotent():
    """main() sets the backend and run() quarantines, so a CLI invocation passes
    through twice. It must not become results/x_dummy_dummy."""
    cfg = src.run.Config.load("configs/pilot.yaml")
    cfg.raw["generator"]["backend"] = "dummy"

    src.run.quarantine_dummy(cfg)
    once = cfg["output"]["results_dir"]
    src.run.quarantine_dummy(cfg)

    assert cfg["output"]["results_dir"] == once
    assert once.count("_dummy") == 1


def test_quarantine_leaves_real_backends_alone():
    cfg = src.run.Config.load("configs/main.yaml")
    before = (cfg["output"]["results_dir"], cfg["cache"]["path"])

    src.run.quarantine_dummy(cfg)

    assert (cfg["output"]["results_dir"], cfg["cache"]["path"]) == before


def test_quarantine_does_not_hijack_an_explicit_cache_path(tmp_path):
    """Steering dummy rows out of the SHARED cache is hygiene; overriding a path
    the caller chose deliberately would break test isolation and silently write
    into the repo. The results-dir suffix still applies -- that one is safety."""
    cfg = src.run.Config.load("configs/pilot.yaml")
    cfg.raw["generator"]["backend"] = "dummy"
    mine = str(tmp_path / "mine.sqlite")
    cfg.raw["cache"]["path"] = mine

    src.run.quarantine_dummy(cfg)

    assert cfg["cache"]["path"] == mine
    assert cfg["output"]["results_dir"].endswith("_dummy")


def _pilot_on_dummy(tmp_path):
    """The shipped pilot config, shrunk, on the backend that needs no GPU."""
    cfg = src.run.Config.load("configs/pilot.yaml")
    cfg.raw["generator"]["backend"] = "dummy"
    cfg.raw["data"]["n_queries"] = 4
    cfg.raw["cache"]["path"] = str(tmp_path / "c.sqlite")
    cfg.raw["output"]["results_dir"] = str(tmp_path / "out")
    return cfg


@pytest.mark.network
def test_instant_arms_print_no_selection_progress(tmp_path, capsys):
    """`full` decides without calling anything, and real time is used here, so
    the whole run finishes inside one interval. A line per arm regardless would
    be pure noise in every local run."""
    src.run.run(_pilot_on_dummy(tmp_path))
    assert "selected" not in capsys.readouterr().out


@pytest.mark.network
def test_a_slow_arm_reports_progress_with_an_eta(tmp_path, monkeypatch, capsys):
    """The case this exists for: cells that take real time must produce a line,
    so an idle-looking log can be told apart from a hung process."""
    ticks = iter([i * 40.0 for i in range(500)])
    monkeypatch.setattr(src.run.time, "time", lambda: next(ticks))

    src.run.run(_pilot_on_dummy(tmp_path))
    out = capsys.readouterr().out
    assert "full: selected" in out
    assert "cells" in out and "left)" in out
