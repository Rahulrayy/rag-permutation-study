"""The determinism audit. It exists to put a number on the one thing a hosted
generator cannot promise, so its own failure modes matter more than most."""

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
