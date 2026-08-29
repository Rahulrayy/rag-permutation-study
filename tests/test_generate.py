"""Prompt construction, the greedy guard, and the cache wrapper."""

import pytest

from src.cache import CachedGeneration, GenerationCache
from src.generate import (
    CachedGenerator,
    DecodeParams,
    DummyGenerator,
    LocalGenerator,
    build_prompt,
)


def test_prompt_preserves_chunk_order(chunks):
    """build_prompt must never re-sort: order is the independent variable."""
    reordered = [chunks[5], chunks[0], chunks[9]]
    prompt = build_prompt("q?", reordered)
    assert prompt.index("title5") < prompt.index("title0") < prompt.index("title9")


def test_prompt_numbers_positions_from_one(chunks):
    prompt = build_prompt("q?", chunks[:3])
    assert "[1] title0" in prompt and "[3] title2" in prompt


def test_prompt_contains_every_chunk(chunks):
    prompt = build_prompt("q?", chunks)
    assert all(c.title in prompt for c in chunks)


def test_empty_context_uses_the_nocontext_template():
    prompt = build_prompt("who?", [])
    assert "who?" in prompt and "passages" not in prompt


def test_local_backend_refuses_sampling():
    """Sampling would confound sampling noise with permutation noise."""
    with pytest.raises(ValueError, match="greedy"):
        LocalGenerator().generate_batch(["x"], DecodeParams(do_sample=True))


def test_cache_wrapper_hits_on_repeat(tmp_path):
    gen = CachedGenerator(DummyGenerator(), GenerationCache(tmp_path / "c.sqlite"))
    params = DecodeParams()
    first = gen.generate("prompt", params)
    second = gen.generate("prompt", params)
    assert first.text == second.text
    assert (gen.hits, gen.misses) == (1, 1)


def test_batch_preserves_order_with_mixed_hits(tmp_path):
    """Half-cached batches must come back in the caller's order, not the
    order the misses happened to be computed in."""
    gen = CachedGenerator(DummyGenerator(), GenerationCache(tmp_path / "c.sqlite"))
    params = DecodeParams()
    prompts = [f"p{i}" for i in range(6)]

    gen.generate(prompts[1], params)
    gen.generate(prompts[4], params)

    batched = gen.generate_batch(prompts, params)
    direct = [DummyGenerator().generate(p, params).text for p in prompts]
    assert [g.text for g in batched] == direct
    assert len(batched) == 6


def test_decode_params_key_is_order_independent():
    a = DecodeParams(max_new_tokens=32, seed=1).as_key()
    assert a == DecodeParams(seed=1, max_new_tokens=32).as_key()


class _CountingBackend:
    """Records exactly which prompts the backend was asked to generate."""

    model = "counting"

    def __init__(self):
        self.seen = []

    def generate(self, prompt, params):
        return self.generate_batch([prompt], params)[0]

    def generate_batch(self, prompts, params):
        self.seen.extend(prompts)
        return [CachedGeneration(text=f"answer for {p}") for p in prompts]

    def score(self, prompt, answer):
        return 0.0


def test_duplicate_prompts_are_generated_once(tmp_path):
    """The `full` arm keeps all ten chunks at every budget, so main.yaml's three
    budgets resolve to one set of prompts and must be paid for once."""
    backend = _CountingBackend()
    gen = CachedGenerator(backend, GenerationCache(tmp_path / "c.sqlite"))
    out = gen.generate_batch(["a", "b", "a", "a", "b"], DecodeParams())

    assert backend.seen == ["a", "b"]
    assert [g.text for g in out] == [
        "answer for a", "answer for b", "answer for a",
        "answer for a", "answer for b",
    ]


def test_short_backend_result_raises_instead_of_misaligning(tmp_path):
    """Silently dropping a generation would shift every downstream row against
    its metadata -- corrupting results rather than crashing."""

    class Short(_CountingBackend):
        def generate_batch(self, prompts, params):
            return super().generate_batch(prompts, params)[:-1]

    gen = CachedGenerator(Short(), GenerationCache(tmp_path / "c.sqlite"))
    with pytest.raises(RuntimeError, match="misaligned"):
        gen.generate_batch(["a", "b"], DecodeParams())


def test_generations_are_cached_before_the_whole_batch_finishes(tmp_path):
    """A crash nine hours into an overnight run must keep what it already paid
    for; a single flush at the end would lose all of it."""

    class Exploding(_CountingBackend):
        def generate_batch(self, prompts, params):
            if "boom" in prompts:
                raise RuntimeError("GPU fell over")
            return super().generate_batch(prompts, params)

    cache = GenerationCache(tmp_path / "c.sqlite")
    gen = CachedGenerator(Exploding(), cache, flush_every=2)
    with pytest.raises(RuntimeError, match="fell over"):
        gen.generate_batch(["a", "b", "c", "boom"], DecodeParams())
    assert len(cache) == 2
