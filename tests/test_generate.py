"""Prompt construction, the greedy guard, and the cache wrapper."""

import pytest

from src.cache import GenerationCache
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
