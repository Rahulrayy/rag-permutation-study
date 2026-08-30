"""Prompt construction, the greedy guard, and the cache wrapper."""

import time

import pytest

from src.cache import CachedGeneration, GenerationCache
from src.generate import (
    CachedGenerator,
    DecodeParams,
    DummyGenerator,
    GroqGenerator,
    LocalGenerator,
    _parse_duration,
    _retry_after,
    _TokenBudget,
    build_prompt,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("27.277s", 27.277),
        ("1h7m40.8s", 4060.8),
        ("2m59.56s", 179.56),
        ("100ms", 0.1),
        ("", None),
        (None, None),
        ("soon", None),
    ],
)
def test_parse_duration(raw, expected):
    """Groq reports resets in a compound format, not seconds. Misreading
    '1h7m40.8s' as 1 second would retry straight back into an exhausted cap."""
    assert _parse_duration(raw) == expected


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


def test_groq_backend_refuses_sampling():
    """Same guard as the local backend, and it must fire before the client is
    built -- otherwise a bad decode config surfaces as 'GROQ_API_KEY is not
    set', which points at the wrong thing entirely."""
    with pytest.raises(ValueError, match="greedy"):
        GroqGenerator().generate_batch(["x"], DecodeParams(do_sample=True))


def test_groq_cannot_score():
    """The LOO oracle needs answer log-probs, which hosted chat APIs do not
    return. This is why the primary generator is local (plan Sec. 4.2)."""
    with pytest.raises(NotImplementedError, match="log-prob"):
        GroqGenerator().score("prompt", "answer")


def test_groq_batch_preserves_order_under_concurrency(monkeypatch):
    """Results are zipped against the caller's key list, so a thread pool that
    returned completion-order would attach every answer to the wrong row."""
    backend = GroqGenerator(concurrency=4)
    monkeypatch.setattr(backend, "_client", lambda: object())

    # Invert completion order against submission order: later prompts finish
    # sooner. If the pool ever yielded as-completed, this test fails.
    def _slow_in_reverse(prompt, params):
        time.sleep((20 - int(prompt[1:])) * 0.002)
        return CachedGeneration(text=f"answer for {prompt}")

    monkeypatch.setattr(backend, "_complete", _slow_in_reverse)

    prompts = [f"p{i}" for i in range(20)]
    out = backend.generate_batch(prompts, DecodeParams())
    assert [g.text for g in out] == [f"answer for {p}" for p in prompts]


def test_groq_empty_batch_needs_no_client():
    """An arm that selects nothing must not cost an API key lookup."""
    assert GroqGenerator().generate_batch([], DecodeParams()) == []


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers


class _FakeStatusError(Exception):
    def __init__(self, headers):
        self.response = _FakeResponse(headers)


def _fake_completion(text, finish_reason="stop", headers=None):
    """Shaped like the SDK's raw response, only the fields we read."""
    message = type("Message", (), {"content": text})()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    parsed = type("Completion", (), {"choices": [choice]})()
    return type(
        "Raw", (), {"headers": headers or {}, "parse": staticmethod(lambda: parsed)}
    )()


def _install_fake_client(backend, monkeypatch, responses):
    """Wire `backend` to a client whose create() replays `responses` in order,
    raising any that are exceptions. Returns the recorded sleep durations."""
    slept = []
    monkeypatch.setattr("src.generate.time.sleep", slept.append)

    queue = list(responses)

    def create(**kwargs):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    raw_ns = type("WithRaw", (), {"create": staticmethod(create)})()
    completions = type("Completions", (), {"with_raw_response": raw_ns})()
    chat = type("Chat", (), {"completions": completions})()
    monkeypatch.setattr(backend, "_client", lambda: type("C", (), {"chat": chat})())
    return slept


def _status_error(status, headers=None):
    import groq
    import httpx

    response = httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/x"),
    )
    return groq.APIStatusError("boom", response=response, body=None)


def test_groq_waits_the_retry_after_a_429_asks_for(monkeypatch):
    """The overnight replication run lives or dies on this: ignoring
    Retry-After spends the daily request cap on retries that are refused."""
    backend = GroqGenerator(concurrency=1)
    slept = _install_fake_client(
        backend,
        monkeypatch,
        [_status_error(429, {"retry-after": "9"}), _fake_completion("Paris")],
    )

    out = backend.generate("prompt", DecodeParams())
    assert out.text == "Paris"
    assert slept == [9.0]


def test_groq_backs_off_exponentially_when_the_server_says_nothing(monkeypatch):
    backend = GroqGenerator(concurrency=1)
    slept = _install_fake_client(
        backend,
        monkeypatch,
        [_status_error(500), _status_error(503), _fake_completion("ok")],
    )

    assert backend.generate("prompt", DecodeParams()).text == "ok"
    assert slept == [1.0, 2.0]


def test_groq_names_the_daily_cap_instead_of_sleeping_through_it(monkeypatch):
    """A Retry-After of hours is the daily request cap, which no wait inside
    this process can clear. Failing fast keeps the cache and says why."""
    backend = GroqGenerator(concurrency=1)
    slept = _install_fake_client(
        backend, monkeypatch, [_status_error(429, {"retry-after": "7200"})]
    )

    with pytest.raises(RuntimeError, match="daily request cap"):
        backend.generate("prompt", DecodeParams())
    assert slept == []


def test_groq_reads_the_daily_cap_off_the_header_not_the_wait(monkeypatch):
    """A daily rejection can still carry a small reset-tokens. Sleeping five
    seconds into an exhausted day would loop to the retry ceiling and then
    report the wrong cause."""
    backend = GroqGenerator(concurrency=1)
    slept = _install_fake_client(
        backend,
        monkeypatch,
        [
            _status_error(
                429,
                {
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-tokens": "5s",
                    "x-ratelimit-reset-requests": "2h13m20s",
                },
            )
        ],
    )

    with pytest.raises(RuntimeError, match="daily request cap"):
        backend.generate("prompt", DecodeParams())
    assert slept == []


def test_groq_waits_the_token_window_named_by_a_tpm_rejection(monkeypatch):
    """The failure that broke the first live run: a TPM 429 with no retry-after.
    The reset header carries the answer, so it must be preferred to doubling."""
    backend = GroqGenerator(concurrency=1)
    slept = _install_fake_client(
        backend,
        monkeypatch,
        [
            _status_error(
                429,
                {
                    "x-ratelimit-remaining-requests": "953",
                    "x-ratelimit-reset-tokens": "5.625s",
                },
            ),
            _fake_completion("Paris"),
        ],
    )

    assert backend.generate("p", DecodeParams()).text == "Paris"
    assert slept == [5.625]


def test_groq_paces_off_the_remaining_token_header(monkeypatch):
    """With the window nearly empty, the next call must wait for the refill
    rather than spend a request the server will refuse."""
    backend = GroqGenerator(concurrency=1)
    backend.budget = _TokenBudget(headroom_tokens=500)
    slept = _install_fake_client(
        backend,
        monkeypatch,
        [
            _fake_completion(
                "first",
                headers={
                    "x-ratelimit-remaining-tokens": "80",
                    "x-ratelimit-reset-tokens": "12s",
                },
            ),
            _fake_completion("second"),
        ],
    )

    params = DecodeParams()
    assert backend.generate("p", params).text == "first"
    assert slept == []  # nothing known yet on the first call
    assert backend.generate("q", params).text == "second"
    assert slept == [12.0]  # 80 remaining cannot cover cost + 500 headroom
    assert backend.budget.waits == 1


def test_groq_does_not_retry_a_client_error(monkeypatch):
    """A bad model id or a rejected key will not fix itself. Six backoffs on one
    buries the real error for a minute."""
    import groq

    backend = GroqGenerator(concurrency=1)
    slept = _install_fake_client(backend, monkeypatch, [_status_error(404)])

    with pytest.raises(groq.APIStatusError):
        backend.generate("prompt", DecodeParams())
    assert slept == []


def test_groq_records_truncation(monkeypatch):
    """finish_reason='length' scores as a wrong answer rather than an error, so
    it has to be recoverable from the row."""
    backend = GroqGenerator(concurrency=1)
    _install_fake_client(
        backend, monkeypatch, [_fake_completion("a very long ans", "length")]
    )

    assert backend.generate("p", DecodeParams()).meta["finish_reason"] == "length"


def test_retry_after_reads_the_header():
    assert _retry_after(_FakeStatusError({"retry-after": "7.5"})) == 7.5


@pytest.mark.parametrize(
    "headers", [{}, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}]
)
def test_retry_after_falls_back_when_the_header_is_absent_or_a_date(headers):
    """Returning None hands the caller its exponential schedule. Guessing at a
    date would mean guessing at clock skew."""
    assert _retry_after(_FakeStatusError(headers)) is None


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
