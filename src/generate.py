"""Generator backends behind one interface.

Sec. 4.2. Two backends, one interface, so swapping a 3B local model for a hosted
70B is a config change rather than a rewrite.

Why local is primary and not a compromise: proper leave-one-out attribution needs
the **log-probability of the answer sequence**, not a string match, and hosted
APIs generally do not expose it. It also takes the rate limit off the critical path.

Greedy decoding, temperature 0, fixed seed, everywhere. If you sample, sampling
noise and permutation noise are confounded and the design collapses.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .cache import CachedGeneration, GenerationCache, cache_key
from .chunks import Chunk


@dataclass
class DecodeParams:
    max_new_tokens: int = 32
    temperature: float = 0.0
    do_sample: bool = False
    seed: int = 20260828

    def as_key(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "seed": self.seed,
        }


# Freeze this verbatim in ANALYSIS_PLAN.md Sec. 4 before the main run.
#
# Chosen over the looser "Reply with the short answer only" phrasing on a 12-query
# x 5-permutation comparison (Qwen2.5-3B-Instruct, 4-bit):
#
#                    mean F1   mean EM   median within-query SD   answer words
#     loose            0.524     0.367                  0.0434            9.0
#     terse (this)     0.611     0.517                  0.0673            3.5
#     gold                                                                2.2
#
# The justification is **accuracy and answer-format match**: EM and token-F1 only
# measure what they claim to when the model emits an answer, not a sentence about
# the answer. At 9 words against a 2.2-word gold, F1 was largely scoring verbosity.
#
# The justification is NOT that this template has the higher permutation SD. That
# it does is a finding, not a selection criterion -- picking the prompt that
# maximises the quantity the study reports would be a garden-of-forking-paths
# error, and it would hand a reviewer an easy shot at the headline number. If the
# terse template had scored *worse* on accuracy, the loose one would be the right
# choice despite its lower SD.
DEFAULT_TEMPLATE = (
    "Answer the question using only the passages below.\n"
    "Reply with the shortest possible answer: a name, a phrase, a date, or "
    "yes/no. Do not write a sentence. Do not explain.\n\n"
    "{context}\n"
    "Question: {question}\n"
    "Short answer:"
)

# The robustness check from plan Sec. 8: permutation effects must not be an
# artifact of the chunk delimiters. This differs from DEFAULT_TEMPLATE in the
# delimiters *only* -- identical instruction wording, identical answer cue -- so
# a difference between the two isolates the delimiter effect. Varying the
# instructions here as well would confound the check with the verbosity effect
# measured above.
ALT_TEMPLATE = (
    "Answer the question using only the passages below.\n"
    "Reply with the shortest possible answer: a name, a phrase, a date, or "
    "yes/no. Do not write a sentence. Do not explain.\n\n"
    "<context>\n{context}\n</context>\n"
    "Question: {question}\n"
    "Short answer:"
)

# Enforced, not just asserted in a comment: ALT must be reachable from DEFAULT by
# substituting the context placeholder alone. Any other drift between the two
# would confound the delimiter robustness check with a wording change.
# A bare `assert` is stripped by `python -O`, which would disable this guard
# exactly when someone runs a long job with optimisations on.
if ALT_TEMPLATE != DEFAULT_TEMPLATE.replace(
    "{context}", "<context>\n{context}\n</context>"
):
    raise AssertionError(
        "ALT_TEMPLATE must differ from DEFAULT_TEMPLATE in the delimiters only"
    )

# Matching instructions, no context. Used by the nocontext arm and therefore by
# the memorization filter, so its answer cue must match the context templates --
# otherwise the filter compares answers elicited under two different formats.
NOCONTEXT_TEMPLATE = (
    "Answer the question.\n"
    "Reply with the shortest possible answer: a name, a phrase, a date, or "
    "yes/no. Do not write a sentence. Do not explain.\n\n"
    "Question: {question}\n"
    "Short answer:"
)


def build_prompt(
    question: str,
    chunks: Sequence[Chunk],
    template: str = DEFAULT_TEMPLATE,
) -> str:
    """Render chunks in the order given.

    That order is the independent variable of the whole study, so nothing in here
    may re-sort them.
    """
    if not chunks:
        return NOCONTEXT_TEMPLATE.format(question=question)
    context = "\n\n".join(f"[{i + 1}] {c.title}: {c.text}" for i, c in enumerate(chunks))
    return template.format(context=context, question=question)


class Generator(Protocol):
    """One interface, three backends."""

    model: str

    def generate(self, prompt: str, params: DecodeParams) -> CachedGeneration:
        ...

    def generate_batch(
        self, prompts: Sequence[str], params: DecodeParams
    ) -> list[CachedGeneration]:
        ...

    def score(self, prompt: str, answer: str) -> float:
        """Log-prob of ``answer`` given ``prompt``. Needed by the LOO oracle."""
        ...


# --------------------------------------------------------------------------- #
# Caching wrapper
# --------------------------------------------------------------------------- #

@dataclass
class CachedGenerator:
    """Wraps any backend with the content-hash cache.

    Always use this, never a bare backend: 36,000 generations, and you will rerun
    the analysis twenty times.
    """

    backend: Generator
    cache: GenerationCache
    # Generations are written to the cache after every block of this many, not
    # once at the end. The main grid is ~36,000 generations over ~10 hours; a
    # single flush at the end means a crash at hour 9 caches nothing and the
    # whole run restarts from zero, which is the opposite of what the cache is
    # for. Blocks also give a long run something to print, so an idle-looking
    # log can be told apart from a hung process.
    flush_every: int = 64
    progress: bool = False
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    @property
    def model(self) -> str:
        return self.backend.model

    def generate(self, prompt: str, params: DecodeParams) -> CachedGeneration:
        key = cache_key(self.backend.model, prompt, params.as_key())
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        gen = self.backend.generate(prompt, params)
        self.cache.put(key, self.backend.model, gen)
        return gen

    def generate_batch(
        self, prompts: Sequence[str], params: DecodeParams
    ) -> list[CachedGeneration]:
        """Resolve hits from cache, batch only the misses, restore original order.

        The distinct prompt, not the call site, is the unit of work: many grid
        cells legitimately resolve to the same prompt -- the `full` arm keeps all
        ten chunks whatever the budget, so main.yaml's three budgets would
        otherwise pay three times over for one set of generations. Duplicates
        within a single call are generated once and fanned back out.
        """
        keys = [cache_key(self.backend.model, p, params.as_key()) for p in prompts]

        # First occurrence of each distinct key drives the work.
        first_idx: dict[str, int] = {}
        for i, k in enumerate(keys):
            first_idx.setdefault(k, i)

        resolved: dict[str, CachedGeneration] = {}
        pending: list[int] = []
        for k, i in first_idx.items():
            hit = self.cache.get(k)
            if hit is None:
                pending.append(i)
            else:
                resolved[k] = hit

        self.misses += len(pending)
        self.hits += len(prompts) - len(pending)

        for start in range(0, len(pending), self.flush_every):
            block = pending[start : start + self.flush_every]
            fresh = self.backend.generate_batch([prompts[i] for i in block], params)
            # A backend returning the wrong count would otherwise be absorbed by
            # zip() and silently shift every downstream row against its metadata.
            if len(fresh) != len(block):
                raise RuntimeError(
                    f"backend returned {len(fresh)} generations for "
                    f"{len(block)} prompts; results would be misaligned"
                )
            for i, gen in zip(block, fresh):
                resolved[keys[i]] = gen
                self.cache.put(keys[i], self.backend.model, gen)
            if self.progress:
                done = min(start + self.flush_every, len(pending))
                print(f"    generated {done}/{len(pending)} new", flush=True)

        return [resolved[k] for k in keys]

    def score(self, prompt: str, answer: str) -> float:
        key = cache_key(self.backend.model, prompt, {"score_of": answer})
        hit = self.cache.get(key)
        if hit is not None and hit.answer_logprob is not None:
            self.hits += 1
            return hit.answer_logprob
        self.misses += 1
        logprob = self.backend.score(prompt, answer)
        self.cache.put(
            key,
            self.backend.model,
            CachedGeneration(text=answer, answer_logprob=logprob),
        )
        return logprob


# --------------------------------------------------------------------------- #
# Local backend
# --------------------------------------------------------------------------- #

@dataclass
class LocalGenerator:
    """4-bit HF transformers on the 3060. Primary backend.

    Model and tokenizer load lazily on first use, so importing this module (and
    running the tests) costs nothing.
    """

    model: str = "Qwen/Qwen2.5-3B-Instruct"
    quantization: str = "4bit"
    batch_size: int = 8
    # None -> standard HF cache. Never point this inside the project:
    # the repo is under OneDrive and weights would sync to the cloud.
    cache_dir: str | None = None
    # Hard ceiling on the share of VRAM this process may allocate, or None for
    # no cap. On a laptop the display runs off the same GPU, so an unbounded
    # run does not fail cleanly: it starves the compositor and the monitor
    # drops out. A cap converts that into an ordinary CUDA OOM, which is
    # recoverable -- the cache keeps every generation already paid for, so a
    # restart at a smaller batch resumes rather than starts over.
    max_vram_fraction: float | None = None
    # Seconds to idle between batches, giving the GPU back to the desktop
    # compositor. On a laptop where the display shares the GPU, Windows resets
    # the display driver if a kernel blocks it past TdrDelay (2s by default) --
    # the monitor blanks and recovers, with no error on the Python side. That is
    # a latency problem, not a memory one, so a VRAM cap alone does not fix it.
    # Pausing costs wall-clock and nothing else.
    batch_pause_s: float = 0.0
    _tok: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._tok, self._model

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.max_vram_fraction is not None:
            if not 0.0 < self.max_vram_fraction <= 1.0:
                raise ValueError(
                    f"max_vram_fraction must be in (0, 1], got {self.max_vram_fraction}"
                )
            if torch.cuda.is_available():
                # Applied before the weights load, so the cap covers loading too.
                torch.cuda.set_per_process_memory_fraction(self.max_vram_fraction)
                total = torch.cuda.get_device_properties(0).total_memory
                print(
                    f"VRAM capped at {self.max_vram_fraction:.0%} "
                    f"({total * self.max_vram_fraction / 1e9:.2f} GB of "
                    f"{total / 1e9:.2f} GB)"
                )

        tok = AutoTokenizer.from_pretrained(self.model, cache_dir=self.cache_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Decoder-only batched generation needs left padding, or the generated
        # continuation starts after the pad run and comes back as garbage.
        tok.padding_side = "left"

        kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "device_map": "auto",
            "dtype": torch.float16,
        }
        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization not in (None, "none", "fp16"):
            raise ValueError(f"unknown quantization: {self.quantization!r}")

        model = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)
        model.eval()

        self._tok, self._model = tok, model
        return tok, model

    def _chat(self, tok: Any, prompt: str) -> str:
        """Apply the instruct chat template, if the checkpoint has one."""
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompt: str, params: DecodeParams) -> CachedGeneration:
        return self.generate_batch([prompt], params)[0]

    def generate_batch(
        self, prompts: Sequence[str], params: DecodeParams
    ) -> list[CachedGeneration]:
        # Validate before importing torch or loading weights: a bad decode config
        # should fail in milliseconds with the real reason, not after a 3B model
        # has been pulled onto the GPU.
        if params.do_sample or params.temperature != 0.0:
            raise ValueError("greedy decoding only; see plan Sec. 4.2")

        import torch

        tok, model = self._load()
        out: list[CachedGeneration] = []

        for start in range(0, len(prompts), self.batch_size):
            batch = [self._chat(tok, p) for p in prompts[start : start + self.batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                # Greedy: do_sample=False and no temperature/top_p passed at all.
                # Passing temperature=0.0 alongside do_sample=False makes
                # transformers emit a warning and, worse, invites someone later
                # to "fix" the warning by enabling sampling.
                generated = model.generate(
                    **enc,
                    max_new_tokens=params.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )

            # Strip the prompt: with left padding every row shares input width.
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            for row in new_tokens:
                text = tok.decode(row, skip_special_tokens=True).strip()
                out.append(CachedGeneration(text=text))

            if self.batch_pause_s:
                time.sleep(self.batch_pause_s)

        return out

    def score(self, prompt: str, answer: str) -> float:
        """Summed log-prob of ``answer`` tokens conditioned on ``prompt``.

        This is the quantity the LOO oracle ranks chunks by, and the reason the
        primary generator runs locally at all.
        """
        import torch

        tok, model = self._load()
        prefix = self._chat(tok, prompt)

        prefix_ids = tok(prefix, return_tensors="pt")["input_ids"]
        # Tokenize the answer on its own and concatenate ids, rather than
        # tokenizing `prefix + answer` and slicing at len(tok(prefix)).
        #
        # Slicing is only safe while the prefix happens to end on a token
        # boundary BPE will not merge across. It does today -- the chat template
        # ends in a newline, and DEFAULT_TEMPLATE ends "Short answer:" with no
        # trailing space -- so this is a latent trap rather than a live bug.
        # Verified against the Qwen2.5 tokenizer: end the prefix with a space
        # instead ("Short answer: ") and " Vil" merges into one token, the
        # prefix length comes out wrong, and every scored position shifts. The
        # oracle would then rank noise with nothing in the output to say so, and
        # the trigger would be an innocuous-looking edit to a frozen template.
        # Concatenating ids removes the dependency; on the current templates it
        # produces byte-identical token sequences to the old path.
        answer_ids = tok(answer, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        if answer_ids.shape[1] == 0:
            raise ValueError("answer contributed no tokens; check the tokenizer")

        n_prefix = prefix_ids.shape[1]
        n_answer = answer_ids.shape[1]
        full_ids = torch.cat([prefix_ids, answer_ids], dim=1).to(model.device)

        # Only the last n_answer+1 positions are ever read, but a plain forward
        # pass materialises logits for all ~1,450 of them: 1450 x 151,936 vocab
        # x 2 bytes is ~440 MB in one allocation, the largest single spike in
        # the codebase. `logits_to_keep` asks the model for just the tail.
        with torch.no_grad():
            try:
                logits = model(full_ids, logits_to_keep=n_answer + 1).logits
                # Returned positions are the last n_answer+1, i.e. n_prefix-1
                # through the end; drop the final one, which predicts past the
                # answer.
                tail = logits[0, :-1]
            except TypeError:  # older transformers without the kwarg
                logits = model(full_ids).logits
                tail = logits[0, n_prefix - 1 : -1]

        # logits[t] predicts token t+1, so the answer token at position t is
        # scored by the distribution at t-1.
        log_probs = torch.log_softmax(tail.float(), dim=-1)
        target_ids = full_ids[0, n_prefix:]
        token_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return float(token_lp.sum().item())


# --------------------------------------------------------------------------- #
# Hosted backend
# --------------------------------------------------------------------------- #

_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def _parse_duration(raw: str | None) -> float | None:
    """Seconds from Groq's compound duration format: ``27.277s``, ``1h7m40.8s``.

    Returns None for anything unparseable, so a header format change degrades to
    the caller's fallback rather than to a crash mid-run.
    """
    if not raw:
        return None
    matches = _DURATION_RE.findall(raw.strip())
    if not matches:
        return None
    return sum(float(value) * _DURATION_UNITS[unit] for value, unit in matches)


class _TokenBudget:
    """Paces requests off the account's token-per-minute window.

    Blind exponential backoff does not work against a TPM limit shared by
    several workers: they sleep independently, wake together, and re-exhaust the
    window. That is exactly how the first live run of this backend failed --
    six retries spanning 63s all landed inside one saturated 60s window and the
    429 escaped to the caller.

    Groq reports ``x-ratelimit-remaining-tokens`` and ``x-ratelimit-reset-tokens``
    on *every* response, so the right amount to wait is a fact to be read, not a
    quantity to be guessed. This tracks that window across threads and blocks
    before a request that would not fit.

    The lock is deliberately held across the sleep. At 8,000 TPM and ~1,500-token
    prompts the ceiling is about five requests a minute, so there is no
    throughput to lose by serialising, and holding it removes the race where a
    waiting worker is overtaken by one that sees a stale budget.
    """

    def __init__(self, headroom_tokens: int = 1000) -> None:
        #: Stay this far clear of the ceiling. A request's cost is estimated
        #: from character count, and undershooting costs a 429 plus a full
        #: window -- far more than the throughput the headroom gives up.
        self._headroom = headroom_tokens
        self._lock = threading.Lock()
        self._remaining: int | None = None
        self._reset_at = 0.0
        self.waited_s = 0.0
        self.waits = 0

    def observe(self, headers: Any) -> None:
        """Update the window from a response's headers. Cheap and idempotent."""
        raw_remaining = headers.get("x-ratelimit-remaining-tokens")
        if raw_remaining is None:
            return
        try:
            remaining = int(float(raw_remaining))
        except (TypeError, ValueError):
            return
        reset = _parse_duration(headers.get("x-ratelimit-reset-tokens"))
        with self._lock:
            self._remaining = remaining
            if reset is not None:
                self._reset_at = time.monotonic() + reset

    def acquire(self, cost: int) -> float:
        """Block until ``cost`` tokens plausibly fit. Returns seconds slept."""
        with self._lock:
            if self._remaining is not None and self._remaining < cost + self._headroom:
                delay = max(0.0, self._reset_at - time.monotonic())
                if delay:
                    time.sleep(delay)
                    self.waited_s += delay
                    self.waits += 1
                # The window has refilled; the next response re-establishes the
                # true figure. Optimistic, but only ever by one request.
                self._remaining = None
            if self._remaining is not None:
                self._remaining -= cost
        return 0.0


def _estimate_tokens(prompt: str, max_new_tokens: int) -> int:
    """Rough token cost of one call, for pacing only.

    Four characters per token is the usual English approximation and does not
    need to be better than that: it feeds a headroom check, not a billing
    figure, and the real number arrives in the next response's headers.
    """
    return len(prompt) // 4 + max_new_tokens


def _daily_cap_error(wait_s: float | None) -> RuntimeError:
    """The one rate-limit condition that is not worth waiting out in-process.

    A long grid is *expected* to cross the daily cap -- that is a schedule, not
    a failure -- so this says what happened, what it costs (nothing) and what to
    do, rather than surfacing an opaque 429 after the retry ceiling.
    """
    when = ""
    if wait_s is not None:
        hours = wait_s / 3600.0
        when = f" It resets in about {hours:.1f}h." if hours >= 1 else ""
    return RuntimeError(
        "Groq's daily request cap for this account is exhausted." + when + " Nothing "
        "already generated is lost: rerun the identical command after the reset "
        "and the cache replays every call paid for so far, resuming where this "
        "stopped. See console.groq.com/settings/limits."
    )


def _daily_requests_exhausted(headers: Any) -> bool:
    """True when the account has no requests left today.

    Distinguished from per-minute throttling by the header rather than by the
    size of the Retry-After: a daily rejection can still carry a small
    ``reset-tokens``, and sleeping five seconds into an exhausted day would loop
    until the retry ceiling and then report the wrong cause.
    """
    raw = headers.get("x-ratelimit-remaining-requests")
    if raw is None:
        return False
    try:
        return int(float(raw)) <= 0
    except (TypeError, ValueError):
        return False


def _retry_after(exc: Any) -> float | None:
    """Seconds the server asked us to wait, when it says.

    Groq answers a 429 with a ``retry-after`` header. Honouring it is the
    difference between waiting once and spending the daily request cap on
    retries that were always going to be refused.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw = headers.get("retry-after")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            # Some proxies send an HTTP-date instead of seconds. Fall through to
            # the reset headers rather than guessing at a clock skew.
            pass

    # A TPM rejection carries the window reset even when retry-after is absent
    # or unparseable, and it is the more precise of the two. Requests-per-day is
    # checked second: it resets in hours, and the caller treats a wait that long
    # as the daily cap rather than as something to sleep through.
    for header in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        seconds = _parse_duration(headers.get(header))
        if seconds is not None:
            return seconds
    return None


@dataclass
class GroqGenerator:
    """Cross-generator check, n=100 (plan Sec. 4.2, Sec. 5).

    A second model on the same queries, to show the finding is not an artifact
    of a 3B local generator. Not a drop-in replacement for `LocalGenerator`, for
    two reasons that are methodological rather than technical.

    **No answer log-probs.** ``score`` is unavailable, so the `loo_oracle` arm
    cannot run here. That is the reason the primary generator is local at all.

    **Greedy is best-effort.** ``temperature=0`` is the closest this API offers,
    and ``seed`` is a hint: hosted serving batches requests across tenants, and
    for a mixture-of-experts model the batch composition changes the arithmetic.
    That matters more here than in most projects, because the headline quantity
    is *within-query SD across permutations* -- API nondeterminism would land in
    exactly that number with nothing to separate it from position bias.

    So it was measured rather than assumed (ANALYSIS_PLAN Sec. 9, 2026-08-30):
    4 HotpotQA queries x 5 identical repeat calls to ``qwen/qwen3.8-27b``,
    greedy, fixed seed, **20/20 byte-identical**. Evidence, not a guarantee --
    one session, one server pool, 3-5 token answers. The generation cache
    narrows the exposure further by paying for each distinct prompt once and
    then freezing it. Treat a non-zero SD here as real; do not treat it as
    interchangeable with the local SD.

    Rate limits: check console.groq.com/settings/limits before sizing a run, and
    note that BOTH ceilings bind. Measured on the development account: 8,000
    tokens/min, which at ~1,500-token prompts is ~5 requests/min and sets the
    pace, and 1,000 requests/day, which caps total volume regardless. A grid
    larger than the daily cap is expected to span days; ``_complete`` raises a
    named error for that case rather than retrying into it, and the cache makes
    the resumed run free.
    """

    #: Not llama-3.3-70b: retired from the catalogue. See the model-selection
    #: note in configs/replication.yaml -- every other chat model on the account
    #: is a reasoning model that returns an empty answer at max_new_tokens=32.
    model: str = "qwen/qwen3.8-27b"
    #: None -> read GROQ_API_KEY from the environment. Never put a key in a
    #: config file: configs are committed, and this one is a portfolio repo.
    api_key: str | None = None
    #: In-flight requests. 1 is right for a token-limited account: at 8,000 TPM
    #: and ~1,500-token prompts the ceiling is ~5 requests a minute, so extra
    #: workers add contention and 429s rather than throughput. Raise it only for
    #: short prompts or a paid tier.
    concurrency: int = 1
    #: A 429 is routine here, not exceptional -- a TPM window refills every
    #: minute and a long grid will cross many of them. This is a ceiling on
    #: consecutive failures for one prompt, not an error budget for the run.
    max_retries: int = 8
    max_backoff_s: float = 90.0
    timeout_s: float = 60.0
    #: Shared across every worker, so pacing is a property of the account rather
    #: than of one thread. Constructed per generator instance.
    budget: _TokenBudget = field(default_factory=_TokenBudget, repr=False)
    _api: Any = field(default=None, init=False, repr=False)

    def _client(self) -> Any:
        if self._api is not None:
            return self._api

        import os

        from .env import DEFAULT_ENV_PATH, load_env

        try:
            import groq
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise RuntimeError(
                "the groq backend needs the SDK: pip install 'groq>=0.11'"
            ) from exc

        # Pull in the repo-root .env if there is one. Done here rather than in
        # run.py so that a notebook or a one-off script driving this backend
        # directly gets the key the same way the CLI does. An already-exported
        # GROQ_API_KEY wins over the file.
        load_env()

        key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Create a key at console.groq.com/keys, "
                f"then either put it in {DEFAULT_ENV_PATH} as "
                "GROQ_API_KEY=gsk_... or export it in this shell."
            )
        # max_retries=0: the retry policy lives in `_complete`, so a Retry-After
        # longer than the SDK's internal cap is actually waited out instead of
        # being truncated into an immediate second 429.
        self._api = groq.Client(api_key=key, max_retries=0, timeout=self.timeout_s)
        return self._api

    def _complete(self, prompt: str, params: DecodeParams) -> CachedGeneration:
        import groq

        client = self._client()
        backoff = 1.0
        cost = _estimate_tokens(prompt, params.max_new_tokens)

        for attempt in range(self.max_retries + 1):
            # Wait for room in the token window before spending a request on a
            # call the server is going to refuse.
            self.budget.acquire(cost)
            try:
                # with_raw_response so the rate-limit headers survive parsing.
                # They are what makes the pacing above self-correcting rather
                # than a guess that drifts.
                raw = client.chat.completions.with_raw_response.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=params.max_new_tokens,
                    temperature=0.0,
                    seed=params.seed,
                    stream=False,
                )
            except groq.APIConnectionError:
                if attempt == self.max_retries:
                    raise
                asked = None
            except groq.APIStatusError as exc:
                # 429 and 5xx are worth waiting out. 400/401/404 are not: a bad
                # model name or a rejected key will not fix itself, and retrying
                # one buries the real error behind a minute of backoff.
                if exc.status_code != 429 and exc.status_code < 500:
                    raise
                headers = getattr(getattr(exc, "response", None), "headers", None)
                if headers:
                    if _daily_requests_exhausted(headers):
                        raise _daily_cap_error(_retry_after(exc)) from exc
                    # Feed the rejection back into the pacer: a 429 carries the
                    # same window headers a success does, so the next attempt
                    # waits the right amount instead of the doubling guess.
                    self.budget.observe(headers)
                if attempt == self.max_retries:
                    raise
                asked = _retry_after(exc)
            else:
                self.budget.observe(raw.headers)
                choice = raw.parse().choices[0]
                return CachedGeneration(
                    text=(choice.message.content or "").strip(),
                    meta={
                        "backend": "groq",
                        # "length" means the answer hit max_new_tokens and was
                        # cut off. That scores as a wrong answer rather than as
                        # an error, so a run with many of them is measuring
                        # truncation, not position. Recorded to be checkable.
                        "finish_reason": choice.finish_reason,
                    },
                )

            if asked is None:
                wait, backoff = backoff, min(backoff * 2, self.max_backoff_s)
            elif asked > self.max_backoff_s:
                # Backstop for the same condition when the headers did not say
                # so outright: a wait this long is not per-minute throttling,
                # and no amount of sleeping inside this process will clear it.
                raise _daily_cap_error(asked)
            else:
                wait = asked
            time.sleep(wait)

        raise RuntimeError("retry loop exited without returning")  # unreachable

    def generate(self, prompt: str, params: DecodeParams) -> CachedGeneration:
        return self.generate_batch([prompt], params)[0]

    def generate_batch(
        self, prompts: Sequence[str], params: DecodeParams
    ) -> list[CachedGeneration]:
        # Validate before opening a client, so a bad decode config fails in
        # milliseconds with the real reason rather than after the first call.
        if params.do_sample or params.temperature != 0.0:
            raise ValueError("greedy decoding only; see plan Sec. 4.2")
        if not prompts:
            return []

        # Construct up front: a missing key should be one clear error, not N
        # identical ones surfacing out of a thread pool.
        self._client()

        if self.concurrency <= 1:
            return [self._complete(p, params) for p in prompts]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            # `map` yields in submission order. CachedGenerator zips the result
            # against its own key list, so returning out of order would silently
            # attach every answer to the wrong row.
            return list(pool.map(lambda p: self._complete(p, params), prompts))

    def score(self, prompt: str, answer: str) -> float:
        raise NotImplementedError(
            "hosted API does not expose answer log-probs; use LocalGenerator"
        )


# --------------------------------------------------------------------------- #
# Test backend
# --------------------------------------------------------------------------- #

@dataclass
class DummyGenerator:
    """Deterministic fake for exercising the pipeline without a GPU.

    Returns a hash-derived token, so it is prompt-sensitive and therefore
    order-sensitive: enough to prove that permutation, caching, scoring and the
    gate computation are wired together correctly.

    Its numbers mean **nothing**. A permutation SD computed from this backend
    measures hash churn, not position bias -- never report it as a gate result.
    """

    model: str = "dummy"
    vocabulary: tuple[str, ...] = ("yes", "no", "Paris", "1994", "unknown")

    def generate(self, prompt: str, params: DecodeParams) -> CachedGeneration:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        word = self.vocabulary[digest[0] % len(self.vocabulary)]
        return CachedGeneration(text=word, meta={"backend": "dummy"})

    def generate_batch(
        self, prompts: Sequence[str], params: DecodeParams
    ) -> list[CachedGeneration]:
        return [self.generate(p, params) for p in prompts]

    def score(self, prompt: str, answer: str) -> float:
        digest = hashlib.sha256((prompt + "||" + answer).encode("utf-8")).digest()
        return -(digest[0] / 255.0) * 10.0
