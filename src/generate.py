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
    _tok: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._tok, self._model

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        full_ids = torch.cat([prefix_ids, answer_ids], dim=1).to(model.device)
        with torch.no_grad():
            logits = model(full_ids).logits

        # logits[t] predicts token t+1, so answer token at position t is scored
        # by the distribution at t-1.
        log_probs = torch.log_softmax(logits[0, n_prefix - 1 : -1].float(), dim=-1)
        target_ids = full_ids[0, n_prefix:]
        token_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return float(token_lp.sum().item())


# --------------------------------------------------------------------------- #
# Hosted backend
# --------------------------------------------------------------------------- #

@dataclass
class GroqGenerator:
    """Cross-generator check, n=100 (plan Sec. 4.2, Sec. 5).

    Before sizing a run against this, check console.groq.com/settings/limits
    yourself. The binding constraint is more likely tokens-per-minute than
    requests-per-day, and both vary per model.

    STATUS: stubbed (week 5). ``score`` is expected to stay unavailable, which is
    exactly why the primary generator is local.
    """

    model: str = "llama-3.3-70b-versatile"

    def generate(self, prompt: str, params: DecodeParams) -> CachedGeneration:
        raise NotImplementedError("groq backend not implemented (week 5)")

    def generate_batch(
        self, prompts: Sequence[str], params: DecodeParams
    ) -> list[CachedGeneration]:
        return [self.generate(p, params) for p in prompts]

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
