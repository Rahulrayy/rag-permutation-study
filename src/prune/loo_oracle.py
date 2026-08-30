"""Causal ceiling: keep the k chunks whose removal costs the answer the most.

Not a method. A yardstick
-------------------------
This arm reads the gold answer. It is therefore not deployable, cannot be
compared against the others as a *method*, and must never enter the confirmatory
Holm family -- `ANALYSIS_PLAN.md` Sec. 8 already fixes it as a secondary
quantity. It exists to put a number on headroom: Oracle Gap is `Q(m) /
Q(loo_oracle)` at matched budget (plan Sec. 4.5).

For each chunk it measures the leave-one-out drop in the reference answer's
log-probability

    drop(c) = logP(answer | all n chunks) - logP(answer | all n except c)

and keeps the k largest. Positive means removing the chunk hurt, so the chunk
was carrying its weight. Needs ``Generator.score`` -- the log-prob of the
reference sequence, not a string match -- which is the reason the primary
generator is local at all (plan Sec. 4.2); hosted APIs generally will not give
you this.

The drop is averaged over orderings, and that is not a detail
------------------------------------------------------------
Removing chunk *c* does not just remove content. It promotes everything after it
into higher-visibility slots -- plan Sec. 3, premise 3, the confound this whole
study is about. A LOO drop measured under one presentation order is therefore
part content and part position, and a "ceiling" built out of it would be one
draw from an order-indexed distribution: precisely the criticism this study
levels at `llm_pruner`, committed by its own oracle.

So every context is scored under the same P orderings the study generates under,
and the drop is the mean across them. Cost is ``(n + 1) x P`` scored forward
passes per query, no decoding -- 300 x 11 x 5 = 16,500 for the main run, which is
the plan's Sec. 5 arithmetic.

The per-order selections are kept as well, and their Jaccard is reported on
``.stats``. It is the same statistic ``llm_pruner.selection_stability`` reports,
and it costs nothing here because the scores are already in hand. If it comes
back near 1.0 the averaging was insurance; if it comes back low, then a
single-order LOO oracle -- which is what a reader would naively build -- is
ranking position as much as content, and that is a finding in its own right.

The random orderings are keyed on the question, so they are *not* the same draws
the generation grid uses (which are keyed on qid). That is deliberate. An oracle
selected under exactly the orderings it will be evaluated under can pick chunks
that happen to suit those draws, which inflates the ceiling; independent draws
remove the shortcut.

Known limitations, both of which belong in the write-up
------------------------------------------------------
**LOO is additive by construction and will misvalue complementary evidence.**
CUE-R (arXiv 2604.05467) found non-additive interaction in ~20% of the HotpotQA
examples it tested, with ~14% fully complementary -- neither single removal hurt,
joint removal broke the answer. On a 2-hop dataset that is the common case, not
the corner case: drop either gold paragraph and the model may still guess from
the other. So this is a strong ceiling, not a true optimum, and it is the reason
``gold_recall`` is counted below rather than assumed.

**The dynamic range may be small.** `HANDOFF.md` Sec. 5 records a probe where
removing *both* gold paragraphs cost 1.79 nats. If per-chunk drops are of that
order, the ranking is close to noise. ``.stats`` reports the drop distribution
and the count of degenerate queries so that this is measured rather than
believed -- check it before trusting the arm.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..chunks import Chunk, permutation_set
from .base import Pruner, validate_selection

#: The study's P=5 protocol (plan Sec. 4.4). Overridable through `arm_params` so
#: a quick pass can run ``orders: ["rank"]`` at a fifth of the cost, but the
#: default is the real thing -- see the module docstring on why one order is not
#: enough.
DEFAULT_ORDERS = ("rank", "reverse", "random", "random", "random")

#: A drop smaller than this is not evidence. Log-probs are summed over answer
#: tokens, so this is nats over the whole answer, not per token.
NOISE_NATS = 0.05

#: Keep the raw-drop lists bounded. 300 queries x 10 chunks is 3,000 floats and
#: fits easily; a larger grid should not grow memory without limit for a summary
#: statistic. The counters are exact regardless of this cap.
STATS_CAP = 50_000


@dataclass
class OracleStats:
    """The arm's own validity check. Reported, not buried."""

    cells: int = 0
    queries_scored: int = 0
    #: Queries where no chunk's removal hurt by more than NOISE_NATS. The oracle
    #: is ranking noise on these, and its selection is the rank-order tie-break.
    degenerate: int = 0
    #: Cells where the k-th and (k+1)-th chunk are separated by less than
    #: NOISE_NATS: the budget boundary fell inside the noise.
    boundary_ties: int = 0
    gold_kept: int = 0
    gold_total: int = 0
    drops: list[float] = field(default_factory=list)
    spreads: list[float] = field(default_factory=list)
    baselines: list[float] = field(default_factory=list)
    order_jaccards: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "cells": self.cells,
            "queries_scored": self.queries_scored,
            "degenerate": self.degenerate,
            "boundary_ties": self.boundary_ties,
        }
        if self.gold_total:
            d["gold_recall"] = self.gold_kept / self.gold_total
        if self.queries_scored:
            d["degenerate_rate"] = self.degenerate / self.queries_scored
        if self.cells:
            d["boundary_tie_rate"] = self.boundary_ties / self.cells
        if self.drops:
            d["mean_drop"] = statistics.fmean(self.drops)
            d["median_drop"] = statistics.median(self.drops)
            d["max_drop"] = max(self.drops)
            d["min_drop"] = min(self.drops)
            # A chunk whose removal *helps* is real and worth knowing about: it
            # is a distractor the generator was better off without.
            d["negative_drop_rate"] = sum(1 for x in self.drops if x < 0) / len(
                self.drops
            )
        if self.spreads:
            d["mean_within_query_spread"] = statistics.fmean(self.spreads)
        if self.baselines:
            d["mean_baseline_logprob"] = statistics.fmean(self.baselines)
        if self.order_jaccards:
            # 1.0 means the selection did not depend on presentation order.
            d["mean_order_jaccard"] = statistics.fmean(self.order_jaccards)
        return d


@dataclass
class Scored:
    """LOO result for one (query, chunk set). Budget-independent."""

    baseline: float
    mean_drop: dict[int, float]
    per_order: dict[int, list[float]]


class LOOOracle(Pruner):
    name = "loo_oracle"
    #: run.py hands this arm the run's own generator, so every scored forward
    #: pass goes through the same cache as everything else and replays for free.
    needs_generator = True
    #: ...and the reference answers, which `select`'s signature has no room for.
    needs_answers = True

    def __init__(
        self,
        generator: Any = None,
        answers: dict[str, str] | None = None,
        orders: Sequence[str] | None = None,
        seed: int = 20260828,
        template: str | None = None,
    ) -> None:
        self.generator = generator
        self.answers = dict(answers or {})
        self.orders = tuple(orders) if orders else DEFAULT_ORDERS
        self.seed = seed
        self.template = template
        self.stats = OracleStats()
        self._cache: dict[tuple[str, tuple[int, ...]], Scored] = {}

    # ------------------------------------------------------------------ wiring

    def attach(self, generator: Any, params: Any = None, **run_state: Any) -> None:
        """Receive the run's generator and prompt state (called by run.py).

        ``params`` is accepted and ignored: scoring does not decode, so
        max_new_tokens and the greedy guard have nothing to act on here. The
        *template* does matter -- the log-prob has to be conditioned on the same
        prompt the generator will actually see, or the oracle is ranking chunks
        for a prompt that never runs.
        """
        self.generator = generator
        if run_state.get("template") is not None:
            self.template = run_state["template"]
        # Only take the run's permutation protocol if the config did not pin one.
        if run_state.get("orders") and self.orders == DEFAULT_ORDERS:
            self.orders = tuple(run_state["orders"])
        if run_state.get("seed") is not None:
            self.seed = run_state["seed"]

    def attach_answers(self, examples: Sequence[Any]) -> None:
        """Map question -> reference answer (called by run.py).

        Keyed on the question because that is all ``select`` is given. Two
        examples sharing a question but not an answer would silently score one of
        them against the other's gold, so that raises instead.
        """
        mapping: dict[str, str] = {}
        for ex in examples:
            prev = mapping.get(ex.question)
            if prev is not None and prev != ex.answer:
                raise ValueError(
                    "two examples share a question with different answers, so "
                    f"the oracle cannot key on it: {ex.question!r} -> "
                    f"{prev!r} / {ex.answer!r}"
                )
            mapping[ex.question] = ex.answer
        self.answers = mapping

    # --------------------------------------------------------------- selection

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        if self.generator is None:
            raise RuntimeError(
                "loo_oracle has no generator; run.py attaches one via attach(). "
                "Constructing this arm standalone requires passing generator=."
            )
        chunks = list(chunks)
        self.stats.cells += 1
        if budget >= len(chunks):
            return validate_selection([c.idx for c in chunks], chunks, budget)

        scored = self._score(query, chunks)
        # Ties break on the as-given rank: deterministic, content-blind, and it
        # cannot smuggle in signal the log-probs did not provide. This is the
        # path a degenerate query takes, so it has to be boring.
        by_rank = {c.idx: c.rank for c in chunks}
        ranked = sorted(
            (c.idx for c in chunks),
            key=lambda i: (-scored.mean_drop[i], by_rank[i]),
        )
        kept = ranked[:budget]

        self._record(scored, chunks, kept, ranked, budget)
        return validate_selection(sorted(kept), chunks, budget)

    def _score(self, query: str, chunks: Sequence[Chunk]) -> Scored:
        """LOO drops for one (query, chunk set). Budget-independent, so cached.

        The three budgets are three slices of one ranking, not three rankings.
        Without this cache the main grid would ask for the same scores three
        times over, and while the SQLite cache would absorb the cost, the
        distribution counters on ``.stats`` would triple-count.
        """
        key = (query, tuple(c.idx for c in chunks))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        answer = self._answer(query)
        baseline = self._logprobs(query, chunks, answer)

        per_order: dict[int, list[float]] = {}
        for c in chunks:
            without = [x for x in chunks if x.idx != c.idx]
            lp = self._logprobs(query, without, answer)
            per_order[c.idx] = [b - l for b, l in zip(baseline, lp)]

        scored = Scored(
            baseline=statistics.fmean(baseline),
            mean_drop={i: statistics.fmean(d) for i, d in per_order.items()},
            per_order=per_order,
        )
        self._cache[key] = scored

        # The drop distribution is a property of the query, not of the budget,
        # so it is accumulated here -- once -- rather than in `_record`, which
        # runs once per (query, budget) cell and would count each query three
        # times over on the shipped budget list.
        self.stats.queries_scored += 1
        drops = list(scored.mean_drop.values())
        if max(drops) <= NOISE_NATS:
            self.stats.degenerate += 1
        if len(self.stats.drops) < STATS_CAP:
            self.stats.drops.extend(drops)
            self.stats.baselines.append(scored.baseline)
            self.stats.spreads.append(max(drops) - min(drops))
        return scored

    def _logprobs(
        self, query: str, subset: Sequence[Chunk], answer: str
    ) -> list[float]:
        """logP(answer | subset) under each of the P orderings."""
        from ..generate import DEFAULT_TEMPLATE, build_prompt

        template = self.template or DEFAULT_TEMPLATE
        orderings = permutation_set(
            subset, list(self.orders), seed=self.seed, key=query
        )
        return [
            self.generator.score(build_prompt(query, ordering, template), answer)
            for ordering in orderings
        ]

    def _answer(self, query: str) -> str:
        try:
            return self.answers[query]
        except KeyError:
            raise KeyError(
                "loo_oracle has no reference answer for this query; run.py "
                "supplies them via attach_answers(). Constructing this arm "
                "standalone requires passing answers={question: answer}."
            ) from None

    def _record(
        self,
        scored: Scored,
        chunks: Sequence[Chunk],
        kept: Sequence[int],
        ranked: Sequence[int],
        budget: int,
    ) -> None:
        """The budget-dependent half of the diagnostics, one cell at a time.

        Everything that does not depend on k is accumulated in `_score`, which
        runs once per query. HANDOFF Sec. 7 asks for all of it to be measured
        rather than assumed.
        """
        drops = [scored.mean_drop[i] for i in ranked]
        if budget < len(ranked) and drops[budget - 1] - drops[budget] < NOISE_NATS:
            self.stats.boundary_ties += 1

        gold = {c.idx for c in chunks if c.is_gold}
        self.stats.gold_total += min(len(gold), budget)
        self.stats.gold_kept += len(gold & set(kept))

        # Free order-sensitivity check: what would each single presentation order
        # have selected on its own? Same tie-break as the real selection, so a
        # flat query reads as agreement rather than as spurious disagreement.
        by_rank = {c.idx: c.rank for c in chunks}
        sels = [
            frozenset(
                sorted(
                    (c.idx for c in chunks),
                    key=lambda i, j=j: (-scored.per_order[i][j], by_rank[i]),
                )[:budget]
            )
            for j in range(len(self.orders))
        ]
        union = frozenset().union(*sels)
        inter = sels[0].intersection(*sels)
        self.stats.order_jaccards.append(len(inter) / len(union) if union else 1.0)

    def close(self) -> None:
        # The generator is the run's, shared with every other arm; unloading it
        # is not this arm's business. The score cache is.
        self._cache.clear()
