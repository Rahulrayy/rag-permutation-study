"""Arm contracts. Every pruner obeys the same budget rules or the
matched-keep-count comparison against the placebo is meaningless.
"""

import pytest

from src.prune import expand_arms, get_pruner, registered_arms
from src.prune.base import validate_selection

IMPLEMENTED = [
    "full", "nocontext", "random_drop", "placebo_pos",
    "provence_rerank", "provence_full", "rerank_topk", "llm_pruner",
    "llmlingua2", "loo_oracle",
]


def test_all_arms_are_registered():
    assert set(registered_arms()) == {
        "full", "nocontext", "rerank_topk", "provence_rerank", "provence_full",
        "llmlingua2", "llm_pruner", "random_drop", "placebo_pos", "loo_oracle",
    }


@pytest.mark.parametrize("arm", ["random_drop", "placebo_pos"])
@pytest.mark.parametrize("budget", [2, 3, 5])
def test_selection_respects_budget(arm, budget, chunks):
    selected = get_pruner(arm).select("q", chunks, budget)
    assert len(selected) == budget
    assert len(set(selected)) == budget
    assert set(selected) <= {c.idx for c in chunks}


@pytest.mark.parametrize("strategy", ["middle_first", "edges_first", "tail_first"])
def test_placebo_ignores_content(strategy, chunks):
    """The placebo must not look at the query or the text. That is the point:
    it isolates positional promotion from content selection."""
    pruner = get_pruner("placebo_pos", strategy=strategy)
    a = pruner.select("who founded Rome?", chunks, 3)
    b = pruner.select("what colour is the sky?", chunks, 3)
    assert a == b


def test_placebo_strategies_differ(chunks):
    picks = {
        s: tuple(get_pruner("placebo_pos", strategy=s).select("q", chunks, 3))
        for s in ("middle_first", "edges_first", "tail_first")
    }
    assert len(set(picks.values())) == 3


def test_middle_first_keeps_the_edges(chunks):
    kept = get_pruner("placebo_pos", strategy="middle_first").select("q", chunks, 4)
    assert 0 in kept and 9 in kept


def test_edges_first_keeps_the_middle(chunks):
    kept = get_pruner("placebo_pos", strategy="edges_first").select("q", chunks, 4)
    assert 0 not in kept and 9 not in kept


def test_tail_first_is_truncation(chunks):
    assert get_pruner("placebo_pos", strategy="tail_first").select("q", chunks, 4) == [0, 1, 2, 3]


def test_random_drop_is_seeded(chunks):
    a = get_pruner("random_drop", seed=1).select("q", chunks, 3)
    b = get_pruner("random_drop", seed=1).select("q", chunks, 3)
    assert a == b


def test_random_drop_varies_by_query(chunks):
    p = get_pruner("random_drop", seed=1)
    assert p.select("q1", chunks, 3) != p.select("q2", chunks, 3)


def test_full_keeps_everything(chunks):
    assert get_pruner("full").select("q", chunks, 3) == list(range(10))


def test_nocontext_keeps_nothing(chunks):
    assert get_pruner("nocontext").select("q", chunks, 3) == []


def test_validate_rejects_over_budget(chunks):
    with pytest.raises(ValueError, match="budget"):
        validate_selection([0, 1, 2, 3], chunks, 3)


def test_validate_rejects_duplicates(chunks):
    with pytest.raises(ValueError, match="duplicate"):
        validate_selection([0, 0, 1], chunks, 3)


def test_no_arm_is_still_a_stub():
    """Every registered arm is implemented, as of week 3.

    A new arm added as a stub fails here until it is listed, which is the
    reminder to give it a NotImplementedError test of its own in the meantime.
    """
    assert set(registered_arms()) == set(IMPLEMENTED)


# --------------------------------------------------------------------------- #
# Provence. A stub stands in for the checkpoint: these tests cover the arm's
# contract, not the model's quality, and must not download 1.74 GB or touch a
# GPU to run.
# --------------------------------------------------------------------------- #

class _StubProvence:
    """Mimics the real `process()` shape: scores plus sentence-pruned text."""

    def __init__(self, scores, pruned):
        self.scores, self.pruned, self.calls = scores, pruned, 0

    def process(self, question, context, title, **kw):
        self.calls += 1
        return {
            "reranking_score": [self.scores],
            "pruned_context": [self.pruned],
            "compression_rate": [[0.0] * len(self.scores)],
        }


@pytest.fixture
def stub_provence(chunks, monkeypatch):
    from src.prune import provence as mod

    # descending relevance for chunks 9..0, so top-k is unambiguous
    scores = [float(i) for i in range(10)]
    pruned = ["" if i in (9, 8) else f"kept{i}" for i in range(10)]
    stub = _StubProvence(scores, pruned)
    monkeypatch.setattr(mod, "_load", lambda checkpoint, device: stub)
    monkeypatch.setattr(mod, "unload_all", lambda: None)
    return stub


def test_provence_selects_top_k_by_score(stub_provence, chunks):
    """Scores ascend with index, so the top 3 are chunks 9, 8, 7."""
    kept = get_pruner("provence_rerank").select("q", chunks, 3)
    assert sorted(kept) == [7, 8, 9]


def test_provence_rerank_leaves_text_untouched(stub_provence, chunks):
    """The reranker arm must stay content-matched against placebo_pos."""
    p = get_pruner("provence_rerank")
    kept = [c for c in chunks if c.idx in p.select("q", chunks, 3)]
    assert [c.text for c in p.rewrite("q", kept, 3)] == [c.text for c in kept]


def test_provence_full_replaces_text_with_pruned(stub_provence, chunks):
    p = get_pruner("provence_full")
    kept = [c for c in chunks if c.idx in p.select("q", chunks, 3)]
    out = p.rewrite("q", kept, 3)
    assert [c.text for c in out] == ["kept7", "", ""]   # 8 and 9 pruned to empty
    assert [c.idx for c in out] == [c.idx for c in kept]


def test_provence_full_keeps_emptied_chunks_in_place(stub_provence, chunks):
    """A chunk pruned to nothing keeps its slot: dropping it would break the
    keep-count match against placebo_pos, and an empty passage is what Provence
    actually returns."""
    p = get_pruner("provence_full")
    kept = [c for c in chunks if c.idx in p.select("q", chunks, 3)]
    assert len(p.rewrite("q", kept, 3)) == 3


def test_provence_runs_the_model_once_per_query(stub_provence, chunks):
    """The result does not depend on the budget, but run.py visits each query
    once per budget. Without caching the model would run three times over."""
    p = get_pruner("provence_rerank")
    for budget in (2, 3, 5):
        p.select("q", chunks, budget)
    assert stub_provence.calls == 1


def test_provence_arms_share_one_model_load(stub_provence, chunks):
    """1.74 GB twice would not fit alongside the generator on a 6 GB card."""
    from src.prune import provence as mod
    assert mod.ProvenceRerank().checkpoint == mod.ProvenceFull().checkpoint


def test_rewrite_may_not_change_the_chunk_set(chunks):
    from src.prune import validate_rewrite

    with pytest.raises(ValueError, match="chunk count"):
        validate_rewrite(chunks[:3], chunks[:2])
    with pytest.raises(ValueError, match="reordered"):
        validate_rewrite(chunks[:3], list(reversed(chunks[:3])))


def test_default_rewrite_is_identity(chunks):
    assert get_pruner("full").rewrite("q", chunks[:3], 3) == chunks[:3]


def test_unknown_arm_raises():
    with pytest.raises(KeyError):
        get_pruner("magic_pruner")


def test_bare_arm_expands_to_its_variants():
    """A config writing `placebo_pos` means all three positional strategies,
    each as its own arm -- they test three different hypotheses."""
    assert expand_arms(["full", "placebo_pos"]) == [
        "full",
        "placebo_pos:middle_first",
        "placebo_pos:edges_first",
        "placebo_pos:tail_first",
    ]


def test_explicit_variant_is_not_re_expanded():
    assert expand_arms(["placebo_pos:tail_first"]) == ["placebo_pos:tail_first"]


def test_variant_suffix_selects_the_strategy(chunks):
    suffixed = get_pruner("placebo_pos:tail_first").select("q", chunks, 4)
    kwarg = get_pruner("placebo_pos", strategy="tail_first").select("q", chunks, 4)
    assert suffixed == kwarg


def test_unknown_variant_fails_at_construction(chunks):
    """A mistyped strategy must fail before the model loads, not hours in."""
    with pytest.raises(ValueError, match="unknown placebo strategy"):
        get_pruner("placebo_pos:sideways_first")


def test_variant_suffix_rejected_on_arms_without_variants():
    with pytest.raises(ValueError, match="variant suffix"):
        get_pruner("full:something")


# --------------------------------------------------------------------------- #
# rerank_topk. Stubbed like Provence: this is the OAE denominator arm, so its
# *contract* matters more than the checkpoint's quality, and the suite must
# stay offline and fast.
# --------------------------------------------------------------------------- #

class _StubCrossEncoder:
    """Returns one relevance logit per (query, passage) pair, like the real one."""

    def __init__(self, scores):
        self.scores, self.calls, self.seen_pairs = scores, 0, None

    def __call__(self, queries, passages, **kw):
        self.calls += 1
        self.seen_pairs = list(zip(queries, passages))
        return self.scores


@pytest.fixture
def stub_reranker(monkeypatch):
    import torch

    from src.prune import rerank_topk as mod

    stub = _StubCrossEncoder(None)

    class _Tok:
        def __call__(self, queries, passages, **kw):
            stub.calls += 1
            stub.seen_pairs = list(zip(queries, passages))
            return _Enc()

    class _Enc(dict):
        """Real tokenizer output is a dict-like BatchEncoding, so `model(**enc)`
        works. A plain object would not unpack."""

        def to(self, device):
            return self

    class _Model:
        device = "cpu"

        def __call__(self, **enc):
            n = len(stub.seen_pairs)
            # ascending relevance with index, so top-k is unambiguous
            return type("O", (), {"logits": torch.arange(n, dtype=torch.float).unsqueeze(-1)})

    monkeypatch.setattr(mod, "_load", lambda model, device: (_Tok(), _Model()))
    monkeypatch.setattr(mod, "unload_all", lambda: None)
    return stub


def test_rerank_selects_top_k(stub_reranker, chunks):
    """Stub scores ascend with index, so the top 3 are chunks 9, 8, 7."""
    assert get_pruner("rerank_topk").select("q", chunks, 3) == [7, 8, 9]


def test_rerank_scores_title_and_text(stub_reranker, chunks):
    """Titles carry real signal on HotpotQA -- the answer is often the title of
    a gold paragraph -- and are part of what the generator is shown."""
    get_pruner("rerank_topk").select("who?", chunks, 3)
    q, passage = stub_reranker.seen_pairs[0]
    assert q == "who?"
    assert passage.startswith("title0: ")


def test_rerank_runs_the_model_once_per_query(stub_reranker, chunks):
    """Scores do not depend on the budget; run.py visits each query per budget."""
    p = get_pruner("rerank_topk")
    for budget in (2, 3, 5):
        p.select("q", chunks, budget)
    assert stub_reranker.calls == 1


def test_rerank_selection_order_carries_no_meaning(stub_reranker, chunks):
    """Returning score order would invite someone downstream to read it as the
    reranker's ranking -- the selection/ordering conflation this study is about."""
    assert get_pruner("rerank_topk").select("q", chunks, 4) == sorted(
        get_pruner("rerank_topk").select("q", chunks, 4)
    )


def test_rerank_does_not_rewrite_content(stub_reranker, chunks):
    """Selection-only: it must stay content-matched against placebo_pos."""
    p = get_pruner("rerank_topk")
    kept = [c for c in chunks if c.idx in p.select("q", chunks, 3)]
    assert p.rewrite("q", kept, 3) == kept


# --------------------------------------------------------------------------- #
# llm_pruner. The arm whose *selection* may depend on the order it was shown,
# and the only one that has to repair a malformed reply without quietly
# changing what the arm is.
# --------------------------------------------------------------------------- #

class _ScriptedGenerator:
    """Returns canned replies, so parsing and repair are tested without a GPU."""

    model = "scripted"

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, params):
        self.prompts.append(prompt)
        from src.cache import CachedGeneration
        return CachedGeneration(text=self.replies.pop(0) if self.replies else "")


def _pruner(*replies, **kw):
    from src.generate import DecodeParams
    from src.prune.llm_pruner import LLMPruner

    p = LLMPruner(**kw)
    gen = _ScriptedGenerator(*replies)
    p.attach(gen, DecodeParams())
    return p, gen


def test_llm_pruner_parses_a_comma_separated_reply(chunks):
    p, _ = _pruner("2, 5, 9")
    assert p.select("q", chunks, 3) == [1, 4, 8]     # 1-indexed reply -> idx
    assert p.stats.under_selected == 0


def test_llm_pruner_truncates_over_selection(chunks):
    """validate_selection would raise; the arm repairs and counts instead."""
    p, _ = _pruner("1, 2, 3, 4, 5, 6")
    assert p.select("q", chunks, 3) == [0, 1, 2]
    assert p.stats.over_selected == 1


def test_llm_pruner_fills_under_selection_to_keep_the_budget_matched(chunks):
    """An arm quietly returning k-1 is no longer matched against placebo_pos,
    which is the study's centerpiece."""
    p, _ = _pruner("7")
    kept = p.select("q", chunks, 3)
    assert len(kept) == 3
    assert 6 in kept                      # the model's pick is honoured
    assert p.stats.under_selected == 1


def test_llm_pruner_handles_an_unparseable_reply(chunks):
    p, _ = _pruner("I cannot help with that.")
    assert len(p.select("q", chunks, 3)) == 3
    assert p.stats.unparseable == 1 and p.stats.under_selected == 1


def test_llm_pruner_ignores_out_of_range_numbers(chunks):
    p, _ = _pruner("3, 47, 5")
    kept = p.select("q", chunks, 3)
    assert 2 in kept and 4 in kept
    assert p.stats.out_of_range == 1


def test_llm_pruner_maps_numbers_through_the_presented_order(chunks):
    """Reply numbers index the order the model was SHOWN, not chunk.idx. With
    reverse presentation, "1" is the last chunk by rank."""
    p, _ = _pruner("1", selection_order="reverse")
    assert 9 in p.select("q", chunks, 1)


def test_llm_pruner_selection_stays_greedy(chunks):
    """Sampling here would put sampling noise inside the independent variable."""
    p, _ = _pruner("1, 2, 3")
    assert p._params.do_sample is False
    assert p._params.temperature == 0.0
    assert p._params.max_new_tokens == 64      # more room than an answer needs


def test_llm_pruner_without_a_generator_says_so(chunks):
    from src.prune.llm_pruner import LLMPruner

    with pytest.raises(RuntimeError, match="no generator"):
        LLMPruner().select("q", chunks, 3)


def test_llm_pruner_caches_per_query_and_budget(chunks):
    p, gen = _pruner("1, 2, 3")
    p.select("q", chunks, 3)
    p.select("q", chunks, 3)
    assert len(gen.prompts) == 1


def test_selection_stability_detects_order_dependence(chunks):
    """The study's thesis, applied to the pruner: if selection moves with
    presentation order, a published LLM-pruner result is one draw from a
    distribution its paper does not mention."""
    from src.prune.llm_pruner import selection_stability

    # same reply "1,2,3" under every presentation -> different chunks each time
    p, _ = _pruner("1,2,3", "1,2,3", "1,2,3")
    out = selection_stability(p, "q", chunks, 3, orders=("rank", "reverse"))
    assert out["jaccard"] < 1.0 and out["stable"] is False

    p2, _ = _pruner("1,2,3", "3,2,1")
    same = selection_stability(p2, "q", chunks, 3, orders=("rank", "rank"))
    assert same["jaccard"] == 1.0 and same["stable"] is True


# --------------------------------------------------------------------------- #
# llmlingua2. Rate-based, so it breaks the keep-k mould deliberately; the tests
# pin the decisions that were made before the arm ran.
# --------------------------------------------------------------------------- #

class _StubCompressor:
    def __init__(self):
        self.calls = []

    def compress_prompt(self, text, rate, force_tokens, use_context_level_filter):
        self.calls.append({"text": text, "rate": rate,
                           "context_filter": use_context_level_filter})
        return {"compressed_prompt": text[: max(1, int(len(text) * rate))]}


@pytest.fixture
def stub_lingua(monkeypatch):
    from src.prune import llmlingua2 as mod

    stub = _StubCompressor()
    monkeypatch.setattr(mod, "_load", lambda model, device: stub)
    monkeypatch.setattr(mod, "unload_all", lambda: None)
    return stub


def test_llmlingua2_keeps_every_chunk(stub_lingua, chunks):
    """It has no chunk ranking; the budget is spent on compression instead."""
    assert get_pruner("llmlingua2").select("q", chunks, 3) == list(range(10))


def test_llmlingua2_is_not_keep_k_matched(stub_lingua):
    """Analysis code must not pool it with the selection arms."""
    assert get_pruner("llmlingua2").budget_is_keep_count is False
    assert get_pruner("full").budget_is_keep_count is False
    assert get_pruner("rerank_topk").budget_is_keep_count is True


def test_llmlingua2_budget_becomes_a_rate(stub_lingua, chunks):
    p = get_pruner("llmlingua2")
    assert p.rate_for(3, 10) == pytest.approx(0.3)
    assert p.rate_for(5, 10) == pytest.approx(0.5)
    p.rewrite("q", chunks, 3)
    assert all(c["rate"] == pytest.approx(0.3) for c in stub_lingua.calls)


def test_llmlingua2_compresses_each_chunk_independently(stub_lingua, chunks):
    """Joint compression makes content a function of order -- measured at 0/100
    chunks surviving identically across orderings. Never enable the
    cross-context filter here."""
    get_pruner("llmlingua2").rewrite("q", chunks, 3)
    assert len(stub_lingua.calls) == 10                      # one call per chunk
    assert all(c["context_filter"] is False for c in stub_lingua.calls)


def test_llmlingua2_rewrite_preserves_the_chunk_set(stub_lingua, chunks):
    out = get_pruner("llmlingua2").rewrite("q", chunks, 3)
    assert [c.idx for c in out] == [c.idx for c in chunks]
    assert all(len(o.text) < len(c.text) for o, c in zip(out, chunks))


def test_llmlingua2_caches_per_chunk_and_rate(stub_lingua, chunks):
    p = get_pruner("llmlingua2")
    p.rewrite("q", chunks, 3)
    p.rewrite("other query", chunks, 3)   # same chunks, same rate -> no new work
    assert len(stub_lingua.calls) == 10
    p.rewrite("q", chunks, 5)             # new rate -> recompressed
    assert len(stub_lingua.calls) == 20


# --------------------------------------------------------------------------- #
# loo_oracle. The ceiling. It reads the gold answer, so it is not a method and
# its arithmetic has to be exactly right: a wrong ceiling silently rescales
# every Oracle Gap in the study.
# --------------------------------------------------------------------------- #

class _ScriptedScorer:
    """A generator whose answer log-prob is a known function of the context.

    Each chunk contributes a fixed number of nats, so the leave-one-out drop for
    a chunk is exactly its value and the oracle's ranking is checkable by hand.
    With ``position_weight`` above zero the contribution decays down the context,
    which is the confound the arm averages over orderings to remove.
    """

    model = "scripted-scorer"

    def __init__(self, values, position_weight=0.0):
        self.values = values
        self.position_weight = position_weight
        self.calls = []

    @staticmethod
    def _rendered_order(prompt):
        seen = [(prompt.index(f"title{i}:"), i) for i in range(10)
                if f"title{i}:" in prompt]
        return [i for _, i in sorted(seen)]

    def score(self, prompt, answer):
        self.calls.append(prompt)
        total = 0.0
        for slot, idx in enumerate(self._rendered_order(prompt)):
            total += self.values.get(idx, 0.0) * (1.0 - self.position_weight * slot)
        return total


def _oracle(values, position_weight=0.0, **kw):
    from src.prune.loo_oracle import LOOOracle

    gen = _ScriptedScorer(values, position_weight)
    p = LOOOracle(generator=gen, answers={"q": "the answer"}, **kw)
    return p, gen


def test_oracle_keeps_the_chunks_whose_removal_hurts_most(chunks):
    p, _ = _oracle({2: 5.0, 5: 4.0, 7: 3.0})
    assert p.select("q", chunks, 3) == [2, 5, 7]


def test_oracle_drop_is_the_lost_logprob_not_the_raw_score(chunks):
    """A chunk the generator is better off without must rank last, not first."""
    p, _ = _oracle({0: -6.0, 2: 5.0, 5: 4.0, 7: 3.0, 9: 1.0})
    assert p.select("q", chunks, 3) == [2, 5, 7]
    assert 0 not in p.select("q", chunks, 9)[:1]
    assert p.stats.as_dict()["negative_drop_rate"] > 0


def test_oracle_ties_break_on_the_as_given_order(chunks):
    """Every drop equal: the selection must be the boring deterministic one."""
    p, _ = _oracle({})
    assert p.select("q", chunks, 3) == [0, 1, 2]
    assert p.stats.degenerate == 1


def test_oracle_counts_the_drop_distribution_once_per_query(chunks):
    """`_record` runs once per (query, budget) cell and the shipped config has
    three budgets, so anything budget-independent has to be counted where the
    scoring happens or every distribution statistic triples."""
    p, _ = _oracle({})
    for budget in (2, 3, 5):
        p.select("q", chunks, budget)
    assert p.stats.cells == 3
    assert p.stats.queries_scored == 1
    assert p.stats.degenerate == 1           # not 3
    assert len(p.stats.drops) == 10          # not 30
    assert len(p.stats.baselines) == 1
    assert p.stats.as_dict()["degenerate_rate"] == 1.0


def test_oracle_selection_is_sorted_and_within_budget(chunks):
    p, _ = _oracle({9: 5.0, 5: 4.0, 1: 3.0})
    selected = p.select("q", chunks, 3)
    assert selected == sorted(selected)
    assert len(set(selected)) == 3


def test_oracle_scores_every_leave_one_out_context_under_every_ordering(chunks):
    p, gen = _oracle({2: 5.0})
    p.select("q", chunks, 3)
    # (10 leave-one-out contexts + 1 baseline) x P orderings
    assert len(gen.calls) == 11 * 5


def test_oracle_scores_once_across_budgets(chunks):
    """Three budgets are three slices of one ranking, not three rankings."""
    p, gen = _oracle({2: 5.0, 5: 4.0})
    p.select("q", chunks, 2)
    p.select("q", chunks, 3)
    p.select("q", chunks, 5)
    assert len(gen.calls) == 11 * 5
    assert p.stats.queries_scored == 1
    assert p.stats.cells == 3


def test_oracle_at_or_above_full_budget_costs_nothing(chunks):
    p, gen = _oracle({2: 5.0})
    assert p.select("q", chunks, 10) == list(range(10))
    assert gen.calls == []


def test_oracle_averages_the_drop_over_orderings(chunks):
    """Position-dependent scoring must not decide the selection on its own.

    Chunk 9 is worth more content but sits last in as-given order, so a
    single-order oracle prefers chunk 0. Averaging over the P orderings, which
    put each of them first, second and last in turn, recovers the content
    ranking.
    """
    values = {0: 3.0, 9: 4.0}
    averaged, _ = _oracle(values, position_weight=0.15)
    single, _ = _oracle(values, position_weight=0.15, orders=["rank"])
    assert averaged.select("q", chunks, 1) == [9]
    assert single.select("q", chunks, 1) == [0]


def test_oracle_reports_its_own_order_sensitivity(chunks):
    """The statistic llm_pruner reports, turned on the oracle. Free: the
    per-order scores are already in hand."""
    flat, _ = _oracle({2: 5.0, 5: 4.0, 7: 3.0})
    flat.select("q", chunks, 3)
    assert flat.stats.as_dict()["mean_order_jaccard"] == 1.0

    positional, _ = _oracle({i: 1.0 + i * 0.1 for i in range(10)},
                            position_weight=0.2)
    positional.select("q", chunks, 3)
    assert positional.stats.as_dict()["mean_order_jaccard"] < 1.0


def test_oracle_counts_gold_recall(chunks):
    """If the ceiling cannot find the gold paragraphs it is not a ceiling."""
    found, _ = _oracle({2: 5.0, 5: 4.0})          # conftest golds are 2 and 5
    found.select("q", chunks, 3)
    assert found.stats.as_dict()["gold_recall"] == 1.0

    missed, _ = _oracle({1: 5.0, 3: 4.0, 4: 3.0})
    missed.select("q", chunks, 3)
    assert missed.stats.as_dict()["gold_recall"] == 0.0


def test_oracle_counts_a_budget_boundary_inside_the_noise(chunks):
    p, _ = _oracle({2: 5.0, 5: 4.0, 7: 3.0, 8: 3.0 - 1e-6})
    p.select("q", chunks, 3)                      # 3rd and 4th are indistinguishable
    assert p.stats.boundary_ties == 1


def test_oracle_without_a_generator_says_so(chunks):
    from src.prune.loo_oracle import LOOOracle

    with pytest.raises(RuntimeError, match="attach"):
        LOOOracle(answers={"q": "a"}).select("q", chunks, 3)


def test_oracle_without_an_answer_says_so(chunks):
    from src.prune.loo_oracle import LOOOracle

    p = LOOOracle(generator=_ScriptedScorer({}))
    with pytest.raises(KeyError, match="attach_answers"):
        p.select("q", chunks, 3)


def test_oracle_attach_answers_rejects_a_colliding_question():
    """Keying on the question is only safe if it is unique; otherwise one
    example would be scored against another's gold, silently."""
    from dataclasses import dataclass

    from src.prune.loo_oracle import LOOOracle

    @dataclass
    class _Ex:
        qid: str
        question: str
        answer: str

    p = LOOOracle()
    p.attach_answers([_Ex("1", "who?", "Ada"), _Ex("2", "who?", "Ada")])
    assert p.answers == {"who?": "Ada"}

    with pytest.raises(ValueError, match="different answers"):
        p.attach_answers([_Ex("1", "who?", "Ada"), _Ex("2", "who?", "Grace")])


def test_oracle_scores_against_the_runs_prompt_template(chunks):
    """The log-prob has to be conditioned on the prompt the generator will
    actually see, or the oracle ranks chunks for a prompt that never runs."""
    from src.generate import ALT_TEMPLATE
    from src.prune.loo_oracle import LOOOracle

    gen = _ScriptedScorer({2: 5.0})
    p = LOOOracle(answers={"q": "a"})
    p.attach(gen, None, template=ALT_TEMPLATE, orders=["rank"], seed=7)
    p.select("q", chunks, 3)
    assert p.orders == ("rank",)
    assert p.seed == 7
    assert all(c.startswith(ALT_TEMPLATE[:20]) for c in gen.calls)


def test_oracle_keeps_a_config_pinned_ordering_over_the_runs(chunks):
    from src.prune.loo_oracle import LOOOracle

    p = LOOOracle(answers={"q": "a"}, orders=["rank"])
    p.attach(_ScriptedScorer({}), None, orders=["rank", "reverse", "random"])
    assert p.orders == ("rank",)


def test_oracle_does_not_rewrite_content(chunks):
    p, _ = _oracle({2: 5.0})
    kept = [c for c in chunks if c.idx in p.select("q", chunks, 3)]
    assert p.rewrite("q", kept, 3) == kept


def test_llmlingua2_does_not_alias_chunks_across_queries(stub_lingua):
    """Two different passages in the same slot must not share compressed text.

    Regression for a defect that silently invalidated a whole arm. `run.py` is
    arm-major: one pruner instance serves every query in the run. The cache was
    keyed on (chunk idx, rate), and rate is k/n which is constant across
    queries, so the arm had only ~30 distinct keys for the entire dataset.
    Query 2 onward received query 1's compressed passages, while `n_gold_kept`
    kept reporting the correct count because it reads chunk metadata rather
    than text. That combination is what makes it invisible: the arm looks like
    it retained the gold passages and simply answered badly.

    One query's chunks are not enough to catch it. The test has to reuse a slot
    across two different queries, which is what the run does.
    """
    from src.chunks import Chunk

    query_a = [Chunk(idx=i, title=f"a{i}", text=f"alpha passage {i} " * 12, rank=i)
               for i in range(10)]
    query_b = [Chunk(idx=i, title=f"b{i}", text=f"bravo passage {i} " * 12, rank=i)
               for i in range(10)]

    pruner = get_pruner("llmlingua2")
    out_a = pruner.rewrite("question a", query_a, 3)
    out_b = pruner.rewrite("question b", query_b, 3)

    # Every compressed passage must derive from its own source, not the slot's
    # first occupant.
    for chunk, compressed in zip(query_b, out_b):
        assert "bravo" in compressed.text, (
            f"chunk {chunk.idx} of the second query came back as "
            f"{compressed.text!r}, which is the first query's content"
        )
    assert [c.text for c in out_a] != [c.text for c in out_b]
    # 20 distinct passages, so 20 compressions: the cache must not have
    # collapsed the second query onto the first.
    assert len(stub_lingua.calls) == 20


def test_llmlingua2_still_memoizes_identical_text(stub_lingua, chunks):
    """The fix must not throw the cache away: identical text is compressed once."""
    pruner = get_pruner("llmlingua2")
    pruner.rewrite("q", chunks, 3)
    pruner.rewrite("a different query", chunks, 3)   # same text, same rate
    assert len(stub_lingua.calls) == 10
