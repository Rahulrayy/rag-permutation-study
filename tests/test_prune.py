"""Arm contracts. Every pruner obeys the same budget rules or the
matched-keep-count comparison against the placebo is meaningless.
"""

import pytest

from src.prune import expand_arms, get_pruner, registered_arms
from src.prune.base import validate_selection

IMPLEMENTED = [
    "full", "nocontext", "random_drop", "placebo_pos",
    "provence_rerank", "provence_full", "rerank_topk",
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


def test_unimplemented_arms_raise_not_implemented(chunks):
    for arm in set(registered_arms()) - set(IMPLEMENTED):
        with pytest.raises(NotImplementedError):
            get_pruner(arm).select("q", chunks, 3)


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
    assert [c.text for c in p.rewrite("q", kept)] == [c.text for c in kept]


def test_provence_full_replaces_text_with_pruned(stub_provence, chunks):
    p = get_pruner("provence_full")
    kept = [c for c in chunks if c.idx in p.select("q", chunks, 3)]
    out = p.rewrite("q", kept)
    assert [c.text for c in out] == ["kept7", "", ""]   # 8 and 9 pruned to empty
    assert [c.idx for c in out] == [c.idx for c in kept]


def test_provence_full_keeps_emptied_chunks_in_place(stub_provence, chunks):
    """A chunk pruned to nothing keeps its slot: dropping it would break the
    keep-count match against placebo_pos, and an empty passage is what Provence
    actually returns."""
    p = get_pruner("provence_full")
    kept = [c for c in chunks if c.idx in p.select("q", chunks, 3)]
    assert len(p.rewrite("q", kept)) == 3


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
    assert get_pruner("full").rewrite("q", chunks[:3]) == chunks[:3]


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
    assert p.rewrite("q", kept) == kept
