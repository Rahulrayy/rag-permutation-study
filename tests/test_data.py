"""Loader invariants.

Only the test that actually downloads HotpotQA carries the `network` marker.
Marking the whole module would hide the subsampling and memorization-filter
tests -- which are pure functions of in-memory fixtures -- from the fast path
that gets run on every change.

    pytest -m "not network"    # skip the download
"""

import pytest

from src.chunks import Chunk
from src.data import Example, memorization_filter, subsample


def _fake(n, hop_types=("bridge", "comparison")):
    return [
        Example(
            qid=f"q{i}",
            question=f"question {i}",
            answer="a",
            chunks=[Chunk(idx=j, title=f"t{j}", text="x", rank=j) for j in range(10)],
            gold_chunk_ids=[0, 1],
            hop_type=hop_types[i % len(hop_types)],
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("n", [10, 50, 100])
def test_subsample_is_nested(n):
    """The pilot set must be a strict prefix of the main set at the same seed,
    or week-1 numbers are not comparable to week-4 numbers."""
    pop = _fake(500)
    small = [e.qid for e in subsample(pop, n, seed=1, stratify_by="hop_type")]
    large = [e.qid for e in subsample(pop, 200, seed=1, stratify_by="hop_type")]
    assert large[:n] == small


def test_subsample_is_deterministic():
    pop = _fake(200)
    a = [e.qid for e in subsample(pop, 50, seed=1, stratify_by="hop_type")]
    b = [e.qid for e in subsample(pop, 50, seed=1, stratify_by="hop_type")]
    assert a == b


def test_subsample_is_stratified():
    from collections import Counter

    pop = _fake(300, hop_types=("bridge", "bridge", "bridge", "comparison"))
    got = Counter(e.hop_type for e in subsample(pop, 100, seed=1, stratify_by="hop_type"))
    assert got["comparison"] == pytest.approx(25, abs=2)


def test_memorization_filter_drops_correct_nocontext_answers():
    """Keep only what the model gets WRONG without context."""
    pop = _fake(3)
    preds = {"q0": "a", "q1": "wrong", "q2": "a"}
    assert [e.qid for e in memorization_filter(pop, preds)] == ["q1"]


def test_memorization_filter_requires_predictions():
    with pytest.raises(KeyError):
        memorization_filter(_fake(2), {"q0": "a"})


@pytest.mark.network
@pytest.mark.slow
def test_hotpotqa_loads_with_fixed_context_size():
    from src.data import load_dataset

    examples = load_dataset("hotpotqa_distractor", n_queries=20, stratify_by="hop_type")
    assert len(examples) == 20
    assert all(len(e.chunks) == 10 for e in examples)
    assert all(len(e.gold_chunk_ids) == 2 for e in examples)
    assert all(e.hop_type in ("bridge", "comparison") for e in examples)


def test_subsample_of_the_whole_population_is_still_ordered():
    """The prefix property must hold at n == len too, not just below it."""
    pop = _fake(60)
    whole = [e.qid for e in subsample(pop, 60, seed=1, stratify_by="hop_type")]
    part = [e.qid for e in subsample(pop, 20, seed=1, stratify_by="hop_type")]
    assert whole[:20] == part


def test_subsample_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        subsample(_fake(10), 0, seed=1)


def test_memorization_filter_threshold_is_configurable():
    """ANALYSIS_PLAN Sec. 3 leaves EM vs a token-F1 cutoff open; both must be
    expressible without rewriting the comparison."""
    from src.metrics import token_f1

    pop = _fake(2)
    for e in pop:
        e.answer = "Vilnius Old Town"
    preds = {"q0": "Old Town", "q1": "Paris"}  # token-F1 0.8 and 0.0

    strict = [e.qid for e in memorization_filter(pop, preds, correct_fn=token_f1)]
    loose = [
        e.qid
        for e in memorization_filter(pop, preds, correct_fn=token_f1, threshold=0.6)
    ]
    assert strict == ["q0", "q1"]  # only a perfect answer counts as recalled
    assert loose == ["q1"]         # partial recall counts too


def _fake_with_gold(n_gold, hop_type="bridge_comparison"):
    return Example(
        qid=f"q-{n_gold}",
        question="q",
        answer="a",
        chunks=[Chunk(idx=j, title=f"t{j}", text="x", rank=j) for j in range(10)],
        gold_chunk_ids=list(range(n_gold)),
        hop_type=hop_type,
    )


def test_require_two_gold_drops_rows_with_four(capsys):
    """The registered exclusion (ANALYSIS_PLAN Sec. 3), which 2Wiki triggers.

    `bridge_comparison` rows carry four gold paragraphs. Keeping them would make
    a matched-keep-count comparison mean something different on those rows: at
    k=2 they cannot retain all their evidence even in principle.
    """
    from src.data import _require_two_gold

    population = [_fake_with_gold(2, "comparison"), _fake_with_gold(4), _fake_with_gold(2)]
    kept = _require_two_gold(population)

    assert [e.qid for e in kept] == ["q-2", "q-2"]
    # The count is reported, never silently absorbed.
    assert "without exactly 2 gold paragraphs" in capsys.readouterr().out


def test_require_two_gold_is_a_no_op_when_every_row_has_two():
    """It must not disturb HotpotQA, whose 7,345 rows all carry exactly two.

    The filter is applied to every dataset rather than only to 2Wiki, so this is
    the property that keeps the completed main run's population intact.
    """
    from src.data import _require_two_gold

    population = [_fake_with_gold(2) for _ in range(5)]
    assert _require_two_gold(population) == population


def test_require_two_gold_refuses_to_empty_the_population():
    from src.data import _require_two_gold

    with pytest.raises(ValueError, match="no rows with exactly 2 gold"):
        _require_two_gold([_fake_with_gold(4), _fake_with_gold(3)])


def test_chunks_from_context_is_shared_by_both_loaders():
    """One text builder, because the text is what the cache key hashes.

    Two copies that drifted by a stripped space would split the cache and make
    the two datasets quietly incomparable.
    """
    from src.data import _chunks_from_context

    context = {"title": ["A", "B"], "sentences": [[" one.", " two."], [" three."]]}
    chunks = _chunks_from_context(context, gold_titles={"B"})

    assert [c.text for c in chunks] == ["one. two.", "three."]
    assert [c.idx for c in chunks] == [0, 1]
    assert [c.rank for c in chunks] == [0, 1]
    assert [c.is_gold for c in chunks] == [False, True]


@pytest.mark.network
@pytest.mark.slow
def test_2wikimultihop_loads_with_the_same_shape_as_hotpotqa():
    """The second dataset has to be a drop-in, or it is a second protocol."""
    from src.data import load_dataset

    examples = load_dataset("2wikimultihop", n_queries=20, stratify_by="hop_type")
    assert len(examples) == 20
    assert all(len(e.chunks) == 10 for e in examples)
    # The registered exclusion has already run, so nothing four-gold survives.
    assert all(len(e.gold_chunk_ids) == 2 for e in examples)
    assert all(e.hop_type in ("comparison", "compositional", "inference") for e in examples)
    assert all(e.question and e.answer for e in examples)
