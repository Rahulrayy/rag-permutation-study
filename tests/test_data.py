"""Loader invariants. Marked `network` -- these download HotpotQA on first run.

    pytest -m "not network"    # skip
"""

import pytest

from src.data import Example, memorization_filter, subsample
from src.chunks import Chunk

pytestmark = pytest.mark.network


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


@pytest.mark.slow
def test_hotpotqa_loads_with_fixed_context_size():
    from src.data import load_dataset

    examples = load_dataset("hotpotqa_distractor", n_queries=20, stratify_by="hop_type")
    assert len(examples) == 20
    assert all(len(e.chunks) == 10 for e in examples)
    assert all(len(e.gold_chunk_ids) == 2 for e in examples)
    assert all(e.hop_type in ("bridge", "comparison") for e in examples)
