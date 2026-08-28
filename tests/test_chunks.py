"""Permutation must be reproducible, distinct, and content-preserving.

If any of these break, the cache silently misses, the paired statistics come
unpaired, and P=5 quietly becomes P=3.
"""

import pytest

from src.chunks import Chunk, keep, permutation_set, permute, positional_bucket

STRATEGIES = ["rank", "reverse", "random", "random", "random"]


def test_rank_is_dataset_order(chunks):
    assert [c.idx for c in permute(chunks, "rank", seed=1)] == list(range(10))


def test_reverse_is_exact_mirror(chunks):
    assert [c.idx for c in permute(chunks, "reverse", seed=1)] == list(reversed(range(10)))


def test_permutation_is_reproducible(chunks):
    a = permutation_set(chunks, STRATEGIES, seed=20260828)
    b = permutation_set(chunks, STRATEGIES, seed=20260828)
    assert [[c.idx for c in p] for p in a] == [[c.idx for c in p] for p in b]


def test_random_replicates_are_distinct(chunks):
    """Three random permutations must not collapse to one ordering."""
    perms = permutation_set(chunks, STRATEGIES, seed=20260828)
    randoms = {tuple(c.idx for c in p) for p in perms[2:]}
    assert len(randoms) == 3


def test_different_seeds_give_different_orders(chunks):
    a = permutation_set(chunks, STRATEGIES, seed=1)
    b = permutation_set(chunks, STRATEGIES, seed=2)
    assert [[c.idx for c in p] for p in a[2:]] != [[c.idx for c in p] for p in b[2:]]


def test_permutation_preserves_content(chunks):
    """Content is held fixed; only order varies. That is the entire design."""
    for perm in permutation_set(chunks, STRATEGIES, seed=7):
        assert sorted(c.idx for c in perm) == list(range(10))
        assert {c.text for c in perm} == {c.text for c in chunks}


def test_permute_does_not_mutate_input(chunks):
    before = [c.idx for c in chunks]
    permute(chunks, "random", seed=3)
    assert [c.idx for c in chunks] == before


def test_keep_preserves_requested_order(chunks):
    assert [c.idx for c in keep(chunks, [7, 2, 5])] == [7, 2, 5]


def test_keep_rejects_unknown_index(chunks):
    with pytest.raises(KeyError):
        keep(chunks, [99])


def test_unknown_strategy_raises(chunks):
    with pytest.raises(ValueError):
        permute(chunks, "spiral", seed=1)


def test_positional_buckets():
    assert [positional_bucket(i, 5) for i in range(5)] == [
        "begin", "middle", "middle", "middle", "end"
    ]
