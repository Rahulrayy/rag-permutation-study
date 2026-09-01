"""Dataset loaders and the memorization filter.

Sec. 4.1. HotpotQA distractor is the primary set precisely because it ships ten
paragraphs per question with two gold: no corpus, no index, no retrieval, and
positions that are meaningful and permutable by construction. That single
decision is what makes this fit on a laptop.

A note on what ``Chunk.rank`` means here. HotpotQA distractor has no retriever,
so there is no retriever rank to speak of: ``rank`` is the order the dataset
ships the paragraphs in, and the ``rank`` permutation strategy reproduces that
as-given order. It is still the right reference ordering -- it is the one any
evaluation on this dataset implicitly uses -- but the write-up should say
"as-given order" rather than "retriever rank" for HotpotQA, and reserve the
latter for the NQ-open arm where a real retriever produces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

from .chunks import Chunk

DatasetName = Literal["hotpotqa_distractor", "2wikimultihop", "nq_open_top10"]

# Default to the standard HF location (~/.cache/huggingface), NOT a path
# inside the project: this repo lives under OneDrive, and a 6 GB checkpoint
# in a synced folder gets uploaded to the cloud in the background.
DEFAULT_HF_CACHE = None


@dataclass
class Example:
    qid: str
    question: str
    answer: str
    chunks: list[Chunk]
    gold_chunk_ids: list[int] = field(default_factory=list)
    hop_type: str | None = None  # stratification key, where the dataset labels it
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Subsampling
# --------------------------------------------------------------------------- #

def _stratified_order(
    keys: Sequence[str],
    seed: int,
) -> list[int]:
    """A deterministic global ordering whose every prefix is roughly stratified.

    Why not "shuffle, then allocate per-stratum quotas": quota rounding at n=100
    and at n=300 can select different examples, so the pilot set would not be a
    subset of the main set and week-1 numbers would not be comparable to week-4
    numbers.

    Instead each stratum is shuffled once, then the strata are interleaved by
    round-robin weighted by stratum size. Taking the first N of the result is
    both nested (prefix property, exactly) and proportional (to within one
    example per stratum, at every N).
    """
    import random

    strata: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        strata.setdefault(key, []).append(i)

    for key, members in strata.items():
        # str seeding goes through sha512: deterministic, PYTHONHASHSEED-independent.
        random.Random(f"{seed}:{key}").shuffle(members)

    total = len(keys)
    # Position each stratum's j-th member on a shared [0, 1) line, spaced by that
    # stratum's share. Sorting by that position interleaves proportionally.
    ranked: list[tuple[float, str, int]] = []
    for key, members in strata.items():
        step = total / len(members)
        for j, idx in enumerate(members):
            ranked.append(((j + 0.5) * step, key, idx))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [idx for _, _, idx in ranked]


def subsample(
    examples: Sequence[Example],
    n_queries: int | None,
    seed: int = 20260828,
    stratify_by: str | None = None,
) -> list[Example]:
    """Seeded, stratified, and nested across n. See ``_stratified_order``."""
    if n_queries is not None and n_queries <= 0:
        raise ValueError(f"n_queries must be positive, got {n_queries}")

    # The full set goes through the same ordering as any subset. Returning the
    # natural order here instead would break the prefix property at exactly one
    # boundary: n=300 would not be a prefix of the whole population, only of
    # every proper subset of it.
    if stratify_by is None:
        keys = ["_all"] * len(examples)
    elif stratify_by == "hop_type":
        keys = [ex.hop_type or "_unlabelled" for ex in examples]
    else:
        keys = [str(ex.meta.get(stratify_by, "_unlabelled")) for ex in examples]

    order = _stratified_order(keys, seed)
    limit = len(examples) if n_queries is None else min(n_queries, len(examples))
    return [examples[i] for i in order[:limit]]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _chunks_from_context(context: dict, gold_titles: set) -> list[Chunk]:
    """Build the chunk list from a `{title[], sentences[][]}` context column.

    Shared by both multi-hop loaders, which ship the identical column layout.
    Kept as one function on purpose: the text here is what gets hashed into the
    generation cache key, so two copies that drifted by a stripped space would
    silently split the cache and make two datasets incomparable for no visible
    reason.
    """
    return [
        Chunk(
            idx=i,
            title=title,
            # Sentences ship with leading spaces; join then collapse.
            text=" ".join(s.strip() for s in sents).strip(),
            rank=i,
            is_gold=title in gold_titles,
        )
        for i, (title, sents) in enumerate(zip(context["title"], context["sentences"]))
    ]


def _load_hotpotqa_distractor(split: str, cache_dir: str | None) -> list[Example]:
    from datasets import load_dataset as hf_load

    ds = hf_load(
        "hotpotqa/hotpot_qa",
        "distractor",
        split=split,
        cache_dir=cache_dir or DEFAULT_HF_CACHE,
    )

    examples: list[Example] = []
    for row in ds:
        chunks = _chunks_from_context(row["context"], set(row["supporting_facts"]["title"]))
        examples.append(
            Example(
                qid=row["id"],
                question=row["question"],
                answer=row["answer"],
                chunks=chunks,
                gold_chunk_ids=[c.idx for c in chunks if c.is_gold],
                hop_type=row["type"],          # comparison | bridge
                meta={"level": row["level"]},  # easy | medium | hard
            )
        )
    return examples


#: The 2WikiMultihopQA mirror this study loads. The dataset's own authors publish
#: at `xanhho/2WikiMultihopQA`, but that repo is script-based and `datasets` 5.x
#: no longer runs dataset scripts, so it cannot be loaded here at all. This mirror
#: ships the same columns as parquet. It is a third party, and the write-up says
#: so rather than implying the result rests on the authors' own release.
_2WIKI_REPO = "framolfese/2WikiMultihopQA"


def _load_2wikimultihop(split: str, cache_dir: str | None) -> list[Example]:
    """2WikiMultihopQA, the secondary dataset (plan Sec. 4.1).

    Same column layout as HotpotQA distractor, and the same ten paragraphs per
    row, which is what makes it a drop-in second dataset rather than a second
    protocol. Two differences matter and neither is cosmetic:

    **Four hop types, not two.** `comparison`, `compositional`, `inference` and
    `bridge_comparison`, against HotpotQA's bridge/comparison. Stratified
    sampling is dict-based and interleaves whatever strata it is given, so this
    needs no special handling and the nested-prefix property still holds.

    **`bridge_comparison` rows carry FOUR gold paragraphs, not two**, and they
    are a substantial share of the split. That is the reason
    `_require_two_gold` exists; see its docstring, and note the exclusion was
    registered in advance rather than chosen once the number was known.
    """
    from datasets import load_dataset as hf_load

    ds = hf_load(_2WIKI_REPO, split=split, cache_dir=cache_dir or DEFAULT_HF_CACHE)

    examples: list[Example] = []
    for row in ds:
        chunks = _chunks_from_context(row["context"], set(row["supporting_facts"]["title"]))
        examples.append(
            Example(
                qid=row["id"],
                question=row["question"],
                answer=row["answer"],
                chunks=chunks,
                gold_chunk_ids=[c.idx for c in chunks if c.is_gold],
                # comparison | compositional | inference | bridge_comparison
                hop_type=row["type"],
                # 2Wiki has no `level`; it ships relation triples instead, which
                # are the dataset's own evidence annotation and are worth keeping
                # even though nothing in this study reads them yet.
                meta={"evidences": row["evidences"]},
            )
        )
    return examples


def load_dataset(
    name: DatasetName,
    split: str = "validation",
    n_queries: int | None = None,
    seed: int = 20260828,
    stratify_by: str | None = None,
    cache_dir: str | None = None,
) -> list[Example]:
    """Load and subsample.

    Subsampling is seeded, stratified and nested, so the pilot set is a strict
    prefix of the main set at the same seed.
    """
    if name == "hotpotqa_distractor":
        examples = _load_hotpotqa_distractor(split, cache_dir)
    elif name == "2wikimultihop":
        examples = _load_2wikimultihop(split, cache_dir)
    elif name == "nq_open_top10":
        raise NotImplementedError(
            "NQ-open loader not implemented; needs a Pyserini prebuilt index "
            "(plan Sec. 4.1, add only if time allows)"
        )
    else:
        raise ValueError(f"unknown dataset: {name!r}")

    examples = _require_fixed_context(examples, expected=10)
    examples = _require_two_gold(examples)
    return subsample(examples, n_queries, seed, stratify_by)


def _require_fixed_context(
    examples: Sequence[Example],
    expected: int = 10,
) -> list[Example]:
    """Drop rows that do not ship exactly ``expected`` paragraphs.

    HotpotQA distractor is not perfectly rectangular: 60 of the 7,405 validation
    rows (0.81%) carry between 2 and 9 paragraphs instead of 10. All 60 still
    contain both gold paragraphs, so this is not a gold-coverage problem -- it is
    a comparability problem. A fixed context size is what makes a position, a
    positional bucket, and a keep-k budget mean the same thing across queries; a
    2-paragraph row at k=5 is not pruned at all, and would silently enter the
    analysis as a free win for every arm.

    Dropping <1% is cheap. Leaving it undocumented is not: record this in
    ANALYSIS_PLAN.md Sec. 3 under Exclusions before registering.
    """
    kept = [ex for ex in examples if len(ex.chunks) == expected]
    dropped = len(examples) - len(kept)
    if dropped:
        print(
            f"excluded {dropped}/{len(examples)} rows "
            f"({dropped / len(examples):.2%}) without exactly {expected} paragraphs"
        )
    if not kept:
        raise ValueError(f"no rows with exactly {expected} paragraphs")
    return kept


def _require_two_gold(
    examples: Sequence[Example],
    expected: int = 2,
) -> list[Example]:
    """Drop rows that do not carry exactly ``expected`` gold paragraphs.

    **Registered in advance, before the number was known.** ANALYSIS_PLAN Sec. 3
    lists "not exactly 2 gold paragraphs" among the candidate exclusions,
    measures it at 0 rows on HotpotQA, and fixes the rule for later datasets:
    *drop the row, report the count, and never drop a row on the basis of the
    answer it produced.* This is that rule, applied uniformly rather than per
    dataset, which is why it is a no-op on HotpotQA and does not disturb the
    completed main run.

    It bites on 2WikiMultihopQA, where `bridge_comparison` rows carry four gold
    paragraphs. Keeping them would break the comparison the study is built on: at
    k=2 a four-gold question cannot retain all its evidence even in principle, so
    gold recall, the Placebo Gap at matched keep-count and the Oracle Gap would
    all mean something different on those rows than on every other row and on
    every HotpotQA row.

    The cost is real and belongs in the write-up rather than in a footnote: the
    2Wiki replication then covers three of the dataset's four question types, and
    the one it drops is the hardest.
    """
    kept = [ex for ex in examples if len(ex.gold_chunk_ids) == expected]
    dropped = len(examples) - len(kept)
    if dropped:
        print(
            f"excluded {dropped}/{len(examples)} rows "
            f"({dropped / len(examples):.2%}) without exactly {expected} gold paragraphs"
        )
    if not kept:
        raise ValueError(f"no rows with exactly {expected} gold paragraphs")
    return kept


# --------------------------------------------------------------------------- #
# Memorization control
# --------------------------------------------------------------------------- #

def memorization_filter(
    examples: Sequence[Example],
    nocontext_predictions: dict[str, str],
    correct_fn: Callable[[str, str], float] | None = None,
    threshold: float = 1.0,
) -> list[Example]:
    """Keep only the queries the generator gets **wrong** with no context.

    Non-negotiable (plan Sec. 4.1): Wikipedia-derived multi-hop benchmarks leak
    into pretraining. Without this you are measuring parametric recall and
    calling it retrieval. Report filtered and unfiltered numbers, always both.

    "Correct" is ``correct_fn(pred, gold) >= threshold``. The default pair --
    exact match at 1.0 -- is the strict reading. ANALYSIS_PLAN.md Sec. 3 leaves
    the choice open between that and a token-F1 cutoff; the threshold is a
    parameter so the F1 option is expressible rather than requiring the
    comparison itself to be rewritten. Whichever is picked has to be registered
    before the main run.
    """
    from .metrics import exact_match

    correct_fn = correct_fn or exact_match
    kept = []
    for ex in examples:
        pred = nocontext_predictions.get(ex.qid)
        if pred is None:
            raise KeyError(f"no nocontext prediction for {ex.qid!r}; run that arm first")
        if correct_fn(pred, ex.answer) < threshold:
            kept.append(ex)
    return kept


def iter_cells(
    examples: Iterable[Example],
    arms: Sequence[str],
    budgets: Sequence[int],
) -> Iterable[tuple[Example, str, int]]:
    """The (query, arm, budget) cell grid. Permutations expand inside each cell."""
    for ex in examples:
        for arm in arms:
            for budget in budgets:
                yield ex, arm, budget
