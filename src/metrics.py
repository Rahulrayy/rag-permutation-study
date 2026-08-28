"""Answer quality and the four derived quantities that are the contribution.

Sec. 4.5. EM and token-F1 are standard (SQuAD/HotpotQA normalisation). OAE, Rank
Flip Rate, Placebo Gap and Oracle Gap are the study's own, and their formulas are
frozen by ANALYSIS_PLAN.md -- change one after registration and it is a protocol
deviation, not a bugfix.

Score container convention throughout: ``PerArmScores`` maps
``arm -> query_id -> [score per permutation]``, with the per-query lists in a
consistent permutation order. Permutations are nested within queries; nothing
here ever flattens that nesting (see stats.py for why).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

PerArmScores = Mapping[str, Mapping[str, Sequence[float]]]


# --------------------------------------------------------------------------- #
# Answer quality
# --------------------------------------------------------------------------- #

def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD convention)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        # Both empty -> agreement; one empty -> no overlap possible.
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def supporting_fact_f1(predicted: Sequence[int], gold: Sequence[int]) -> float:
    """Secondary HotpotQA signal, over kept chunk indices vs gold chunk indices."""
    pred_set, gold_set = set(predicted), set(gold)
    if not pred_set or not gold_set:
        return float(pred_set == gold_set)
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #

def within_query_sd(scores: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """SD across permutations, per query. RQ1, and the week-1 kill criterion."""
    return {q: float(np.std(list(v), ddof=1)) for q, v in scores.items() if len(v) > 1}


def order_adjusted_effect(
    scores: PerArmScores,
    method: str,
    baseline: str,
) -> float:
    """OAE: the headline number.

        OAE(m) = mean_q[ mean_pi Q(m,q,pi) - mean_pi Q(b,q,pi) ]
                 / mean_q[ SD_pi Q(b,q,pi) ]

    Read as: how many orderings-worth of noise does this method actually buy you.
    Paired -- restricted to queries present in both arms.
    """
    shared = sorted(set(scores[method]) & set(scores[baseline]))
    if not shared:
        raise ValueError(f"no shared queries between {method!r} and {baseline!r}")

    deltas = [
        float(np.mean(scores[method][q])) - float(np.mean(scores[baseline][q]))
        for q in shared
    ]
    sds = [float(np.std(scores[baseline][q], ddof=1)) for q in shared]

    denom = float(np.mean(sds))
    if denom == 0.0:
        # The premise is dead at this scale, not a division bug. See plan Sec. 9.
        return float("nan")
    return float(np.mean(deltas)) / denom


def rank_flip_rate(scores: PerArmScores, arms: Sequence[str] | None = None) -> float:
    """RFR: fraction of method-pair comparisons whose sign, under some *single*
    ordering, disagrees with the permutation-averaged ranking.

    High RFR means single-order evaluation -- what every published pruner is
    evaluated with -- is unsound. Ties under either view are not counted as flips.
    """
    arms = list(arms) if arms is not None else sorted(scores)
    if len(arms) < 2:
        raise ValueError("need at least two arms to compare")

    n_perms = {len(v) for arm in arms for v in scores[arm].values()}
    if len(n_perms) != 1:
        raise ValueError(f"ragged permutation counts across arms/queries: {n_perms}")
    n_perm = n_perms.pop()

    def mean_over(arm: str, perm: int | None) -> float:
        vals = [
            float(np.mean(v)) if perm is None else float(v[perm])
            for v in scores[arm].values()
        ]
        return float(np.mean(vals))

    averaged = {a: mean_over(a, None) for a in arms}

    flips = total = 0
    for perm in range(n_perm):
        single = {a: mean_over(a, perm) for a in arms}
        for a, b in combinations(arms, 2):
            ref = np.sign(averaged[a] - averaged[b])
            obs = np.sign(single[a] - single[b])
            if ref == 0:
                continue
            total += 1
            if obs != ref:
                flips += 1
    return flips / total if total else 0.0


def placebo_gap(scores: PerArmScores, method: str, placebo: str = "placebo_pos") -> float:
    """Q(m) - Q(placebo_pos) at matched keep-count. RQ4, the centerpiece.

    Near zero means the method is not doing content selection -- its apparent gain
    is positional promotion, which dropping by position alone reproduces.
    """
    return _paired_mean_delta(scores, method, placebo)


def oracle_gap(scores: PerArmScores, method: str, oracle: str = "loo_oracle") -> float:
    """Q(m) / Q(loo_oracle) at matched budget. Headroom left on the table."""
    shared = sorted(set(scores[method]) & set(scores[oracle]))
    if not shared:
        raise ValueError(f"no shared queries between {method!r} and {oracle!r}")
    num = float(np.mean([np.mean(scores[method][q]) for q in shared]))
    den = float(np.mean([np.mean(scores[oracle][q]) for q in shared]))
    return num / den if den else float("nan")


def _paired_mean_delta(scores: PerArmScores, a: str, b: str) -> float:
    shared = sorted(set(scores[a]) & set(scores[b]))
    if not shared:
        raise ValueError(f"no shared queries between {a!r} and {b!r}")
    return float(
        np.mean([np.mean(scores[a][q]) - np.mean(scores[b][q]) for q in shared])
    )
