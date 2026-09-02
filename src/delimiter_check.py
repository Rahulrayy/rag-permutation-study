"""The registered prompt-template robustness check (ANALYSIS_PLAN Sec. 7).

RQ1 says reordering an identical context changes the score for half of all
questions. `generate.build_prompt` renders each passage as `[i] Title: text`,
where `i` is its slot in the *current* ordering -- what a deployed pipeline
does, and what stops the permutation from being inferable. It does mean a
reordering changes the numeric labels as well as the semantic positions, so RQ1
as measured is the effect of presentation order inclusive of its labelling.

`ALT_TEMPLATE` differs from `DEFAULT_TEMPLATE` in the context fencing alone --
`generate.py` enforces that at import with a real raise, not an assert, so the
two cannot drift. If the within-question SD survives the change, the effect is
not an artifact of one delimiter style.

**What this does not establish.** It varies the fencing, not the `[i]` numbering,
so it bounds the delimiter contribution and not the numbering contribution. A
template that dropped the indices would be a different prompt rather than a
delimiter variant, and would confound the check with a change in how the model
is told the passages are separate.

    python -m src.run --config configs/robustness_delimiter.yaml
    python -m src.delimiter_check
"""

from __future__ import annotations

from pathlib import Path

from .matched_comparison import compare, figure, load_rows, write

MAIN = Path("results/main_hotpotqa/generations.csv")
ALT = Path("results/robustness_delimiter/generations.csv")
OUT = Path("results/robustness_delimiter/delimiter_check.json")
FIG = Path("results/robustness_delimiter/figures/delimiter_sd.png")

#: All five. Both runs use the same permutation seed and the same five
#: strategies, so unlike the 27B replication there is no prefix restriction to
#: make -- the orders match cell for cell, which `compare` asserts.
PERMS = ("0", "1", "2", "3", "4")

#: `full` only. RQ1 is a property of the un-pruned context, which has the most
#: permutable slots and is the arm the headline number is quoted from.
ARMS = ("full",)


def main(budget: str = "3", n_replicates: int = 10_000, seed: int = 20260828) -> dict:
    out = compare(
        load_rows(MAIN),
        load_rows(ALT),
        "default",
        "alt",
        arms=ARMS,
        budget=budget,
        perms=PERMS,
        n_replicates=n_replicates,
        seed=seed,
    )
    figure(
        out,
        FIG,
        title="The permutation effect is not an artifact of the context delimiters",
        subtitle=(
            f"same {out['n_shared_queries']} questions, same five orderings, "
            f"k={budget}; only the context fencing differs"
        ),
        series_labels=("default:  passages, bare", "alt:  <context> … </context>"),
        xlabel="mean within-question SD of token-F1 across the five orderings",
        footnote=(
            "*  the paired default-alt difference excludes zero. A registered "
            "robustness check (ANALYSIS_PLAN Sec. 7), reported without "
            "multiplicity correction."
        ),
    )
    write(out, OUT)
    return out


if __name__ == "__main__":
    main()
