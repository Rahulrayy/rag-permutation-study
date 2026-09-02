"""Matched 3B-vs-27B comparison of the RQ1 permutation SD.

The replication ran P=3 on its own memorization-filtered population, and the
main run ran P=5 on a larger one, so their headline SDs are not directly
comparable: fewer orderings give a score fewer chances to move, and the two
populations are filtered by different generators. ANALYSIS_PLAN Sec. 9 records
the P mismatch as a removable confound and says to remove it.

It is removable exactly, not approximately. Both configs seed the permutation
draw identically and the replication's three strategies are the prefix of the
main run's five, so for every shared query the two runs present *byte-identical*
passage orders in perms 0-2 -- asserted by `matched_comparison.compare` rather
than assumed. Restricting the main run to those three perms and to the queries
both populations retain leaves the generator as the only difference.

    python -m src.generator_comparison
"""

from __future__ import annotations

from pathlib import Path

from .matched_comparison import compare, figure, load_rows, write

MAIN = Path("results/main_hotpotqa/generations.csv")
REPL = Path("results/replication_groq/generations.csv")
OUT = Path("results/replication_groq/matched_generator_comparison.json")
FIG = Path("results/replication_groq/figures/matched_generator_sd.png")

#: The replication's permutations, as indices into the main run's five.
SHARED_PERMS = ("0", "1", "2")
ARMS = ("full", "rerank_topk", "llm_pruner", "placebo_pos:middle_first")


def main(budget: str = "3", n_replicates: int = 10_000, seed: int = 20260828) -> dict:
    out = compare(
        load_rows(MAIN),
        load_rows(REPL),
        "3B",
        "27B",
        arms=ARMS,
        budget=budget,
        perms=SHARED_PERMS,
        n_replicates=n_replicates,
        seed=seed,
    )
    figure(
        out,
        FIG,
        title="Order sensitivity survives a 9x scale jump, at a smaller size",
        subtitle=(
            f"same {out['n_shared_queries']} questions, same three orderings, "
            f"k={budget}; only the generator differs"
        ),
        series_labels=("Qwen2.5-3B, local", "Qwen3.8-27B, hosted"),
        xlabel="mean within-question SD of token-F1 across the three orderings",
        star_note="*  the paired 3B-27B difference excludes zero.",
        footnote=(
            "Exploratory: these four comparisons are not in the registered "
            "confirmatory family and carry no multiplicity correction."
        ),
    )
    write(out, OUT)
    return out


if __name__ == "__main__":
    main()
