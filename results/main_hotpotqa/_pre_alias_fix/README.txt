Aggregates produced BEFORE the llmlingua2 cache-aliasing fix (commit 5807566),
kept so the defect and its effect stay auditable from the repo alone.

The llmlingua2 per-chunk compression cache was keyed on (chunk idx, rate).
Neither identifies the chunk, and run.py is arm-major, so one instance served
every query and every query after the first received query 1's compressed
passages. Every llmlingua2 figure in these files is therefore invalid.

Every OTHER arm in these files is unaffected: provence, llm_pruner and
loo_oracle all key their caches on the query.

permutation_analysis.json here is also superseded, because llmlingua2 supplies
two of the nine confirmatory Holm comparisons and Holm adjusts across the whole
family.

Do not pool these numbers with post-fix results.
