# A3 cold-prefill and exact top-one echo runtime v4

Frozen before v4 inference on 22 October 2026. Use every correctness and reversed
order check in a3-cold-validation.md with unchanged validate_a3_cold.py. V3 cold
HTTP was stopped as an incomplete diagnostic when exact echo ranking proved the
latency bottleneck; its first eight candidates passed all strict checks. Preserve
its snapshots and never call it a complete gate.

The fresh v4 environment is cloned from the original pinned wrapper using
prepare_a3_scoring_runtime.py and applies python-cold-echo.patch followed by
python-echo-top1.patch. The latter changes only selection of the single greatest
logprob alternative when logprobs=1: reversed argmax returns the same descending
(score, token ID) maximum as full lexsort. Every other count keeps the old path.
Full/supplied logprobs, softmax, task, client and scorer remain unchanged.

CPU tests compare selected IDs and full echoed top-probability dictionaries,
including tied scores, signed zero, infinities and NaNs. Their isolated selection
timing is not API throughput. Measure live API latency separately. Rerun the
complete independent cold oracle, full HTTP gate and reversed-order gate in
both arms under fresh manifests; do not relabel v1/v2/v3 outputs. Preserve legacy
A2 numerical failures and all prediction/margin changes as explicit comparisons.

After both arms pass, independently review the evidence and merge the small
runtime and upstream-runner PRs after CI before locked full scoring. Canonical
contract choice and affected baseline decisions still require user review before
the precision sweep.
