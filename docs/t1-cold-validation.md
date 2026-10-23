# T1 expanded cold-likelihood validation

This protocol prepares a future live gate. No model validation has run merely
because this file or its tests exist. Run only after the reviewed launch package
and user collection authorization. Preserve the original A3 protocol and evidence.

Use the validated v4 cold-prefill runtime, unchanged upstream LocalCompletionsAPI,
the pinned tokenizer, model/native/wrapper/client identities, and the exact new
900-row original-order T1 request lock. Pin this protocol and validator hashes
before the first request. Never modify the lock to make a failed gate pass.

For each language select indices 0, 100 and 299, plus the row with the longest
full candidate token sequence (first row wins a length tie). Deduplicate, sort
the row indices, and check all four candidates. This covers 9–12 contexts and
36–48 candidates per arm. Both model artifacts must pass independently.

1. Direct mode loads the manifest's model settings in the pinned wrapper and
   performs independent cold forwards. Derive full-vocabulary log-softmax in
   float64 from raw logits. Check exact model token IDs, one BOS, and repeat the
   first selected context's candidates to measure repeat drift.
2. Stop the recorded owned direct process before starting the scoring server;
   keep exactly one GPU owner. Client mode checks the exact served model identity,
   unchanged client encode/request/parser path, echoed strings and array lengths,
   token counts, and generated suffix against the saved direct oracle.
3. A separate client invocation reverses the entire selected row/candidate list
   against that same oracle. Preserve both request orders and every raw response.

All continuations must be exactly one model token. Check each continuation token
and supplied echoed probability to 0.0002 nats; the separate summed-continuation
bound is max(0.001, continuation-token count × 0.0002) nats. Both constraints
apply, so the looser sum bound cannot excuse a failed token check. Parser agreement
with the same supplied array is 1e-8. Generated-suffix probability uses 0.0002 nats.
Reject nonfinite values, mismatched bytes/IDs/counts, and any winner inversion.
Numerical tolerances never excuse token alignment errors.

The direct, client and reversed-order reports bind the exact lock, runtime,
validator and protocol hashes. Client reports additionally bind the oracle,
tokenizer and upstream source/version identities. Successful complete reports are
required before full collection; a partial file or empty record list is not a gate.
Do not overwrite failed/partial attempts. Preserve them and diagnose discrepancies.

This extends the selected-case A3 evidence to a predeclared T1 selection; it is not
proof of correctness for every possible input, generation quality, or runtime.
The 100-item-per-language A3 overlap is checked separately on freshly collected
scores under exact request/token/model/runtime identities. Historical A1 deltas
are descriptive and never serve as tolerance bounds.

Run `python setup/validate_t1_cold.py --help` for arguments. Direct mode performs
model inference; client mode sends scoring requests. Neither is an offline smoke.
Use fresh output paths and the explicit reviewed lock hash. The script refuses
optimized Python because its inherited validation assertions must remain active.
