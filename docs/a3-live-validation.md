# A3 live validation preregistration

Frozen before live scoring on 22 October 2026. This diagnostic does not select a
canonical scoring contract and never reads holdout data.

Use the frozen request lock SHA256
`75486c22c0c25f7c6c9d3da6abfaaa0165babf4a5de24970147ab9d1488d9844`.
Select index zero for every language, variant and rotation, plus the longest full
input and longest continuation per language/variant. Deduplicate by lock row.
Score every candidate in each selected row, in both recorded runtime arms.

The unchanged upstream LocalCompletionsAPI uses the isolated HF tokenizer,
string requests, no client BOS, and batch size one. Direct-logit evidence uses
the pinned wrapper/native library and complete locked model IDs, independently
of the echo parser. Reset KV state before each direct forward. Preserve direct
float64 log-softmax probabilities, generated greedy token, expected per-token
echo strings, complete HTTP response, parser output and source identity.

Require exact supplied IDs, automatic single BOS, echo array lengths, all token
strings (including empty marker renderings), and precisely one generated suffix.
An ID/count mismatch cannot pass under numerical tolerance. Compare every scored
continuation token with direct logits: absolute error at most 0.0002 nats AND
absolute difference between summed likelihoods at most
max(0.001, continuation tokens * 0.0002).
Compare the parser sum against math.fsum of the same supplied echo slice with
absolute tolerance 1e-8. No relative tolerance is used. Repeat direct forwards on
the first selected row to expose GPU forward variability.

For plain-letter rows bridge all four candidates to A2 by source/question and
rotation, absolute tolerance 0.001 nats. Record candidate deltas, prediction and
margin changes. Record every oracle/client argmax inversion. A direct top-two
margin at most twice the sum tolerance identifies a tie-sensitive inversion,
not exact behavioral parity. Failure stops full inference and is reported;
tolerances are never loosened retrospectively. No custom scorer or adapter.

Full inference starts only after independent review of validation evidence and
the resumable upstream-task runner. New manifests preserve original A2 identity.

Inspection found an EOS documentation discrepancy: the older conversion manifest
requires an EOS override to 3; the recorded A2 Python-server commands contain no
override. This bridge reproduces the exact A2 commands, records effective EOS,
and does not silently add an override. Any subsequent runtime fix has a fresh
identity and separate validation.
