# Stage 2 compression plan

Approved by the user on 20 October 2026.

Compression is the primary experimental objective: measure per-language damage under deployment formats and test precision allocation at matched shipped bytes. Adaptation is secondary and requires an explicit user decision after compression evidence; sensitivity results guide placement rather than presuming attention or MLP targets.

## Task order

- A1: https://github.com/grinder81/polyglot-quant-eval/issues/5
- P12a: https://github.com/grinder81/tiny-aya-edge/issues/18
- P12b: https://github.com/grinder81/tiny-aya-edge/issues/19
- A2: https://github.com/grinder81/polyglot-quant-eval/issues/17
- A3: https://github.com/grinder81/polyglot-quant-eval/issues/18
- A4: https://github.com/grinder81/polyglot-quant-eval/issues/19
- B7: https://github.com/grinder81/polyglot-quant-eval/issues/20
- B5: https://github.com/grinder81/tiny-aya-edge/issues/24
- B6: https://github.com/grinder81/polyglot-quant-eval/issues/21
- C8: https://github.com/grinder81/tiny-aya-edge/issues/25
- C9: https://github.com/grinder81/tiny-aya-edge/issues/26
- C10: https://github.com/grinder81/tiny-aya-edge/issues/27
- C11: https://github.com/grinder81/tiny-aya-edge/issues/28
- P13: https://github.com/grinder81/tiny-aya-edge/issues/29
- tracker: https://github.com/grinder81/polyglot-quant-eval/issues/22

A1 then A2 then A3 then A4. Reserve B7 before the sweep. B5 and B6 proceed together; then C8, C9, C10, C11. Upstream contributions and baseline iPhone validation are separate tracks; final-device comparison waits for a finalist. No experiment starts as part of this documentation PR.

## Governing gates

- Keep 2.2 GB total shipped bytes, counting model, nonembedded tokenizer and adapters once. A tighter official-Q4 cap remains a user decision. Freeze equal-byte tolerance and retention bounds before candidate selection.
- On 100 development questions per language, validate option rotations and candidate scoring variants before the sweep. Keep raw and adjusted metrics separate. A smaller letter bias alone does not select the scoring winner. No custom adapter without approval. Freeze canonical settings explicitly, rerun affected baselines if changed.
- Verify 424 source-disjoint holdout groups per language using membership metadata; reserve before subsequent fitting and touch once with frozen finalists. Do not use the 176 remaining questions sharing development passages as an untouched test.
- Verify real quantizer API support and quantize/dequantize fidelity; no generic fake rounding substituted for GGUF. C9 has six extreme-group builds plus one random, seven total; about 0.7 rank correlation is a provisional criterion to preregister with uncertainty, not automatic proof.
- After two map-validation failures, use measured GGUF diagnostics rather than simulation-led allocation.
- If sweep plus generation excludes a practically meaningful language-specific effect through Q3, bring an adaptation-primary decision to the user. Failure to reach significance alone does not establish equivalence.
- Existing upstream fixes are local and tested; external acceptance cannot be promised. No claim of phone performance before measurement.

## Completed starting evidence

The local-Q4 900-question run is complete: English226/300, Bangla189/300, Hindi176/300, merged evaluation PR15. BF16 at300 and the new controls are next; old n=50 results retain their own runtime/settings. The raw doc-ID caveat is tracked in evaluation issue16; use verified source/question identities.

Older Mac and kernel tasks remain supporting backlog, not gates on this sequence. Existing generic reporter and other-task audits remain explicitly open. All code and repository documentation changes use small PRs; ready PRs may merge under standing approval after CI. Update issues and clean branches/worktrees without deleting unrelated user files.
