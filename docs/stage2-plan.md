# Stage 2 compression plan

Approved by the user on 20 October 2026.

Current execution amendment, 1 November 2026: [pilot tracker 47](https://github.com/grinder81/polyglot-quant-eval/issues/47) governs the bounded decision gate before full C8-C10. The user authorized implementation and pilot collection after the Astra review. Freeze the relevant specification and pass technical gates first; preserve all holdout and A4 seals. The [T3 specification](https://github.com/grinder81/polyglot-quant-eval/blob/main/docs/t3-preregistration.md), including its November amendment, resolves the prior draft defaults and separates the primary language contrast from practical superiority. Existing unfinished issues remain open.

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

## 23 October 2026 status addendum

Fresh T1 affected comparator arms are complete under new cold-runtime identities: 300 original-order development questions per language, BF16 GGUF and local Q4_K_M. See [complete results and evidence](../results/t1-cold-baselines/README.md). This supersedes the prospective baseline status above, preserving its historical measurements. The original BF16 vLLM reference remains separate. A4 human labels and all remaining B5/B6 prerequisites remain gates; no sweep or holdout scoring is implied.

## 26 October 2026 amendment: A4 and T3 sequencing

The user decided on 26 October 2026 that T3, the staged precision curve that opens B5, may build and run inference before the A4 human negation labels are merged. Until the adjudicated labels are merged with a recorded hash, no negation-slice analysis of any arm, T1 or T3, is computed, read or reported. The sealed analyses are named in the [T3 preregistration](t3-preregistration.md) and run once, after the labels, on the frozen rows. A4's acceptance criteria and its human-only labelling rule are unchanged, and B5's other prerequisites still apply: the committed preregistration, per-build readiness gates and a separate collection authorization. The same day's decisions on the allocation operating point, the byte ceiling, endpoint roles, the meaningful margin, Tiny Aya Fire, adaptation and the T3 precision points are recorded in that preregistration; none of them changes a governing gate above.
