# Article 2 pilot readiness evidence

Measured 1 November 2026 for [pilot tracker47](https://github.com/grinder81/polyglot-quant-eval/issues/47) and [resource task42](https://github.com/grinder81/tiny-aya-edge/issues/42). This package contains readiness measurements, not a completed allocation or language-specific improvement.

## Byte feasibility

The pinned native quantizer's dry-run assignments, combined with exact native row sizes and GGUF metadata/alignment, predict the following file bytes. Q4_K_M agrees exactly with its existing real export; the new Q6_K and Q3_K_M builds also agree with their predictions.

| Layout | Bytes | Evidence status |
| --- | ---: | --- |
| Q6_K | 2,759,532,032 | Actual build and SHA256 recorded |
| Default Q4_K_M | 2,143,976,960 | Existing actual export checked |
| Uniform Q4_K_M with `--pure` | 1,896,078,848 | Native dry run |
| Default Q3_K_M | 1,799,740,928 | Actual build and SHA256 recorded |
| Uniform Q3_K_M with `--pure` | 1,451,269,632 | Native dry run |
| Q3_K_S | 1,660,984,832 | Native dry run |
| Q2_K | 1,481,973,248 | Native dry run |
| IQ3_S | 1,665,998,336 | Native dry run |

These results prevent describing untouched default and uniform exports as byte matched. Primary language allocations can instead preserve the same per-shape format counts or solve under a frozen narrow tolerance. Which controls achieve that constraint must be established before C10. No file padding supplies a scientific control.

The actual GGUF has one stored `token_embd.weight` with 536,870,912 parameters and no separate `output.weight`. Count it once in shipped bytes. This does not establish that a runtime has only one resident copy. Q6_K exceeds the2.2GB final ceiling and remains a diagnostic.

## Streamed KL profile

The existing BF16 and Q4 models were profiled sequentially on exactly original development index0 in English, Bangla and Hindi, under the [frozen resource specification](../../docs/article2-resource-profile.md). All tokens matched the T1 lock. Their twelve final letter probabilities per arm matched the verified T1 scores with maximum absolute error2.12e-7nats, below the0.0002 gate. Full-vocabulary KL was reduced in float64 chunks of at most8rows.

The recorded profile interval was 26.00 seconds, including runtime verification and model execution. It cached 628 content-token distributions, approximately 628 MiB plus array headers. Across the complete 900-prompt lock, 176,756 content-token distributions would require 185,342,099,456 raw float32 bytes, about 172.6 GiB, before headers. This is a storage projection, not a measured full-900 runtime. Three prompts do not estimate a language effect. Individual forward timings include warm-up and are not phone performance or an inference throughput benchmark.

## Fisher resource profile

A separate TED talk supplied one English example of64content tokens. The float32 differentiable reference's entire layer17MLP down matrix matched the dequantized BF16 GGUF values exactly. All parameters except that matrix were frozen; mean next-token NLL was differentiated without updating any weights. Per-example squared gradients were finite. The load/weight check took53.19seconds, forward0.696seconds and backward0.080seconds; Torch peak allocated13,984,115,200bytes. These measure one selected-matrix example, not the full C8 cost.

This sequence-level empirical Fisher proxy differs from token-wise Fisher and the exact output-KL Hessian. The HF float32/eager path is separate from native GGUF execution; matching one weight matrix does not demonstrate whole-runtime numerical equivalence or authorize proxy substitution.

## Reproduction

Run the scripts from the repository root with explicit paths. `builds/profile_pilot_bytes.py` performs CPU dry runs; `builds/profile_pilot_kl.py` performs the three-prompt model profile; `builds/profile_pilot_fisher.py` performs the one-example backward profile; `python -m builds.make_t3_build` makes a frozen T3 artifact. Their manifests include the actual commands or inputs, implementation hashes and model/runtime identities. Scripts refuse existing output paths. Weights and raw vocabulary logits stay outside Git.

Q6_K subsequently passed its complete direct/client/reversed-order gates and its task checkpoint is in the companion evaluation repository. Its full KL comparison and the rest of T3 remain in progress. Q3_K_M has passed its vocabulary check only. No C8 ranking, generation judgment, A4 slice or final holdout outcome is supplied by this readiness package.

The original Q6_K and Q3_K_M manifests pinned actual model hashes and bytes, alongside the native dry-run layout, but did not attach a readable actual-file tensor inventory. The inventory supplements under `builds/` were published later: after Q6 task scoring and before any Q3 inference. They match all predicted tensor types, shapes and payload sizes. This is a documentation timing deviation; the original manifests and collected Q6 identity remain unchanged. Remaining T3 builds attach their actual inventories before inference.
