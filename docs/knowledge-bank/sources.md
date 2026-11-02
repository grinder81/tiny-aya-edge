# Sources and evidence boundaries

This bank describes **Tiny Aya Global at the inspected checkpoint**, its recorded GGUF artifacts and an existing frozen pilot. The code, artifact inventory, mathematical derivations and measured outcomes support different kinds of claims. Dates refer to this knowledge-bank review on 1 November 2026, not a new experiment.

## Primary project evidence

| Source | What it establishes | Boundary |
| --- | --- | --- |
| [Original file verification](../../reports/stage1-downloaded/verified-files.json) | Checkpoint revision, original config/tokenizer/shard filenames, bytes and hashes | Historical authorized download evidence; no weights were downloaded or opened to create this bank |
| [Original inspection](../../reports/stage1-downloaded/inspection.json) | Exact configuration and original tensor membership, shapes and dtypes | Describes the inspected model; not an activation or sensitivity study |
| [Actual controlled GGUF inventory](../../reports/control-tensor-inventory.json) and [explanation](../../reports/control-tensor-inventory.md) | Complete BF16/local-Q4 file bytes, tensor types, shared-source accounting and payloads | Q4 is the local controlled export, distinct from the separately downloaded official file |
| [Frozen companion execution plan snapshot](sources/c8-plan.json) | The selected operating inventory, exact eight targets, formats and restricted pair eligibility | An unchanged snapshot of another repository's frozen plan; it is not a new execution authority |
| [Companion scientific design](https://github.com/grinder81/polyglot-quant-eval/blob/34db052fc98de4c15b511d831fe23f77d8c3153f/docs/c8-bounded-pilot.md) | Sampling, estimators, nomination, confirmation and cumulative caps | The original protocol remains authoritative; this bank explains its choices without amending them |
| [Complete development precision curve](https://github.com/grinder81/polyglot-quant-eval/blob/34db052fc98de4c15b511d831fe23f77d8c3153f/results/t3-precision-curve/final/README.md) | Task/fidelity results and English-only operating-point selection | Development results, not phone results or established allocation benefit |

The checkpoint revision is `af89d219b53ed9b13b8a4645f9c8028973510324`. The source config SHA256 is `458a9265b6a96e5ad4bbe1857716bd69e434f062f8b32fae48f5f06197d26384`. The control BF16 GGUF SHA256 is `0e2f666ac8fb50a7f53d14164147beeca35905b76b93eb90e0e3a963eb203b4b`. The [machine-readable pins](sources/source-pins.json) collect these and the source-code identities.

The companion plan snapshot has SHA256 `e7ccdb50be254ee8bdde307f73c58bc9a9c9d68882a40aa453b9b516ec8be64f` and comes from evaluation commit `34db052fc98de4c15b511d831fe23f77d8c3153f`. The generator rejects a changed snapshot. Only public IDs, hashes, numeric configuration and paths already published in that plan are present; no transcript excerpts, credentials or weights are added.

## Implementation sources

### Native execution graph

The [pinned llama.cpp Cohere2 implementation](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/src/models/cohere2.cpp) was retrieved directly from its public raw URL and read for this bank. Raw file SHA256 is `b5c4fdf1c3050f2ddbb9881aed09ddcfc41c61e0ed14476c444cb36cc2fa6732`.

Its tensor loader creates the input embedding and output from the same source tensor. Its graph preserves the normalized input separately for MLP, computes attention, computes the gated MLP from that preserved input, and adds residual plus both outputs. RoPE is inside the local-block condition. Final normalization and the shared-source output projection produce logits. These are architecture/source facts, not proof that every backend has identical allocations or floating-point outputs.

### Transformers reference

The [Cohere2 reference at v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/cohere2/modeling_cohere2.py) was also retrieved directly and read. Raw file SHA256 is `81750a9746f8abba97c3bf36340851bfd31a37010896d42e3da504bd9999ec9b`.

Relevant definitions are `Cohere2LayerNorm`, `Cohere2Attention`, `Cohere2MLP` and `Cohere2DecoderLayer`. They establish centering normalization, logical head layout, local-only RoPE, gate/up/down multiplication and parallel residual combination. The original inspected config names Transformers 4.51.3. The later differentiable pilot environment uses a separately pinned version; this bank does not claim the reference tag is that executed environment or validate their full numerical equivalence.

The input/output storage claim is anchored in actual GGUF evidence and the native loader rather than inferred only from an absent config field. New differentiable interventions must verify actual parameter tying at load time.

## Foundational and method references

| Primary reference | Use here |
| --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Attention's score/softmax/value-mixture formulation; Tiny Aya's particular graph comes from its source |
| [GQA](https://arxiv.org/abs/2305.13245) | Grouped-query sharing terminology and motivation |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | Gated feed-forward background; exact SiLU placement checked in source |
| [AWQ](https://arxiv.org/abs/2306.00978) | Prior art for activation-aware equivalent scaling and low-bit numerical range control |
| [Activation patching metrics and methods](https://arxiv.org/abs/2309.16042) | Why intervention construction and endpoint choice affect localization conclusions |
| [Official llama.cpp imatrix documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md) | Text-derived calibration statistics and their use in quantization; this mutable documentation is conceptual background, not the locked executable version |
| [Tiny Aya technical report](https://arxiv.org/abs/2603.11510v1) and [model card](https://huggingface.co/CohereLabs/tiny-aya-global) | Family context and release information; no new vendor benchmark replication |

The MLP scaling/permutation proofs and cache element counts are explicit derivations from the stated graph/dimensions. Their arithmetic is not a new measured result. More speculative transfers in the intervention chapter are framed as questions requiring compatibility checks, not claims of Tiny Aya performance.

## What the bank verifies

`build_tiny_aya_atlas.py` maps every original tensor to its GGUF name, checks reversed matrix dimensions, recomputes all 290 tensors' parameter sums and per-group payloads, checks complete file accounting, verifies the 36-block pattern and eight target membership, and binds the companion snapshot hash. Canonical JSON hashes in the atlas make semantic source identities portable across CRLF/LF checkouts.

`draw_tiny_aya_knowledge.py` generates six architecture/study diagrams. SVG is the portable canonical diagram; PNG previews were generated from the same drawing primitives and visually inspected. Neither generator imports model libraries, launches WSL or accesses the GPU.

These checks validate the reference's internal consistency. They do not rehash the inaccessible native model files after the storage incident, repair Ubuntu, repeat numerical inference gates, or supply missing sensitivity outcomes. Historical artifact evidence remains historical until execution recovery can verify it again.

Local verification passed for all 290 cross-mapped tensors, 36 blocks, eight selected sites, complete file accounting, regeneration of all six SVGs, XML parsing and 46 relative links. All six PNG previews were visually inspected for flow, labels and clipping. Existing archive verification preserved 19 original evidence identities, the CPU echo/ranking checks passed, and all four existing cold-validation tests passed. These are documentation, static-data and CPU checks; no model inference occurred. The pull request's CI records the separate Linux validation.

## Keep observation separate from explanation

Use **structural fact** for matrix shapes and graph order; **derivation** for consequences of those equations; **observation** for a measured result with its runtime/data identity; and **hypothesis** for a proposed behavioral role or intervention. A role name such as “value projection” is structural. A claim such as “this matrix preserves Bangla reasoning” is a behavioral hypothesis until independently supported.
