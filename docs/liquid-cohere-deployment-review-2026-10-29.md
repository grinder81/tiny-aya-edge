# Liquid AI, Cohere and the multilingual deployment opportunity

Research date: **29 October 2026, America/Los_Angeles**. Prepared by Codex for Mo Al Mamun and independent review. Repository baseline: `8a0d922ef40f86ba3ea11e0b4a2656d25d4bcea4`.

This document records the material findings and corrections from the Liquid AI/Cohere discussion. It is a source-backed research assessment and a proposed opportunity, not an independently reviewed report, completed comparison experiment, commercial commitment or modification of the approved study. No inference, phone benchmark, training, holdout access or outreach was performed for this document.

Read alongside the [source and access ledger](liquid-cohere-source-ledger-2026-10-29.md) and [revision/provenance manifest](liquid-cohere-source-pins-2026-10-29.json). Source IDs below resolve in the ledger. Repository heads were resolved through the public Hugging Face API; resolving a head does not verify downloaded weights or reproduce its results. The report date remains 29 October locally even though final source checks crossed midnight UTC into 30 October.

## 1. Assessment

1. **Cohere has an on-device initiative.** Tiny Aya includes phone inference measurements, official quantizations and multilingual quality evaluation. North Micro Vision adds a compact vision model. Community deployment and compression projects already exist. Saying Cohere has no initiative was incorrect.
2. **Liquid publicly treats device efficiency as a central design objective.** Its disclosed approach spans architecture search, tokenizer adaptation, quantization-aware distillation, multimodal models and deployment measurement. This is an assessment of published work, not evidence about either company's undisclosed roadmap.
3. **Liquid's current text model has a smaller file footprint.** Its published Q4_K_M is about 1.10 GB versus our verified Tiny Aya Q4_K_M artifact at 2.144 GB. These are different models with different parameter counts. This does not establish twice the speed, half the runtime RAM or equivalent language quality.
4. **Published vision scores favor LFM3-VL on the selected shared benchmarks below.** The evaluations were performed separately, with different settings. They justify a controlled comparison, not a universal winner claim.
5. **No controlled Tiny Aya–LFM3 Bangla quality/device comparison was found in the sources reviewed.** This is a bounded search result, not an exhaustive absence claim. Tokenizer compression, task accuracy and deployment speed are distinct endpoints.
6. **There is a plausible engineering opportunity in validated multilingual deployment.** A generic local chat app overlaps existing work. A reproducible, device-specific quality/latency/memory pipeline may add value. Cohere demand, willingness to fund it, employment and commercial viability remain unestablished.

## 2. Evidence rules

| Label | Meaning here |
|---|---|
| Published specification | Primary model card, technical report, product documentation or source listing was inspected |
| Author-reported result | A vendor/research team reports a measurement; we did not reproduce it |
| Local artifact evidence | Existing repository inspection or result was read; this review did not recollect it |
| Assessment | Our interpretation of sources, with its limits stated |
| Proposal | A possible next experiment/product; no execution or plan change is implied |
| Unresolved | Evidence was missing, inaccessible, incomplete or not audited deeply enough |

Marketing descriptions are not independent validation. A download format appearing in a model family does not prove every modality works in every runtime. Automatic library snippets on hosting pages can be less reliable than the model authors' explicit support notes. Dynamic `main` URLs can change; use the manifest's model revisions for later checks.

## 3. Liquid AI: what it does and how

### 3.1 Model family and backbone

Liquid announced LFM3 on 27 October 2026, with text, vision, audio, Omni and two decision variants derived from a shared small backbone. The published architecture has **24 layers: 13 gated short-convolution, five sliding-window attention and six global-attention layers**. The local attention window is 256 tokens; convolution kernel size is seven. Architecture search includes measurements across 14 device/backend combinations. Attention output gates, partial positional rotation in global attention and input/output normalization are disclosed design choices; gates are intended in part to control quantization-unfriendly outliers. Training is reported at 22T tokens. [L1]

Interpretation: the hybrid reduces the amount of history-dependent work compared with using global attention throughout. It does not eliminate attention, feed-forward computation, cache storage or hardware-specific kernel constraints. Do not transfer LFM2's exact layer counts or continuous-time liquid-network descriptions onto LFM3 without checking the released implementation. LFM2 is a prior hybrid generation, not the current architecture. [L2]

### 3.2 Multimodal components

| Model | Published scope | Relevant architectural distinction |
|---|---|---|
| LFM3-1.7B | Text; 256K context | Shared language backbone [L3] |
| LFM3-VL-2B | Images/video and text to text; 128K context | About 1.92B total: 1.73B language, 164M vision, 25M projector; the vision encoder itself mixes convolution and attention [L4] |
| LFM3-Audio-2B | Text/audio to text and/or speech; 32K context | Causal audio encoder, streaming prefill and LiquidCodec; 16 kHz audio input and 24 kHz speech output [L5] |
| LFM3-Omni-2B | Text, images, video and audio to text/speech; 32K context | 2.17B total; separate vision/audio encoders and connectors feed the backbone, with additional speech decoding [L6] |

The Omni flow is approximately:

```text
text -> tokenizer/embedding ----------------------+
image/video -> vision encoder -> connector -------+-> backbone -> text head
audio -> causal audio encoder -> connector -------+           -> speech decoder/codec
```

This is an integrated model with modality-specific components, not proof that one ordinary text GGUF file handles every input/output. Count encoder/projector/codec artifacts when sizing a deployment. The approximately 1 GB text checkpoint must not be presented as total Omni memory. [L6, L9]

The Audio card advertises 18 speech languages, with separate English locales. Bengali does not appear in the inspected Audio speech list or Omni text/speech lists. This means Bengali support is not established by those lists; it does not mean Bengali is untokenizable or that all tasks fail. Earlier broad language claims must retain this distinction. [L5, L6]

### 3.3 Tokenizer adaptation

Liquid's published LFM2.5-8B-A1B procedure expands its approximately 65K byte-level BPE vocabulary to 128K. It preserves the original merge ordering, continues learning merges on multilingual text, copies retained embedding rows and initializes new rows from the mean of their constituent old-token rows. It first trains only new rows, then unfreezes the model. Liquid reports 3.4x fewer Bengali tokens, 2.4x fewer Hindi tokens, 2.6x fewer Vietnamese tokens and 4.0x fewer Thai tokens **relative to its old tokenizer**. These are not Tiny Aya comparisons. [L7]

The technical report specifies 600B embedding-only training tokens and 400B full-model continued-pretraining tokens before subsequent training. Its latency analysis separates device per-token measurements from synthesized per-character speedups; do not relabel the latter as direct end-to-end application measurements. Larger vocabularies increase output-head cost, and previously efficient languages can regress in decode speed. [L8]

LFM3 is described as using that 128K vocabulary. A larger vocabulary can reduce fragmentation while increasing matrix size and work per generated token. Therefore vocabulary size, token count, model bytes and observed time must be measured separately. A tokenizer swap is not a safe file replacement: it changes the representation the model was trained to consume. [L1, L7]

### 3.4 Weight compression and quantization-aware distillation

Liquid released QAD checkpoints for four LFM2.5 sizes in August. A high-precision teacher guides training of a quantized student to recover capability lost through quantization. The reported benchmark-average retention is 96.5–97.4% of BF16. That is **not** 97% of the lost gap recovered: reported gap recovery ranges from 48.4% to 73.4%. These are LFM2.5 results, not a measured LFM3/Bangla outcome. [L10]

The current LFM3 text GGUF repository identifies its Q4_0 file as a QAD checkpoint. Its listed sizes include Q4_0 1.05 GB, Q4_K_M 1.10 GB, Q8_0 1.85 GB and BF16 3.47 GB. File labels and sizes do not establish uniform tensor precision or peak inference memory. [L9]

### 3.5 Deployment tooling and its limits

Liquid documents mainstream paths including llama.cpp, MLX, ONNX, Transformers and other serving/integration tools. Exact model/modality/version support must be checked per path. Its documentation index provides mobile embedding guidance. **LEAP SDK and its model bundling service are deprecated**; official migration guidance recommends direct llama.cpp integration. Old LEAP examples remain references, not a currently maintained proprietary advantage. [L11, L12]

Pipette is open-source device benchmark infrastructure developed with Artificial Analysis. It associates versioned model/quantization/runtime/device/workload configurations with latency, throughput and memory. Its initial published quality tasks are IFBench, GPQA Diamond and MATH-500; quality runs occur separately on designated GPU systems. A quality score beside phone timing does not mean that quality evaluation ran on the phone. Android coverage is CPU-focused in the described release; consistent cross-model NPU comparisons are absent. These are useful tools and explicit coverage gaps, not comprehensive multilingual or multimodal validation. [L13, L14]

## 4. Cohere: architecture, deployment and existing initiatives

### 4.1 Tiny Aya is deliberately local-capable

Cohere presents Tiny Aya as a compact multilingual model family, provides local weights and official Global/Earth/Fire/Water GGUFs, and also offers API access. Its launch explicitly discusses consumer devices and efficient tokenization. It is an initiative, not simply a community port of a server-only release. [C1, C2]

Our existing configuration/artifact inspection establishes:

- 3,349,227,520 stored parameters; 36 Cohere2 transformer blocks.
- Repeating three local-attention blocks then one global block; local window 4,096.
- Hidden width 2,048; 16 query heads and four key/value heads; vocabulary 262,144.
- Shared embedding/output source weights. In the pinned llama.cpp implementation, output loading duplicates the embedding source tensor.
- The shared matrix has 536,870,912 stored parameters, about 16% of the total. A precision change affects both uses.
- The verified local Q4_K_M artifact is 2,143,976,960 bytes. The inspected official file is 2,143,977,056 bytes; they are distinct identities, despite nearly equal size.

Sources: [local inspection](../reports/stage1-downloaded/inspection.json), [download inspection notes](../reports/stage1-downloaded/README.md), [tensor inventory](../reports/control-tensor-inventory.md), and the pinned implementation linked in the ledger [P1–P3]. These are existing inspections, not new sensitivity or phone results.

Cohere describes language/script-aware tokenizer sampling and a 262K vocabulary trained from a 50 GB sample. Tiny Aya's text-only input/output and local/global attention pattern are public. The paper reports phone decoding near 10 tokens/s on iPhone 13 and 32 tokens/s on iPhone 17 Pro, using a 100-input/100-output-token workload and MLX-converted models. Its Q4_K_M discussion gives 2.14 GB and 32.4 tokens/s, with a 1.4-point average mDolly judge-score loss. These are author-reported results; 2.14 GB must not be treated as an independently measured process-RAM ceiling. [C3, C4]

### 4.2 Compact multimodal offering and broader products

North Micro Vision is a 2.4B image/text-to-text model: an in-house approximately 2B language model plus a 400M encoder continued from SigLIP2. The vision training preserves native aspect ratios and document detail; the projector injects features from multiple vision depths. Cohere presents it as a customizable foundation for OCR, charts, documents and visual understanding. The backbone context is 128K, but multimodal validation is stated up to 8K. It is not a reasoning model; tool calling is not supported. [C5, C6]

Its card lists community MLX-VLM weights, an NVIDIA AutoModel fine-tuning/deployment recipe and Axolotl support. This contradicts a claim that no deployment ecosystem exists. Explicit text says public vLLM support is coming soon, despite generic hosting-page usage snippets; validate the actual version instead of assuming support. A compact downloadable model is not proof of a production phone application or measured mobile performance. [C5]

North is marketed for enterprise workflows with VPC/on-premises/private hosting. Those are deployment products, but private hosting does not imply execution inside a phone. Our interpretation is that Cohere's broader public commercial emphasis is enterprise/private deployment, while Liquid presents device efficiency as its central model-family proposition. This is positioning, not a ranking of internal investment. [C7]

### 4.3 Community work is direct prior art

Cohere's Expedition overview identifies:

- **Tiny Facade:** multilingual on-device tool calling, Android infrastructure and quantized models.
- **Language-Aware Quantization:** Tiny Aya compression across languages, mentored by Saurabh Dash.
- **Tiny Aya Vision:** connecting a visual encoder with much of Tiny Aya frozen.
- **Inside Tiny Aya:** sparse-autoencoder analysis of multilingual internal representations.

The overview establishes that these projects exist and overlap our interests. It does not expose sufficient detail to establish their exact allocation procedures, budgets, data separation or replication quality. In particular, the quantization summary has no directly linked detailed artifact in the inspected passage. Our exact experimental novelty remains unresolved. [C8]

Tiny Facade's own write-up describes an Android service/application and reports substantial template-related effects. These are team-reported findings, not a general rule that 4-bit always beats FP16 or that one sampler is universally best. It is a useful deployment prior-art lead and a potential collaboration starting point. [C9]

## 5. How open are the internals?

| Dimension | Tiny Aya | Liquid LFM3 | What access does not establish |
|---|---|---|---|
| Network/configuration | Inspectable; downloaded config already checked | Public config and implementation files | A complete causal account of language capability |
| Weights | Available subject to access/license terms | Available under LFM terms | Unrestricted commercial use |
| Inference implementation | Cohere2 implementations exist in common runtimes; pinned local path inspected | Published backbone, modality and codec Python files | Every accelerated backend has equal coverage |
| Training account | Substantial report and data-mixture descriptions | Architecture/training descriptions and tokenizer adaptation paper | Full reproducibility of original pretraining |
| Complete original corpus, training infrastructure and exact QAD reproduction | Not established in this review | Not established in this review | Absence from our inspection is not proof of secrecy |

Tiny Aya weights use CC-BY-NC with an acceptable-use addendum; its model card directs commercial inquiries to Cohere. North Micro Vision weights are Apache 2.0. Liquid's LFM Open License has a commercial-use revenue threshold provision around $10M; inspect its exact entity definitions and Sections 1/5 for a concrete deployment. Open weights/source visibility should not be conflated with unrestricted open-source licensing. The license on our original code does not relicense model weights. [C4, C5, L15]

Known wiring permits activation capture, tensor interventions and quantization studies. It does not tell us in advance which tensors causally preserve Bengali or whether a language-informed allocation will help. That remains an empirical question.

## 6. What can currently be compared?

### 6.1 Text footprint and device performance

| Observation | Tiny Aya Global | LFM3-1.7B | Interpretation |
|---|---|---|---|
| Parameters | 3.35B | 1.7B | Different capacities and training objectives |
| Q4_K_M file | Local verified 2.144 GB | Published rounded 1.10 GB | Roughly 1.95x file-size difference, not runtime-RAM or speed ratio |
| Published phone timing | About 10/32 tokens/s on iPhone 13/17 Pro | At least 1.8x decode and 2.5x prefill versus its selected comparison set at 32K context | Tiny Aya is absent from Liquid's comparison set |

Liquid's specified text device setup uses Q4_0, llama.cpp, batch one, 32,768-token context, 100 output tokens and GPU offload on AMD Ryzen AI Max+ 395, Apple M5 Max and iPhone 17 Pro. Its competitors include MiniCPM5, Qwen3.5 and Gemma 4 variants. Cohere's MLX/short-prompt setup is different. Do not calculate a Tiny Aya–LFM3 speedup by combining these charts. [L3, L9, C3, P2]

LFM3 text's reported capabilities include BFCL V4 71.39, IFEval 96.66 and SWE-bench Verified 43.40 in its stated agent/evaluation harnesses. There is no matched Tiny Aya evaluation here. These scores cannot substitute for Bangla task quality or compare models with unmatched reasoning/token budgets. [L3]

### 6.2 Vision quality: separate published evaluations

Numbers are percentages or normalized 0–100 scores, higher is better. Cohere fractions were multiplied by 100. Use the non-thinking Liquid column to reduce one obvious inference-mode difference; settings are still unmatched.

| Benchmark | North Micro Vision | LFM3-VL non-thinking | LFM3-VL thinking |
|---|---:|---:|---:|
| DocVQA validation | 92.1 | 94.9 | 96.7 |
| ChartQA test | 80.8 | 81.4 | 92.7 |
| RealWorldQA | 62.2 | 73.1 | 79.3 |
| OCRBench v1 | 79.2 | 82.2 | 86.3 |

Cohere used VLMEvalKit and capped generation at 1,024 tokens. Liquid used its stated vLLM evaluation and model-specific settings. These are not independently reproduced, matched quantized phone results. Thinking adds a variable compute/latency cost; the larger score alone is not a deployment comparison. [C5, L4]

Liquid's vision timing uses a Q4_0 language model plus F16 projector, one 512x512 image, 1,024 text tokens and 256 generated tokens on its selected devices. That does not establish performance for a native-resolution A4 document or include North Micro Vision as a direct timing competitor. [L4]

### 6.3 Bangla and our completed evidence

There is no verified Bangla winner in this review. Liquid's 3.4x tokenizer result compares its own old/new tokenizers; it is neither a Tiny Aya comparison nor a task-accuracy result. Tiny Aya's larger vocabulary likewise does not prove faster generation. Compare the same text in characters, tokens and elapsed time.

The companion repository's existing fresh T1 development result is BF16 GGUF versus local Q4_K_M, **not Liquid versus Cohere**:

| Language | BF16 correct / 300 | Q4 correct / 300 | Q4 minus BF16, pp [95% interval] |
|---|---:|---:|---:|
| English | 230 | 229 | -0.33 [-3.03, +2.36] |
| Bangla | 187 | 190 | +1.00 [-2.01, +4.15] |
| Hindi | 184 | 180 | -1.33 [-4.70, +2.03] |

These source-cluster-bootstrap intervals include zero. They establish neither equivalence nor a Bangla-specific quantization deficit. Bangla choice churn was 11% versus English 7%, approximately 1.57x, not twice. Read the [pinned complete T1 report](https://github.com/grinder81/polyglot-quant-eval/blob/c0ea61dc2d30fc7fcfd18e0b874563bf2056f918/results/t1-cold-baselines/README.md) for identities, transition counts and limitations. This review read that report but did not rerun its statistics. [P4]

## 7. Relation to our planned optimization

The approved [Stage 2 plan](stage2-plan.md) remains the authority. This research does not alter its gates, 2.2 GB total shipped-byte limit, endpoint roles or holdout policy.

| Intervention | What changes | Relation to this project |
|---|---|---|
| Tokenizer expansion | Vocabulary, embeddings and training | Liquid precedent; not our current fixed-tokenizer allocation experiment |
| Precision allocation | Storage precision per tensor | Our planned primary intervention; original model/tokenizer stay fixed |
| Quantizer calibration | Statistics from calibration text | Separate variable; language mix must be recorded |
| QAD/QAT | Training weights under low-precision constraints | Possible later recovery track; not equivalent to allocation |
| Architecture change | Operators, layer mixture and cache behavior | Liquid's foundation-model work; not a cheap conversion of Tiny Aya |
| Runtime engineering | Kernels, loading, templates, cache, scheduling | Deployment track; must preserve and test semantics |

The proposed sensitivity work measures real quantization damage, constructs tensor-by-format costs and allocates bytes. KL can be used to **score damage** in our probes, while teacher-matching losses can **train weights** in distillation. Sharing a metric does not make the methods identical. Precision allocation and QAD could eventually be combined, but their independent effects need controls.

The 29 October user-supplied mechanism cross-check proposed Bangla-, English- and pooled-informed allocations; a KL/Fisher/activation-proxy pilot; separate equal-budget rank and magnitude validation; and per-language KL alongside task endpoints. These remain proposals unless adopted in the governing plan. That note also reports direct prior art for degrade-one probes and budget allocation, including *Ranking Is Enough*. This review did not independently re-audit that preprint, its reported code absence, the other papers or license assessments. They must not be promoted to newly verified findings here.

The [existing research landscape](https://github.com/grinder81/polyglot-quant-eval/blob/c0ea61dc2d30fc7fcfd18e0b874563bf2056f918/docs/research-landscape.md) already rejects broad novelty claims for multilingual damage, mixed precision and mobile Tiny Aya. The newly located Cohere-mentored quantization project makes direct overlap checking still more necessary. [P5, C8]

## 8. Deployment opportunity: a hypothesis to validate

### 8.1 Concrete contribution

Proposed positioning: **a reproducible deployment pipeline for multilingual local models, with measured language quality and device cost**.

The deliverable would let a developer choose a device, language and workload and receive a tested model/precision/runtime configuration with its evidence. It could include:

- Artifact hashes, explicit runtime builds, model/template/tokenizer identities, supported operations and fallback records.
- A working offline integration with resumable verified downloads, load/unload, streaming, cancellation, context limits, memory-pressure handling and actionable failures.
- Per-language task-quality and fidelity checks, plus time to first token, completion latency, characters/s, peak memory and sustained behavior.
- Reproducible app/CLI builds and small regression fixtures that catch template, tokenizer and runtime changes.
- A configuration report that exposes tradeoffs instead of choosing solely by advertised tokens/s or parameter count.

These are proposed requirements, not evidence they are all absent from Cohere or other tools. Audit Tiny Facade, upstream apps, MLX support and Pipette before building equivalents. Kernel/backend contributions should follow profiling rather than assuming an NPU will be faster.

### 8.2 Possible directions and tradeoffs

| Direction | Why it fits | What would weaken the opportunity |
|---|---|---|
| Bangla offline assistant/translation | Native-language review and existing Tiny Aya controls | Poor measured quality or an existing solution already meeting the same constraints |
| Device validation/integration toolkit | Reuses our artifact/evaluation discipline | Existing open tooling covers the same needs with little integration friction |
| North Micro Vision document pipeline | Compact model, document focus, Apache 2.0 weights | Native-resolution cost exceeds target hardware or extraction quality misses requirements |
| Upstream runtime contribution | A reproducible issue/fix can benefit many users | No material bottleneck after profiling or correctness cannot be preserved |

The user's GPU workstation is useful for preparing artifacts, running quality evaluations and testing training-scale pilots. Its availability is user-reported; no new hardware inventory was collected. It is not evidence of phone throughput or sufficient capacity to reproduce Liquid's trillion-token tokenizer adaptation.

Potential outcomes include an open-source portfolio, research artifact, integration service or a concrete proposal to Cohere. No paying customer, partnership, grant, job or unmet internal need has been verified. Before claiming a business opportunity, demonstrate a measurable improvement and validate that users actually experience the problem.

Licensing affects the product route: Tiny Aya's noncommercial weights require resolving commercial terms for commercial use. North Micro Vision's Apache 2.0 release is a different starting point. A generic wrapper cannot remove either model's license conditions. [C4, C5, L15]

### 8.3 Smallest useful validation, proposed only

1. Pick one target device already available and one task. Record actual hardware/OS/build access; preserve the approved iPhone protocol when using that study track. An Android pilot is a separate run identity, not a substitution for the planned iPhone result.
2. Reproduce unmodified Tiny Aya behavior on that device first. Make loading, templates, tokenization, timing boundaries and failures observable. Compare with a trusted desktop path using the same artifact; record disagreement rather than assuming bit-identical GPU execution.
3. Add LFM3 as an **external model comparator**, using the same semantic prompts and matched operating constraints. Keep each model's correct chat template. Report task quality and actual output lengths; use character-based throughput alongside native tokens/s. A fixed-token synthetic benchmark is separate from the same-text user workload.
4. Within Tiny Aya, compare default and language-informed allocations at equal measured shipped bytes under the existing gates. External-model comparisons answer a different question; identical quantization labels are neither equal bytes nor equal quality.
5. Publish raw evidence and a usable integration, with clear wins, losses and unsupported cases. Investigate sustained-use behavior and memory at increasing context before expanding to more devices or modalities.

For every configuration, pin model revision and file hashes; tokenizer/template; quantizer and calibration; runtime/kernel/backend; context/KV precision; thread/offload settings; sampling; prompt membership; output stop reason; and measurement units/scope. Separate cold loading, prompt prefill, first token and generation. Observe power/cooling conditions and repeat measurements with spread.

Follow [the existing device protocol](../device/README.md): no phone results before hardware runs. Battery percentage is not joules; categorical thermal state is not Celsius; file size is not resident memory. Missing measurements remain unavailable. A model/runtime discrepancy must be resolved or bounded before transferring desktop quality to phone claims. [P6]

Use Pipette's compatible clients/protocols where they fit, with attributed reuse under the actual code license. Its published quality suite needs additional Bangla and application endpoints for this project. Do not build a new dashboard merely because one can be built. [L13, L14]

## 9. Corrections and nonclaims for the next reviewer

| Earlier statement or possible inference | Correct record |
|---|---|
| Cohere has no on-device initiative | False: Tiny Aya quantization/phone work, compact vision release and mentored projects exist |
| Liquid has a uniquely necessary deployment SDK | LEAP is deprecated; current guidance favors direct mainstream runtime integration |
| Tiny Aya/LFM weights are unrestricted open source | Source visibility, weights and licenses must be distinguished; North Micro Vision differs from Tiny Aya |
| Liquid's Bengali token reduction proves it beats Tiny Aya | It compares Liquid's old/new tokenizer, not Tiny Aya or Bengali task accuracy |
| Bengali absent from an advertised support list means the tokenizer cannot encode it | Unsupported inference; tokenizer coverage and demonstrated capability differ |
| Approximately 1 GB text means 1 GB Omni deployment | False: modality artifacts, caches, activations and runtime memory must be counted |
| 97% BF16 retention means 97% quantization damage recovered | Different denominators; see QAD retention versus gap recovery |
| Liquid is a measured multiple faster than Tiny Aya | Not established by the different published timing setups |
| Higher vision scores establish better quantized phone behavior | Not established without matched configuration, task and inference budget |
| Tiny Aya's observed English/Bangla quality gap proves compression harms Bangla more | Existing paired development evidence does not establish that interaction |
| Language-aware allocation, Tiny Aya Android deployment or multilingual damage are new | Direct prior art exists; exact contribution needs review |
| Cohere will fund/hire/buy this deployment work | No such evidence or contact exists |

The narrower defensible opportunity is to improve a specified deployment and show evidence that matters to its users. Claims about market size, competitor superiority, novelty, model licensing and unannounced roadmaps remain outside what this review establishes.

## 10. Independent review request

Suggested prompt to send with this PR:

> Review the report, source ledger and revision manifest independently. Separate primary specifications, vendor results, existing local evidence and the author's proposals. Verify architecture/model identities, tokenizer old/new denominators, QAD retention versus recovered loss, license wording, and benchmark units/settings. Check whether the Cohere community projects or existing runtime tools already implement the proposed deployment contribution. Do not assume the absence of a public product proves unmet demand. Assess whether the smallest validation is useful and whether it preserves the current study's controls. Return supported corrections and unresolved questions; do not treat this document as authority to change protocols or run the holdout.

Priority questions:

1. Can the current model-card claims be reproduced at the pinned revisions, and are any source statements internally inconsistent?
2. Is there now a truly matched Tiny Aya–LFM3 device/Bangla evaluation that this bounded search missed?
3. What exactly did the Cohere-mentored Language-Aware Quantization team implement, on which languages and budgets?
4. Which deployment features are already covered by Tiny Facade, upstream mobile apps, MLX and Pipette?
5. Which one task/device delivers the most informative low-cost proof, and what result would reject the opportunity?
6. Does commercial usage require different model selection or a separate license agreement?

This PR is intended to remain reviewable before any implementation or commercial claim follows from it.

Author validation: nine model-repository revision strings and four existing local-evidence hashes checked; sixteen relative Markdown links resolved; JSON parsed; eleven selected rows matched three revision-pinned raw model READMEs; `git diff --check` passed. These mechanical/source-transcription checks do not constitute an independent scientific review. No executable code changed, so no model or software test suite was run.

## Appendix: collaborator pitch draft

I am looking for a PhD student or research collaborator interested in multilingual model compression and practical on-device deployment. I am studying Tiny Aya with a focus on Bangla and paired English controls: how quantization changes task performance, whether language-informed precision allocation helps at the same shipped size, and whether those gains survive real-device deployment. I have a GPU workstation available for experiments and existing artifact/evaluation infrastructure. The proposed work includes reproducible benchmarks, controlled interventions and an open research/software contribution. Prior-art review, experimental design and contribution expectations would be agreed together; a publication outcome is not guaranteed.

This is an unpublished draft, not an employment or funding offer. No collaborator/community message was sent as part of this task.
