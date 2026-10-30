# Liquid/Cohere source and access ledger

Companion to the [review](liquid-cohere-deployment-review-2026-10-29.md). Inspected on 29 October 2026 America/Los_Angeles; final metadata checks on 30 October UTC. The [manifest](liquid-cohere-source-pins-2026-10-29.json) records observed Hugging Face heads and local file hashes. A primary source verifies that an author published a claim; it does not independently validate that claim.

## Liquid AI primary sources

| ID | Source | What was inspected / supports | Limits |
|---|---|---|---|
| L1 | [LFM3 announcement](https://www.liquid.ai/blog/lfm3), 27 October 2026 | Family, 24-layer operator mixture, hardware search, vocabulary reuse, training scale, normalization/gating | Vendor account; no training or timing reproduced |
| L2 | [LFM2 technical report](https://arxiv.org/abs/2511.23404) and [LFM2 launch](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models) | Earlier convolution/attention family and multiplicative-gate mechanism | Earlier generation; not LFM3 layer counts |
| L3 | [LFM3-1.7B model card](https://huggingface.co/LiquidAI/LFM3-1.7B) | Specifications; capability table; edge inference setup and named comparators | Selected sections directly read; benchmark claims are not replicated |
| L4 | [LFM3-VL-2B model card](https://huggingface.co/LiquidAI/LFM3-VL-2B) | Components; non-thinking/thinking scores; vision input and inference setup | Separate vendor evaluation; no Cohere timing comparison |
| L5 | [LFM3-Audio-2B model card](https://huggingface.co/LiquidAI/LFM3-Audio-2B) | Audio architecture, streaming, sample rates, context and speech list | No ASR/TTS/audio benchmark reproduced; task support is not uniform language support |
| L6 | [LFM3-Omni-2B model card](https://huggingface.co/LiquidAI/LFM3-Omni-2B) | Modality flow, 2.17B total, 32K context, text/speech lists | Diagram/specification inspected; deployment not run |
| L7 | [Tokenizer expansion engineering article](https://www.liquid.ai/blog/tokenizer-expansion), 21 July 2026 | Continued BPE, row initialization, old/new token reductions | Bengali 3.4x is relative to Liquid's source tokenizer |
| L8 | [In-Place Tokenizer Expansion for Pre-trained LLMs, v1](https://arxiv.org/html/2607.15232v1), 16 July 2026 | Selected method/training/latency/availability sections; 600B/400B adaptation | Entire training/data implementation not audited; synthesized latency is labeled |
| L9 | [LFM3 text GGUF release](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) | QAD Q4_0 designation and displayed file sizes | Rounded sizes; artifact files not downloaded/hashed here |
| L10 | [QAD announcement](https://www.liquid.ai/blog/qad), 19 August 2026 | Four LFM2.5 sizes; retention/gap-recovery denominators; deployment framing | Not a Bangla result or LFM3 replication; no complete training reproduction package established |
| L11 | [Liquid documentation index](https://docs.liquid.ai/llms.txt) | Current runtime, fine-tuning and mobile-documentation navigation | Index presence does not certify each model/backend combination |
| L12 | [Official deprecations](https://docs.liquid.ai/lfm/help/deprecations), [LEAP migration](https://docs.liquid.ai/deployment/on-device/llama-cpp/migrating-from-leap-sdk), [publisher's PyPI notice](https://pypi.org/project/leap-bundle/) | LEAP SDK/bundling deprecation; direct llama.cpp replacement | Older index examples still mention LEAP; do not infer maintenance from an old example |
| L13 | [Pipette launch](https://www.liquid.ai/blog/pipette-on-device-ai-benchmarking-by-liquid-ai), 24 August 2026 | Open benchmark clients/data, quality-task scope, Android/NPU limitations | Launch coverage is time-specific; not an independently rerun model leaderboard |
| L14 | [Pipette overview](https://pipette.liquid.ai/docs/introduction), [comparison rules](https://pipette.liquid.ai/docs/reading-results), [replication guide](https://pipette.liquid.ai/docs/methodology/replication-guide) | Configuration matching, separation of GPU quality and device timing, reproducibility paths | No complete data export or submission replication in this review |
| L15 | [LFM Open License](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE) | Definitions and Sections 2/5, including commercial revenue threshold | Factual summary only; exact legal-entity/use facts must be assessed for a real product |
| L16 | [Text implementation](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py), [Omni files](https://huggingface.co/LiquidAI/LFM3-Omni-2B/tree/main) | Public forward-model source and configuration/codec/processor file availability | Source listing and selected implementation inspected; no full code, kernel or supply-chain audit |

## Cohere primary and project-authored sources

| ID | Source | What was inspected / supports | Limits |
|---|---|---|---|
| C1 | [Tiny Aya launch](https://cohere.com/blog/cohere-labs-tiny-aya) | Local/mobile intent, regional family and access | Product/research positioning, not an independent quality ranking |
| C2 | [Tiny Aya documentation](https://docs.cohere.com/docs/tiny-aya) | API, local models and official GGUF variants | Runtime execution not performed |
| C3 | [Tiny Aya technical report, v1](https://arxiv.org/html/2603.11510v1) | Tokenizer sampling; Section 6 quantization and iPhone measurements | Selected sections directly read; author-reported results, not ours |
| C4 | [Tiny Aya Global model card](https://huggingface.co/CohereLabs/tiny-aya-global) | Open-weight research status, text scope, attention pattern and noncommercial terms | Gated files require accepted terms; unrelated templated “111 billion” phrase conflicts with 3.35B specification and was not adopted |
| C5 | [North Micro Vision model card](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct) | Specifications, benchmarks, limitations, Apache 2.0 and explicit ecosystem notes | Generic vLLM snippets conflict with author text “coming soon”; actual current support needs verification |
| C6 | [North Micro Vision technical article](https://huggingface.co/blog/CohereLabs/meet-north-micro-vision-instruct), 12 August 2026 | Native-resolution design, encoder origin and DeepStack-style feature injection | Selected architecture/introduction sections; not an independent phone demonstration |
| C7 | [North product page](https://cohere.com/north), [model catalog](https://cohere.com/models-overview) | Enterprise/private deployment and presence of compact model offerings | Neither establishes phone execution of North itself or internal roadmap priorities |
| C8 | [Tiny Aya in the wild](https://cohere.com/blog/blog/tiny-aya-in-the-wild) | Named mentored Android, language-aware quantization, vision and interpretability projects | Project-summary depth; exact quantization method and detailed artifact unresolved |
| C9 | [Tiny Facade team write-up](https://huggingface.co/blog/Bronsn/tiny-facade), 20 April 2026 | Existing Android service/app and template/quantization evaluation claims | Project-authored report; numerical claims not independently audited or generalized |

## Existing repository evidence

| ID | Evidence | Depth and boundary |
|---|---|---|
| P1 | [Downloaded configuration](../reports/stage1-downloaded/inspection.json), [inspection notes](../reports/stage1-downloaded/README.md) | Read actual config fields and existing original/official tensor inspection; no new weight download |
| P2 | [Controlled tensor inventory](../reports/control-tensor-inventory.md) | Existing complete-file bytes, parameter counts, formats and shared-matrix evidence read; no new model execution |
| P3 | [Pinned llama.cpp Cohere2 source](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/src/models/cohere2.cpp) | Pointer backing existing inventory's tied-source explanation; not a new complete forward-pass audit |
| P4 | [Fresh T1 report](https://github.com/grinder81/polyglot-quant-eval/blob/c0ea61dc2d30fc7fcfd18e0b874563bf2056f918/results/t1-cold-baselines/README.md) | Existing counts, intervals and nonclaims read; statistics/raw archives not rerun |
| P5 | [Existing research landscape](https://github.com/grinder81/polyglot-quant-eval/blob/c0ea61dc2d30fc7fcfd18e0b874563bf2056f918/docs/research-landscape.md) | Prior-art/context pointer; not a fresh audit of every linked paper |
| P6 | [Device protocol/readiness](../device/README.md), [Stage 2 plan](stage2-plan.md) | Existing local project rules and implementation status read; protocol remains unchanged |

## Access failures, source conflicts and exclusions

- The web reader could not directly retrieve the large Liquid text `tokenizer.json`. The BPE recipe is supported by the engineering article and paper; this document does not claim a complete tokenizer-file audit.
- A direct Tiny Aya config request encountered gating in the earlier investigation. Existing, authorized downloaded configuration evidence was used. Public API metadata retrieval did not bypass gated weight access.
- Liquid's text edge plot asset failed through its redirected CDN in the web reader. The model card's textual ratios, comparator list and method were readable. Exact plot values were not invented or visually transcribed.
- One revision-pinned Liquid vision README also failed in the web reader. A subsequent ordinary HTTPS retrieval of the pinned raw README succeeded; selected table rows were parsed and checked. This did not require credentials or bypass access controls.
- The Markdown mobile-documentation URL returned an unsupported-content-type error in the web reader. The documentation index and readable official migration pages support the limited runtime/navigation claims made here.
- A guessed Cohere North Micro documentation URL failed. The model card and official author technical article were used instead.
- Some web search summaries exposed older support language. Current explicit author instructions take precedence over generic auto-generated snippets; conflicts remain review items.
- No authenticated Discord search was performed for this Liquid/Cohere review. Earlier conversation/community readings do not become independently verified evidence through repetition.
- The user-provided 29 October mechanism cross-check is prior context. Its Crossref/preprint checks and software-license conclusions were not independently repeated here. The report labels its design suggestions as proposals.
- General press, comparison sites, search-result rankings, download counts and an unofficial impact-report domain were not used as authority for technical superiority or business demand. A community transcription wrapper was inspected as a deployment lead but adds no evidence for the specific mobile-text opportunity and is not used to assert it.
- No complete systematic-review search, runtime compatibility matrix, enterprise contract, commercial license approval, private roadmap or customer interview was obtained.

## Verification boundaries for another agent

The manifest pins model repository heads observed through `/api/models/<id>`. It does not pin mutable blogs or preserve full copyrighted pages. Three revision-pinned raw READMEs were fetched over HTTPS, with eleven selected benchmark rows checked against the report and UTF-8 text hashes recorded. This is a transcription check, not benchmark replication or a full README/code audit. Other pinned config/implementation contents should be checked by the reviewer. Local file SHA256 values bind Windows CRLF working-tree bytes; the additional Git blob IDs identify the portable repository versions, so an LF checkout need not match those raw working-tree hashes. URLs establish inspectability; a source being public is not permission to copy code under a different license.
