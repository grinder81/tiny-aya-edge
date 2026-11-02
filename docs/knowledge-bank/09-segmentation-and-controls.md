# Segmentation inside a matrix

## Choose the unit that matches the question

“Which layer matters?” can refer to several different experiments. A whole-block intervention changes both branches and their residual update. A branch intervention changes attention or MLP. A matrix intervention changes one projection. A head/channel intervention changes a structured subset of coefficients. A single-weight intervention changes one scalar.

These units have different costs, interaction patterns and meanings. The eight-site pilot chooses **whole stored matrices** because the native exporter supports a clear tensor-type intervention. It does not identify the important heads or neurons within those matrices.

| Unit | Tiny Aya interpretation | A useful question | Main interpretation risk |
| --- | --- | --- | --- |
| Block | Parallel attention + MLP + residual update | Does bypassing this update affect a specified capability? | Changes several mechanisms at once; skipped cache work needs care |
| Branch | One attention or MLP path | Does restoring this branch rescue a damaged trajectory? | Teacher-state injection can repair earlier damage too |
| Matrix | One of seven per block | What damage comes from this projection's native quantization? | Cannot resolve which internal subgroup caused the effect |
| Query head | 128 Q coordinates plus corresponding attention output contribution | Which routed information is useful? | KV sharing and output recombination must be respected |
| KV head | 128 K coordinates and 128 V coordinates shared by four query heads | Which shared memory content matters? | Intervention affects multiple query heads |
| MLP channel | A gate row, up row and matching down column | Does a channel group carry a recoverable feature interaction? | A coordinate can participate in many behaviors |
| Embedding row | One token's input vector and output classifier row | Is damage concentrated on a defined vocabulary slice? | Tied dual use and token-frequency differences confound interpretation |
| Residual feature | One of 2,048 hidden coordinates | Is a representation direction causal under an intervention? | Normalization and all downstream projections interact |

## Attention segmentation in logical PyTorch coordinates

The reference Q projection emits 2,048 numbers and reshapes them into 16 heads of width 128. In that logical row ordering, query head j occupies rows `128*j` through `128*(j+1)-1` of W_Q. K and V emit 512 numbers each, organized as four KV heads of width 128.

The reference repeats each KV head for a contiguous group of four query heads. In its logical grouping, query heads 0–3 share KV head 0, 4–7 share KV head 1, 8–11 share KV head 2 and 12–15 share KV head 3. The attention output concatenates the 16 query-head results before W_O. A selected query-head contribution therefore corresponds to its appropriate input columns of W_O, not its rows.

These statements describe the source's logical tensor layout. A fused or packed native backend may store or expose a different layout. Conversion, head reshaping and positional-coordinate conventions must be checked before editing packed rows. The existing whole-matrix pilot verifies native target-row parity rather than assuming a generic NumPy rounding operation is equivalent.

RoPE further couples coordinate pairs in local Q/K computation. A coordinate permutation that changes those pairings without compensation changes the function. A whole-head intervention also needs to preserve the intended GQA mapping. “These rows have the same count” is not enough for a valid architecture transformation.

## MLP segmentation couples three matrices

For intermediate channel c, the contributing parameters are row c of W_gate, row c of W_up and column c of W_down in PyTorch's output-by-input convention. Their product and recombination define the channel's contribution. Changing all three consistently can implement a valid permutation or diagonal rebalancing, as derived in [chapter 3](03-mlp-and-residual.md).

This coupling explains why labeling a gate row as a standalone “neuron” can obscure the actual computation. Gate and up factors interact before down mixes their products into every residual coordinate. A feature direction can be distributed across channels, and one channel can participate in multiple behaviors depending on its input.

Ablating a channel, reducing intermediate width, rescaling it and quantizing its weights answer different questions. Ablation changes the function; a compensated scale can preserve the exact function before quantization; native precision changes its approximation. Their results should not be pooled as one sensitivity score.

## Quantizer blocks are another segmentation

Packed low-bit formats approximate groups of numerical values with shared metadata. Those storage groups need not coincide with attention heads, linguistic features or whole MLP channels. A row permutation may change which values share a quantization block, while another permutation may leave the grouping effectively unchanged.

For a given intervention, determine the native row direction, block length, scales, packing and supported kernels from the pinned implementation and exported bytes. The existing atlas lists both shape conventions and exact payloads. It deliberately does not equate “one quantization block” with “one neuron.”

## Embedding slices need a defined denominator

An embedding row can be assigned to a vocabulary slice using a reproducible tokenizer analysis, such as the token string's script or actual occurrence in a corpus. Those definitions answer different questions: a token with Latin characters can occur in Bangla text, and Bangla text can contain digits, punctuation and mixed-language names.

Report how the slice was constructed, its row count, parameter count, token frequency and shared-token membership. A larger summed error on a larger or more frequent slice is not sufficient evidence of a language-specific defect. A token's output role also affects its probability relative to every other candidate through softmax normalization.

The knowledge bank does not assign new rows to language slices or access protected labels. Such a study would need its own source and data definitions.

## A useful progression of granularity

Begin with a component-level question that the exporter and measurement instrument can validate. If a role/depth effect repeats, ask whether finer segmentation would change a practical intervention. Move to heads or channels only with a layout-aware mechanism, matched controls and an export/cost hypothesis.

Fine-grained inspection is useful when it resolves a decision. It is not automatically more causal or more useful than a complete-model result. The intended progression is understanding the graph, selecting a testable unit, measuring under a defined background, confirming independently and only then explaining the observed behavior at the level the evidence supports.
