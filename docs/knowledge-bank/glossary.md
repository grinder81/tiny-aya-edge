# Glossary

| Term | Meaning in this project |
| --- | --- |
| Activation | An input-dependent intermediate value computed during a forward pass. |
| Activation proxy | An approximate error score that weights squared weight error by input activation energy. |
| Allocation | An assignment of actual stored tensor formats under a shipped-byte constraint. |
| Attention head | A projection subspace and attention operation; not necessarily an independent stored tensor. |
| Background precision | The formats used by the rest of the model during a local intervention. |
| BF16 | A 16-bit floating-point representation; the GGUF BF16 control also contains F32 normalization vectors. |
| Block | One of Tiny Aya's 36 residual updates, containing parallel attention and MLP branches. |
| Branch | One path contributing an output to the block's residual update. |
| Calibration | Collecting representative-data statistics to guide a quantizer; not inherently optimizer training. |
| Causal mask | Restricts each position to permitted past/current positions, excluding the future. |
| Channel | One coordinate of a representation; its meaning depends on the basis and jointly learned computation. |
| Confirmation | Evaluation on independent sources after a choice has been frozen. |
| Decode | Generate tokens autoregressively after prefill, reusing compatible cached state. |
| Degrade-one | Compress exactly one selected tensor in an otherwise BF16 model to measure its introduced damage. |
| Diagonal empirical Fisher | A statistic of squared per-example gradients used here as a damage proxy; not an exact KL Hessian. |
| Embedding | A vocabulary-by-hidden matrix used for token lookup; Tiny Aya shares its source with the output projection. |
| FFN or MLP | The token-position-wise gated feed-forward branch using gate, up and down matrices. |
| Fidelity | Agreement with a reference model's predictions; distinct from task correctness. |
| Gate | The nonlinear factor in Tiny Aya's MLP product; SiLU outputs are not binary or restricted to [0,1]. |
| GGUF | A model artifact format containing metadata, tensor descriptions, payloads and alignment. |
| Global attention | Causal access over the available prefix, as opposed to a fixed local window. |
| GQA | Grouped-query attention: 16 query heads share four KV heads in this checkpoint. |
| Gradient | Derivative of an objective with respect to parameters; computing it does not itself update weights. |
| Hidden width | Number of features per residual vector: 2,048 here. |
| Imatrix | Text-derived importance statistics used by supported native quantization paths. |
| Intervention | A defined change to weights, state or computation, with an explicit comparison and held-fixed conditions. |
| KL divergence | A directional distribution difference; this pilot uses KL(BF16 || candidate). |
| KV cache | Stored input-dependent keys and values reused during autoregressive decoding. |
| LayerNorm | Centers and rescales features with a learned scale here; different from RMSNorm. |
| Logit | An unnormalized next-token score before softmax. |
| Low rank | Factoring a matrix through a smaller intermediate dimension, changing its parameterization. |
| Native format | The actual supported packed tensor representation and kernels, rather than generic simulated rounding. |
| Operating point | A development-selected complete model that supplies a fixed starting byte budget and inventory. |
| Payload | Bytes that encode tensor values, excluding artifact metadata and alignment. |
| Prefill | Process a supplied token prefix and initialize its compatible attention state. |
| Proxy | A cheaper estimate whose relationship to the intended endpoint must be validated. |
| Q/K/V/O | Query, key, value and attention output projections. |
| Quantization | Approximation of numbers with lower-storage codes plus required metadata. |
| Residual stream | The evolving hidden representation updated by each block. |
| Restoration | Replacing selected damaged computation with a reference operation under a specified trajectory. |
| RoPE | Position-dependent rotations of Q/K coordinates, applied only at local blocks here. |
| Shipped bytes | All artifacts required for the proposed deployment, counting each stored artifact once. |
| SiLU | The activation s × sigmoid(s), used on the MLP gate projection. |
| Softmax | Converts scores into a normalized distribution; changing candidate vocabulary changes its denominator. |
| Source cluster | Related examples grouped by source for separation or uncertainty calculations. |
| Tensor | A numerical array; matrices and normalization vectors are both tensors. |
| Tied weights | Multiple computational uses share an intended parameter/source; runtime allocations need separate verification. |
| Token | A vocabulary ID, not necessarily a word or Unicode character. |
| Up/down projection | Expand the MLP representation to 11,008 channels, then project it back to 2,048. |

In equations, h is an incoming residual column vector, x is its normalized version, E is the shared vocabulary matrix, W denotes a learned projection, and elementwise multiplication is written `*`. PyTorch matrix shapes are output × input. GGUF shape order is listed separately in the atlas.
