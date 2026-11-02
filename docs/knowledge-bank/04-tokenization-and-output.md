# Tokenization and the shared language interface

## Text and tokens are different units

A tokenizer turns text into a sequence of vocabulary IDs. A token can represent a whole word, part of a word, punctuation, whitespace or another encoded unit. Unicode characters, words and tokens are different counting units. A Bangla word may use a different number of tokens from an English translation, and a visible character can contain multiple Unicode code points.

For any claim about Tiny Aya's segmentation, inspect the actual tokenizer output under the pinned tokenizer rather than guessing from the script. The current knowledge bank does not tokenize fresh examples. Its numeric vocabulary size comes from the inspected configuration; no illustrative Bangla token boundaries are presented as measurements.

Tokenization affects context consumption, number of decoding steps and calibration coverage. It does not by itself determine understanding or correctness. More characters per token can increase character throughput even if tokens per second falls; comparing models by token speed alone can obscure the user-visible trade-off.

## The embedding table has two jobs

The shared matrix E has shape **262,144 × 2,048** and 536,870,912 parameters. At input, selecting row t supplies the initial vector for token ID t. At output, multiplying the final normalized hidden vector by E produces one score for every vocabulary token:

```text
input vector for token t:  h_0 = E[t, :]^T
final vocabulary scores:  z = E n
probability(token v):      exp(z_v) / sum_u exp(z_u)
```

The same stored source weights participate in both operations. The GGUF has `token_embd.weight` and no separately stored `output.weight`; the pinned native loader explicitly initializes its output matrix from that same tensor with its duplication flag. See the [storage inventory](../../reports/control-tensor-inventory.md) and [pinned native source](sources.md).

This establishes shared storage/source semantics. A backend can still maintain separate allocations or transformed copies, so it does not prove runtime RAM equals one copy. A differentiable implementation must also verify that both uses share the intended parameter and receive the combined gradient.

## Why this matrix deserves attention

The embedding contains **16.03% of all stored parameters**. Its BF16 payload is 1,073,741,824 bytes; its actual Q6_K payload in both our Q4 and operating Q3 exports is 440,401,920 bytes. These are payloads for one stored tensor, excluding file metadata and runtime state. The actual file labels therefore hide a substantial higher-precision component.

Its two uses create two possible damage paths. Perturbed input rows change a token's starting representation. Perturbed output rows change how final representations are scored against the vocabulary. An embedding intervention can affect both simultaneously; an improvement or failure cannot be automatically attributed to only one side.

For the proxy in the frozen pilot, input lookup frequency and final-head activation contributions are reported separately before their approximate combination. Even that accounting does not establish that their effects on complete output KL are independent. A later side-specific mechanism experiment would need to untie execution carefully, keep a matching baseline and account for any additional stored/runtime copy.

## Vocabulary size and logit memory

One full float32 vocabulary-score row contains 262,144 × 4 = 1,048,576 bytes, exactly one MiB. Saving many rows therefore consumes far more storage than saving the token IDs or four answer scores. The completed T3 reference required 176,756 content-token distributions: 185,342,099,456 raw bytes before array headers.

That cache supported repeated full-vocabulary comparisons. It is distinct from a model file and from the attention KV cache. This distinction explains why a roughly 1.8 GB model can coexist with hundreds of GB of research artifacts. A future measurement strategy can stream or recompute distributions, but changing one after outcomes requires a versioned and numerically verified implementation.

## Language comparisons need two views of coverage

Equal token counts answer a resource-controlled question: each arm receives the same amount of model-token evidence. Parallel translations answer a content-controlled question: each arm starts from aligned material. Truncating each translation to 64 model tokens can break exact semantic equivalence because the languages consume that budget differently.

The frozen pilot uses aligned source spans and equal token budgets, then explicitly acknowledges truncation. Future studies could report both fixed-token and fixed-content comparisons, with lengths and truncation recorded separately. A larger English sample or a pooled sample with triple the evidence would confound the language choice with the amount of information used.

Script, domain, named entities, translation quality and code mixing also matter. A statistical difference associated with Bangla text could reflect orthography, token fragmentation, subject matter or content coverage. Establishing a language-specific computational mechanism requires controls that distinguish these possibilities.

## Tokenizer modification is a separate intervention

Changing bit widths preserves token IDs and model dimensions. Changing the tokenizer may introduce new IDs, change which sequences the model sees, or require new embedding/output rows and training. A smaller token count is not automatically a smaller shipped model: a larger vocabulary can increase embedding and head costs.

Deleting unused vocabulary rows is also not a neutral storage operation. Even if the surviving rows are numerically unchanged, their softmax denominator changes when other candidates disappear. Byte fallback, special tokens, punctuation, numbers, names and code mixing need explicit coverage. A restricted-vocabulary scoring result does not establish unrestricted-generation equivalence.

Liquid AI's tokenizer and quantization-aware distillation work can inform a separate research question, as recorded in the [deployment review](https://github.com/grinder81/tiny-aya-edge/blob/2883795/docs/liquid-cohere-deployment-review-2026-10-29.md) on its branch. It does not supply an unmodified drop-in tokenizer for Tiny Aya or a matched performance result. The current model's weights, vocabulary and chat template remain the compatibility boundary.
