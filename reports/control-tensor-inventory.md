# Controlled tensor storage inventory

This is a read-only prerequisite for C8, not a sensitivity result. No calibration,
held-out data, activations or scoring were used. Full per-tensor shapes, types,
parameter counts and payload bytes are in `control-tensor-inventory.json`.

Reproduce in the pinned converter environment (gguf 0.19.0):

```sh
python builds/inventory_control_tensors.py \
  /home/shohag/tiny-aya-stage1/models/tiny-aya-global-control-bf16.gguf \
  /home/shohag/tiny-aya-stage1/models/tiny-aya-global-local-q4_k_m.gguf
```

The script requires identical tensor names, shapes and parameter counts across
artifacts, rejects duplicate names, and hashes both complete files. It counts
unique stored tensors once, rather than counting input/output uses twice.

| Group | Tensors | Parameters | BF16 payload bytes | Local Q4 payload bytes |
|---|---:|---:|---:|---:|
| Shared embedding/output | 1 | 536,870,912 | 1,073,741,824 | 440,401,920 |
| MLP down | 36 | 811,597,824 | 1,623,195,648 | 561,143,808 |
| MLP gate | 36 | 811,597,824 | 1,623,195,648 | 456,523,776 |
| MLP up | 36 | 811,597,824 | 1,623,195,648 | 456,523,776 |
| Attention Q | 36 | 150,994,944 | 301,989,888 | 84,934,656 |
| Attention K | 36 | 37,748,736 | 75,497,472 | 21,233,664 |
| Attention V | 36 | 37,748,736 | 75,497,472 | 26,099,712 |
| Attention output | 36 | 150,994,944 | 301,989,888 | 84,934,656 |
| Block normalization | 36 | 73,728 | 294,912 | 294,912 |
| Final normalization | 1 | 2,048 | 8,192 | 8,192 |

Both artifacts contain 290 tensors and 3,349,227,520 stored parameters. BF16 has
253 BF16 and 37 F32 tensors; local Q4 has 216 Q4_K, 37 Q6_K and 37 F32 tensors.
The embedding is Q6_K. Some attention-V and MLP-down matrices are Q6_K; consult
individual rows rather than treating an entire group as uniformly four-bit.

BF16 occupies 6,710,484,480 bytes and Q4 occupies 2,143,976,960 bytes. Each has
11,877,888 bytes outside tensor payload (metadata and alignment). Allocation
budgets must use complete shipped artifacts, not just payload estimates.

## Shared input/output handling

The config does not explicitly set `tie_word_embeddings`; absence alone is not
proof of tying. The stored artifact has `token_embd.weight` and no independent
`output.weight`. More importantly, the pinned llama.cpp source at
`797de63b2fa7396dcc8caca76834c699ac84e1c3`, `src/models/cohere2.cpp`, loads
`tok_embd` from `LLM_TENSOR_TOKEN_EMBD` and loads `output` from the same tensor
name with `TENSOR_DUPLICATED`. The forward path uses `model.output` for logits.
Thus a stored-weight intervention changes the source for both uses. This does
not claim a single allocation in every backend or predict runtime memory.

Any later BF16 intervention implementation must verify its own parameter tying
and update/restore both uses consistently. The inventory establishes storage
and the pinned GGUF loader behavior; it does not identify a sensitive tensor.
The actual C8 scan remains gated on the preceding evaluation phases.
