# Download verification and tensor inspection

All nine files selected by the merged Stage 1 metadata audit are now downloaded.
`verified-files.json` records their exact revisions, byte sizes and locally computed
SHA256 hashes. All sizes match the pinned audit. The two original weight shards,
tokenizer.json and official GGUF match the advertised Hub LFS SHA256 values.
For small non-LFS files, local hashes establish our recorded artifact identity;
there was no independently advertised SHA256 to compare (`hub_lfs_match: null`).

## Measured contents

| Artifact | Result |
| --- | --- |
| Original checkpoint | 290 BF16 tensors; 3,349,227,520 stored parameters |
| Original weight shards | 6,698,488,528 bytes combined, excluding tokenizer/configuration |
| Official Q4_K_M | 2,143,977,056 bytes |
| GGUF tensor mix | 216 Q4_K, 37 Q6_K, 37 F32 tensors |
| GGUF stored parameters | 3,349,227,520 |
| GGUF tensor-payload bits/weight | 5.09275421694851 |
| GGUF whole-file bits/weight | 5.121126094174695 (includes metadata/alignment) |

The actual configuration confirms cohere2, hidden width 2048, 36 layers, four KV
heads, vocabulary 262144, 4096-token sliding window and 8192 configured positions.
The full configuration and tensor inventory are in `inspection.json`.
Q4_K_M is a mixed format: it does not mean every parameter occupies four bits.
Matching parameter totals alone does not establish the official GGUF's base revision.
Its calibration, quantizer revision and command remain unknown.

## Reproduction

Run in the installed converter environment:

```bash
python builds/inspect_stage1_artifacts.py /absolute/path/to/models > inspection.json
```

The models directory must contain `tiny-aya-global` and `tiny-aya-global-GGUF`,
with the filenames pinned in `reports/stage1-artifact-audit.json`.
The inspector reads safetensors headers and memory-maps GGUF; it does not load a
model onto the GPU. It checks the original tensor names and shard assignments
against the index. Downloads used Hugging Face `hf_hub_download` with explicit
revisions and filenames, then streamed SHA256 over each complete local file.

Validation: inspector executed successfully on the real downloaded artifacts;
file sizes and available LFS hashes matched; JSON evidence parsed; diff whitespace
checks passed. No weights or credentials are included in this repository.

Remaining: full build manifests, tokenizer/chat-template behavior validation,
same-runtime BF16 conversion, model-loading smoke tests, scoring parity and benchmarks.
This is partial progress on issue #3; checkpoint C1 is not claimed complete.
