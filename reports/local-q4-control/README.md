# Local Q4 control build

Created from the previously verified BF16 control with the pinned native llama.cpp quantizer. The recipe checks the source hash and quantizer revision and refuses to overwrite artifacts. No importance matrix, calibration data, training, or metadata override was used.

The local file is 2,143,976,960 bytes. Source, local output and official Q4 have matching tensor names/shapes and 3,349,227,520 stored parameters. Both Q4 files contain 216 Q4_K, 37 Q6_K and 37 F32 tensors. The full Q6 allocation is in inspection.json. Equal type counts do not establish equal weight values or model quality.

manifest.json records exact command, source/revision and binary/output hashes. Original and official files are unchanged. The quantizer uses the original pinned native checkout; evaluation will use the already validated patched Python runtime, as for the previous GGUF arms.

Pending: tokenizer metadata/token checks, runtime smoke validation, then the same frozen 150-question comparison in the evaluation repository. No accuracy result exists for this new arm yet. This is an experimental control, not a claimed improvement.
