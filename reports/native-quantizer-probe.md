# Native K-quant API prerequisite

The installed gguf-py 0.19.0 raises `NotImplementedError` for Q4_K and Q6_K
quantization. Its dequantization paths are available. A sensitivity experiment
must therefore use the pinned native quantizer rather than substituting generic
four-bit rounding or assuming the Python writer implements these formats.

`builds/probe_native_quantizer.py` calls the exported `ggml_quantize_chunk` API
from the same native toolchain used to produce our local Q4_K_M model. It extracts
BF16 source rows, converts them to float32, quantizes without an importance matrix,
and compares packed bytes to the corresponding exported tensor rows. It then
compares native and Python dequantization of those packed bytes.

Six rows passed exact byte and dequantized-value equality:

- Q4_K: `blk.0.attn_k.weight`, rows 0, 256 and 511.
- Q6_K: `token_embd.weight`, rows 0, 131072 and 262143.

Each row contains 2,048 weights. These are representative first/middle/last-row
checks on two tensors, not an exhaustive validation of every tensor, format,
importance-matrix setting or platform. Source/model/library hashes and row hashes
are in `native-quantizer-probe.json`. Weight-space error is descriptive and does
not estimate language quality or task loss.

Run using the pinned converter environment (gguf and NumPy installed):

```sh
python builds/probe_native_quantizer.py --source BF16_GGUF --exported LOCAL_Q4_GGUF \
  --library PINNED_BUILD/bin/libggml-base.so --output NEW_REPORT
```

The output must not already exist. The native API signatures come from the pinned
toolchain headers, and the probe verifies that these formats do not require an
importance matrix. This is CPU-only artifact validation; it neither builds a
sensitivity map nor selects calibration data. C8 remains gated on the reviewed
precision/generation results and a source-disjoint calibration specification.
Before relying on a future map, extend fidelity checks to the actual tensors and
quantizer settings used, then validate predictions with real exported builds.
