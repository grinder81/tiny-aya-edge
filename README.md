# tiny-aya-edge

Quantized builds of Cohere Labs' Tiny Aya for real devices, their per-language quality, and
what they cost on phones. Companion to `polyglot-quant-eval` (the harness).

**Question.** Which real deployment quantization formats of a 3B multilingual model break which
languages, at what bit-width each language breaks, which tensors carry the loss, what each format
costs on an iPhone 15 and a Snapdragon 8 Elite phone (TTFT, decode speed, sustained throughput,
memory, energy), and whether a targeted repair under a fixed size budget buys capability back.

**Method.** Every experiment follows one loop: baseline, inspect, intervene, measure, explain.
Baseline is BF16 and the standard builds per language. Inspect means locating the error:
embedding-row reconstruction error by script (`builds/embedding_error.py`) and per-tensor
sensitivity via mixed-precision diagnostic builds (`builds/make_mixed.sh`). Interventions run in a
fixed order, selective precision allocation, then multilingual calibration, then QAT, each judged
at the same deployment budget. Bangla is the deep dive (native-speaker evaluation lives in the
harness repo); the 14-language matrix is the breadth.

**Deployment budget.** A final repaired candidate must be at most 5.0 effective bits per weight
and at most 2.2 GB on disk, so it is interchangeable with Q4_K_M on both phones. Diagnostic builds
may exceed it; they exist to locate the loss, not to ship.

**Layout.**
- `setup/` machine setup: RTX 5090 box, Mac, Android phone, Hugging Face access.
- `builds/` scripts that produce every format in the matrix, the mixed-precision diagnostics, the
  embedding-row inspection, and a provenance manifest per build.
- `bench/q4_matvec/` a Q4_0 matvec CUDA microbenchmark against the memory-bandwidth roofline, a dense
  baseline, and the decode roofline calculator for each device.
- `device/` the on-device runner: fixed prompt set, TTFT, decode tok/s, 10-minute sustained
  run, peak RSS, battery counters, N=3.
- `results/` manifests and raw measurements. Quality rows live in the harness repo.

**Format matrix.** BF16 reference; AWQ INT4 g128, GPTQ INT4 g128, FP8 (server); GGUF sweep Q8_0,
Q6_K, Q4_K_M, Q4_K_S, Q4_0, Q3_K_M, Q3_K_S, Q2_K, IQ2_M, with Q4_K_M and Q3_K_M also built with
imatrix; MLX 8-bit, 4-bit g64, 4-bit g32; ExecuTorch 8da4w (after the Optimum-ExecuTorch Cohere2
work); diagnostic mixed-precision GGUFs (a flat 4-bit base with one tensor group restored to Q8_0);
repaired candidates from selective precision, multilingual calibration and QAT.

**Devices.** iPhone 15 (6 GB RAM, LPDDR5, about 51 GB/s; BF16 impossible, Q8_0 tight) and a
Snapdragon 8 Elite phone (Hexagon NPU, Adreno GPU, CPU). Every table that shows a non-English delta
shows the English delta beside it.

**Model facts.** `CohereLabs/tiny-aya-global`, 3.35B params, `cohere2` architecture (three
sliding-window layers with RoPE per one global layer without positional embeddings, window 4096),
8K context, 70+ languages, CC-BY-NC with Cohere Acceptable Use Policy, gated download.
Regional variants: tiny-aya-fire (South Asia), tiny-aya-earth (Africa, West Asia), tiny-aya-water.

Derived weights published from this repo keep the CC-BY-NC terms. Code is Apache-2.0.
