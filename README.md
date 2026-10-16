# tiny-aya-edge

Quantized builds of Cohere Labs' Tiny Aya for real devices, their per-language quality, and
what they cost on phones. Companion to `polyglot-quant-eval` (the harness).

**Question.** Which real deployment quantization formats of a 3B multilingual model break which
languages, what does each cost on an iPhone and a Snapdragon 8 Elite phone (TTFT, decode speed,
sustained throughput, memory, energy), and can multilingual calibration or QAT buy capability back?

**Layout.**
- `setup/` machine setup: RTX 5090 box, Mac, Android phone, Hugging Face access.
- `builds/` scripts that produce every format in the matrix plus a provenance manifest per build.
- `bench/q4_matvec/` a Q4_0 matvec CUDA microbenchmark against the memory-bandwidth roofline, a dense
  baseline, and the decode roofline calculator for each device.
- `device/` the on-device runner: fixed prompt set, TTFT, decode tok/s, 10-minute sustained
  run, peak RSS, battery counters, N=3.
- `results/` manifests and raw measurements. Quality rows live in the harness repo.

**Format matrix.** BF16 reference; AWQ INT4 g128, GPTQ INT4 g128, FP8 (server); GGUF Q8_0, Q6_K,
Q4_K_M, Q4_0, Q3_K_M, Q2_K with and without imatrix; MLX 8-bit, 4-bit g64, 4-bit g32; ExecuTorch
8da4w (needs Cohere2 support in Optimum-ExecuTorch, in progress); mitigated builds with multilingual calibration and QAT.

**Model facts.** `CohereLabs/tiny-aya-global`, 3.35B params, `cohere2` architecture (three
sliding-window layers with RoPE per one global layer without positional embeddings, window 4096),
8K context, 70+ languages, CC-BY-NC with Cohere Acceptable Use Policy, gated download.
Regional variants: tiny-aya-fire (South Asia), tiny-aya-earth (Africa, West Asia), tiny-aya-water.

Derived weights published from this repo keep the CC-BY-NC terms. Code is Apache-2.0.
