# CLAUDE.md — tiny-aya-edge

Purpose: quantized builds of Cohere Labs' Tiny Aya for real devices, their provenance, the
on-device cost protocol, and the matvec microbenchmarks. Quality numbers live in the companion harness repo
github.com/grinder81/polyglot-quant-eval and its results schema; this repo produces artifacts and
device measurements, not eval tables.

## Model facts (do not guess, these were checked)
- `CohereLabs/tiny-aya-global`: 3.35B params, `cohere2` architecture (three sliding-window layers
  with RoPE per one global layer without positional embeddings, window 4096), 8K context, 70+
  languages, CC-BY-NC with Cohere Acceptable Use Policy, gated. Regional: tiny-aya-fire (South
  Asia), tiny-aya-earth (Africa, West Asia), tiny-aya-water. Base: tiny-aya-base-32K.
- Already supported upstream: transformers, vLLM, SGLang, llama.cpp (cohere2), mlx-lm (cohere2.py).
  An official GGUF exists (CohereLabs/tiny-aya-global-GGUF). Not supported: Optimum-ExecuTorch
  (no Cohere2), LiteRT-LM. Adding Cohere2 support there is part of this work.
- llama.cpp Hexagon backend is experimental with a limited op set; fallback ops get reported upstream with numbers.

## Layout
- `setup/` 5090 box (three venvs: eval, vllm, sglang), Mac, Android, HF access.
- `builds/` one script per format family; each build gets a manifest from `manifest_template.json`
  with base revision, tool version, exact command, calibration set hash, artifact sha256.
- `bench/q4_matvec/` Q4_0 matvec CUDA microbenchmark, dense baseline, decode roofline.
- `device/` runner and protocol. Rows go to the harness schema with task `device_cost`.
- `results/` manifests and raw device measurements (CC-BY-4.0). Weights are never committed.

## Fixed rules of the study
- Loop: baseline, inspect, intervene, measure, explain. One change per experiment.
- Interventions in order: selective precision allocation, then multilingual calibration, then QAT.
- Deployment budget for a final candidate: at most 2.2 GB total shipped bytes (model file plus
  tokenizer plus any adapter). Bits per weight is descriptive. Diagnostic builds may exceed it and
  are named `diag-*` so nobody ships them.
- Every table with a non-English delta shows the English delta beside it.
- Bangla deep, fourteen wide: Bangla gets human eval and the repair target; the matrix stays at 14.
- Validation phones: iPhone 15 (6 GB RAM) now; a Snapdragon 8 Elite device when it arrives. The
  Hexagon work is a time-boxed systems track and never gates the main results.
- Compression is the primary experimental objective: measure per-language damage under deployment formats and test precision allocation at matched shipped bytes. Adaptation is secondary and requires an explicit user decision after compression evidence; sensitivity results guide placement rather than presuming attention or MLP targets.
- Follow `docs/stage2-plan.md` for the approved sequence and gates. Historical Stage 1 scoring remains unchanged; canonical changes require explicit validation and new run identities.

## Conventions
- Every artifact published to the Hub carries its manifest and keeps CC-BY-NC terms.
- Calibration sets are experiment variables: record language mix and sha256, never "default".
- Device protocol is fixed: airplane mode, fixed brightness, battery 40 to 80 percent, 30-minute
  cool-down, N=3, report spread. Do not change it mid-study; add a column instead.
- Kernel work: profile first, pick the op farthest from the roofline, PR with before/after numbers.

## What is verified and what is not
- Verified here: roofline.py runs; Python scripts compile.
- Not verified until hardware runs it: setup scripts (versions drift), q4_matvec.cu (no nvcc in the
  authoring environment), llm-compressor API names in builds/make_awq_gptq.py and make_fp8.py.
