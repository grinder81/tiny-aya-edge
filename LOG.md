# LOG

Dated entries: what changed, the numbers, open problems.

## 2026-10-16
- Repository created: model facts, format matrix, machine setup, build scripts for GGUF, MLX, AWQ,
  GPTQ and FP8, provenance manifest, device protocol, Q4 matvec microbenchmark and decode roofline.
- Open: first microbenchmark numbers on the RTX 5090 (GB/s, percent of spec); first llama-bench
  numbers on both phones.

## 2026-10-16 (later)
- Study aligned to the loop baseline, inspect, intervene, measure, explain. Added the GGUF bit-width
  sweep, mixed-precision diagnostic recipes (`builds/make_mixed.sh`), the embedding-row error
  inspection (`builds/embedding_error.py`), the deployment budget (5.0 bpw, 2.2 GB), and the
  iPhone 15 as the iOS validation device.

## 2026-10-17
- Added read-only RTX preflight and recorded reports/stage1-environment-preflight.json.
- Existing-environment CUDA calculation and vLLM import passed; CMake, lm_eval and pinned llama.cpp setup remain open. No dependency installation or model downloads. See the companion preflight report.
