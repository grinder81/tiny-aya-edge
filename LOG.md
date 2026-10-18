# LOG

## 2026-10-17 (isolated setup completed)
- Executed the reviewed LF-corrected setup recipe: all three isolated environments installed, dependency checks passed, and pinned llama.cpp built with CUDA.
- Eval import, converter help, vLLM import, GPU calculation, and server version checks passed; setup exited 0.
- Recorded installed package freezes and final preflight in reports/stage1-installed, including build warnings and remaining gates. No Tiny Aya weights downloaded or model inference performed.

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

## 2026-10-17 (setup recipe)
- Prepared separate hash-pinned eval, vLLM and converter dependency locks and a fixed llama.cpp revision.
- Dependency resolution, shell syntax, no-argument and existing-directory guards checked.
- Recipe has not been installed or build-validated; issue #2 remains open.

## 2026-10-17 (Windows checkout correction)
- First setup execution stopped at Bash option parsing because the Windows checkout used CRLF. No environments or packages were created.
- Added Git attributes forcing LF for shell scripts and Python/requirements files. Verified syntax and no-argument guard in WSL; full installation awaits this fix review.
