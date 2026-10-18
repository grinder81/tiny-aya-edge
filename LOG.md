# LOG

## Stage 1 manifest consolidation
- Added three artifact manifests using the repository template fields, preserving per-file original hashes and unknown official provenance.
- Linked C1 environment/artifact evidence and recorded the mandatory BF16 control EOS override. Endpoint/scoring checkpoint C2 remains open.

## BF16 stopping correction (same PR)
- Isolated EOS metadata mismatch: converted GGUF uses token 6, source generation config and official GGUF use token 3. Runtime EOS override alone restored clean stops in both smoke prompts.
- Preserved failed and corrected evidence. Smoke runner now rejects output-limit termination; verified positive and negative real-model runs. Artifact hash unchanged.

## Stage 1 BF16 GGUF control
- Converted the verified original with pinned llama.cpp to a 6,710,484,480-byte GGUF; recorded SHA256, tensor types and parameter count.
- BF16 GGUF loaded and generated nonempty English/Bangla text, but both responses repeated and hit the output limit. Stopping/readiness failed; investigate before matched evaluation. No accuracy, warmed speed or concurrent-capacity claim.

## Stage 1 runtime smoke
- Both original BF16/vLLM and official Q4_K_M/llama.cpp generated nonempty English and Bangla answers and stopped normally. Synthetic development prompts only; no benchmark claim.
- vLLM required V2 runner disabled and environment/CUDA binaries on PATH. No package changes. Recorded the llama.cpp EOS/EOG warning for follow-up.
- Added reproducible smoke runner and outputs/logs in reports/stage1-smoke. BF16 runtime control and scoring parity remain pending.

## Stage 1 downloads and inspection
- Downloaded all nine pinned files and verified sizes and available Hub LFS SHA256 values. Recorded local SHA256 for every file.
- Inspected 290 original BF16 tensors and the official GGUF mix: 216 Q4_K, 37 Q6_K, 37 F32. Both contain 3,349,227,520 stored parameters.
- Added a reusable CPU/header inspection tool and evidence in reports/stage1-downloaded. No inference or benchmark results yet.

## 2026-10-17 (artifact access verified)
- After local Hugging Face login, metadata HEAD requests passed for the pinned original configuration and both weight shards, with matching sizes. The earlier access blocker is cleared.
- No weights downloaded or local checksums verified; artifact inspection and checkpoint C1 remain pending.

## 2026-10-17 (artifact metadata audit)
- Pinned public Hub revisions and selected file metadata for Global and official Q4_K_M. No model weights downloaded; hashes remain remote expected values.
- Original checkpoint access from WSL returned HTTP 401. Artifact verification and checkpoint C1 remain blocked on usable gated access; issue #3 stays open.

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
