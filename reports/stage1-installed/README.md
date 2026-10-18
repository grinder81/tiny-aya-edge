# Stage 1 isolated setup: measured results

Recorded 2026-10-17 (America/Los_Angeles). Setup completed with exit code 0.
Recipe source: repository commit `92ceb20823ebf944e60a6a40362b00852ce96f91`,
`setup/stage1_setup.sh` and its hash-pinned dependency locks.

## Scope and evidence

The approved recipe created new eval, vLLM, and converter environments in WSL
Ubuntu 26.04, with Python 3.13.15 and uv 0.12.5. Existing project environments
were not installation targets. No driver or system package installation was needed.
SGLang was not installed.

| Check | Observed result |
| --- | --- |
| Dependency installation | All three environments installed with required hashes; each `uv pip check` passed |
| Evaluation tools | `lm_eval` import passed in the eval environment |
| Inference tools | vLLM import and a small PyTorch CUDA calculation passed |
| Converter | `convert_hf_to_gguf.py --help` exited successfully |
| llama.cpp | Release CUDA build completed all 589 build tasks; server version command passed |
| Final recipe gate | All six preflight checks passed; script exited 0 |

The adjacent freeze files record installed distributions for each environment.
`preflight.json` is the final machine-readable check from the vLLM environment.
Its `lm_eval: null` is expected: lm-eval is installed in the separate eval environment,
whose import was checked separately by the recipe. These are installation/import
checks, not successful model inference or benchmark runs.

## Build and host

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB reported VRAM; driver 610.88.
- WSL kernel: 6.6.114.1; glibc 2.43.
- CUDA compiler: 13.2.86; host compiler: GNU 15.2.0; CMake: 4.4.4.
- PyTorch reports compiled CUDA 13.0 in the vLLM environment. This differs from
  the 13.2 toolkit used to compile llama.cpp; the recorded CUDA calculation passed.
- llama.cpp revision: `797de63b2fa7396dcc8caca76834c699ac84e1c3` (verified checkout).
- CMake options: Release, Ninja, `GGML_CUDA=ON`, requested CUDA architecture 120;
  upstream configuration selected 120a. Build parallelism: 4.
- Configuration warned that NCCL and OpenSSL were unavailable. Distributed GPU
  execution and HTTPS support were not validated. The intended baseline is local,
  single-GPU execution with separately downloaded artifacts.
- Final preflight reported 964,425,154,560 free bytes on the WSL filesystem.
  A subsequent Windows C: check reported 94,061,084,672 free bytes. Virtual Linux
  capacity is not a guarantee of physical host capacity; recheck before downloads.

## Reproduction and remaining gates

Follow `setup/stage1/README.md` from the repository root with a new absolute Linux
installation directory and the existing CUDA toolkit path. The recipe intentionally
refuses to overwrite an existing installation. The earlier Windows CRLF failure was
fixed through the reviewed Git attributes change before this successful run.

Still untested: Tiny Aya loading, actual BF16-to-GGUF conversion, inference in either
runtime, scoring parity, and phone deployment. No model weights were downloaded by
this setup. Next: audit and pin model artifacts and provenance before measurements.
Historical preflight reports remain unchanged so the before/after state is inspectable.
