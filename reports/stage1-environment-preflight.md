# Stage 1 environment preflight

Read-only inventory, 2026-10-17. Partial progress toward issue #2, not C1 completion.
All repository changes go through a pull request. Installation recipes will be reviewed
in a separate PR before environment changes are executed.

## Observed

- Windows 11 Pro, build 10.0.26200; Ubuntu 26.04 LTS on WSL2.
- WSL kernel 6.6.114.1-microsoft-standard-WSL2.
- RTX 5090 visible from Windows and Ubuntu: 32607 MiB VRAM, driver 610.88.
- CUDA compiler 13.2.86 exists under `/usr/local/cuda/bin`, but `nvcc` is not on PATH.
- Ubuntu system Python 3.14.4; uv 0.12.5; GCC 15.2.0. CMake not on PATH.
- Existing unrelated Python environment: torch 2.13.0+cu132 and vLLM 0.27.1.
  GPU float32 matrix multiplication matched CPU for a small exact-integer fixture;
  vLLM import succeeded. This is not a model-load or inference test.
- lm_eval is absent from that environment. No planned `~/.venvs` environments or
  `~/src/llama.cpp` checkout were found in the inspected locations.
- WSL filesystem reports about 911 GiB available, but the Windows C: backing volume
  has only about 88 GiB free. Virtual filesystem capacity is not download capacity.
- Passwordless sudo is unavailable. No privileged action was attempted beyond checking it.

Existing environments and drivers were left unchanged. No model files were downloaded.

## Reproduce the inventory

Inside Ubuntu, select the Python interpreter explicitly. Do not activate or change an
unrelated environment's dependencies. The probe imports optional libraries only when asked.

```bash
"$TARGET_PYTHON" setup/rtx_preflight.py --cuda-root /usr/local/cuda \
  --disk-path . --check-torch --check-vllm
```

The script emits JSON to stdout and does not install or write files. GPU checks allocate
a tiny tensor. An inventory exit code is not a readiness verdict: inspect each `ok` field
and `not_checked` status. Supply `--llama-server` when a built binary exists.
Check Windows backing-volume free space separately with `Get-PSDrive` in PowerShell.

## Next PR: isolated, pinned setup recipe

The existing setup script pulls moving llama.cpp HEAD, installs unpinned dependencies,
adds optional quantizers/SGLang, and suppresses converter installation failures. Do not
run it as the Stage 1 reproducibility recipe yet. Prepare isolated eval, vLLM and converter
environments; select compatible Python and explicit versions, pin llama.cpp, and record
the installed dependency resolution. Resolve CMake and CUDA PATH without replacing the
Windows driver. Verify CUDA compiler support and a real build rather than changing GPU
architecture flags to make compilation pass.

Open gates: lm_eval import/task audit; pinned llama.cpp build and server version;
isolated vLLM/PyTorch checks; dependency locks; model loading in the later artifact/runtime
tasks. The successful existing-environment probes are feasibility evidence only.

References: [vLLM GPU installation](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/),
[NVIDIA WSL guidance](https://docs.nvidia.com/cuda/archive/13.0.0/wsl-user-guide/index.html).
