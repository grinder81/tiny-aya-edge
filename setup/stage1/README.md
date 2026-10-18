# Isolated Stage 1 setup recipe

Status: dependency resolution and script checks only. **Not installed or build-validated.**
This PR prepares the recipe; review before executing system/environment changes.
It advances issue #2 without closing its working-runtime acceptance criteria.

Use `setup/stage1_setup.sh`, not the older broad `setup/5090_setup.sh`, for this stage.
The new entry point creates three environments in a new absolute directory:

| Environment | Purpose |
|---|---|
| eval | lm-eval API client and result tools |
| vllm | inference runtime and its compatible PyTorch dependency set |
| converter | pinned llama.cpp converter dependencies, CMake and Ninja |

Python is pinned to 3.13.15; uv must be 0.12.5. llama.cpp is fixed to
`797de63b2fa7396dcc8caca76834c699ac84e1c3`. This revision was selected for
validation, not yet established as working with Tiny Aya. CUDA architecture stays 120.
The installed CUDA toolkit and GCC remain external prerequisites recorded by the
preflight; they are not installed or upgraded here. CUDA 13.2.86 and GCC 15.2.0 were
observed on the target host. Their build compatibility still needs execution evidence.

Each `.in` file lists direct requirements; each `.lock` pins the resolved dependencies
and allowed artifact hashes. Resolution targets Linux x86_64 / Python 3.13. Converter
requirements mirror the two upstream converter requirement files at the pinned commit.
Separate environments allow its older Transformers/PyTorch requirements to coexist
with vLLM. No SGLang, QAT or extra quantization packages are installed in this stage.

Regenerate locks with uv 0.12.5 from the repository root (replace ENV for each environment):

```bash
uv pip compile setup/stage1/ENV.in --python-version 3.13 \
  --python-platform x86_64-manylinux_2_39 --generate-hashes \
  -o setup/stage1/ENV.lock
```

The initial eval resolution rejected huggingface-hub 1.27.0 with datasets 5.1.0;
the recipe pins Hub 1.31.0 instead. All three revised locks resolve successfully.
Resolution is not proof that imports, CUDA kernels or model inference work.

After review, run inside Ubuntu WSL, with uv on PATH:

```bash
bash setup/stage1_setup.sh --execute /absolute/new/stage1-root /usr/local/cuda
```

Check both the destination filesystem and its Windows backing-volume space first.
The latter had approximately 88 GiB available during preflight; WSL's virtual free
space is larger and must not be used as the sole capacity estimate. Three isolated
environments and their package cache can consume substantial storage. No model
download is included. There is no sudo or driver installation.

The script refuses any existing destination, isolates uv caches/Python downloads,
and stops on install/build/converter failures. A partial failed destination remains
for diagnosis; it is not automatically deleted or overwritten. Record any revised
recipe in a PR before retrying. Build parallelism is limited to four jobs.

On success, retain the environment freeze files, compiler/runtime preflight and
converter help output for the execution report. Gate failures must be investigated;
do not weaken architecture flags or suppress errors to obtain a successful exit.
Actual model loading, tokenizer parity and task scoring remain later tasks.
