# Stage 1 runtime smoke: both runtimes generated text

The pinned original BF16 model loaded in vLLM 0.27.1 and the verified official
Q4_K_M loaded in llama.cpp revision 797de63b2fa7396dcc8caca76834c699ac84e1c3.
Both answered the synthetic English and Bangla library-closure prompts, produced
10 output tokens per response, and returned a normal stop. Both processes exited 0.
The output JSON files preserve the exact prompts and responses.

These are development smoke examples, not benchmark or held-out evaluation data.
The automated acceptance condition is nonempty text for both prompts. Inspection
also confirms both answers identify Friday. This is not evidence of general
Bangla quality, compression parity, scoring parity or production performance.

## Reproduce in WSL

From the repository root, using the already installed environments and models:

```bash
root=/home/shohag/tiny-aya-stage1
PATH="$root/vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin:/usr/lib/wsl/lib" \
  VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 \
  "$root/vllm/bin/python" setup/stage1_smoke.py vllm \
  --model "$root/models/tiny-aya-global" --output /tmp/new-vllm-smoke.json

"$root/converter/bin/python" setup/stage1_smoke.py llama \
  --model "$root/models/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf" \
  --server "$root/llama.cpp/build/bin/llama-server" --output /tmp/new-llama-smoke.json
```

Choose new output paths: the runner refuses to overwrite existing JSON results.
Run the backends sequentially. llama-server binds to loopback and is terminated
after requests; its adjacent log is saved. Context is 1024, output cap 64,
temperature zero and seed 42. vLLM uses BF16, eager mode and 50% GPU-memory
utilization. llama.cpp requests 99 GPU layers; this run's default log does not
enumerate offloaded layers, so it is not a verified layer-placement profile.

## Startup findings and limits

1. Default vLLM V2 runner failed during initialization: `UVA is not available`.
2. Selecting `VLLM_USE_V2_MODEL_RUNNER=0` passed that point but initially failed
   because `ninja` was not on PATH. It was already installed in the vLLM environment.
3. Including the environment and CUDA toolkit binaries on PATH succeeded, without
   dependency changes. The successful log records FlashAttention 2 and model loading.

The successful vLLM cold run took substantial startup/first-generation time.
No warmed throughput comparison is claimed. Eager mode and the compatibility
runner must be recorded in future experiments rather than treated as defaults.

llama.cpp warned that its special EOS ID is not in its EOG IDs. Both observed
responses stopped, but tokenizer/end-of-generation behavior needs a dedicated
audit before scoring or long generations. Each backend used its own chat-template
path; template equivalence is not yet established.

Next: same-source BF16 GGUF conversion, provenance manifests, tokenizer/template
and stopping checks, then matched scoring. Issues #3 and #4 remain incomplete.
