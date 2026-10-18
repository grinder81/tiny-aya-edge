# Upstream likelihood compatibility investigation

This is a diagnostic gate, not a Belebele accuracy or performance result. No custom
evaluation adapter is implemented. The existing native llama-server did not return
the echoed prompt probability fields required by the installed lm-eval GGUF path.

## Isolated candidate

- llama-cpp-python: `1652066e0af45f2313b339670ef9555e8a54e545` (0.3.36).
- Bundled llama.cpp: `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
- New environment: `$STAGE1_ROOT/llama-python-compat`, Python 3.13.15.
- Source: `$STAGE1_ROOT/llama-python-source`, detached commit with pinned submodule.
- CUDA build: `CMAKE_ARGS=-DGGML_CUDA=ON`, `CMAKE_BUILD_PARALLEL_LEVEL=4`.
- Installed from local source with the `server` extra using uv. Existing eval,
  vLLM, converter and native llama.cpp environments are unchanged.
- Transitive dependencies were resolved during this exploratory install, not
  hash-locked beforehand. An installed package inventory must accompany results.

The bundled runtime differs from the deployment runtime. Any later evaluation
must label it explicitly and run both GGUF precision arms through the same build.
Speed measurements on this integration are not deployment speed measurements.

## Validation sequence

1. `setup/probe_vllm_likelihood.py` records original-model prompt token IDs and
   candidate log probabilities for two synthetic contexts, each with A/B/C/D.
   There is no chat template. These are mechanical probes, not test questions.
2. `setup/probe_llama_python.py` checks echoed probabilities and continuation
   boundaries. It independently indexes the same model's raw logits and checks
   the returned probabilities with absolute tolerance 0.0001.
3. Compare tokenizer IDs against the vLLM reference. Report numerical differences
   rather than assuming distinct runtimes or quantizations are exactly equivalent.
4. Start the package's upstream server and use `setup/probe_upstream_gguf.py`
   to call the installed, unmodified lm-eval `GGUFLM.loglikelihood` implementation.
   Compare with the same-model direct probe, absolute tolerance 0.0001.
5. Real Belebele prompt/token boundary checks and sample selection remain a
   separate gate before the pilot. Passing these short probes is insufficient
   to claim full benchmark compatibility.

Both GGUF files are existing verified artifacts; no weights or tokenizer are
modified. The BF16 file's documented EOS mismatch still applies to generation.
One-token diagnostic completions do not validate stopping behavior.

## Results

The isolated CUDA build succeeded. All eight synthetic cases passed for each
GGUF arm: echoed candidate probabilities agree with independently indexed raw
logits within 0.0001. The unchanged installed lm-eval GGUF implementation passed
the HTTP server checks for both arms against their direct-probe references.
Both servers were stopped after testing.

All eight full token ID sequences match the vLLM reference for both GGUF files.
Maximum absolute continuation log-probability differences from vLLM are:

| GGUF arm | Maximum absolute difference (natural-log units) |
| --- | ---: |
| BF16 control | 0.04369020462036133 |
| Official Q4_K_M | 0.7675952911376953 |

These are observations on synthetic cases, not accepted numerical parity bounds
or quality metrics. The official Q4 artifact has unmatched provenance. Do not
attribute its differences solely to compression. BF16 runtime differences also
need examination on actual task prompts before interpreting benchmark deltas.

Raw evidence and installed dependency inventory are in `llama-python-compatibility/`.
The first direct Q4 probe reached serialization after its checks but failed on
NumPy float serialization; converting scalars for JSON fixed only evidence output.
The rerun passed and saved evidence. No scoring code or weights were patched.

## Reproduction

Run `probe_vllm_likelihood.py` with the original model and a fresh output path in
the existing vLLM environment, with `VLLM_USE_V2_MODEL_RUNNER=0` and that environment
plus CUDA on PATH. Run `probe_llama_python.py` once per verified GGUF in the new
environment. Both accept `--model` and `--output`.

For each GGUF, launch the upstream server in the new environment:

```sh
python -m llama_cpp.server --model "$MODEL" --n_gpu_layers -1 --n_ctx 512 \
  --verbose False --host 127.0.0.1 --port 18093
```

Then, in the existing eval environment:

```sh
python setup/probe_upstream_gguf.py --base-url http://127.0.0.1:18093 \
  --reference "$DIRECT_PROBE_JSON" --output "$NEW_RESULT_JSON"
```

These diagnostic scripts do not register an adapter, alter the upstream task, or
produce benchmark accuracy. Actual Belebele prompt checks and the pilot remain
pending. Issue 4 is not complete.
