# Local tokenizer fix for the Stage 1 pilot

The isolated patched runtime matches the original tokenizer on all 150 frozen
pilot contexts and all 600 candidate requests for each GGUF arm. The previous
five context mismatches are resolved on this sample. Model files, prompts,
candidate labels, and the evaluation harness are unchanged. This is a local
runtime patch, not an upstream release or a Belebele accuracy result.

## Cause and scope

Tiny Aya's isolated digit regex was dispatched to the AFMoE splitter. That
splitter included preceding text in a digit segment and split years followed by
letters even though Tiny Aya's rule requires a trailing word boundary. The patch
opts Tiny Aya into prefix isolation and suffix checks; the AFMoE call retains its
old behavior. Tests include spaces before numbers and years such as `1480s`.

This is a scoped experimental patch. Passing the pilot is not proof of complete
Unicode regex equivalence: number categories and all Unicode word-boundary
classes need broader testing before proposing a general upstream fix.

## Pinned build

- llama-cpp-python: `1652066e0af45f2313b339670ef9555e8a54e545` (0.3.36).
- Patched llama.cpp base: `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
- Patch: `patches/tiny-aya-digit-prefix.patch`.
- Patch SHA256: `b15f791e924888bde063d99701f967779709cbfab7a742118bff23248b7c2616`.
- Built libllama SHA256: `366adfc21af16044a8b00a4a01e74d8f528898a3dcd9c3c6b06e69ae9c74de50`.
- Separate build: `$STAGE1_ROOT/llama-tokenizer-pinned-fix/build`.
- CUDA enabled, release build, architecture 120a, Ninja, four build workers.

Apply the patch in a clean checkout at the pinned base using `git apply --check`
followed by `git apply`. Configure with `GGML_CUDA=ON`,
`CMAKE_CUDA_ARCHITECTURES=120a`, `CMAKE_BUILD_TYPE=Release`, and build target
`llama`. Select its `build/bin` directory with `LLAMA_CPP_LIB_PATH` when invoking
the existing Python package. The installed original library is not overwritten.
Unset that variable to use the original runtime again. Both GGUF comparison arms
must use the same patched build and record that fact in their results.

## Evidence

- `setup/test_digit_prefix.cpp`: ten Tiny Aya cases pass; AFMoE preservation
  check passes. The unpatched source fails the prefix regression.
- `local-tokenizer-fix/*-token-audit-patched.json`: 600/600 candidate sequences
  match, zero context mismatches in each arm. These use the frozen sample lock
  from polyglot-quant-eval's pilot investigation; no examples were dropped.
- `local-tokenizer-fix/patched-*-echo.json`: eight synthetic cases per arm;
  echoed candidate probabilities agree with independently indexed logits within
  absolute tolerance 0.0001.
- `local-tokenizer-fix/patched-*-harness.json`: unchanged installed lm-eval GGUF
  integration compared against those same-model direct probes.

The probes use 512-token contexts and are correctness checks, not speed tests.
The pilot uses its separately frozen 8192-token context configuration. Existing
BF16 stopping metadata requirements remain unchanged. No custom scoring adapter
was added. Next: review this local runtime change, then run the matched pilot.
