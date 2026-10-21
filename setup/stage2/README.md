# Isolated A3 API-client environment

Recipe validation: uv resolved 80 packages while retaining all 70 Stage 1 eval
pins unchanged. Shell syntax, missing-opt-in refusal and existing-path preservation
checks passed in WSL. This evidence alone does not claim installation or imports.

This recipe prepares the existing upstream `LocalCompletionsAPI` with an HF
tokenizer. It is a client, not a model server, custom adapter or scoring-method
change. A2 continues in its original environment. Do not install into that
environment or modify the running native library.

The Stage 1 API-only client has no Transformers tokenizer. This separate client
retains the Stage 1 dependency constraints and adds Transformers 5.20.1 and
tokenizers 0.23.3, the versions already installed in the original vLLM environment.
This selection does **not** establish equivalence with the 4.57.6 tokenizer used
to prepare A3's lock. Compare every prepared request against that lock before use.
Use local tokenizer files only; do not fetch a different model revision.

Generate the Linux x86_64 / Python 3.13 lock with uv 0.12.5:

```bash
uv pip compile setup/stage2/client.in -c setup/stage1/eval.lock \
  --python-version 3.13 --python-platform x86_64-manylinux_2_39 \
  --generate-hashes -o setup/stage2/client.lock
```

After merging the recipe and checking disk capacity, execute inside WSL Ubuntu:

```bash
bash setup/stage2_client_setup.sh --execute /absolute/new/a3-client-root
```

The destination must not exist (including a symlink). The parent must exist.
The script pins Python 3.13.15 and uv 0.12.5, verifies package hashes/dependencies
and imports, and records the freeze and lock hash. It installs no CUDA, drivers,
weights or native runtime. A partial failed directory is retained for diagnosis;
the script never overwrites it or automatically removes files. No existing
environment is modified. Review any revised recipe through a PR before retrying.

Import success is not inference validation. After A2 finishes, validate the exact
server request text, BOS and echoed token-array alignment against direct logits,
and bridge plain-letter probabilities to the original scorer before A3 accuracy
runs. The evaluation repository owns that protocol and its canonical-contract
review gate. Do not use a newer tokenizer to silently change prepared requests.
