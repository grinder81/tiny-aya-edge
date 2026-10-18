# Same-source BF16 GGUF control

**Resolved in this PR:** the unchanged artifact passes the two-prompt stopping
check when served with `--override-kv tokenizer.ggml.eos_token_id=int:3`.
This explicit runtime correction is required; it does not rewrite the file.

Converted the locally verified original Global checkpoint at revision
`af89d219b53ed9b13b8a4645f9c8028973510324` using llama.cpp revision
`797de63b2fa7396dcc8caca76834c699ac84e1c3` and the installed converter environment.
Source hashes are in `../stage1-downloaded/verified-files.json`.

Executed in WSL (output path checked absent before conversion):

```bash
root=/home/shohag/tiny-aya-stage1
"$root/converter/bin/python" "$root/llama.cpp/convert_hf_to_gguf.py" \
  "$root/models/tiny-aya-global" --outtype bf16 \
  --outfile "$root/models/tiny-aya-global-control-bf16.gguf"

"$root/converter/bin/python" setup/stage1_smoke.py llama \
  --model "$root/models/tiny-aya-global-control-bf16.gguf" \
  --server "$root/llama.cpp/build/bin/llama-server" \
  --override-kv tokenizer.ggml.eos_token_id=int:3 \
  --output "$root/smoke/llama-bf16.json"
```

Both commands exited 0. The output is 6,710,484,480 bytes, contains 290 tensors
(253 BF16 and 37 F32), and retains 3,349,227,520 stored parameters. The converter
keeps small tensors in F32 rather than making every tensor BF16. This is a
precision-preserving control, not a low-bit quantization. Tensor counts and
the streamed local SHA256 are recorded in `inspection.json`.

Initially English and Bangla smoke requests returned nonempty responses but both repeated
text and reached the 64-token limit (`finish_reason: length`). The smoke runner's
nonempty-output check passed, but the stopping/readiness check failed. Exact outputs
and server log are adjacent. This does not establish numerical or
quality parity with vLLM. The same EOS/EOG warning appears as for the official GGUF;
tokenizer, template, stopping and scoring parity require follow-up before benchmark
claims. All three baseline artifacts now exist and have generated text sequentially,
but matched evaluations still require template/scoring validation.
Concurrent GPU capacity, warmed speed, peak memory and accuracy are unmeasured.

The converted file differs from the independently published official BF16 GGUF;
we use our conversion to retain a known source revision and conversion command.
No weights are committed. No training or calibration was performed.

## Controlled diagnosis and correction

The pinned source has conflicting stopping metadata: `config.json` and
`generation_config.json` specify EOS ID 3 (`<EOS_TOKEN>`), whereas
`tokenizer_config.json` names `<|END_OF_TURN_TOKEN|>` (ID 6) as its EOS token.
The converter emitted GGUF EOS ID 6. The official Q4_K_M uses EOS ID 3.
Tokenization/architecture metadata comparison also found context-length metadata
8192 in our control versus 500000 in the official file; we do not adopt that larger
value. Both smoke runs explicitly use context 1024.

Changing only the BF16 server's EOS metadata to ID 3 produced normal stops for
both prompts: English 9 tokens, Bangla 10 tokens. See `fixed-smoke.json` and
`fixed-server.log`; original failed evidence is retained in `smoke.json`.
This supports the stopping mismatch as the cause of these observed continuations,
not a claim of complete numerical parity or absence of other conversion issues.

The runner now records overrides and fails if any response does not stop normally,
after saving the output evidence. Validation on the real BF16 control: override
enabled exits 0; override omitted reproduces the limit-hit behavior and exits 1.
No dependency or model-weight changes were made. Future control runs must retain
the override, including benchmarking and evaluation commands.
