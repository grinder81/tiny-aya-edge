# Same-source BF16 GGUF control

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
  --output "$root/smoke/llama-bf16.json"
```

Both commands exited 0. The output is 6,710,484,480 bytes, contains 290 tensors
(253 BF16 and 37 F32), and retains 3,349,227,520 stored parameters. The converter
keeps small tensors in F32 rather than making every tensor BF16. This is a
precision-preserving control, not a low-bit quantization. Tensor counts and
the streamed local SHA256 are recorded in `inspection.json`.

English and Bangla smoke requests returned nonempty responses but both repeated
text and reached the 64-token limit (`finish_reason: length`). The smoke runner's
nonempty-output check passed, but the stopping/readiness check failed. Exact outputs
and server log are adjacent. This does not establish numerical or
quality parity with vLLM. The same EOS/EOG warning appears as for the official GGUF;
tokenizer, template, stopping and scoring parity require follow-up before benchmark
claims. All three baseline artifacts now exist and have generated text sequentially,
but the converted control is not yet ready for matched evaluations.
Concurrent GPU capacity, warmed speed, peak memory and accuracy are unmeasured.

The converted file differs from the independently published official BF16 GGUF;
we use our conversion to retain a known source revision and conversion command.
No weights are committed. No training or calibration was performed.
