# Local Python decoding repair

The reduced BF16 pilot stopped after 24 of 40 candidates in the unchanged
lm-eval GGUF scorer. The Python wrapper's `LlamaModel.detokenize` allocated 32
bytes per token and treated a negative required-size return as a Python slice.
Two Bangla tokens require 34 bytes and were silently omitted. An affected prompt
shrunk from 899 to 851 characters on decoding, invalidating echoed answer offsets.
The original Hugging Face tokenizer round trip retained all 899 characters.

`patches/python-decode-buffer.patch` retries with the required allocation and
raises if the retry still fails. It applies to llama-cpp-python source revision
`1652066e0af45f2313b339670ef9555e8a54e545`. The same patch is applied to the
isolated `llama-python-compat` installation; this is explicitly a locally patched
package, not a stock upstream release. The separate llama.cpp tokenizer patch
remains selected through `LLAMA_CPP_LIB_PATH`.

Validation: seven mocked boundary sizes (0, 1, 31, 32, 33, 34, 128 bytes) pass.
All 750 strings (150 frozen contexts plus their four candidates) round trip
exactly with the patched BF16 runtime. No model weights, questions, token IDs,
or scoring rule were changed. The failed pilot produced no completed accuracy
result and must be restarted.

Future upstream work is tracked separately in issues #18 (llama.cpp digit
splitting) and #19 (Python buffer handling). No upstream submission was made.
The sibling `token_to_piece` convenience method also uses a fixed buffer and
should be reviewed before an upstream proposal; this local patch targets the
`detokenize` path used by our server.
