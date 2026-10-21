# A3 isolated client execution

Executed the merged setup recipe from runtime PR34 in
`/home/shohag/tiny-aya-stage1/a3-api-client`. The new environment uses Python
3.13.15. All 80 hash-pinned packages installed, `uv pip check` passed, and imports
of the upstream `LocalCompletionsAPI` and `AutoTokenizer` succeeded. The freeze,
import versions and source lock hash are retained here.

The Transformers warning that PyTorch is absent is expected for this API-only
client. It does not run model weights locally. The existing A2 client/server
environments and native libraries were not changed.

`setup/probe_a3_client.py` then exercised the **unchanged upstream** client's
tokenization, context/continuation splitting, request construction, payload flags
and parser on the frozen A3 request lock. Every one of 3,600 contexts and 14,400
candidate requests passed, across all three variants and languages. It used only
local tokenizer files, offline Hugging Face mode and an HTTP-blocking guard.
`results/a3-client-offline.json` pins source modules, tokenizer files, package
versions and the request lock.

This verifies that Transformers 5.20.1/tokenizers 0.23.3 reproduce the prepared
4.57.6 token IDs and exact wire strings on this request set. Artificial logprob
arrays exercised parser plumbing; **they are not model probabilities** and do not
validate what the live server returns. No inference or accuracy was measured.

Reproduce with the isolated client Python:

```text
python setup/probe_a3_client.py --lock A3_REQUEST_LOCK.json \
  --tokenizer ORIGINAL_TOKENIZER_DIRECTORY --output NEW_REPORT.json
```

The next gate, after A2, is a live check of echoed token-array alignment, candidate
probabilities against direct logits and the plain-letter bridge to GGUFLM. The
upstream token-indexed parser may avoid the character-offset problem, but this
offline result alone does not establish that. No custom adapter has been written.
Canonical-contract selection still belongs to the user after the diagnostics.
