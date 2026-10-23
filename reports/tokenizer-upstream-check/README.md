# Upstream tokenizer parity check

Checked on 23 October 2026 for the future llama.cpp contribution tracked in issue #18.
This is a runtime parity check, not an accuracy result and not an upstream submission.

`setup/tokenizer_digit_reproducer.py` tokenizes 66 digit-boundary cases and 9 numeral
probes with the original Tiny Aya tokenizer (`tokenizers` library reading the verified local
`tokenizer.json`) and with any `llama-tokenize` binary, then compares token IDs. Both sides
run without BOS and without special-token parsing. Case classes: leading whitespace before
digits, punctuation-adjacent digits, letter and underscore suffixes, Bengali, Devanagari,
fullwidth and Arabic-Indic digits, join controls, combining marks, digit-run lengths 1 to 8,
and mixed text. Probes cover No/Nl numerals (superscripts, Roman numerals, circled digits,
a vulgar fraction) where `\d` and `\p{N}` differ.

| Runtime | Regular mismatches | Probe differences | File |
| --- | --- | --- | --- |
| pinned native llama.cpp `797de63`, CUDA build, unpatched | 15 of 66 | 0 of 9 | `pinned-797de63-unpatched.json` |
| upstream master `9ff0d7e` (23 October 2026), CPU build, unpatched | 15 of 66 | 0 of 9 | `master-9ff0d7e-unpatched.json` |
| upstream master `9ff0d7e` plus `patches/tiny-aya-digit-prefix.patch` | 0 of 66 | 0 of 9 | `master-9ff0d7e-patched.json` |

The 15 mismatching cases and their token IDs are identical on both unpatched builds:
two or three spaces before digits, letter-suffixed years (`1990s`, `1480s`, `1970s and
1930s`, `2007x`), a combining mark after digits, a mixed sentence containing `1990s`, and
the leading-space variants of Bengali, Devanagari, fullwidth and Arabic-Indic digits.
Bare digit runs of every length, punctuation-adjacent digits and join controls without a
leading space already matched before the patch.

The patch applied to master with line offsets only (hunks at 1114, 1156 and 1258 of
`src/unicode.cpp`). Upstream `test-unicode` passes on the patched tree.
`setup/test_digit_prefix.cpp` compiled against the patched master `unicode.cpp` and
`unicode-data.cpp` passes its 15 cases and the AFMoE preservation check; compiled against
the unpatched master sources it fails.

Equal token IDs on the probes do not prove equal pre-tokenization segments, because
byte-level merges can coincide for rare characters. The official Q4_K_M GGUF was opened
vocab-only; no model computation ran. Result files record only file basenames, sizes,
revisions and token IDs.

Run from the repository root with explicit paths:

```
python setup/tokenizer_digit_reproducer.py --tokenizer <tiny-aya checkout> \
    --gguf <any tiny-aya gguf> --llama-tokenize <build/bin/llama-tokenize> \
    --runtime-label "<llama.cpp revision and patch state>" --json <output.json>
```

Exit status 1 means at least one regular case mismatched.
