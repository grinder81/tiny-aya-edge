#!/usr/bin/env bash
# GGUF matrix with and without imatrix. Run on the 5090 box after setup/5090_setup.sh and hf download.
# Usage: builds/make_gguf.sh models/tiny-aya-global out/gguf [calibration.txt]
set -euo pipefail
SRC="${1:?hf model dir}"; OUT="${2:?output dir}"; CALIB="${3:-}"
LL=~/src/llama.cpp
mkdir -p "$OUT"
BF16="$OUT/tiny-aya-global-bf16.gguf"
[ -f "$BF16" ] || python "$LL/convert_hf_to_gguf.py" "$SRC" --outtype bf16 --outfile "$BF16"

TYPES=(Q8_0 Q6_K Q4_K_M Q4_0 Q3_K_M Q2_K)
for T in "${TYPES[@]}"; do
  [ -f "$OUT/tiny-aya-global-$T.gguf" ] || "$LL/build/bin/llama-quantize" "$BF16" "$OUT/tiny-aya-global-$T.gguf" "$T"
done

if [ -n "$CALIB" ]; then
  # imatrix: the calibration text's language mix is part of the experiment. Record its sha256 in the manifest.
  IM="$OUT/imatrix-$(basename "$CALIB" .txt).gguf"
  [ -f "$IM" ] || "$LL/build/bin/llama-imatrix" -m "$BF16" -f "$CALIB" -o "$IM" -ngl 999 -c 2048
  for T in Q4_K_M Q4_0 Q3_K_M Q2_K; do
    [ -f "$OUT/tiny-aya-global-$T-imx.gguf" ] || "$LL/build/bin/llama-quantize" --imatrix "$IM" "$BF16" "$OUT/tiny-aya-global-$T-imx.gguf" "$T"
  done
fi
echo "sanity:"; "$LL/build/bin/llama-cli" -m "$OUT/tiny-aya-global-Q4_K_M.gguf" -ngl 999 -n 32 -p "আপনি কেমন আছেন?" -no-cnv 2>/dev/null | tail -3
ls -la "$OUT"
