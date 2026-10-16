#!/usr/bin/env bash
# MLX builds on the Mac. Usage: builds/make_mlx.sh out/mlx
set -euo pipefail
OUT="${1:?output dir}"; HF=CohereLabs/tiny-aya-global
mkdir -p "$OUT"
mlx_lm.convert --hf-path $HF -q --q-bits 8 --q-group-size 64 --mlx-path "$OUT/tiny-aya-global-mlx-8bit"
mlx_lm.convert --hf-path $HF -q --q-bits 4 --q-group-size 64 --mlx-path "$OUT/tiny-aya-global-mlx-4bit-g64"
mlx_lm.convert --hf-path $HF -q --q-bits 4 --q-group-size 32 --mlx-path "$OUT/tiny-aya-global-mlx-4bit-g32"
mlx_lm.generate --model "$OUT/tiny-aya-global-mlx-4bit-g64" --prompt "আপনি কেমন আছেন?" --max-tokens 32
echo "serve for the harness: mlx_lm.server --model $OUT/tiny-aya-global-mlx-4bit-g64 --port 8081"
