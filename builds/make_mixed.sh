#!/usr/bin/env bash
# Mixed-precision diagnostic builds for the sensitivity stage.
# Base: a flat 4-bit model (every tensor Q4_K via --pure). Each recipe restores ONE tensor group to Q8_0.
# Comparing each build against the flat base on bn, hi, en shows which group carries the loss.
# These are diagnostics (diag-*); most exceed the deployment budget and are never shipped.
#
# Usage: builds/make_mixed.sh out/gguf/tiny-aya-global-bf16.gguf out/diag [recipe ...]
# Tensor-name patterns are regexes matched against GGUF tensor names (token_embd, blk.N.attn_q, blk.N.ffn_down ...).
# Verify flag syntax on your llama.cpp build with:  llama-quantize --help   (look for --tensor-type, --pure,
# --token-embedding-type, --output-tensor-type). cohere2 ties the output head to token_embd, so the
# embedding recipe sets both.
set -euo pipefail
BF16="${1:?bf16 gguf}"; OUT="${2:?output dir}"; shift 2
LQ=~/src/llama.cpp/build/bin/llama-quantize
mkdir -p "$OUT"
BASE_TYPE=Q4_K_S   # with --pure this becomes "all tensors Q4_K", about 4.5 bits per weight

declare -A RECIPE
RECIPE[flat]=""
RECIPE[embed]="--token-embedding-type q8_0 --output-tensor-type q8_0"
RECIPE[attn]="--tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 --tensor-type attn_output=q8_0"
RECIPE[attn_v]="--tensor-type attn_v=q8_0"
RECIPE[ffn]="--tensor-type ffn_gate=q8_0 --tensor-type ffn_up=q8_0 --tensor-type ffn_down=q8_0"
RECIPE[ffn_down]="--tensor-type ffn_down=q8_0"
RECIPE[early]="--tensor-type blk\.([0-9]|1[01])\..*=q8_0"
RECIPE[mid]="--tensor-type blk\.(1[2-9]|2[0-3])\..*=q8_0"
RECIPE[late]="--tensor-type blk\.(2[4-9]|3[0-5])\..*=q8_0"
# cohere2 layer_types: every 4th layer (1-indexed 4, 8, ... 36) is full attention = 0-indexed 3, 7, ... 35
RECIPE[global]="--tensor-type blk\.(3|7|11|15|19|23|27|31|35)\..*=q8_0"
RECIPE[sliding]="--tensor-type blk\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30|32|33|34)\..*=q8_0"

ORDER=(flat embed attn attn_v ffn ffn_down early mid late global sliding)
[ $# -gt 0 ] && ORDER=("$@")

PARAMS=3349227520
for name in "${ORDER[@]}"; do
  [ -n "${RECIPE[$name]+x}" ] || { echo "unknown recipe $name"; exit 1; }
  f="$OUT/diag-$name-q8.gguf"; [ "$name" = flat ] && f="$OUT/diag-flat-q4k.gguf"
  if [ ! -f "$f" ]; then
    echo "== $name =="
    # shellcheck disable=SC2086
    "$LQ" --pure ${RECIPE[$name]} "$BF16" "$f" "$BASE_TYPE"
  fi
  bytes=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
  bpw=$(python3 -c "print(round($bytes*8/$PARAMS, 3))")
  cat > "${f%.gguf}.manifest.json" <<JSON
{"build_id": "$(basename "${f%.gguf}")", "kind": "diagnostic", "base_type": "$BASE_TYPE (--pure)",
 "restored_group": "$name", "restored_type": "q8_0", "flags": "${RECIPE[$name]}",
 "artifact": {"path": "$f", "size_bytes": $bytes, "effective_bpw": $bpw},
 "ships": false, "note": "sensitivity diagnostic; compare against diag-flat-q4k on bn, hi, en first"}
JSON
  echo "$name: $bytes bytes, $bpw bpw"
done
