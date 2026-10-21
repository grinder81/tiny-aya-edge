#!/usr/bin/env bash
# A new API-client environment only. Never touch a running scorer or model server.
set -euo pipefail
if [[ $# -ne 2 || $1 != --execute ]]; then
  echo 'Usage: bash setup/stage2_client_setup.sh --execute NEW_ABSOLUTE_ROOT'
  exit 2
fi
root=$2
[[ $root == /* && ! -e $root && ! -L $root ]] || {
  echo 'Destination must be an absolute, nonexistent path'; exit 1;
}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || exit 1
command -v uv >/dev/null
[[ $(uv --version) == 'uv 0.12.5'* ]] || { echo 'Expected uv 0.12.5'; exit 1; }
recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -f $recipe/stage2/client.lock ]] || exit 1
mkdir -- "$root"
export UV_CACHE_DIR="$root/cache"
export UV_PYTHON_INSTALL_DIR="$root/python"
uv venv "$root/client" --python 3.13.15
uv pip sync --python "$root/client/bin/python" --require-hashes "$recipe/stage2/client.lock"
uv pip check --python "$root/client/bin/python"
uv pip freeze --python "$root/client/bin/python" > "$root/client.freeze.txt"
"$root/client/bin/python" - <<'PY' > "$root/import-check.json"
import importlib.metadata as m
import json
from lm_eval.models.openai_completions import LocalCompletionsAPI
from transformers import AutoTokenizer
print(json.dumps(dict(imports_passed=True, versions={name:m.version(name) for name in
    ('lm-eval', 'transformers', 'tokenizers', 'huggingface-hub', 'datasets')}), indent=2))
PY
sha256sum "$recipe/stage2/client.lock" > "$root/lock.sha256.txt"
echo 'Client imports passed. Token parity and endpoint scoring remain unverified.'
