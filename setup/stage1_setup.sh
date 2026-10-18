#!/usr/bin/env bash
# Explicit opt-in setup. No sudo, drivers, model downloads or existing-env changes.
set -euo pipefail
if [[ $# -ne 3 || $1 != --execute ]]; then
  echo 'Usage: bash setup/stage1_setup.sh --execute NEW_ABSOLUTE_ROOT CUDA_ROOT'
  echo 'Creates eval, vllm and converter environments, then builds pinned llama.cpp.'
  echo 'Review setup/stage1/README.md first. No changes made.'
  exit 2
fi
root=$2
cuda_root=$3
[[ $root == /* && $cuda_root == /* ]] || { echo 'Absolute paths required'; exit 1; }
[[ ! -e $root && ! -L $root ]] || { echo 'Root already exists; refusing reuse'; exit 1; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || exit 1
command -v uv >/dev/null
[[ $(uv --version) == 'uv 0.12.5'* ]] || { echo 'Expected uv 0.12.5'; exit 1; }
command -v git >/dev/null
command -v g++ >/dev/null
[[ -x $cuda_root/bin/nvcc ]] || { echo 'CUDA compiler missing'; exit 1; }
recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
revision=797de63b2fa7396dcc8caca76834c699ac84e1c3
mkdir -p -- "$root"
export PATH="$cuda_root/bin:$PATH"
export UV_CACHE_DIR="$root/cache"
export UV_PYTHON_INSTALL_DIR="$root/python"
for env in eval vllm converter; do
  uv venv "$root/$env" --python 3.13.15
  uv pip sync --python "$root/$env/bin/python" --require-hashes "$recipe/stage1/$env.lock"
  uv pip check --python "$root/$env/bin/python"
  uv pip freeze --python "$root/$env/bin/python" > "$root/$env.freeze.txt"
done
git init "$root/llama.cpp"
git -C "$root/llama.cpp" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$root/llama.cpp" fetch --depth 1 origin "$revision"
git -C "$root/llama.cpp" checkout --detach FETCH_HEAD
[[ $(git -C "$root/llama.cpp" rev-parse HEAD) == "$revision" ]]
export PATH="$root/converter/bin:$PATH"
cmake -S "$root/llama.cpp" -B "$root/llama.cpp/build" -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/llama.cpp/build" --parallel 4
"$root/eval/bin/python" -c 'import lm_eval; print("lm_eval import passed")'
"$root/converter/bin/python" "$root/llama.cpp/convert_hf_to_gguf.py" --help > "$root/converter-help.txt"
"$root/vllm/bin/python" "$recipe/rtx_preflight.py" --cuda-root "$cuda_root" \
  --llama-server "$root/llama.cpp/build/bin/llama-server" --disk-path "$root" \
  --check-torch --check-vllm > "$root/preflight.json"
"$root/vllm/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert all(d[k]["ok"] for k in ("nvidia_smi","nvcc","cmake","llama_server","torch_cuda","vllm_import")), d' "$root/preflight.json"
echo 'Setup checks passed. Model loading and scoring are still separate gates.'
