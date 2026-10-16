#!/usr/bin/env bash
# RTX 5090 box (Blackwell, sm_120). Ubuntu 22.04/24.04 assumed. Driver >= 570 and CUDA 12.8+ required.
# Three venvs on purpose: vllm and sglang pin different torch builds; keep the eval venv independent.
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/.venvs ~/src

echo "== eval venv (harness, quantizers) =="
uv venv ~/.venvs/eval --python 3.12
source ~/.venvs/eval/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
uv pip install "lm_eval[api]" lighteval sacrebleu pyyaml pandas huggingface_hub hf_transfer datasets
uv pip install llmcompressor            # AWQ, GPTQ, FP8 recipes (builds/make_awq_gptq.py, make_fp8.py)
uv pip install torchao                  # QAT
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
deactivate

echo "== vllm venv =="
uv venv ~/.venvs/vllm --python 3.12
source ~/.venvs/vllm/bin/activate
uv pip install vllm                     # if Blackwell wheels lag, follow vLLM docs for the cu128 nightly
python -c "import vllm; print('vllm', vllm.__version__)"
deactivate

echo "== sglang venv =="
uv venv ~/.venvs/sglang --python 3.12
source ~/.venvs/sglang/bin/activate
uv pip install "sglang[all]"
deactivate

echo "== llama.cpp with CUDA =="
cd ~/src
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git pull
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
echo "binaries: ~/src/llama.cpp/build/bin/{llama-server,llama-quantize,llama-imatrix,llama-bench,llama-cli}"
pip install -q -r ~/src/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt || true

echo "== CUDA toolchain =="
nvcc --version | tail -1
echo "then: hf auth login  (see setup/hf_access.md)"
