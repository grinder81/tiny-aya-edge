#!/usr/bin/env bash
# Mac (Apple silicon) with Xcode installed. Used for MLX builds and quality runs, llama.cpp Metal,
# and as the host for the iPhone apps (mlx-swift-examples, ExecuTorch etLLM demo).
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/.venvs ~/src
uv venv ~/.venvs/mlx --python 3.12
source ~/.venvs/mlx/bin/activate
uv pip install mlx mlx-lm "lm_eval[api]" sacrebleu pyyaml pandas huggingface_hub hf_transfer
python -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__)"
cd ~/src
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git pull && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j
cd ~/src
[ -d mlx-swift-examples ] || git clone https://github.com/ml-explore/mlx-swift-examples   # LLMEval / MLXChatExample apps for iPhone
[ -d executorch ] || git clone https://github.com/pytorch/executorch                        # examples/demo-apps/apple_ios/LLaMA (etLLM)
[ -d optimum-executorch ] || git clone https://github.com/huggingface/optimum-executorch  # Cohere2 support target
echo "iPhone note: physical-device ExecuTorch runs need the Increased Memory Limit entitlement (paid developer account)."
