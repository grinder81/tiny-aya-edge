# Q4 matvec microbenchmark and decode roofline

Decode on a 3B model is memory-bound: every weight is read once per generated token. These tools
measure how close a quantized matvec gets to a device's bandwidth roofline, and what that bound
implies for tokens per second on each target device.

## Contents
- `q4_matvec.cu` Q4_0-style blocked matvec (32 weights per block, fp16 scale, nibbles minus 8).
  One warp per row, SoA layout for aligned 16-byte quant loads, fp32 accumulation. Checks the result
  against a host dequantized reference and reports GB/s and percent of peak.
- `bench_torch_baseline.py` dense fp16 and bf16 `torch.mv` bandwidth on the same GPU.
- `roofline.py` theoretical and expected decode tok/s per device and format, from parameter count,
  effective bits per weight and KV-cache bytes per token (cohere2's sliding window caps three of
  every four layers at 4096 tokens).

## Usage
```bash
make ARCH=sm_120 && make run              # RTX 5090; use ARCH=native if supported
python bench_torch_baseline.py 4096 4096
python roofline.py --config ~/models/tiny-aya-global/config.json --context 4096
```

## Reading the numbers
- Dense `torch.mv` usually reaches 70 to 85 percent of spec bandwidth. A straightforward Q4 kernel
  lands lower; the gap is the first thing to explain.
- The roofline table is the ceiling for every device measurement. `llama-bench` tg128 on a phone
  should be compared against the same device's row.

## Design notes and references
- ggml-cuda `mmvq.cu` and `vecdotq.cuh`: int8-quantized `x` with `dp4a`, 18-byte AoS blocks,
  different warp partitioning.
- ggml-metal `kernel_mul_mv_q4_0_f32`: SIMD-group partitioning on Apple GPUs.
- MLX `mx.fast.metal_kernel` for a Metal port; compare against `mx.quantized_matmul` at 4 bits with
  group sizes 32 and 64.
- Variants worth measuring next: int8 `x` with `dp4a`, two rows per warp, shared-memory staging of `x`.
