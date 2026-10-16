#!/usr/bin/env python3
"""Dense fp16/bf16 matvec bandwidth on the same GPU, for comparison with q4_matvec.
Usage: python bench_torch_baseline.py [M] [K] [peak_GBps]
"""
import sys, torch
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
peak = float(sys.argv[3]) if len(sys.argv) > 3 else 1792.0
dev = "cuda"
for dt in (torch.float16, torch.bfloat16):
    W = torch.randn(M, K, device=dev, dtype=dt); x = torch.randn(K, device=dev, dtype=dt)
    for _ in range(10): torch.mv(W, x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    iters = 200; s.record()
    for _ in range(iters): torch.mv(W, x)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    bytes_ = W.numel() * W.element_size() + x.numel() * x.element_size() + M * W.element_size()
    gbps = bytes_ / (ms * 1e-3) / 1e9
    print(f"{str(dt):16s} M={M} K={K}  {ms:.4f} ms  {gbps:.1f} GB/s  = {100*gbps/peak:.1f}% of peak")
print(torch.cuda.get_device_name(0))
