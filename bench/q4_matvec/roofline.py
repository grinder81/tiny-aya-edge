#!/usr/bin/env python3
"""Decode-time roofline: the memory-bandwidth upper bound on tokens/second per device and format.

Decode reads every weight once per token (plus the KV cache), so tok/s <= bandwidth / bytes_per_token.
Usage:
  python roofline.py --params 3.35e9 [--context 4096 --layers N --kv-heads H --head-dim D]
  python roofline.py --config models/tiny-aya-global/config.json --context 4096
Bandwidths below are spec or estimates; replace with measured numbers (e.g. from q4_matvec) as you get them.
"""
import argparse, json

# effective bits per weight including scales/zeros (ggml and MLX conventions; verify against your files)
FORMATS = {
    "bf16": 16.0, "fp8": 8.0, "gguf-q8_0": 8.5, "gguf-q6_k": 6.56, "mlx-4bit-g64": 4.5,
    "gguf-q4_k_m": 4.85, "gguf-q4_0": 4.5, "awq/gptq-int4-g128": 4.25, "et-8da4w-g32": 4.5,
    "gguf-q3_k_m": 3.91, "gguf-q2_k": 3.35,
}
# GB/s. spec = vendor number; est = estimate, verify. Achievable decode bandwidth is typically 55-80% of these.
DEVICES = {
    "rtx5090 (spec)": 1792.0,
    "m4-max (spec)": 546.0,
    "m4-pro (spec)": 273.0,
    "iphone-15 LPDDR5-6400 x64 (est)": 51.2,
    "snapdragon-8-elite LPDDR5X-9600 x64 (spec)": 76.8,
    "android-tv-box DDR4 (est)": 12.0,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, default=3.35e9, help="total params read per token (tied lm_head counts once)")
    ap.add_argument("--config", help="HF config.json; overrides --params/--layers/--kv-heads/--head-dim when fields exist")
    ap.add_argument("--context", type=int, default=4096, help="tokens of KV cache read per decode step")
    ap.add_argument("--layers", type=int, default=None); ap.add_argument("--kv-heads", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None); ap.add_argument("--kv-bits", type=float, default=16.0)
    ap.add_argument("--efficiency", type=float, default=0.65, help="fraction of peak bandwidth you expect to achieve")
    a = ap.parse_args()
    layers, kvh, hd, params = a.layers, a.kv_heads, a.head_dim, a.params
    if a.config:
        c = json.load(open(a.config))
        layers = c.get("num_hidden_layers", layers); kvh = c.get("num_key_value_heads", kvh)
        hd = c.get("head_dim") or (c["hidden_size"] // c["num_attention_heads"] if "hidden_size" in c else hd)
        print(f"config: layers={layers} kv_heads={kvh} head_dim={hd} hidden={c.get('hidden_size')} vocab={c.get('vocab_size')}")
    kv_bytes = 0.0
    if layers and kvh and hd:
        # cohere2: 3 of every 4 layers use a 4096 sliding window, so KV read is capped at the window for those.
        kv_bytes = 2 * layers * kvh * hd * min(a.context, 4096) * (a.kv_bits / 8)
        print(f"KV cache read per token at ctx={a.context}: {kv_bytes/1e6:.1f} MB (fp16, window-capped)")
    print(f"\nparams={params:.3e}  efficiency={a.efficiency:.0%}\n")
    hdr = f"{'format':22s}{'MB/token':>10s}" + "".join(f"{d.split(' ')[0]:>26s}" for d in DEVICES)
    print(hdr); print("-" * len(hdr))
    for f, bits in FORMATS.items():
        mb = (params * bits / 8 + kv_bytes) / 1e6
        row = f"{f:22s}{mb:10.0f}"
        for d, bw in DEVICES.items():
            tps = bw * 1e3 / mb
            row += f"{tps:12.1f} ({tps*a.efficiency:6.1f})"
        print(row)
    print("\ncolumns: theoretical max tok/s (achievable at the given efficiency). Compare with llama-bench tg128.")

if __name__ == "__main__":
    main()
