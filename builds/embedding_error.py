#!/usr/bin/env python3
"""Inspect step: how much does block quantization distort each embedding row, grouped by script?

Simulates Q4_0 (32-weight blocks, d = max/-8, 4-bit) and Q8_0 (32-weight blocks, 8-bit) on the
embedding table and reports per-row relative reconstruction error, grouped by the Unicode script of
the token. The table is tied to the output head in cohere2, so this error hits both lookup and logits.

Usage (real model, needs the gated download and torch + safetensors):
  python builds/embedding_error.py --model-dir ~/models/tiny-aya-global --tokenizer ~/models/tiny-aya-global/tokenizer.json --out results/embedding_error.csv
Usage (test / CI with a synthetic matrix):
  python builds/embedding_error.py --embedding-npy some.npy --tokenizer tokenizer.json --out /tmp/e.csv
"""
import argparse, csv, glob, json, os, sys
import numpy as np

SCRIPTS = [
    ("Bengali", 0x0980, 0x09FF), ("Devanagari", 0x0900, 0x097F), ("Arabic", 0x0600, 0x06FF),
    ("Arabic", 0x0750, 0x077F), ("Hangul", 0xAC00, 0xD7AF), ("Hangul", 0x1100, 0x11FF),
    ("Kana", 0x3040, 0x30FF), ("Han", 0x4E00, 0x9FFF), ("Han", 0x3400, 0x4DBF),
    ("Cyrillic", 0x0400, 0x04FF), ("Thai", 0x0E00, 0x0E7F), ("Greek", 0x0370, 0x03FF),
    ("Hebrew", 0x0590, 0x05FF), ("Tamil", 0x0B80, 0x0BFF), ("Telugu", 0x0C00, 0x0C7F),
    ("Gujarati", 0x0A80, 0x0AFF), ("Gurmukhi", 0x0A00, 0x0A7F), ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F), ("Ethiopic", 0x1200, 0x137F), ("Latin", 0x0041, 0x024F),
]

def script_of(text):
    counts = {}
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or ch.isdigit():
            continue
        name = None
        for nm, lo, hi in SCRIPTS:
            if lo <= cp <= hi:
                name = nm; break
        if name is None:
            name = "Other" if ch.isalpha() else "Symbol"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "Whitespace/Digit"
    return max(counts.items(), key=lambda kv: kv[1])[0]

def q4_0(block):
    # block: (..., 32) float32. ggml rule: d = max-magnitude signed value / -8
    idx = np.argmax(np.abs(block), axis=-1, keepdims=True)
    mx = np.take_along_axis(block, idx, axis=-1)
    d = mx / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0)
    q = np.clip(np.floor(block * inv + 8.5), 0, 15)
    d16 = d.astype(np.float16).astype(np.float32)
    return (q - 8.0) * d16

def q8_0(block):
    amax = np.max(np.abs(block), axis=-1, keepdims=True)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0)
    q = np.clip(np.rint(block * inv), -128, 127)
    d16 = d.astype(np.float16).astype(np.float32)
    return q * d16

def load_embedding(a):
    if a.embedding_npy:
        return np.load(a.embedding_npy).astype(np.float32)
    import torch
    from safetensors import safe_open
    for f in sorted(glob.glob(os.path.join(a.model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if "embed_tokens" in k:
                    return sf.get_tensor(k).to(torch.float32).numpy()
    sys.exit("no embed_tokens tensor found under " + a.model_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir"); ap.add_argument("--embedding-npy")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--out", required=True, help="CSV of per-row results; a .summary.json is written beside it")
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    if not (a.model_dir or a.embedding_npy):
        sys.exit("give --model-dir or --embedding-npy")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    E = load_embedding(a)
    n, dim = E.shape
    if dim % 32:
        sys.exit("hidden size must be a multiple of 32")
    vocab_n = tok.get_vocab_size()
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "token", "script", "norm", "relerr_q4_0", "relerr_q8_0"])
        for s in range(0, n, a.chunk):
            blk = E[s:s + a.chunk]
            b = blk.reshape(blk.shape[0], -1, 32)
            e4 = q4_0(b).reshape(blk.shape); e8 = q8_0(b).reshape(blk.shape)
            norm = np.linalg.norm(blk, axis=1) + 1e-12
            r4 = np.linalg.norm(blk - e4, axis=1) / norm; r8 = np.linalg.norm(blk - e8, axis=1) / norm
            for i in range(blk.shape[0]):
                tid = s + i
                text = tok.decode([tid]) if tid < vocab_n else ""
                sc = script_of(text) if tid < vocab_n else "Padding"
                rows.append((sc, float(norm[i]), float(r4[i]), float(r8[i])))
                w.writerow([tid, text.replace("\n", "\\n"), sc, f"{norm[i]:.4f}", f"{r4[i]:.5f}", f"{r8[i]:.5f}"])
    summary = {}
    for sc, nm, r4, r8 in rows:
        d = summary.setdefault(sc, {"count": 0, "norm": [], "q4": [], "q8": []})
        d["count"] += 1; d["norm"].append(nm); d["q4"].append(r4); d["q8"].append(r8)
    table = []
    for sc, d in summary.items():
        table.append({"script": sc, "count": d["count"], "mean_norm": float(np.mean(d["norm"])),
                      "relerr_q4_0_mean": float(np.mean(d["q4"])), "relerr_q4_0_p90": float(np.percentile(d["q4"], 90)),
                      "relerr_q8_0_mean": float(np.mean(d["q8"]))})
    table.sort(key=lambda r: -r["relerr_q4_0_mean"])
    json.dump(table, open(os.path.splitext(a.out)[0] + ".summary.json", "w"), indent=1)
    print(f"{'script':16s}{'rows':>8s}{'mean norm':>11s}{'Q4_0 err':>10s}{'Q4_0 p90':>10s}{'Q8_0 err':>10s}")
    for r in table:
        print(f"{r['script']:16s}{r['count']:8d}{r['mean_norm']:11.3f}{r['relerr_q4_0_mean']:10.4f}{r['relerr_q4_0_p90']:10.4f}{r['relerr_q8_0_mean']:10.4f}")
    print("relative error = ||w - dequant(w)|| / ||w|| per row; compare scripts, then compare with the per-language eval deltas.")

if __name__ == "__main__":
    main()
