#!/usr/bin/env python3
"""AWQ and GPTQ INT4 g128 builds with llm-compressor. Calibration set is a first-class argument
because the English-vs-multilingual calibration comparison is one of the study's results.

Usage:
  python builds/make_awq_gptq.py --method gptq --calib en-ultrachat --out out/gptq-int4-g128-en
  python builds/make_awq_gptq.py --method awq  --calib multi14 --calib-file data/calib_multi14.jsonl --out out/awq-int4-g128-multi14
Serve with vLLM:  vllm serve out/gptq-int4-g128-en --served-model-name tiny-aya-gptq
API names drift between llm-compressor releases; check `from llmcompressor import oneshot` docs if imports fail.
"""
import argparse, json, os, hashlib
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def load_calib(name, path, n, max_len, tok):
    if name == "en-ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{n}]")
        texts = [tok.apply_chat_template(x["messages"], tokenize=False) for x in ds]
    elif name == "multi14":
        # JSONL with {"text": ..., "lang": ...}; built from Aya Collection, FLORES dev and web text,
        # balanced across the 14 languages. Hash it into the manifest.
        rows = [json.loads(l) for l in open(path)][:n]
        texts = [r["text"] for r in rows]
    else:
        raise SystemExit(f"unknown calib {name}")
    return Dataset.from_dict({"text": texts})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--method", choices=["gptq", "awq"], required=True)
    ap.add_argument("--calib", choices=["en-ultrachat", "multi14"], default="en-ultrachat")
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from llmcompressor import oneshot
    if a.method == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], group_size=a.group_size)
    else:
        from llmcompressor.modifiers.awq import AWQModifier
        recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], group_size=a.group_size)

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_calib(a.calib, a.calib_file, a.n, a.max_len, tok)
    oneshot(model=a.model, dataset=ds, recipe=recipe, output_dir=a.out,
            max_seq_length=a.max_len, num_calibration_samples=a.n)

    manifest = {
        "base_model": a.model, "format": f"{a.method}-int4-g{a.group_size}",
        "quantizer": {"tool": "llm-compressor", "version": __import__("llmcompressor").__version__},
        "calibration": {"set": a.calib, "samples": a.n, "max_len": a.max_len,
                        "sha256": hashlib.sha256(open(a.calib_file, "rb").read()).hexdigest() if a.calib_file else None},
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
