#!/usr/bin/env python3
"""FP8 dynamic (W8A8 fp8) build for vLLM on Blackwell. No calibration data needed.
Usage: python builds/make_fp8.py --out out/fp8-dynamic
"""
import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=a.model, recipe=recipe, output_dir=a.out)
    print("wrote", a.out)
if __name__ == "__main__":
    main()
