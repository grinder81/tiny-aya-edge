"""Diagnostic of upstream echo likelihoods; not an evaluation adapter."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from llama_cpp import Llama


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = Llama(model_path=args.model, n_gpu_layers=-1, n_ctx=512,
                  logits_all=True, seed=42, verbose=False)
    rows = []
    for context in ["The library is closed on Friday.\nAnswer:",
                    "পাঠাগার শুক্রবার বন্ধ থাকে।\nAnswer:"]:
        for candidate in [" A", " B", " C", " D"]:
            text = context + candidate
            result = model.create_completion(prompt=text, max_tokens=1,
                                             temperature=0, logprobs=1, echo=True)
            choice = result["choices"][0]
            lp = choice["logprobs"]
            tokens = model.tokenize(text.encode(), add_bos=True)
            prefix = model.tokenize(context.encode(), add_bos=True)
            assert tokens[:len(prefix)] == prefix, "Boundary merges with context"
            assert len(lp["token_logprobs"]) > 1, "Missing echoed probabilities"
            assert choice["text"].startswith(text), "Prompt not echoed"
            indexes = [i for i, offset in enumerate(lp["text_offset"][:-1])
                       if offset >= len(context)]
            assert indexes, "No candidate positions"
            scores = [lp["token_logprobs"][i] for i in indexes]
            assert all(value is not None and math.isfinite(value) for value in scores)
            # Independent index check against raw logits from the same model.
            # This diagnostic never supplies scores to the evaluation harness.
            model.reset()
            model.eval(tokens)
            direct = []
            for position in range(len(prefix), len(tokens)):
                logits = model.scores[position - 1].astype(np.float64)
                maximum = logits.max()
                normalizer = maximum + np.log(np.exp(logits - maximum).sum())
                direct.append(float(logits[tokens[position]] - normalizer))
            assert len(scores) == len(direct), "Echo candidate token count mismatch"
            assert np.allclose(scores, direct, atol=1e-4, rtol=0), "Echo/logit mismatch"
            rows.append(dict(context=context, candidate=candidate, token_ids=tokens,
                             context_token_ids=prefix, direct_logprobs=direct,
                             echo=lp, candidate_indexes=indexes,
                             continuation_loglikelihood=sum(scores)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(results=rows,
        scope="Echo contract probe only; no benchmark or cross-runtime parity claim"),
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("Eight echo likelihood probes passed; equivalence still requires validation.")


if __name__ == "__main__":
    main()
