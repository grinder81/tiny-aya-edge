"""Synthetic reference probabilities only; not a benchmark adapter."""
import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = LLM(model=args.model, dtype="bfloat16", max_model_len=512,
                gpu_memory_utilization=0.4, enforce_eager=True, seed=42,
                enable_prefix_caching=False)
    tokenizer = model.get_tokenizer()
    pairs = [(context, candidate)
             for context in ["The library is closed on Friday.\nAnswer:",
                             "পাঠাগার শুক্রবার বন্ধ থাকে।\nAnswer:"]
             for candidate in [" A", " B", " C", " D"]]
    outputs = model.generate([context + candidate for context, candidate in pairs],
                             SamplingParams(temperature=0, max_tokens=1,
                                            prompt_logprobs=1))
    rows = []
    for (context, candidate), output in zip(pairs, outputs):
        ids = list(output.prompt_token_ids)
        prefix = tokenizer.encode(context, add_special_tokens=True)
        assert ids[:len(prefix)] == prefix, "Boundary merges with context"
        scores = [output.prompt_logprobs[i][ids[i]].logprob
                  for i in range(len(prefix), len(ids))]
        rows.append(dict(context=context, candidate=candidate, token_ids=ids,
                         context_token_ids=prefix, candidate_logprobs=scores,
                         continuation_loglikelihood=sum(scores)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(results=rows,
        scope="Synthetic vLLM reference only; no benchmark score"),
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
