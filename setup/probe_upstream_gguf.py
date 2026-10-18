"""Exercise the installed lm-eval GGUF integration unchanged on synthetic cases."""
import argparse
import json
import math
from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.models.gguf import GGUFLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    requests = [Instance(request_type="loglikelihood", doc={},
                         arguments=(row["context"], row["candidate"]), idx=i)
                for i, row in enumerate(reference["results"])]
    model = GGUFLM(base_url=args.base_url, max_length=512)
    results = model.loglikelihood(requests)
    assert len(results) == len(requests)
    errors = [abs(score - row["continuation_loglikelihood"])
              for (score, _), row in zip(results, reference["results"])]
    assert all(math.isfinite(error) and error < 1e-4 for error in errors)
    args.output.write_text(json.dumps(dict(results=results, absolute_errors=errors,
        scope="Unmodified upstream harness vs same-model direct probe; synthetic only"),
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
