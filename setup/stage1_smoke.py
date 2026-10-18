"""Two synthetic prompts: runtime smoke only, never a quality benchmark."""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    ("en", "The library is closed on Friday. On which day is the library closed? Answer briefly."),
    ("bn", "পাঠাগার শুক্রবার বন্ধ থাকে। পাঠাগার কোন দিন বন্ধ থাকে? সংক্ষেপে উত্তর দিন।"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["vllm", "llama"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server")
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--override-kv", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        model = LLM(model=args.model, dtype="bfloat16", max_model_len=1024,
                    gpu_memory_utilization=0.5, enforce_eager=True, seed=42)
        tokenizer = model.get_tokenizer()
        rendered = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True) for _, prompt in PROMPTS]
        results = model.generate(rendered, SamplingParams(temperature=0, max_tokens=64))
        for (language, prompt), result in zip(PROMPTS, results):
            output = result.outputs[0]
            rows.append(dict(language=language, prompt=prompt, text=output.text,
                             finish_reason=output.finish_reason,
                             generated_tokens=len(output.token_ids)))
    else:
        if not args.server:
            parser.error("--server is required for llama")
        command = [args.server, "-m", args.model, "-ngl", "99", "-c", "1024",
                   "--host", "127.0.0.1", "--port", str(args.port)]
        for override in args.override_kv:
            command.extend(["--override-kv", override])
        base = f"http://127.0.0.1:{args.port}"
        with args.output.with_suffix(".server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("llama-server exited; inspect server log")
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (OSError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("llama-server readiness timeout")
                    time.sleep(1)
                for language, prompt in PROMPTS:
                    payload = dict(messages=[dict(role="user", content=prompt)],
                                   temperature=0, max_tokens=64, seed=42)
                    request = urllib.request.Request(base + "/v1/chat/completions",
                        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=180) as response:
                        result = json.load(response)
                    choice = result["choices"][0]
                    rows.append(dict(language=language, prompt=prompt,
                        text=choice["message"]["content"], finish_reason=choice["finish_reason"],
                        generated_tokens=result["usage"]["completion_tokens"]))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if len(rows) != len(PROMPTS) or any(not row["text"].strip() for row in rows):
        raise RuntimeError("Missing or empty output")
    args.output.write_text(json.dumps(dict(backend=args.backend, temperature=0,
        max_tokens=64, seed=42, context_limit=1024,
        vllm_use_v2_model_runner=os.environ.get("VLLM_USE_V2_MODEL_RUNNER") if args.backend == "vllm" else None,
        overrides=args.override_kv,
        results=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    if any(row["finish_reason"] != "stop" for row in rows):
        raise RuntimeError("Outputs saved, but normal stopping failed")
    print("Smoke outputs saved; nonempty generation and normal stopping passed. Not a quality score.")


if __name__ == "__main__":
    main()
