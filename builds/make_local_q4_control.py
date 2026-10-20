"""Build a provenance-controlled Q4_K_M from the audited Stage 1 BF16 GGUF."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASE_SHA = "0e2f666ac8fb50a7f53d14164147beeca35905b76b93eb90e0e3a963eb203b4b"
QUANTIZER_REV = "797de63b2fa7396dcc8caca76834c699ac84e1c3"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    source = root / "models/tiny-aya-global-control-bf16.gguf"
    output = root / "models/tiny-aya-global-local-q4_k_m.gguf"
    checkout = root / "llama.cpp"
    binary = checkout / "build/bin/llama-quantize"
    if output.exists() or args.evidence.exists():
        raise FileExistsError("Refusing to overwrite existing artifact or evidence")
    assert sha(source) == BASE_SHA, "BF16 source hash mismatch"
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    assert revision == QUANTIZER_REV, "Unexpected quantizer source revision"
    diff = subprocess.check_output(["git", "-C", str(checkout), "diff", "HEAD"], text=True)
    assert not diff, "Quantizer checkout has tracked modifications"
    args.evidence.mkdir(parents=True)
    command = [str(binary), str(source), str(output), "Q4_K_M", "8"]
    manifest = {
        "base_artifact_sha256": BASE_SHA,
        "original_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
        "quantizer_revision": revision,
        "quantizer_binary_sha256": sha(binary),
        "command": command,
        "calibration": None,
        "importance_matrix": None,
        "note": "Default Q4_K_M tensor allocation; no training or importance matrix. Runtime tokenizer patches are separate.",
    }
    with (args.evidence / "quantize.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest.update(output_sha256=sha(output), output_bytes=output.stat().st_size)
    (args.evidence / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
