"""Read-only environment probe. Run with the Python environment being assessed."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"ok": result.returncode == 0, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--disk-path", type=Path, default=Path.cwd())
    parser.add_argument("--check-torch", action="store_true")
    parser.add_argument("--check-vllm", action="store_true")
    args = parser.parse_args()
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "disk_free_bytes": shutil.disk_usage(args.disk_path).free,
              "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                     "--format=csv"])}
    nvcc = str(args.cuda_root / "bin" / "nvcc") if args.cuda_root else "nvcc"
    report["nvcc"] = command([nvcc, "--version"])
    report["cmake"] = command(["cmake", "--version"])
    report["packages"] = {}
    for name in ("torch", "vllm", "lm_eval"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    if args.llama_server:
        report["llama_server"] = command([str(args.llama_server), "--version"])
    else:
        report["llama_server"] = {"status": "not_checked"}
    if args.check_torch:
        try:
            import torch
            x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
            y = x @ x
            torch.cuda.synchronize()
            report["torch_cuda"] = {"ok": torch.equal(y.cpu(), x.cpu() @ x.cpu()),
                                    "cuda": torch.version.cuda,
                                    "gpu": torch.cuda.get_device_name(0)}
        except Exception as exc:
            report["torch_cuda"] = {"ok": False, "error": str(exc)}
    if args.check_vllm:
        try:
            import vllm
            report["vllm_import"] = {"ok": True, "version": vllm.__version__}
        except Exception as exc:
            report["vllm_import"] = {"ok": False, "error": str(exc)}
    print(json.dumps(report, indent=2))
    # This is an inventory, not an inference-ready gate. Missing tools remain visible.


if __name__ == "__main__":
    main()
