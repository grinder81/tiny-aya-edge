"""Inspect downloaded artifacts without loading weights onto a GPU.

Run with the Stage 1 converter environment. Output is JSON on stdout.
"""

import argparse
import collections
import json
import math
from pathlib import Path

from gguf import GGUFReader
from safetensors import safe_open


def inspect(root: Path) -> dict:
    base = root / "tiny-aya-global"
    config = json.loads((base / "config.json").read_text())
    index = json.loads((base / "model.safetensors.index.json").read_text())
    dtypes = collections.Counter()
    params = collections.Counter()
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(base / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate tensor: {name}")
                value = handle.get_slice(name)
                shape = value.get_shape()
                dtype = value.get_dtype()
                tensors[name] = {"shape": shape, "dtype": dtype, "shard": shard}
                dtypes[dtype] += 1
                params[dtype] += math.prod(shape)
    if set(tensors) != set(index["weight_map"]):
        raise ValueError("Shard contents differ from tensor index")
    for name, shard in index["weight_map"].items():
        if tensors[name]["shard"] != shard:
            raise ValueError(f"Wrong shard for {name}")

    path = root / "tiny-aya-global-GGUF" / "tiny-aya-global-q4_k_m.gguf"
    reader = GGUFReader(path)
    types = collections.Counter()
    elements = collections.Counter()
    storage = collections.Counter()
    for tensor in reader.tensors:
        kind = tensor.tensor_type.name
        types[kind] += 1
        elements[kind] += int(tensor.n_elements)
        storage[kind] += int(tensor.n_bytes)
    total = sum(elements.values())
    return {
        "configuration": config,
        "original": {
            "tensor_counts_by_dtype": dict(dtypes),
            "parameter_counts_by_dtype": dict(params),
            "stored_parameter_count": sum(params.values()),
            "tensor_inventory": tensors,
        },
        "official_gguf": {
            "tensor_counts_by_type": dict(types),
            "parameter_counts_by_type": dict(elements),
            "tensor_bytes_by_type": dict(storage),
            "stored_parameter_count": total,
            "tensor_payload_bits_per_weight": 8 * sum(storage.values()) / total,
            "whole_file_bits_per_weight": 8 * path.stat().st_size / total,
            "note": "Whole-file ratio includes metadata and alignment. Provenance is not inferred from tensor types.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.models_root), indent=2))
