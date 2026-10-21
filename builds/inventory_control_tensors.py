"""Inventory BF16/local-Q4 tensor storage without inference or calibration."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from gguf import GGUFReader


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(path):
    reader = GGUFReader(path)
    tensors = {}
    for tensor in reader.tensors:
        if tensor.name in tensors:
            raise ValueError(f'Duplicate tensor {tensor.name}')
        tensors[tensor.name] = {
            'shape_gguf': tensor.shape.tolist(),
            'parameters': int(tensor.n_elements),
            'type': tensor.tensor_type.name,
            'payload_bytes': int(tensor.n_bytes),
        }
    return tensors


def inventory(bf16, q4):
    left, right = inspect(bf16), inspect(q4)
    if left.keys() != right.keys():
        raise ValueError('Tensor names differ')
    groups = {}
    rows = []
    for name, original in left.items():
        compressed = right[name]
        for field in ('shape_gguf', 'parameters'):
            if original[field] != compressed[field]:
                raise ValueError(f'{name}: {field} differs')
        group = name.split('.')[-2] if name.startswith('blk.') else name.split('.')[0]
        total = groups.setdefault(group, collections.Counter())
        total.update(tensors=1, parameters=original['parameters'],
                     bf16_payload_bytes=original['payload_bytes'],
                     q4_payload_bytes=compressed['payload_bytes'])
        rows.append({'name': name, 'group': group, 'bf16': original, 'local_q4': compressed})
    artifacts = {}
    for label, path, tensors in [('bf16', bf16, left), ('local_q4', q4, right)]:
        payload = sum(t['payload_bytes'] for t in tensors.values())
        artifacts[label] = {
            'name': path.name, 'sha256': digest(path), 'file_bytes': path.stat().st_size,
            'tensor_payload_bytes': payload,
            'non_payload_bytes': path.stat().st_size - payload,
            'parameters': sum(t['parameters'] for t in tensors.values()),
            'tensor_types': dict(collections.Counter(t['type'] for t in tensors.values())),
        }
    return {'artifacts': artifacts, 'groups': groups, 'tensors': rows,
            'storage_note': 'Unique stored tensors counted once; shapes use GGUF dimension order. '
                            'Non-payload includes metadata and alignment. Storage alone does not prove runtime tying.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bf16', type=Path)
    parser.add_argument('local_q4', type=Path)
    args = parser.parse_args()
    print(json.dumps(inventory(args.bf16, args.local_q4), indent=2))
