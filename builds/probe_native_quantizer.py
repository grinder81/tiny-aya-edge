"""Probe real K-quant APIs against representative exported tensor rows (CPU only)."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path

import gguf
import numpy as np


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def probe(source, exported, library):
    native = ctypes.CDLL(str(library))
    native.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    native.ggml_row_size.restype = ctypes.c_size_t
    native.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
    native.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
    native.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    native.ggml_quantize_chunk.restype = ctypes.c_size_t
    original = {t.name: t for t in gguf.GGUFReader(str(source)).tensors}
    compressed = gguf.GGUFReader(str(exported))
    results = []
    for kind in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
        try:
            gguf.quantize(np.linspace(-1, 1, 256, dtype=np.float32).reshape(1, 256), kind)
            python_support = 'implemented'
        except NotImplementedError:
            python_support = 'NotImplementedError'
        tensor = next(t for t in compressed.tensors if t.tensor_type == kind)
        src = original[tensor.name]
        assert src.tensor_type == gguf.GGMLQuantizationType.BF16
        assert np.array_equal(src.shape, tensor.shape) and len(src.shape) == 2
        width, height = map(int, src.shape)
        assert not native.ggml_quantize_requires_imatrix(int(kind))
        size = native.ggml_row_size(int(kind), width)
        decode = getattr(native, 'dequantize_row_' + kind.name.lower().replace('_k', '_K'))
        decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        decode.restype = None
        for row in sorted({0, height // 2, height - 1}):
            values = np.ascontiguousarray(gguf.dequantize(src.data[row:row+1], src.tensor_type).ravel(), dtype=np.float32)
            assert len(values) == width
            packed = np.empty(size, dtype=np.uint8)
            written = native.ggml_quantize_chunk(int(kind), values.ctypes.data, packed.ctypes.data, 0, 1, width, None)
            assert written == size
            expected = tensor.data[row:row+1].ravel()
            native_values = np.empty(width, dtype=np.float32)
            decode(packed.ctypes.data, native_values.ctypes.data, width)
            python_values = gguf.dequantize(packed.reshape(1, -1), kind).ravel()
            record = dict(type=kind.name, tensor=tensor.name, row=row, columns=width,
                python_quantizer=python_support, quantized_bytes=int(size),
                matches_exported_bytes=bool(np.array_equal(packed, expected)),
                native_python_dequant_max_abs_difference=float(np.max(np.abs(native_values-python_values))),
                mean_squared_weight_error=float(np.mean((values-native_values)**2)),
                exported_row_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                native_row_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
            assert np.isfinite(native_values).all() and np.isfinite(python_values).all()
            results.append(record)
    return dict(scope='Two representative tensors, first/middle/last rows only; no sensitivity map or quality evaluation',
                gguf_python_version=importlib.metadata.version('gguf'),
                artifacts={str(p): sha(p) for p in (source, exported, library)}, rows=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--exported', required=True, type=Path)
    p.add_argument('--library', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = probe(args.source, args.exported, args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    if not all(r['matches_exported_bytes'] and r['native_python_dequant_max_abs_difference'] == 0 for r in result['rows']):
        raise ValueError('Native/exported or dequantizer parity failed; inspect report')
    print('Six sampled rows match exported bytes and dequantization exactly.')


if __name__ == '__main__':
    main()
