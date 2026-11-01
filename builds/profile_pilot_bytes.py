"""Measure native quantizer dry-run layouts and derive exact shipped-byte budgets."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import gguf

QUANTIZER_REV = '797de63b2fa7396dcc8caca76834c699ac84e1c3'
QUANTIZER_SHA = '6a2c2efac5f38ebe2c1994befd0dee745cb24ec5c33d6a2c5a50fb8b9fe1e5f0'
SOURCE_SHA = '0e2f666ac8fb50a7f53d14164147beeca35905b76b93eb90e0e3a963eb203b4b'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_layout(log, tensors):
    """Use native decisions, never rounded MiB strings, for each tensor's type."""
    result = {}
    for line in log.splitlines():
        match = re.match(r'^\[\s*\d+/\s*\d+\]\s+(\S+)\s+-.*?type\s*=\s*(\w+)', line)
        if not match:
            continue
        name, original = match.groups()
        if name in result or name not in tensors:
            raise ValueError('Unknown or repeated native tensor: ' + name)
        converted = re.search(r'->.*\((\w+)\)\s*$', line)
        result[name] = (converted.group(1) if converted else original).upper()
    if set(result) != set(tensors):
        raise ValueError('Incomplete native dry-run tensor decisions')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'q4', 'checkout', 'output'):
        p.add_argument('--'+key, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    binary = a.checkout/'build/bin/llama-quantize'
    native_file = a.checkout/'build/bin/libggml-base.so'
    revision = subprocess.check_output(['git','-C',str(a.checkout),'rev-parse','HEAD'],text=True).strip()
    if revision != QUANTIZER_REV or sha(binary) != QUANTIZER_SHA or sha(a.source) != SOURCE_SHA:
        raise ValueError('Pinned source/quantizer identity failed')
    if subprocess.check_output(['git','-C',str(a.checkout),'diff','HEAD'],text=True):
        raise ValueError('Tracked quantizer modifications')
    reader = gguf.GGUFReader(str(a.source))
    tensors = {t.name:t for t in reader.tensors}
    overhead = a.source.stat().st_size - sum(t.n_bytes for t in tensors.values())
    native = ctypes.CDLL(str(native_file))
    native.ggml_row_size.argtypes = [ctypes.c_int,ctypes.c_int64]
    native.ggml_row_size.restype = ctypes.c_size_t
    a.output.mkdir(parents=True)
    records = []
    configs = [(f,False) for f in ('Q6_K','Q4_K_M','Q3_K_M','Q3_K_S','Q2_K','IQ3_S')]
    configs += [(f,True) for f in ('Q4_K_M','Q3_K_M')]
    for file_type,pure in configs:
        label = file_type.lower()+('-pure' if pure else '')
        command = [str(binary),'--dry-run']+(['--pure'] if pure else [])+[str(a.source),file_type,'8']
        started = time.monotonic()
        run = subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        elapsed = time.monotonic()-started
        log = a.output/(label+'.log')
        log.write_text(run.stdout,encoding='utf-8')
        layout = parse_layout(run.stdout,tensors)
        payloads = {}
        for name,kind in layout.items():
            tensor = tensors[name]
            width = int(tensor.shape[0])
            nbytes = int(native.ggml_row_size(int(gguf.GGMLQuantizationType[kind]),width))*(int(tensor.n_elements)//width)
            if nbytes % 32:
                raise ValueError('Non-aligned tensor needs an explicit offset model')
            payloads[name] = nbytes
        predicted = overhead+sum(payloads.values())
        record = dict(label=label,command=command,seconds=elapsed,log_sha256=sha(log),
            non_payload_bytes=overhead,predicted_file_bytes=predicted,
            tensors={n:dict(type=layout[n],payload_bytes=payloads[n]) for n in layout})
        if label == 'q4_k_m':
            actual = gguf.GGUFReader(str(a.q4))
            if {t.name:(t.tensor_type.name,int(t.n_bytes)) for t in actual.tensors} != {
                    n:(layout[n],payloads[n]) for n in layout} or a.q4.stat().st_size != predicted:
                raise ValueError('Exact Q4 export validation failed')
            record['validated_against_actual_file'] = dict(bytes=a.q4.stat().st_size,sha256=sha(a.q4))
        records.append(record)
        print(label,predicted,flush=True)
    report = dict(scope='CPU native dry-run feasibility; no new weights or model forwards',
        source=dict(path=str(a.source),sha256=SOURCE_SHA,bytes=a.source.stat().st_size),
        quantizer=dict(revision=revision,binary_sha256=QUANTIZER_SHA,library_sha256=sha(native_file)),
        implementation_sha256=sha(__file__),records=records,
        stored_embedding_output=dict(tensor='token_embd.weight',output_weight_present='output.weight' in tensors,
            parameters=int(tensors['token_embd.weight'].n_elements)),
        byte_note='Exact native row sizes plus source metadata/alignment; Q4 validated against actual export. Other predicted sizes require their own export check.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
