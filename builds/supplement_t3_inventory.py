"""Bind an actual T3 file inventory to its immutable original build manifest."""
import argparse
from collections import Counter
import json
from pathlib import Path

from builds.native_gguf import NativeGGUF
from builds.profile_pilot_kl import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','source','library','dry-run-report','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--timing',choices=('after-scoring','before-inference'),required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest=json.loads(a.manifest.read_text())
    if sha(manifest['output_path'])!=manifest['output_sha256'] or sha(a.source)!=manifest['source_sha256']:
        raise ValueError('Immutable build/source artifact changed')
    actual=NativeGGUF(manifest['output_path'],a.library)
    original=NativeGGUF(a.source,a.library)
    predicted=next(row for row in json.loads(a.dry_run_report.read_text())['records'] if row['label']==manifest['file_type'].lower())
    if actual.tensors.keys()!=original.tensors.keys() or actual.tensors.keys()!=predicted['tensors'].keys():
        raise ValueError('Actual tensor membership differs')
    for name,row in actual.tensors.items():
        if row['shape_gguf']!=original.tensors[name]['shape_gguf'] or any(row[k]!=predicted['tensors'][name][k] for k in ('type','payload_bytes')):
            raise ValueError('Actual inventory differs from pre-run native layout: '+name)
    if manifest['output_bytes']!=predicted['predicted_file_bytes']:
        raise ValueError('Actual export byte prediction differs')
    result=dict(build_manifest_sha256=sha(a.manifest),model_sha256=manifest['output_sha256'],
        file_type=manifest['file_type'],actual_file_bytes=manifest['output_bytes'],timing=a.timing,
        implementation_sha256=sha(__file__),library_sha256=sha(a.library),
        pre_run_layout_report_sha256=sha(a.dry_run_report),
        tensor_types=dict(Counter(row['type'] for row in actual.tensors.values())),tensors=actual.tensors,
        interpretation='Actual file matches every pre-run native type/payload decision and original shape. Original build manifest bytes remain unchanged; timing is explicit.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(file_type=manifest['file_type'],types=result['tensor_types'],timing=a.timing)),flush=True)


if __name__=='__main__':
    main()
