"""Bind a new precision artifact to the unchanged T1 runtime; no inference."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('original','build','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    original=json.loads(a.original.read_text())
    build=json.loads(a.build.read_text())
    if build['status']!='complete' or sha(build['output_path'])!=build['output_sha256']:
        raise ValueError('Build identity failed')
    for path,digest in original['files'].items():
        if sha(path)!=digest:
            raise ValueError('Original runtime changed: '+path)
    new=json.loads(json.dumps(original))
    command=new['server_command']
    index=command.index('--model')+1
    old_model=command[index]
    command[index]=build['output_path']
    del new['files'][old_model]
    new['files'][build['output_path']]=build['output_sha256']
    new['arm']=a.arm
    new['t3_provenance']=dict(original_runtime_sha256=sha(a.original),build_manifest_sha256=sha(a.build),
        implementation_sha256=sha(__file__),scope='Only model artifact changes; T1 wrapper, native library and settings retained')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(new,stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    main()
