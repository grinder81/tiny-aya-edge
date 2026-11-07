"""Build one frozen T3 precision point with native provenance and no overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from builds.profile_pilot_bytes import QUANTIZER_REV,QUANTIZER_SHA,SOURCE_SHA,sha
from builds.native_gguf import NativeGGUF

FREEZE_BLOB='700a0f601e6503d95b485ad5d048c20b4ef96910'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','checkout','output','evidence','specification'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--file-type',choices=('Q6_K','Q3_K_M','Q3_K_S','Q2_K','IQ3_S'),required=True)
    a=p.parse_args()
    content=a.specification.read_bytes().replace(b'\r\n',b'\n')
    blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if blob!=FREEZE_BLOB:
        raise ValueError('T3 frozen specification differs')
    binary=a.checkout/'build/bin/llama-quantize'
    if sha(a.source)!=SOURCE_SHA or sha(binary)!=QUANTIZER_SHA:
        raise ValueError('Source/quantizer hash mismatch')
    if subprocess.check_output(['git','-C',str(a.checkout),'rev-parse','HEAD'],text=True).strip()!=QUANTIZER_REV:
        raise ValueError('Quantizer revision differs')
    if subprocess.check_output(['git','-C',str(a.checkout),'diff','HEAD'],text=True):
        raise ValueError('Quantizer tracked modifications')
    temporary=a.output.with_suffix('.partial.gguf')
    if a.output.exists() or temporary.exists() or a.evidence.exists():
        raise FileExistsError('Build or evidence already exists')
    a.evidence.mkdir(parents=True)
    command=[str(binary),str(a.source),str(temporary),a.file_type,'8']
    manifest=dict(source_sha256=SOURCE_SHA,source_bytes=a.source.stat().st_size,
        quantizer_revision=QUANTIZER_REV,quantizer_binary_sha256=QUANTIZER_SHA,
        specification_git_blob=blob,implementation_sha256=sha(__file__),command=command,
        file_type=a.file_type,importance_matrix=None,calibration=None,scope='T3 diagnostic precision point; no training')
    started=time.monotonic()
    try:
        with (a.evidence/'quantize.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=NativeGGUF(temporary,a.checkout/'build/bin/libggml-base.so')
        original=NativeGGUF(a.source,a.checkout/'build/bin/libggml-base.so')
        if actual.tensors.keys()!=original.tensors.keys() or any(row['shape_gguf']!=original.tensors[name]['shape_gguf'] for name,row in actual.tensors.items()):
            raise ValueError('T3 inventory membership or shapes changed')
        manifest['tensor_inventory']=actual.tensors
        manifest['native_inventory_library_sha256']=sha(a.checkout/'build/bin/libggml-base.so')
        manifest.update(output_sha256=sha(temporary),output_bytes=temporary.stat().st_size,
            output_path=str(a.output),seconds=time.monotonic()-started,status='complete')
        temporary.rename(a.output)
        (a.evidence/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(manifest,indent=2),flush=True)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),seconds=time.monotonic()-started)
        (a.evidence/'failure.json').write_text(json.dumps(manifest,indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
