"""Export the one nominated equal-byte pair with every other tensor type locked."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from builds.native_gguf import NativeGGUF
from builds.profile_pilot_bytes import SOURCE_SHA,QUANTIZER_REV,QUANTIZER_SHA,sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','nomination','source','checkout','directory'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    plan,nomination=read(a.plan),read(a.nomination)
    pair=nomination['pair']
    if pair is None or nomination['plan_sha256']!=sha(a.plan) or pair not in plan['pairs'] or not pair['eligible']:
        raise ValueError('No valid frozen feasible nominee')
    if a.directory.exists():
        raise FileExistsError(a.directory)
    binary=a.checkout/'build/bin/llama-quantize'
    library=a.checkout/'build/bin/libggml-base.so'
    if sha(a.source)!=SOURCE_SHA or sha(binary)!=QUANTIZER_SHA or sha(library)!=plan['native_library_sha256']:
        raise ValueError('Pinned export inputs differ')
    if subprocess.check_output(['git','-C',str(a.checkout),'rev-parse','HEAD'],text=True).strip()!=QUANTIZER_REV or subprocess.check_output(['git','-C',str(a.checkout),'diff','HEAD'],text=True):
        raise ValueError('Quantizer source changed')
    a.directory.mkdir(parents=True)
    low,high=plan['formats']['lower'],plan['formats']['higher']
    source=NativeGGUF(a.source,library)
    cases=[]
    hashes={}
    for label,first_type,second_type in [('mapA',high,low),('mapB',low,high)]:
        layout={name:row['type'] for name,row in plan['base_inventory'].items()}
        layout[pair['first']]=first_type
        layout[pair['second']]=second_type
        assignment=a.directory/(label+'-tensor-types.txt')
        assignment.write_text('\n'.join('^'+re.escape(name)+'$='+kind.lower() for name,kind in layout.items())+'\n')
        output=a.directory/(label+'.gguf')
        command=[str(binary),'--tensor-type-file',str(assignment),str(a.source),str(output),'Q4_K_M','8']
        started=time.monotonic()
        with (a.directory/(label+'-quantize.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        model=NativeGGUF(output,library)
        if model.tensors.keys()!=source.tensors.keys():
            raise ValueError('Exported model membership differs')
        payloads={}
        for name,tensor in model.tensors.items():
            if tensor['type']!=layout[name] or tensor['shape_gguf']!=source.tensors[name]['shape_gguf']:
                raise ValueError('Locked complete-map layout differs: '+name)
            if name not in (pair['first'],pair['second']):
                payloads[name]=hashlib.sha256(np.memmap(output,mode='r',dtype=np.uint8,
                    offset=tensor['offset'],shape=(tensor['payload_bytes'],))).hexdigest()
        predicted=pair['map_a_bytes'] if label=='mapA' else pair['map_b_bytes']
        actual=output.stat().st_size
        if actual!=predicted or actual>plan['operating_point']['bytes'] or actual>2_200_000_000:
            raise ValueError('Actual complete-map bytes violate frozen budget')
        hashes[label]=payloads
        cases.append(dict(label=label,path=str(output),sha256=sha(output),bytes=actual,
            command=command,assignment_sha256=sha(assignment),seconds=time.monotonic()-started))
    if cases[0]['bytes']!=cases[1]['bytes'] or hashes['mapA']!=hashes['mapB']:
        raise ValueError('Maps differ outside their pair or fail exact byte matching')
    report=dict(schema='c8-nominated-pair-export-v1',plan_sha256=sha(a.plan),nomination_sha256=sha(a.nomination),
        source_sha256=SOURCE_SHA,quantizer_revision=QUANTIZER_REV,quantizer_binary_sha256=QUANTIZER_SHA,
        implementation_sha256=sha(__file__),pair=pair,cases=cases,
        unchanged_nonpair_payload_sha256=hashes['mapA'],
        note='Exact actual-byte equality; all nonpair format assignments and payloads match. No padding or importance matrix.')
    with (a.directory/'pair-export.json').open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(cases=cases)),flush=True)


if __name__=='__main__':
    main()
