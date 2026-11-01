"""Export a native degrade-one GGUF and verify every non-target tensor unchanged."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import gguf
import numpy as np

from builds.profile_pilot_bytes import QUANTIZER_REV,QUANTIZER_SHA,SOURCE_SHA,sha
from builds.native_gguf import NativeGGUF


from builds.c8_budget import BudgetOperation,require_passed_operation
from contextlib import nullcontext


def run(a,budget):
    if budget:
        budget.check()
        budget.reserve(a.source.stat().st_size+1_000_000)
    binary=a.checkout/'build/bin/llama-quantize'
    if sha(a.source)!=SOURCE_SHA or sha(binary)!=QUANTIZER_SHA:
        raise ValueError('Pinned source/binary identity mismatch')
    if subprocess.check_output(['git','-C',str(a.checkout),'rev-parse','HEAD'],text=True).strip()!=QUANTIZER_REV:
        raise ValueError('Quantizer revision changed')
    if subprocess.check_output(['git','-C',str(a.checkout),'diff','HEAD'],text=True):
        raise ValueError('Quantizer tracked modifications')
    if a.output.exists() or a.evidence.exists():
        raise FileExistsError('Existing diagnostic or evidence')
    native_file=a.checkout/'build/bin/libggml-base.so'
    original=NativeGGUF(a.source,native_file)
    before=original.tensors
    if a.tensor not in before or len(before[a.tensor]['shape_gguf'])!=2:
        raise ValueError('Expected one two-dimensional stored tensor')
    a.evidence.mkdir(parents=True)
    temporary=a.output.with_suffix('.partial.gguf')
    if temporary.exists():
        raise FileExistsError(temporary)
    command=[str(binary),'--tensor-type','^'+re.escape(a.tensor)+'$='+a.native_type.lower(),
        '--tensor-type','.*=bf16',str(a.source),str(temporary),'Q4_K_M','8']
    report=dict(scope='Native mostly-BF16 degrade-one diagnostic, not a shipped Q4 model',
        source_sha256=SOURCE_SHA,quantizer_revision=QUANTIZER_REV,quantizer_binary_sha256=QUANTIZER_SHA,
        tensor=a.tensor,native_type=a.native_type,command_file_type='Q4_K_M',command=command,
        implementation_sha256=sha(__file__),importance_matrix=None)
    if budget:
        report.update(budget_plan_sha256=sha(a.budget_plan),budget_operation_id=budget.operation_id)
    started=time.monotonic()
    try:
        with (a.evidence/'quantize.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=budget.remaining() if budget else None)
        exported=NativeGGUF(temporary,native_file)
        after=exported.tensors
        if before.keys()!=after.keys():
            raise ValueError('Stored tensor membership changed')
        unchanged={}
        for name,src in before.items():
            if budget:
                budget.check(False)
            dst=after[name]
            if src['shape_gguf']!=dst['shape_gguf']:
                raise ValueError('Tensor shape changed: '+name)
            if name==a.tensor:
                if dst['type']!=a.native_type:
                    raise ValueError('Native target override was not applied')
            else:
                source_hash=hashlib.sha256(np.memmap(a.source,mode='r',dtype=np.uint8,offset=src['offset'],shape=(src['payload_bytes'],))).hexdigest()
                target_hash=hashlib.sha256(np.memmap(temporary,mode='r',dtype=np.uint8,offset=dst['offset'],shape=(dst['payload_bytes'],))).hexdigest()
                if src['type']!=dst['type'] or source_hash!=target_hash:
                    raise ValueError('Non-target tensor changed: '+name)
                unchanged[name]=source_hash
        native=ctypes.CDLL(str(native_file))
        native.ggml_quantize_chunk.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
        native.ggml_quantize_chunk.restype=ctypes.c_size_t
        native.ggml_row_size.argtypes=[ctypes.c_int,ctypes.c_int64]
        native.ggml_row_size.restype=ctypes.c_size_t
        src,dst=before[a.tensor],after[a.tensor]
        width,height=src['shape_gguf']
        sampled=[]
        for row in sorted({0,height//2,height-1}):
            values=np.ascontiguousarray(original.rows(a.tensor,row,row+1).ravel(),dtype=np.float32)
            packed=np.empty(native.ggml_row_size(dst['type_id'],width),dtype=np.uint8)
            written=native.ggml_quantize_chunk(dst['type_id'],values.ctypes.data,packed.ctypes.data,0,1,width,None)
            if written!=packed.nbytes or not np.array_equal(packed,exported.raw_rows(a.tensor,row,row+1).ravel()):
                raise ValueError('Native/exported target row disagreement')
            sampled.append(dict(row=row,bytes=packed.nbytes,sha256=hashlib.sha256(packed).hexdigest()))
        report.update(output_sha256=sha(temporary),output_bytes=temporary.stat().st_size,
            output_path=str(a.output),target_payload_bytes=dst['payload_bytes'],
            native_library_sha256=sha(native_file),native_row_parity=sampled,
            unchanged_tensor_payload_sha256=unchanged,seconds=time.monotonic()-started,status='complete')
        if budget:
            budget.check()
        # Readers mmap the partial file; rename is safe on the Linux build host.
        temporary.rename(a.output)
        (a.evidence/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('tensor','native_type','output_bytes','output_sha256','seconds','status')}),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),seconds=time.monotonic()-started)
        (a.evidence/'failure.json').write_text(json.dumps(report,indent=2)+'\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','checkout','output','evidence'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tensor',required=True)
    p.add_argument('--native-type',choices=('Q2_K','Q3_K','Q4_K'),required=True)
    p.add_argument('--budget-plan',type=Path)
    a=p.parse_args()
    context=BudgetOperation(a.budget_plan,'native','export-'+a.tensor+'-'+a.native_type,(a.output,a.evidence)) if a.budget_plan else nullcontext(None)
    with context as budget:
        run(a,budget)


if __name__=='__main__':
    main()
