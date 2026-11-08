"""Verify and time two equivalent float64 reductions on fixed real-logit fixtures."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from builds.blocked_kl import reduce_chunk
from builds.profile_pilot_kl import chunked_kl,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    reference=np.asarray(np.load(a.reference,mmap_mode='r',allow_pickle=False)[:8])
    candidate=(reference*np.float32(.997)+np.sin(np.arange(reference.shape[1],dtype=np.float32))[None,:]*np.float32(.03)).astype(np.float32)
    left=torch.tensor(reference,device='cuda',dtype=torch.float64)
    right=torch.tensor(candidate,device='cuda',dtype=torch.float64)
    def original():
        pa,pb=torch.log_softmax(left,-1),torch.log_softmax(right,-1)
        return (pa.exp()*(pa-pb)).sum(-1),(left.argmax(-1)==right.argmax(-1)).sum()
    def blocked():
        return reduce_chunk(torch,left,right)
    times={}
    values={}
    for name,call in [('original',original),('blocked',blocked)]:
        call();call();torch.cuda.synchronize()
        elapsed=[]
        for _ in range(10):
            before=time.perf_counter()
            value,agreement=call()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter()-before)
        times[name]=dict(median_seconds=median(elapsed),samples=elapsed)
        values[name]=value.cpu().numpy()
    expected=chunked_kl(reference,candidate)
    errors={name:float(np.max(np.abs(value-expected['token_kl']))) for name,value in values.items()}
    if max(errors.values())>1e-10:
        raise ValueError('Blocked reduction failed independent CPU oracle')
    report=dict(status='passed',implementation_sha256=sha(__file__),kernel_sha256=sha(Path(__file__).with_name('blocked_kl.py')),
        reference_sha256=sha(a.reference),rows=8,vocabulary=reference.shape[1],dtype='float64',
        torch=torch.__version__,times=times,cpu_oracle_max_errors=errors,
        interpretation='Isolated reduction timings on fixed perturbed real logits; excludes model inference, disk I/O and host/device copies.')
    with a.output.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
