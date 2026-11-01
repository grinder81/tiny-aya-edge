"""Separate cache verification and full host-to-GPU reduction costs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from builds.profile_pilot_kl import sha
from builds.t3_fidelity import TorchReducer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    before=time.perf_counter()
    digest=sha(a.reference)
    sha_seconds=time.perf_counter()-before
    reference=np.load(a.reference,mmap_mode='r',allow_pickle=False)
    candidate=(np.asarray(reference)*np.float32(.997)).astype(np.float32)
    reducer=TorchReducer(Path(torch.__file__).parents[1])
    original_threads=torch.get_num_threads()
    results=[]
    for threads in (original_threads,1):
        torch.set_num_threads(threads)
        torch.cuda.synchronize()
        before=time.perf_counter()
        value=reducer(reference,candidate)
        torch.cuda.synchronize()
        results.append(dict(torch_cpu_threads=threads,seconds=time.perf_counter()-before,
            tokens=value['tokens'],cpu_oracle_error=value['first_chunk_cpu_oracle_max_error']))
    report=dict(implementation_sha256=sha(__file__),reference_sha256=digest,
        reference_bytes=a.reference.stat().st_size,sha_seconds=sha_seconds,results=results,
        scope='Fixed-logit fixture; complete host/device reduction with CPU oracle, no model forward or language comparison')
    with a.output.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
