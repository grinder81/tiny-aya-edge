"""Check optional GPU float64 reduction and native inference after Torch import."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from builds.profile_pilot_kl import chunked_kl,main as profile,sha
from builds.t3_fidelity import TorchReducer


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--torch-site',type=Path,required=True)
    a,remaining=p.parse_known_args()
    # Initialize the native CUDA backend before Torch can resolve competing
    # CUDA/cuBLAS shared libraries from its separate environment.
    runtime_path=Path(remaining[remaining.index('--bf16-runtime')+1])
    runtime=json.loads(runtime_path.read_text())
    model_path=runtime['server_command'][runtime['server_command'].index('--model')+1]
    from llama_cpp import Llama
    vocabulary=Llama(model_path=model_path,vocab_only=True,n_gpu_layers=0,verbose=False)
    vocabulary.close()
    reducer=TorchReducer(a.torch_site)
    sys.argv=[sys.argv[0]]+remaining
    profile()  # Existing T1 three-prompt bridge, now with Torch already imported.
    output=Path(remaining[remaining.index('--output')+1])
    report=json.loads((output/'report.json').read_text())
    checks=[]
    for lang in ('eng_Latn','ben_Beng','hin_Deva'):
        source=np.load(output/(lang+'-bf16-logits.npy'),mmap_mode='r',allow_pickle=False)
        reference=np.asarray(source[:11])
        candidate=(reference*np.float32(.997)+np.sin(np.arange(reference.shape[1],dtype=np.float32))[None,:]*np.float32(.03)).astype(np.float32)
        expected=chunked_kl(reference,candidate)
        actual=reducer(reference,candidate)
        error=max(abs(x-y) for x,y in zip(expected['token_kl'],actual['token_kl'],strict=True))
        if error>1e-10:
            raise ValueError('Optional GPU reduction disagrees with independent CPU oracle')
        checks.append(dict(language=lang,tokens=11,max_error_nats=error))
    torch=reducer.torch
    receipt=dict(status='passed',implementation_sha256=sha(__file__),
        native_profile_sha256=sha(output/'report.json'),checks=checks,
        native_t1_max_letter_error=max(r['t1_letter_max_error_nats'] for r in report['records']),
        torch_version=torch.__version__,torch_source_sha256=sha(torch.__file__),torch_extension_sha256=sha(torch._C.__file__),
        reducer_source_sha256=sha(Path(__file__).with_name('t3_fidelity.py')),
        native_backend_initialized_before_torch=True,
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        interpretation='Nontrivial synthetic perturbations of actual logits verify reduction math; three T1 prompts bound native inference compatibility after Torch import. No C8 language rankings or new task results.')
    with (output/'torch-verification.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
        stream.write('\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
