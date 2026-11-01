"""Use verified native-first initialization around the immutable T3 cache engine."""
import argparse
import json
from pathlib import Path
import sys

from builds.profile_pilot_kl import sha
from builds import t3_fidelity


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    for key in ('verification','reference-runtime','torch-site'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reference-manifest-sha256',required=True)
    a,remaining=p.parse_known_args()
    if not remaining or remaining[0]!='compare':
        raise ValueError('Verified wrapper is for comparison against the frozen reference')
    def arg(name):
        return Path(remaining[remaining.index('--'+name)+1])
    reference=arg('reference')
    runtime=json.loads(arg('runtime').read_text())
    prior=json.loads(a.reference_runtime.read_text())
    manifest=json.loads((reference/'manifest.json').read_text())
    receipt=json.loads(a.verification.read_text())
    if sha(reference/'manifest.json')!=a.reference_manifest_sha256 or manifest['identity']['runtime_sha256']!=sha(a.reference_runtime):
        raise ValueError('Wrong frozen BF16 reference lineage')
    core=Path(t3_fidelity.__file__)
    math_file=core.with_name('profile_pilot_kl.py')
    if manifest['identity']['implementation_sha256']!=sha(core) or manifest['identity']['math_module_sha256']!=sha(math_file):
        raise ValueError('Reference extraction implementation changed')
    current_model=runtime['server_command'][runtime['server_command'].index('--model')+1]
    prior_model=prior['server_command'][prior['server_command'].index('--model')+1]
    expected=list(prior['server_command'])
    expected[expected.index('--model')+1]=current_model
    prior_files={str(Path(k).resolve()):v for k,v in prior['files'].items() if k!=prior_model}
    current_files={str(Path(k).resolve()):v for k,v in runtime['files'].items() if k!=current_model}
    if runtime['server_command']!=expected or current_files!=prior_files or manifest['identity']['imported_runtime']!=prior_files:
        raise ValueError('Non-model inference settings or native imports changed')
    if receipt['status']!='passed' or not receipt['native_backend_initialized_before_torch'] or receipt['reducer_source_sha256']!=sha(core) or receipt['native_t1_max_letter_error']>.0002 or any(r['max_error_nats']>1e-10 for r in receipt['checks']):
        raise ValueError('Optional GPU path lacks its numerical verification')
    if sha(a.verification.parent/'report.json')!=receipt['native_profile_sha256']:
        raise ValueError('Native compatibility evidence changed')
    from llama_cpp import Llama
    vocabulary=Llama(model_path=current_model,vocab_only=True,n_gpu_layers=0,verbose=False)
    vocabulary.close()
    reducer=t3_fidelity.TorchReducer(a.torch_site)
    torch=reducer.torch
    if sha(torch.__file__)!=receipt['torch_source_sha256'] or sha(torch._C.__file__)!=receipt['torch_extension_sha256']:
        raise ValueError('Verified Torch implementation changed')
    sys.argv=[sys.argv[0]]+remaining+['--torch-site',str(a.torch_site)]
    t3_fidelity.main()
    directory=arg('directory')
    if (directory/'manifest.json').exists():
        result=dict(status='passed',wrapper_sha256=sha(__file__),
            verification_sha256=sha(a.verification),reference_manifest_sha256=a.reference_manifest_sha256,
            manifest_sha256=sha(directory/'manifest.json'),runtime_sha256=sha(arg('runtime')),
            native_backend_initialized_before_torch=True)
        path=directory.parent/(directory.name+'-verified.json')
        if path.exists():
            if json.loads(path.read_text())!=result:
                raise ValueError('Fidelity verification receipt differs')
        else:
            with path.open('x') as stream:
                json.dump(result,stream,indent=2)
                stream.write('\n')


if __name__=='__main__':
    main()
