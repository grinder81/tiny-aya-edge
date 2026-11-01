"""Resume a bounded float32 reference cache or full-prompt T3 KL summaries."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from builds.profile_pilot_kl import LOCK_SHA,chunked_kl,sha


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


class TorchReducer:
    """Float64 math on the same process's GPU, checked against CPU per question."""
    def __init__(self,site):
        sys.path.append(str(site))
        import torch
        self.torch=torch
        if torch.__version__!='2.13.0+cu130':
            raise ValueError('Unexpected optional Torch reducer version')

    def __call__(self,reference,candidate):
        torch=self.torch
        if reference.shape!=candidate.shape or reference.ndim!=2 or not len(reference):
            raise ValueError('Invalid KL shapes')
        values=[]
        agree=0
        with torch.no_grad():
            for start in range(0,len(reference),8):
                a=torch.tensor(np.asarray(reference[start:start+8]),device='cuda',dtype=torch.float64)
                b=torch.tensor(np.asarray(candidate[start:start+8]),device='cuda',dtype=torch.float64)
                if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                    raise ValueError('Nonfinite GPU KL logits')
                pa,pb=torch.log_softmax(a,dim=-1),torch.log_softmax(b,dim=-1)
                part=(pa.exp()*(pa-pb)).sum(dim=-1).cpu().tolist()
                if min(part)<-1e-10:
                    raise ValueError('Negative GPU KL beyond roundoff')
                values.extend(max(x,0.) for x in part)
                agree+=int((a.argmax(dim=-1)==b.argmax(dim=-1)).sum().cpu())
        # Same raw actual-model arrays, independent CPU implementation, every question.
        oracle=chunked_kl(reference[:8],candidate[:8])
        error=max(abs(a-b) for a,b in zip(values[:8],oracle['token_kl'],strict=True))
        if error>1e-10:
            raise ValueError('GPU/CPU float64 KL disagreement: '+str(error))
        return dict(mean_kl_nats=float(np.mean(values)),tokens=len(values),top1_agreement=agree/len(values),
            token_kl=values,first_chunk_cpu_oracle_max_error=error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('reference','compare'))
    for key in ('lock','runtime','directory'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--torch-site',type=Path)
    p.add_argument('--max-new-questions',type=int)
    a=p.parse_args()
    if a.mode=='compare' and not a.reference:
        raise ValueError('Comparison needs complete reference cache')
    if a.max_new_questions is not None and a.max_new_questions<0:
        raise ValueError('Negative question limit')
    if sha(a.lock)!=LOCK_SHA:
        raise ValueError('Frozen request lock differs')
    lock=read(a.lock)
    rows=lock['rows']
    if len(rows)!=900:
        raise ValueError('Full T3 lock required')
    runtime=read(a.runtime)
    for file,digest in runtime['files'].items():
        if sha(file)!=digest:
            raise ValueError('Runtime changed: '+file)
    from llama_cpp.server.model import LlamaProxy
    from llama_cpp.server.settings import ModelSettings
    import llama_cpp.llama as wrapper
    import llama_cpp._internals as internals
    import llama_cpp.llama_cpp as native
    import llama_cpp.server.model as server_model
    import llama_cpp.server.settings as server_settings
    imported={str(Path(f).resolve()):sha(f) for f in (wrapper.__file__,internals.__file__,native._lib._name,server_model.__file__,server_settings.__file__)}
    frozen={str(Path(f).resolve()):h for f,h in runtime['files'].items()}
    if any(frozen.get(f)!=h for f,h in imported.items()):
        raise ValueError('Actually imported native runtime differs')
    model_path=runtime['server_command'][runtime['server_command'].index('--model')+1]
    if a.mode=='reference' and runtime['files'][model_path]!='0e2f666ac8fb50a7f53d14164147beeca35905b76b93eb90e0e3a963eb203b4b':
        raise ValueError('KL reference must be frozen BF16 control')
    identity=dict(schema='t3-fidelity-v1',mode=a.mode,lock_sha256=LOCK_SHA,runtime_sha256=sha(a.runtime),
        implementation_sha256=sha(__file__),math_module_sha256=sha(Path(__file__).with_name('profile_pilot_kl.py')),
        imported_runtime=imported,model_sha256=runtime['files'][model_path],
        backend='torch2.13-float64' if a.torch_site else 'numpy-float64',torch_site=str(a.torch_site) if a.torch_site else None)
    if a.reference:
        reference_manifest=read(a.reference/'manifest.json')
        if reference_manifest['questions']!=900 or reference_manifest['identity']['lock_sha256']!=LOCK_SHA or reference_manifest['identity']['model_sha256']!='0e2f666ac8fb50a7f53d14164147beeca35905b76b93eb90e0e3a963eb203b4b':
            raise ValueError('Incomplete or wrong reference cache')
        identity['reference_manifest_sha256']=sha(a.reference/'manifest.json')
    projected=sum(len(r['model_context_ids'])-1 for r in rows)*262144*4
    if projected>220_000_000_000:
        raise ValueError('Frozen raw reference-cache cap exceeded')
    if not a.directory.exists() and a.mode=='reference' and shutil.disk_usage(a.directory.parent).free<270_000_000_000:
        raise ValueError('Insufficient free space for frozen cache cap and50GB reserve')
    a.directory.mkdir(parents=True,exist_ok=True)
    marker=a.directory/'identity.json'
    if marker.exists():
        if read(marker)!=identity:
            raise ValueError('Fidelity identity changed; cannot resume')
    else:
        if list(a.directory.iterdir()):
            raise ValueError('Unidentified fidelity files')
        write_new(marker,identity)
    names={f"{r['language']}-{r['index']:03}" for r in rows}
    allowed={'identity.json','manifest.json'}|{n+'.json' for n in names}
    if a.mode=='reference':
        allowed|={n+'.npy' for n in names}
    if any(f.name not in allowed for f in a.directory.iterdir()):
        raise ValueError('Unexpected or partial fidelity evidence')
    settings={}
    command=runtime['server_command']
    for key,value in zip(command[3::2],command[4::2],strict=True):
        key=key.removeprefix('--')
        if key in ModelSettings.model_fields:
            settings[key]=value
    reducer=TorchReducer(a.torch_site) if a.torch_site else chunked_kl
    model=None
    count=0
    records=[]
    started=time.monotonic()
    try:
        for row in rows:
            name=f"{row['language']}-{row['index']:03}"
            path=a.directory/(name+'.json')
            expected={k:row[k] for k in ('language','index','source','question','prompt_sha256')}
            expected['tokens']=len(row['model_context_ids'])-1
            if path.exists():
                record=read(path)
                if any(record.get(k)!=v for k,v in expected.items()):
                    raise ValueError('Fidelity sample identity changed')
                if a.mode=='reference' and sha(a.directory/(name+'.npy'))!=record['cache_sha256']:
                    raise ValueError('Saved reference cache changed')
                records.append(record)
                continue
            if a.max_new_questions is not None and count>=a.max_new_questions:
                continue
            cache=a.directory/(name+'.npy')
            if a.mode=='reference' and cache.exists():
                raise ValueError('Unidentified interrupted raw cache; preserve and inspect')
            if model is None:
                model=LlamaProxy.load_llama_from_model_settings(ModelSettings(**settings))
            ids=row['model_context_ids']
            if model.tokenize(row['prompt'].encode(),add_bos=True,special=True)!=ids:
                raise ValueError('Native fidelity token mismatch')
            model.reset()
            before=time.monotonic()
            model.eval(ids)
            record=expected|dict(forward_seconds=time.monotonic()-before)
            scores=model._scores[:len(ids)-1]
            if scores.shape!=(len(ids)-1,262144) or not np.isfinite(scores).all():
                raise ValueError('Invalid direct logits')
            before=time.monotonic()
            if a.mode=='reference':
                with cache.open('xb') as stream:
                    np.save(stream,scores,allow_pickle=False)
                record.update(cache_sha256=sha(cache),cache_bytes=cache.stat().st_size,cache_seconds=time.monotonic()-before)
            else:
                meta=read(a.reference/(name+'.json'))
                cache=a.reference/(name+'.npy')
                if any(meta.get(k)!=v for k,v in expected.items()) or sha(cache)!=meta['cache_sha256']:
                    raise ValueError('Reference sample/cache changed')
                reference=np.load(cache,mmap_mode='r',allow_pickle=False)
                record.update(reducer(reference,scores))
                record['cache_and_reduction_seconds']=time.monotonic()-before
                del reference
            write_new(path,record)
            records.append(record)
            count+=1
            if count%10==0:
                print(json.dumps(dict(new_questions=count,completed=len(records),seconds=time.monotonic()-started)),flush=True)
        if len(records)==900:
            manifest=dict(identity=identity,questions=900,tokens=sum(r['tokens'] for r in records),
                per_question_files={f"{r['language']}-{r['index']:03}.json":sha(a.directory/f"{r['language']}-{r['index']:03}.json") for r in records})
            if a.mode=='compare':
                manifest['languages']={lang:dict(mean_kl_nats=float(np.mean([r['mean_kl_nats'] for r in records if r['language']==lang])),
                    p50_question_kl=float(np.quantile([r['mean_kl_nats'] for r in records if r['language']==lang],.5)),
                    p95_question_kl=float(np.quantile([r['mean_kl_nats'] for r in records if r['language']==lang],.95)),
                    mean_question_top1_agreement=float(np.mean([r['top1_agreement'] for r in records if r['language']==lang]))) for lang in ('eng_Latn','ben_Beng','hin_Deva')}
            if (a.directory/'manifest.json').exists():
                if read(a.directory/'manifest.json')!=manifest:
                    raise ValueError('Final fidelity manifest changed')
            else:
                write_new(a.directory/'manifest.json',manifest)
        print(json.dumps(dict(completed=len(records),new_questions=count,seconds=time.monotonic()-started)),flush=True)
    finally:
        if model is not None:
            model.close()
        gc.collect()


if __name__=='__main__':
    main()
