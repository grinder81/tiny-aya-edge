"""Bounded existing-model KL resource profile, with exact T1 identity gates."""
import argparse
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

LOCK_SHA='6dfdd304dcba7bd772c37a7634e74c0c16d1e4b70728be6dcb0e218f3847aa19'
REPORT_SHA='c1ffee6b6030b1d5beb6c6afe3d855c1e1703df3e2fd6329270b06d55a1941a1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def log_softmax(values):
    values=np.asarray(values,dtype=np.float64)
    shifted=values-values.max(axis=-1,keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))


def chunked_kl(reference,candidate,chunk=8):
    if reference.shape!=candidate.shape or reference.ndim!=2 or not reference.shape[0] or chunk<1:
        raise ValueError('Invalid logit shape or chunk')
    values=[]
    agreements=0
    for start in range(0,len(reference),chunk):
        a=np.asarray(reference[start:start+chunk],dtype=np.float64)
        b=np.asarray(candidate[start:start+chunk],dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite logits')
        pa,pb=log_softmax(a),log_softmax(b)
        part=(np.exp(pa)*(pa-pb)).sum(axis=-1)
        if np.min(part)<-1e-10:
            raise ValueError('Negative KL beyond roundoff')
        values.extend(np.maximum(part,0).tolist())
        agreements+=int(np.sum(np.argmax(a,axis=-1)==np.argmax(b,axis=-1)))
    return dict(mean_kl_nats=math.fsum(values)/len(values),tokens=len(values),
        top1_agreement=agreements/len(values),token_kl=values)


def gpu_boundary():
    return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('lock','t1-report','bf16-runtime','q4-runtime','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.lock)!=LOCK_SHA:
        raise ValueError('Request lock identity failed')
    payload=gzip.decompress(a.t1_report.read_bytes())
    if hashlib.sha256(payload).hexdigest()!=REPORT_SHA:
        raise ValueError('T1 report identity failed')
    t1=json.loads(payload)
    rows=json.loads(a.lock.read_text())['rows']
    selected=[r for r in rows if r['index']==0]
    if [r['language'] for r in selected]!=['eng_Latn','ben_Beng','hin_Deva']:
        raise ValueError('Unexpected profile selection')
    from llama_cpp.server.model import LlamaProxy
    from llama_cpp.server.settings import ModelSettings
    import llama_cpp.llama as wrapper
    import llama_cpp._internals as internals
    import llama_cpp.llama_cpp as native
    import llama_cpp.server.model as server_model
    import llama_cpp.server.settings as server_settings
    runtime_paths=[wrapper.__file__,internals.__file__,native._lib._name,server_model.__file__,server_settings.__file__]
    a.output.mkdir(parents=True)
    report=dict(implementation_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).resolve().parents[1]/'docs/article2-resource-profile.md'),
        lock_sha256=LOCK_SHA,t1_report_sha256=REPORT_SHA,records=[],runtimes={},
        projected_900_raw_float32_bytes=sum(len(r['model_context_ids'])-1 for r in rows)*262144*4,
        projected_900_tokens=sum(len(r['model_context_ids'])-1 for r in rows),
        hardware_boundary_before=gpu_boundary(),scope='G1 resource profile; three index-0 development prompts, no holdout or C8 rankings')
    started=time.monotonic()
    try:
        for arm,manifest in [('bf16-gguf',a.bf16_runtime),('local-q4',a.q4_runtime)]:
            runtime=json.loads(manifest.read_text())
            for file,digest in runtime['files'].items():
                if sha(file)!=digest:
                    raise ValueError('Runtime file changed: '+file)
            frozen={str(Path(f).resolve()):h for f,h in runtime['files'].items()}
            imported={str(Path(f).resolve()):sha(f) for f in runtime_paths}
            if any(frozen.get(f)!=h for f,h in imported.items()):
                raise ValueError('Actually imported runtime differs')
            settings={}
            command=runtime['server_command']
            for key,value in zip(command[3::2],command[4::2],strict=True):
                key=key.removeprefix('--')
                if key in ModelSettings.model_fields:
                    settings[key]=value
            before=time.monotonic()
            model=LlamaProxy.load_llama_from_model_settings(ModelSettings(**settings))
            report['runtimes'][arm]=dict(manifest_sha256=sha(manifest),imported_files=imported,
                load_seconds=time.monotonic()-before,settings=ModelSettings(**settings).model_dump())
            refs={(r['language'],r['source'],str(r['question'])):r for r in t1['fresh_rows'][arm]}
            for row in selected:
                if time.monotonic()-started>1800:
                    raise TimeoutError('G1 profile time cap')
                ids=row['model_context_ids']
                if model.tokenize(row['prompt'].encode(),add_bos=True,special=True)!=ids:
                    raise ValueError('Profile token mismatch')
                model.reset()
                before=time.monotonic()
                model.eval(ids)
                elapsed=time.monotonic()-before
                lp=log_softmax(model._scores[len(ids)-1:len(ids)])[0]
                scores=[float(lp[full[-1]]) for full in row['model_token_ids']]
                expected=refs[row['language'],row['source'],str(row['question'])]['scores']
                error=max(abs(x-y) for x,y in zip(scores,expected,strict=True))
                if error>.0002:
                    raise ValueError('Existing T1 score bridge failed: '+str(error))
                entry=dict(arm=arm,language=row['language'],source=row['source'],question=row['question'],
                    tokens=len(ids)-1,forward_seconds=elapsed,t1_letter_max_error_nats=error,
                    gpu_boundary=gpu_boundary(),cpu_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                path=a.output/(row['language']+'-bf16-logits.npy')
                before=time.monotonic()
                if arm=='bf16-gguf':
                    with path.open('xb') as stream:
                        np.save(stream,model._scores[:len(ids)-1],allow_pickle=False)
                    entry.update(cache_bytes=path.stat().st_size,cache_write_seconds=time.monotonic()-before,cache_sha256=sha(path))
                else:
                    reference=np.load(path,mmap_mode='r',allow_pickle=False)
                    entry.update(chunked_kl(reference,model._scores[:len(ids)-1]))
                    entry['cache_read_and_kl_seconds']=time.monotonic()-before
                    del reference
                report['records'].append(entry)
                (a.output/'partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(arm,row['language'],'forward',elapsed,'bridge',error,flush=True)
            model.close()
            del model
            gc.collect()
        report.update(status='complete',elapsed_seconds=time.monotonic()-started,hardware_boundary_after=gpu_boundary())
        with (a.output/'report.json').open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
            stream.write('\n')
    except BaseException as error:
        report.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-started)
        (a.output/'failure.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        raise


if __name__=='__main__':
    main()
