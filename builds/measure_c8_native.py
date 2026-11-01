"""Measure frozen C8 native KL probes without task labels or holdout access."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from builds.profile_pilot_kl import chunked_kl,sha
from builds.prepare_c8_plan import POOL_SHA,SPEC_BLOB


def read(path):
    return json.loads(Path(path).read_text())


def save(path,data):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('reference','diagnostics','pair'))
    for key in ('plan','pool','runtime','directory'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3),action='append',required=True)
    p.add_argument('--nomination',type=Path)
    p.add_argument('--pair-export',type=Path)
    p.add_argument('--reference',type=Path)
    a=p.parse_args()
    if len(a.repeat)!=len(set(a.repeat)):
        raise ValueError('Repeated repetition')
    plan,pool=read(a.plan),read(a.pool)
    if plan['private_pool_sha256']!=POOL_SHA or sha(a.pool)!=POOL_SHA or plan['specification_git_blob']!=SPEC_BLOB:
        raise ValueError('Frozen C8 plan or pool differs')
    if 3 in a.repeat:
        if not a.nomination or read(a.nomination)['plan_sha256']!=sha(a.plan):
            raise ValueError('Freeze discovery nomination, including none, before repetition3')
    if a.mode!='reference' and not a.reference:
        raise ValueError('Reference cache required')
    runtime=read(a.runtime)
    for file,digest in runtime['files'].items():
        if sha(file)!=digest:
            raise ValueError('Reference runtime file changed: '+file)
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
        raise ValueError('Actually imported C8 native runtime differs')
    settings={}
    command=runtime['server_command']
    for key,value in zip(command[3::2],command[4::2],strict=True):
        key=key.removeprefix('--')
        if key in ModelSettings.model_fields:
            settings[key]=value
    if runtime['files'][settings['model']]!=plan['source_sha256']:
        raise ValueError('Wrong BF16 native reference')
    examples=[]
    for repeat in a.repeat:
        for index,row in enumerate(pool['pools'][f'rep{repeat}_probe']):
            for language in ('eng_Latn','ben_Beng','hin_Deva'):
                examples.append(dict(repeat=repeat,index=index,source=row['source'],language=language,
                    ids=row['model_ids'][language],text=row['text'][language]))
    if a.mode=='reference':
        cases=[dict(label='bf16',path=settings['model'],sha256=plan['source_sha256'])]
    elif a.mode=='diagnostics':
        cases=[]
        for entry in plan['diagnostics']:
            manifest=read(Path(entry['evidence_path'])/'manifest.json')
            if manifest['status']!='complete' or manifest['source_sha256']!=plan['source_sha256'] or manifest['tensor']!=entry['tensor'] or manifest['native_type']!=entry['native_type']:
                raise ValueError('Diagnostic manifest mismatch')
            cases.append(dict(label=entry['tensor']+'::'+entry['native_type'],path=entry['output_path'],
                sha256=manifest['output_sha256'],manifest_sha256=sha(Path(entry['evidence_path'])/'manifest.json')))
    else:
        if a.repeat!=[3] or not a.pair_export or not a.nomination:
            raise ValueError('Complete-map confirmation uses repetition3only')
        exported=read(a.pair_export)
        nominee=read(a.nomination)
        if nominee['pair'] is None or exported['nomination_sha256']!=sha(a.nomination) or exported['plan_sha256']!=sha(a.plan):
            raise ValueError('Wrong frozen nominee/export')
        cases=exported['cases']
        if exported['pair']!=nominee['pair'] or len(cases)!=2 or {c['label'] for c in cases}!={'mapA','mapB'} or any(c['bytes']!=nominee['pair']['map_a_bytes'] or c['bytes']>plan['operating_point']['bytes'] for c in cases):
            raise ValueError('Confirmation models violate nominated pair or equal-byte budget')
    a.directory.mkdir(parents=True,exist_ok=True)
    marker=a.directory/'identity.json'
    identity=dict(schema='c8-native-measurement-v1',mode=a.mode,repetitions=a.repeat,
        plan_sha256=sha(a.plan),pool_sha256=POOL_SHA,runtime_sha256=sha(a.runtime),
        implementation_sha256=sha(__file__),math_sha256=sha(Path(__file__).with_name('profile_pilot_kl.py')),
        imported_runtime=imported,nomination_sha256=sha(a.nomination) if a.nomination else None,
        cases=cases)
    if a.mode!='reference':
        reference_report=read(a.reference/'report.json')
        reference_identity=reference_report['identity']
        if reference_identity['mode']!='reference' or reference_identity['plan_sha256']!=sha(a.plan) or reference_identity['repetitions']!=a.repeat:
            raise ValueError('Wrong frozen native reference report')
        if len(reference_identity['cases'])!=1 or reference_identity['cases'][0]['sha256']!=plan['source_sha256']:
            raise ValueError('Native reference is not the BF16 control')
        identity['reference_report_sha256']=sha(a.reference/'report.json')
    if marker.exists():
        if read(marker)!=identity:
            raise ValueError('Native C8 checkpoint identity changed')
    else:
        if list(a.directory.iterdir()):
            raise ValueError('Unidentified C8 output files')
        save(marker,identity)
    started=time.monotonic()
    results=[]
    for number,case in enumerate(cases):
        result_file=a.directory/(f'case-{number:02}.json')
        partial=a.directory/(f'case-{number:02}.partial.json')
        if partial.exists():
            raise ValueError('Interrupted C8 evidence; preserve and inspect')
        if sha(case['path'])!=case['sha256']:
            raise ValueError('C8 model artifact changed')
        if result_file.exists():
            saved=read(result_file)
            if saved['identity_sha256']!=sha(marker) or saved['case']!=case:
                raise ValueError('Saved case identity differs')
            results.append(saved)
            continue
        if time.monotonic()-started>5400:
            raise TimeoutError('Native C8 time cap')
        settings['model']=case['path']
        model=LlamaProxy.load_llama_from_model_settings(ModelSettings(**settings))
        report=dict(identity_sha256=sha(marker),case=case,records=[])
        before=time.monotonic()
        try:
            for example in examples:
                ids=example['ids']
                if len(ids)!=65 or ids[0]!=2 or model.tokenize(example['text'].encode(),add_bos=True,special=True)[:65]!=ids:
                    raise ValueError('Frozen C8 native token mismatch')
                model.reset()
                forward=time.monotonic()
                model.eval(ids)
                logits=model._scores[:64]
                if logits.shape!=(64,262144) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
                    raise ValueError('Invalid native C8 logits')
                record={k:example[k] for k in ('repeat','index','source','language')}
                record.update(tokens=64,forward_seconds=time.monotonic()-forward)
                label=f"r{example['repeat']}-{example['index']}-{example['language']}"
                if a.mode=='reference':
                    cache=a.directory/(label+'.npy')
                    if cache.exists():
                        raise FileExistsError('Interrupted/unidentified reference cache')
                    with cache.open('xb') as stream:
                        np.save(stream,logits,allow_pickle=False)
                    record.update(cache_sha256=sha(cache),cache_bytes=cache.stat().st_size)
                else:
                    cache=a.reference/(label+'.npy')
                    metadata=reference_report['cases'][0]
                    expected=next(r for r in metadata['records'] if all(r[k]==record[k] for k in ('repeat','index','source','language')))
                    if sha(cache)!=expected['cache_sha256']:
                        raise ValueError('C8 reference cache changed')
                    reference=np.load(cache,mmap_mode='r',allow_pickle=False)
                    record.update(chunked_kl(reference,logits))
                    del reference
                report['records'].append(record)
                partial.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            report['seconds']=time.monotonic()-before
            save(result_file,report)
            partial.unlink()
            results.append(report)
            print(json.dumps(dict(case=case['label'],examples=len(examples),seconds=report['seconds'])),flush=True)
        finally:
            model.close()
            del model
            gc.collect()
    final=dict(identity=identity,cases=results,seconds_this_invocation=time.monotonic()-started)
    path=a.directory/'report.json'
    if not path.exists():
        save(path,final)
    print(json.dumps(dict(complete_cases=len(results),seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    main()
