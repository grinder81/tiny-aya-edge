"""Per-example Fisher and activation proxies with native quantization errors."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from builds.native_gguf import NativeGGUF
from builds.prepare_c8_plan import POOL_SHA,SPEC_BLOB
from builds.profile_pilot_kl import sha


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def errors(plan,plan_path,source,library,directory,budget):
    marker=directory/'manifest.json'
    identity=dict(plan_sha256=sha(plan_path),implementation_sha256=sha(__file__),
        native_reader_sha256=sha(Path(__file__).with_name('native_gguf.py')),
        source_sha256=sha(source),library_sha256=sha(library))
    if identity['source_sha256']!=plan['source_sha256'] or identity['library_sha256']!=plan['native_library_sha256']:
        raise ValueError('Native error source/library changed')
    if marker.exists():
        result=read(marker)
        if result['identity']!=identity:
            raise ValueError('Native error cache identity changed')
        for entry in result['entries']:
            if sha(entry['path'])!=entry['sha256']:
                raise ValueError('Quantization-error cache changed')
        return result['entries']
    if directory.exists():
        raise FileExistsError('Incomplete error cache; preserve and inspect')
    directory.mkdir(parents=True)
    original=NativeGGUF(source,library)
    entries=[]
    for number,diagnostic in enumerate(plan['diagnostics']):
        budget.check()
        manifest=read(Path(diagnostic['evidence_path'])/'manifest.json')
        if manifest['status']!='complete' or manifest['tensor']!=diagnostic['tensor'] or manifest['native_type']!=diagnostic['native_type'] or sha(diagnostic['output_path'])!=manifest['output_sha256']:
            raise ValueError('Native error diagnostic differs')
        if manifest.get('budget_plan_sha256')!=sha(plan_path):
            raise ValueError('Native diagnostic export was not budgeted')
        require_passed_operation(plan_path,manifest['budget_operation_id'],'native')
        candidate=NativeGGUF(diagnostic['output_path'],library)
        name=diagnostic['tensor']
        width,height=original.tensors[name]['shape_gguf']
        if candidate.tensors[name]['shape_gguf']!=[width,height]:
            raise ValueError('Quantized error shape changed')
        path=directory/(f'error-{number:02}.npy')
        budget.reserve(height*width*4+2_000_000)
        error=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(height,width))
        columns=np.zeros(width,dtype=np.float64)
        for start in range(0,height,512):
            budget.check(False)
            end=min(start+512,height)
            delta=original.rows(name,start,end)-candidate.rows(name,start,end)
            squared=delta*delta
            error[start:end]=squared
            columns+=squared.sum(axis=0,dtype=np.float64)
        error.flush()
        del error
        entries.append(dict(tensor=name,hf_parameter=diagnostic['hf_parameter'],native_type=diagnostic['native_type'],
            path=str(path),sha256=sha(path),shape=[height,width],column_error_squared_sum=columns.tolist(),
            diagnostic_sha256=manifest['output_sha256']))
        print('cached native errors',name,diagnostic['native_type'],flush=True)
    save(marker,dict(identity=identity,entries=entries))
    budget.check()
    return entries


from builds.c8_budget import BudgetOperation,require_passed_operation
from builds.c8_nomination import validate_nomination


def run(a,budget):
    plan,pool,proof=read(a.plan),read(a.pool),read(a.checkpoint_proof)
    if plan['private_pool_sha256']!=POOL_SHA or sha(a.pool)!=POOL_SHA or plan['specification_git_blob']!=SPEC_BLOB:
        raise ValueError('Frozen proxy plan/pool changed')
    if len(a.repeat)!=len(set(a.repeat)):
        raise ValueError('Duplicate repetition')
    if 3 in a.repeat:
        if not a.nomination:
            raise ValueError('Freeze discovery nomination before repetition3 proxies')
        validate_nomination(a.nomination,a.plan)
    entries=errors(plan,a.plan,a.source,a.library,a.directory/'errors',budget)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if torch.__version__!='2.13.0+cu130' or transformers.__version__!='5.20.1':
        raise ValueError('Differentiable proxy environment changed')
    for file,digest in proof['checkpoint_files'].items():
        if sha(a.model/file)!=digest:
            raise ValueError('Source checkpoint changed')
    if sha(a.model/'config.json')!=proof['config_sha256']:
        raise ValueError('Source model configuration changed')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    identity=dict(schema='c8-proxy-v1',plan_sha256=sha(a.plan),pool_sha256=POOL_SHA,
        implementation_sha256=sha(__file__),error_manifest_sha256=sha(a.directory/'errors/manifest.json'),
        checkpoint_proof_sha256=sha(a.checkpoint_proof),repetitions=a.repeat,
        nomination_sha256=sha(a.nomination) if a.nomination else None,torch=torch.__version__,transformers=transformers.__version__,
        dtype='float32',tf32=False,objective='Per-example mean64-token NLL; gradient squared before averaging',
        budget_operation_id=budget.operation_id)
    run=a.directory/('repeats-'+'-'.join(map(str,a.repeat)))
    if run.exists():
        raise FileExistsError('Proxy run already exists; preserve and inspect')
    run.mkdir()
    save(run/'identity.json',identity)
    started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,dtype=torch.float32,attn_implementation='eager')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError('Expected tied source embedding/output')
    parameters=dict(model.named_parameters())
    original=NativeGGUF(a.source,a.library)
    for name,hf in plan['targets'].items():
        budget.check()
        parameter=parameters[hf]
        width,height=original.tensors[name]['shape_gguf']
        if list(parameter.shape)!=[height,width]:
            raise ValueError('HF/native target shape differs: '+name)
        values=parameter.detach().numpy()
        for start in range(0,height,512):
            end=min(start+512,height)
            if not np.array_equal(values[start:end],original.rows(name,start,end)):
                raise ValueError('Whole selected HF/native matrix differs: '+name)
        parameter.requires_grad_(True)
    activations={}
    handles=[]
    modules=dict(model.named_modules())
    def hook(key):
        def collect(module,args):
            x=args[0][:,:64].detach().float()
            activations[key]=x.square().mean(dim=(0,1)).cpu().numpy().astype(np.float64)
        return collect
    for name,hf in plan['targets'].items():
        if name!='token_embd.weight':
            handles.append(modules[hf.removesuffix('.weight')].register_forward_pre_hook(hook(name)))
    handles.append(model.get_output_embeddings().register_forward_pre_hook(hook('output_head')))
    model.to('cuda')
    torch.cuda.reset_peak_memory_stats()
    mmap={entry['path']:np.load(entry['path'],mmap_mode='r',allow_pickle=False) for entry in entries}
    records=[]
    try:
        for repeat in a.repeat:
            for index,example in enumerate(pool['pools'][f'rep{repeat}_construction']):
                for language in ('eng_Latn','ben_Beng','hin_Deva'):
                    budget.check()
                    if time.monotonic()-started>1800:
                        raise TimeoutError('Frozen proxy time cap')
                    ids=example['model_ids'][language]
                    if len(ids)!=65 or tokenizer(example['text'][language],add_special_tokens=True)['input_ids'][:65]!=ids:
                        raise ValueError('Proxy token identity changed')
                    model.zero_grad(set_to_none=True)
                    activations.clear()
                    before=time.monotonic()
                    tokens=torch.tensor([ids],device='cuda',dtype=torch.long)
                    logits=model(input_ids=tokens,use_cache=False).logits[:,:64,:]
                    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens[:,1:].reshape(-1),reduction='mean')
                    loss.backward()
                    torch.cuda.synchronize()
                    record=dict(repeat=repeat,index=index,source=example['source'],language=language,tokens=64,
                        mean_next_token_nll=float(loss.detach().cpu()),forward_backward_seconds=time.monotonic()-before,scores=[])
                    for entry in entries:
                        budget.check(False)
                        gradient=parameters[entry['hf_parameter']].grad
                        if gradient is None or not bool(torch.isfinite(gradient).all()):
                            raise ValueError('Missing/nonfinite per-example gradient')
                        squared=mmap[entry['path']]
                        weighted=torch.zeros((),device='cuda',dtype=torch.float64)
                        for start in range(0,squared.shape[0],512):
                            budget.check(False)
                            end=min(start+512,squared.shape[0])
                            e=torch.tensor(np.asarray(squared[start:end]),device='cuda',dtype=torch.float64)
                            weighted+=torch.sum(gradient[start:end].double().square()*e)
                        columns=np.asarray(entry['column_error_squared_sum'])
                        if entry['tensor']=='token_embd.weight':
                            lookup=float(np.mean(np.sum(squared[np.asarray(ids[:64])],axis=1,dtype=np.float64)))
                            head=float(np.dot(columns,activations['output_head']))
                            components=dict(lookup=lookup,output_head=head)
                            activation=lookup+head
                        else:
                            activation=float(np.dot(columns,activations[entry['tensor']]))
                            components=dict(linear_output=activation)
                        record['scores'].append(dict(tensor=entry['tensor'],native_type=entry['native_type'],
                            fisher_score=.5*float(weighted.detach().cpu()),activation_score=activation,activation_components=components))
                    records.append(record)
                    budget.check()
                    save(run/(f'r{repeat}-{index}-{language}.json'),record)
                    print(repeat,index,language,'proxy complete',flush=True)
        budget.check()
        save(run/'report.json',dict(identity=identity,whole_selected_matrices_exact=list(plan['targets']),records=records,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
            interpretation='Descriptive sequence-Fisher and activation proxies; no optimizer steps or native-runtime equivalence claim.'))
    finally:
        for handle in handles:
            handle.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','pool','model','source','library','checkpoint-proof','directory'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3),action='append',required=True)
    p.add_argument('--nomination',type=Path)
    a=p.parse_args()
    context=BudgetOperation(a.plan,'proxy','proxies-'+str(a.repeat),(a.directory,))
    with context as budget:
        run(a,budget)


if __name__=='__main__':
    main()
