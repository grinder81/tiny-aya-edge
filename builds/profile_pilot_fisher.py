"""Profile one precisely defined per-example Fisher proxy without fitting weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

POOL_SHA='c17a79fede7ff993f247d4a23cce533ea796887b0ce416c227407bc471997fd3'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','source-gguf','pool','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.pool)!=POOL_SHA:
        raise ValueError('Frozen profiling pool differs')
    pool=json.loads(a.pool.read_text())
    example=pool['pools']['g1_fisher_profile'][0]
    if example['source']!='TED2020:talk:127':
        raise ValueError('Wrong profiling source')
    import torch
    import transformers
    import gguf
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if torch.__version__!='2.13.0+cu130' or transformers.__version__!='5.20.1':
        raise ValueError('Unexpected differentiable environment')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    ids=example['model_ids']['eng_Latn']
    if tokenizer(example['text']['eng_Latn'],add_special_tokens=True)['input_ids'][:65]!=ids:
        raise ValueError('Profile tokenizer mismatch')
    report=dict(scope='G1 one-example sequence-level empirical Fisher proxy resource profile; no training or language ranking',
        implementation_sha256=sha(__file__),pool_sha256=POOL_SHA,source=example['source'],language='eng_Latn',
        tokens=64,model_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        versions=dict(torch=torch.__version__,transformers=transformers.__version__),dtype='float32',attention='eager',tf32=False,
        parameter='model.layers.17.mlp.down_proj.weight',
        objective='Mean next-token NLL over64 content tokens; square the per-example gradient before any across-example average',
        checkpoint_files={f.name:sha(f) for f in sorted(a.model.glob('*.safetensors'))},
        config_sha256=sha(a.model/'config.json'),source_gguf_sha256=sha(a.source_gguf))
    a.output.mkdir(parents=True)
    started=time.monotonic()
    try:
        before=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,dtype=torch.float32,attn_implementation='eager')
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        parameter=dict(model.named_parameters())[report['parameter']]
        reader=gguf.GGUFReader(str(a.source_gguf))
        native=next(t for t in reader.tensors if t.name=='blk.17.ffn_down.weight')
        values=gguf.dequantize(native.data,native.tensor_type).reshape(tuple(parameter.shape))
        if not np.array_equal(values,parameter.detach().numpy()):
            raise ValueError('Whole selected HF/GGUF weight correspondence failed')
        report['whole_selected_weight_exact']=True
        del values,native,reader
        parameter.requires_grad_(True)
        model.to('cuda')
        torch.cuda.synchronize()
        report['load_and_weight_gate_seconds']=time.monotonic()-before
        torch.cuda.reset_peak_memory_stats()
        report['loaded_torch_allocated_bytes']=torch.cuda.memory_allocated()
        tokens=torch.tensor([ids],device='cuda',dtype=torch.long)
        before=time.monotonic()
        logits=model(input_ids=tokens,use_cache=False).logits[:,:-1,:]
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens[:,1:].reshape(-1),reduction='mean')
        torch.cuda.synchronize()
        report['forward_seconds']=time.monotonic()-before
        before=time.monotonic()
        loss.backward()
        torch.cuda.synchronize()
        report['backward_seconds']=time.monotonic()-before
        if not torch.isfinite(parameter.grad).all():
            raise ValueError('Nonfinite per-example gradient')
        report.update(mean_next_token_nll=float(loss.detach().cpu()),
            selected_parameters=parameter.numel(),gradient_l2=float(torch.linalg.vector_norm(parameter.grad).detach().cpu()),
            squared_per_example_gradient_sum=float(parameter.grad.double().square().sum().detach().cpu()),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),status='complete',elapsed_seconds=time.monotonic()-started)
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-started)
        (a.output/'failure.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        raise


if __name__=='__main__':
    main()
