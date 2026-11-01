"""Derive C8 formats and feasible swaps mechanically from the final English point."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import gguf

from builds.profile_pilot_bytes import SOURCE_SHA,sha

SPEC_BLOB='cd43458f25672a0522958a4d1de0e1a88c307dc7'
POOL_SHA='c17a79fede7ff993f247d4a23cce533ea796887b0ce416c227407bc471997fd3'
TARGETS={
    'token_embd.weight':'model.embed_tokens.weight',
    'blk.0.attn_v.weight':'model.layers.0.self_attn.v_proj.weight',
    'blk.3.attn_v.weight':'model.layers.3.self_attn.v_proj.weight',
    'blk.0.ffn_down.weight':'model.layers.0.mlp.down_proj.weight',
    'blk.17.ffn_down.weight':'model.layers.17.mlp.down_proj.weight',
    'blk.35.ffn_down.weight':'model.layers.35.mlp.down_proj.weight',
    'blk.17.ffn_gate.weight':'model.layers.17.mlp.gate_proj.weight',
    'blk.17.ffn_up.weight':'model.layers.17.mlp.up_proj.weight',
}
PAIRS=[('blk.0.attn_v.weight','blk.3.attn_v.weight'),
    ('blk.0.ffn_down.weight','blk.17.ffn_down.weight'),
    ('blk.17.ffn_down.weight','blk.35.ffn_down.weight'),
    ('blk.17.ffn_gate.weight','blk.17.ffn_up.weight')]


def choose_formats(kind,block_bytes):
    return ('Q3_K','Q4_K') if kind=='Q4_K' or block_bytes>144 else ('Q2_K','Q3_K')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('curve','operating-model','source','pool','specification','library','artifacts','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    curve=json.loads(a.curve.read_text())
    point=curve['final_operating_point']
    if point is None:
        raise ValueError('Complete T3 operating-point decision required')
    expected=curve['evidence'][point['arm']]['build']['output_sha256'] if point['arm']!='local-q4' else '5823bd1e52be1027a7ba6f31b43edfd40ae1766020cddec967f0d76618948290'
    if sha(a.operating_model)!=expected or a.operating_model.stat().st_size!=point['bytes'] or point['bytes']>2_200_000_000:
        raise ValueError('Operating-point artifact or budget differs')
    content=a.specification.read_bytes().replace(b'\r\n',b'\n')
    blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if blob!=SPEC_BLOB or sha(a.pool)!=POOL_SHA or sha(a.source)!=SOURCE_SHA:
        raise ValueError('Frozen specification/pool/source mismatch')
    model=gguf.GGUFReader(str(a.operating_model))
    tensors={t.name:t for t in model.tensors}
    base={t.name:dict(type=t.tensor_type.name,payload_bytes=int(t.n_bytes),shape_gguf=t.shape.tolist()) for t in model.tensors}
    gate=tensors['blk.17.ffn_gate.weight'].tensor_type
    block_size,block_bytes=gguf.GGML_QUANT_SIZES[gate]
    if block_size!=256:
        raise ValueError('Unspecified operating-point gate format')
    low,high=choose_formats(gate.name,block_bytes)
    native=ctypes.CDLL(str(a.library))
    native.ggml_row_size.argtypes=[ctypes.c_int,ctypes.c_int64]
    native.ggml_row_size.restype=ctypes.c_size_t
    def payload(name,kind):
        tensor=tensors[name]
        width=int(tensor.shape[0])
        size=int(native.ggml_row_size(int(gguf.GGMLQuantizationType[kind]),width))*(int(tensor.n_elements)//width)
        if size%32:
            raise ValueError('Unsupported alignment model')
        return size
    diagnostics=[]
    for name,hf in TARGETS.items():
        for kind in (low,high):
            label=name.replace('.','-')+'-'+kind.lower()
            diagnostics.append(dict(tensor=name,hf_parameter=hf,native_type=kind,
                target_payload_bytes=payload(name,kind),
                output_path=str(a.artifacts/(label+'.gguf')),evidence_path=str(a.artifacts/(label+'-evidence'))))
    pairs=[]
    for first,second in PAIRS:
        if base[first]['shape_gguf']!=base[second]['shape_gguf']:
            raise ValueError('Prespecified pair shapes differ')
        rest=point['bytes']-base[first]['payload_bytes']-base[second]['payload_bytes']
        left=rest+payload(first,high)+payload(second,low)
        right=rest+payload(first,low)+payload(second,high)
        pairs.append(dict(first=first,second=second,map_a_bytes=left,map_b_bytes=right,
            eligible=left==right and left<=point['bytes'] and left<=2_200_000_000))
    result=dict(schema='c8-bounded-plan-v1',specification_git_blob=blob,private_pool_sha256=POOL_SHA,
        curve_report_sha256=sha(a.curve),operating_point=point,operating_model_sha256=expected,
        source_sha256=SOURCE_SHA,implementation_sha256=sha(__file__),native_library_sha256=sha(a.library),
        formats=dict(lower=low,higher=high,operating_gate_type=gate.name),base_inventory=base,
        targets=TARGETS,diagnostics=diagnostics,pairs=pairs,
        probe_sources={str(rep):[row['source'] for row in json.loads(a.pool.read_text())['pools'][f'rep{rep}_probe']] for rep in (1,2,3)},
        construction_sources={str(rep):[row['source'] for row in json.loads(a.pool.read_text())['pools'][f'rep{rep}_construction']] for rep in (1,2,3)},
        cap=dict(native_minutes=90,proxy_minutes=30,local_artifact_bytes=400_000_000_000),
        note='Mechanical pre-outcome derivation. No C8 language score used for formats, sites, budgets or pair eligibility.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(formats=result['formats'],pairs=pairs,plan_sha256=sha(a.output))),flush=True)


if __name__=='__main__':
    main()
