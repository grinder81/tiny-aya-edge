"""Build a read-only architecture atlas from committed inspection evidence.

No model loading, GPU access, WSL launch, tokenization, or outcome collection.
Run from the repository root; --check verifies generated files without writing.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'docs/knowledge-bank'
PLAN_SHA = 'e7ccdb50be254ee8bdde307f73c58bc9a9c9d68882a40aa453b9b516ec8be64f'
ROLES = {
    'token_embd': ('shared embedding and output', 'model.embed_tokens.weight'),
    'output_norm': ('final LayerNorm scale', 'model.norm.weight'),
    'attn_norm': ('shared branch input LayerNorm scale', 'input_layernorm.weight'),
    'attn_q': ('attention query projection', 'self_attn.q_proj.weight'),
    'attn_k': ('attention key projection', 'self_attn.k_proj.weight'),
    'attn_v': ('attention value projection', 'self_attn.v_proj.weight'),
    'attn_output': ('attention output projection', 'self_attn.o_proj.weight'),
    'ffn_gate': ('MLP gate projection before SiLU', 'mlp.gate_proj.weight'),
    'ffn_up': ('MLP feature expansion', 'mlp.up_proj.weight'),
    'ffn_down': ('MLP projection back to residual width', 'mlp.down_proj.weight'),
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build():
    inspection = json.loads((ROOT / 'reports/stage1-downloaded/inspection.json').read_text())
    inventory = json.loads((ROOT / 'reports/control-tensor-inventory.json').read_text())
    raw_plan = (BANK / 'sources/c8-plan.json').read_bytes()
    if hashlib.sha256(raw_plan).hexdigest() != PLAN_SHA:
        raise ValueError('Frozen companion plan snapshot changed')
    plan = json.loads(raw_plan)
    config = inspection['configuration']
    original = inspection['original']['tensor_inventory']
    if (config['num_hidden_layers'], config['hidden_size'], config['intermediate_size'],
        config['vocab_size'], config['num_attention_heads'], config['num_key_value_heads'],
        config['head_dim']) != (36, 2048, 11008, 262144, 16, 4, 128):
        raise ValueError('Architecture changed; update the knowledge bank explicitly')
    if not config['use_parallel_block'] or config['use_qk_norm'] or config['hidden_act'] != 'silu':
        raise ValueError('Documented branch structure no longer matches inspection')
    rows = []
    seen = set()
    groups = {}
    for tensor in inventory['tensors']:
        name, group = tensor['name'], tensor['group']
        if name in seen or group not in ROLES:
            raise ValueError('Duplicate or undocumented tensor role')
        seen.add(name)
        match = re.fullmatch(r'blk\.(\d+)\.[^.]+\.weight', name)
        block = int(match[1]) if match else None
        role, suffix = ROLES[group]
        hf_name = f'model.layers.{block}.{suffix}' if block is not None else suffix
        if hf_name not in original:
            raise ValueError('HF name mapping absent: ' + hf_name)
        shape = original[hf_name]['shape']
        expected = list(reversed(shape)) if len(shape) == 2 else shape
        for label in ('bf16', 'local_q4'):
            stored = tensor[label]
            if expected != stored['shape_gguf'] or math.prod(shape) != stored['parameters']:
                raise ValueError('HF/GGUF dimensions disagree: ' + name)
        operating = plan['base_inventory'][name]
        if operating['shape_gguf'] != expected:
            raise ValueError('Operating model shape mismatch: ' + name)
        selected = name in plan['targets']
        if selected and plan['targets'][name] != hf_name:
            raise ValueError('Frozen target name mismatch')
        attention = config['layer_types'][block] if block is not None else 'outside decoder blocks'
        row = dict(gguf_name=name, hf_name=hf_name, role=role, group=group,
                   block_index=block, human_layer=block + 1 if block is not None else None,
                   attention_type=attention,
                   hf_shape=shape, gguf_shape=expected, parameters=math.prod(shape),
                   bf16_type=tensor['bf16']['type'], bf16_payload_bytes=tensor['bf16']['payload_bytes'],
                   q4_type=tensor['local_q4']['type'], q4_payload_bytes=tensor['local_q4']['payload_bytes'],
                   operating_q3_type=operating['type'], operating_q3_payload_bytes=operating['payload_bytes'],
                   pilot_selected=selected)
        rows.append(row)
        total = groups.setdefault(group, dict(tensors=0, parameters=0, bf16_payload_bytes=0,
                                             q4_payload_bytes=0, operating_q3_payload_bytes=0))
        total['tensors'] += 1
        for field in ('parameters', 'bf16_payload_bytes', 'q4_payload_bytes', 'operating_q3_payload_bytes'):
            total[field] += row[field]
    if len(rows) != 290 or len(seen) != len(original) or len(plan['base_inventory']) != len(rows):
        raise ValueError('Tensor coverage changed')
    if sum(row['parameters'] for row in rows) != 3349227520:
        raise ValueError('Stored parameter count mismatch')
    if sum(row['pilot_selected'] for row in rows) != 8:
        raise ValueError('Pilot subset changed')
    for group, total in groups.items():
        for field in ('tensors', 'parameters', 'bf16_payload_bytes'):
            if total[field] != inventory['groups'][group][field]:
                raise ValueError('Original group aggregate disagrees')
        if total['q4_payload_bytes'] != inventory['groups'][group]['q4_payload_bytes']:
            raise ValueError('Q4 group aggregate disagrees')
    for label, prefix in (('bf16', 'bf16'), ('local_q4', 'q4')):
        artifact = inventory['artifacts'][label]
        if sum(row[prefix + '_payload_bytes'] for row in rows) + artifact['non_payload_bytes'] != artifact['file_bytes']:
            raise ValueError('Complete file accounting differs')
    operating_payload = sum(row['operating_q3_payload_bytes'] for row in rows)
    overhead = plan['operating_point']['bytes'] - operating_payload
    if overhead != inventory['artifacts']['bf16']['non_payload_bytes']:
        raise ValueError('Operating model overhead changed')
    by_name = {row['gguf_name']: row for row in rows}
    layers = []
    for block in range(36):
        members = [row for row in rows if row['block_index'] == block]
        if len(members) != 8 or sum(len(row['hf_shape']) == 2 for row in members) != 7:
            raise ValueError('Expected seven matrices and one norm per block')
        kind = config['layer_types'][block]
        if kind != ('full_attention' if (block + 1) % 4 == 0 else 'sliding_attention'):
            raise ValueError('Layer pattern differs')
        layers.append(dict(block_index=block, human_layer=block + 1, attention_type=kind,
                           local_rope=kind == 'sliding_attention',
                           selected=[row['group'] for row in members if row['pilot_selected']]))
    selected_parameters = sum(row['parameters'] for row in rows if row['pilot_selected'])
    atlas = dict(schema='tiny-aya-architecture-atlas-v1',
                 scope='Static committed artifacts; no model execution or sensitivity outcomes',
                 source_canonical_json_sha256=dict(inspection=canonical_sha(inspection), inventory=canonical_sha(inventory)),
                 frozen_plan_sha256=PLAN_SHA, configuration=config, tensor_count=len(rows),
                 parameters=sum(row['parameters'] for row in rows),
                 selected_parameters=selected_parameters,
                 selected_parameter_fraction=selected_parameters / 3349227520,
                 non_payload_bytes=overhead, groups=groups, layers=layers, tensors=rows,
                 equal_byte_pairs=plan['pairs'])
    files = {'data/atlas.json': json.dumps(atlas, indent=2, ensure_ascii=False) + '\n'}
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    files['data/tensor-map.csv'] = stream.getvalue()
    lines = ['# All decoder blocks', '', 'Generated from committed configuration and tensor evidence. Human layer = code block + 1.', '',
             '| Human layer | Code block | Attention | Explicit RoPE | Selected pilot matrices |',
             '| ---: | ---: | --- | --- | --- |']
    for row in layers:
        lines.append(f"| {row['human_layer']} | {row['block_index']} | {'Global causal' if row['attention_type']=='full_attention' else 'Local causal, window 4096'} | {'Yes' if row['local_rope'] else 'No'} | {', '.join(row['selected']) or 'None'} |")
    files['data/layer-map.md'] = '\n'.join(lines) + '\n'
    lines = ['# Stored tensor groups', '', 'Payload bytes exclude metadata and alignment. Group sizes are not importance scores.', '',
             '| Group | Tensors | Parameters | Parameter share | BF16 payload bytes | Q4 payload bytes | Operating Q3 payload bytes |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group, total in groups.items():
        lines.append(f"| {group} | {total['tensors']} | {total['parameters']:,} | {100*total['parameters']/3349227520:.3f}% | {total['bf16_payload_bytes']:,} | {total['q4_payload_bytes']:,} | {total['operating_q3_payload_bytes']:,} |")
    files['data/group-map.md'] = '\n'.join(lines) + '\n'
    lines = ['# Exact eight pilot sites', '', 'Shapes below use the PyTorch output by input convention. Formats are actual stored tensor types in each reference export.', '',
             '| GGUF tensor | Human layer | Shape | Parameters | Local Q4 type | Operating Q3 type |',
             '| --- | ---: | --- | ---: | --- | --- |']
    for name in plan['targets']:
        row = by_name[name]
        lines.append(f"| `{name}` | {row['human_layer'] or 'Shared interface'} | {' x '.join(map(str,row['hf_shape']))} | {row['parameters']:,} | {row['q4_type']} | {row['operating_q3_type']} |")
    lines += ['', f'These eight stored matrices contain {selected_parameters:,} parameters ({100*selected_parameters/3349227520:.3f}% of the model). This fraction is dominated by the embedding and is not representative coverage of behavior.', '',
              '| Prespecified pair | Predicted bytes per complete orientation | Eligible |', '| --- | ---: | --- |']
    for pair in plan['pairs']:
        if pair['map_a_bytes'] != pair['map_b_bytes']:
            raise ValueError('Frozen pair byte equality failed')
        lines.append(f"| `{pair['first']}` / `{pair['second']}` | {pair['map_a_bytes']:,} | {pair['eligible']} |")
    files['data/eight-sites.md'] = '\n'.join(lines) + '\n'
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files = build()
    for name, content in files.items():
        path = BANK / name
        if args.check:
            if not path.exists() or path.read_bytes() != content.encode():
                raise ValueError('Generated atlas is stale: ' + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode())
    print(json.dumps(dict(status='verified' if args.check else 'generated', files=len(files),
                          tensors=290, blocks=36, pilot_sites=8, model_execution=False)))


if __name__ == '__main__':
    main()
