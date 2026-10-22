"""Independent direct-logit oracle and unchanged upstream HTTP validation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

LOCK = '75486c22c0c25f7c6c9d3da6abfaaa0165babf4a5de24970147ab9d1488d9844'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected(rows):
    keys = {i for i, r in enumerate(rows) if r['index'] == 0}
    for language in ('eng_Latn', 'ben_Beng', 'hin_Deva'):
        for variant in ('plain-letter', 'chat-letter', 'plain-option-text'):
            subset = [i for i, r in enumerate(rows) if r['language'] == language and r['variant'] == variant]
            keys.add(max(subset, key=lambda i: max(map(len, rows[i]['model_token_ids']))))
            keys.add(max(subset, key=lambda i: max(len(t)-len(rows[i]['model_context_ids']) for t in rows[i]['model_token_ids'])))
    return sorted(keys)


def save(path, report):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def prediction(scores):
    order = sorted(range(4), key=lambda i: -scores[i])
    return dict(prediction=order[0], margin=scores[order[0]]-scores[order[1]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('direct', 'client'))
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--oracle', type=Path)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--a2', type=Path)
    p.add_argument('--base-url', default='http://127.0.0.1:8098/v1/completions')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    assert sha(a.lock) == LOCK
    rows = json.loads(a.lock.read_text())['rows']
    runtime = json.loads(a.runtime.read_text())
    assert all(sha(k) == v for k, v in runtime['files'].items())
    report = dict(lock_sha256=LOCK, runtime_sha256=sha(a.runtime), script_sha256=sha(__file__),
                  selection=selected(rows), records=[], failures=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for language in ('eng_Latn', 'ben_Beng', 'hin_Deva'):
        for variant in ('plain-letter', 'chat-letter', 'plain-option-text'):
            subset = [rows[i] for i in report['selection'] if rows[i]['language']==language and rows[i]['variant']==variant]
            assert {r['shift'] for r in subset} == set(range(4))
            if variant == 'plain-option-text':
                assert any(len(t)-len(r['model_context_ids']) > 1 for r in subset for t in r['model_token_ids'])
                if language != 'eng_Latn':
                    assert any(not c.isascii() for r in subset for c in r['candidates'])
    if a.mode == 'direct':
        import numpy as np
        from llama_cpp.server.model import LlamaProxy
        from llama_cpp.server.settings import ModelSettings
        command = runtime['server_command']
        settings = {}
        for key, value in zip(command[3::2], command[4::2], strict=True):
            key = key.removeprefix('--')
            if key in ModelSettings.model_fields:
                settings[key] = value
            else:
                assert key in ('host', 'port')
        settings = ModelSettings(**settings)
        report['server_model_settings'] = settings.model_dump()
        model = LlamaProxy.load_llama_from_model_settings(settings)
        report['bos_id'] = model.token_bos()
        report['eos_id'] = model.token_eos()
        for ri in report['selection']:
            row = rows[ri]
            for ci, ids in enumerate(row['model_token_ids']):
                assert model.tokenize((row['prompt']+row['candidates'][ci]).encode(), add_bos=True, special=True) == ids
                assert ids[0] == model.token_bos()
                model.reset()
                model.eval(ids)
                scores = model._scores[:len(ids)].astype(np.float64)
                def lp(logits, token):
                    maximum = float(logits.max())
                    return float(logits[token] - maximum - np.log(np.exp(logits-maximum).sum()))
                probs = [lp(scores[i-1], ids[i]) for i in range(1, len(ids))]
                generated = int(np.argmax(scores[len(ids)-1]))
                all_ids = ids[1:] + [generated]
                strings = [model.detokenize([t], prev_tokens=all_ids[:i]).decode('utf-8', errors='ignore') for i, t in enumerate(all_ids)]
                rec = dict(row=ri, candidate=ci, model_ids=ids, supplied_logprobs=probs,
                           generated_id=generated, generated_logprob=lp(scores[-1], generated), echo_strings=strings)
                if ri == report['selection'][0]:
                    model.reset()
                    model.eval(ids)
                    repeated = model._scores[:len(ids)].astype(np.float64)
                    rec['repeat_max_logprob_error'] = max(abs(lp(repeated[i-1], ids[i])-probs[i-1]) for i in range(1,len(ids)))
                    if rec['repeat_max_logprob_error'] > 0.0002:
                        report['failures'].append(dict(row=ri, candidate=ci, error='direct repeat drift'))
                assert all(math.isfinite(x) for x in probs+[rec['generated_logprob']])
                report['records'].append(rec)
                save(a.output, report)
                print(f'direct row {ri} candidate {ci}', flush=True)
    else:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import requests
        import inspect
        import lm_eval.models.openai_completions as api
        import lm_eval.models.api_models as base
        from lm_eval.api.model import TemplateLM
        import importlib.metadata
        from lm_eval.models.openai_completions import LocalCompletionsAPI
        oracle = json.loads(a.oracle.read_text())
        assert oracle['selection'] == report['selection'] and oracle['runtime_sha256'] == report['runtime_sha256']
        assert oracle['lock_sha256'] == LOCK and oracle['script_sha256'] == sha(__file__) and not oracle['failures']
        assert [(r['row'],r['candidate']) for r in oracle['records']] == [(i,c) for i in report['selection'] for c in range(4)]
        model = LocalCompletionsAPI(model='a3-validation', base_url=a.base_url,
            tokenizer=str(a.tokenizer), tokenizer_backend='huggingface', tokenized_requests=False,
            add_bos_token=False, max_length=8192, batch_size=1)
        report['source_files'] = {f:sha(f) for f in (api.__file__, base.__file__, inspect.getfile(TemplateLM), inspect.getfile(type(model.tokenizer)))}
        report['tokenizer_files'] = {n:sha(a.tokenizer/n) for n in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')}
        report['versions'] = {n:importlib.metadata.version(n) for n in ('lm-eval','transformers','tokenizers','datasets')}
        for ref in oracle['records']:
            row = rows[ref['row']]
            ci = ref['candidate']
            context, cont = model._encode_pair(row['prompt'], row['candidates'][ci])
            inputs, lengths, _ = model.batch_loglikelihood_requests([[(None, context, cont)]])
            text = model.create_message(inputs)
            assert inputs == [row['wire_token_ids'][ci]] and context == row['wire_context_ids']
            assert text == row['prompt']+row['candidates'][ci]
            payload = model._create_payload(text, generate=False)
            assert payload['temperature'] == 0 and payload['max_tokens'] == 1 and payload['echo']
            response = requests.post(a.base_url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            save(a.output, report | dict(pending_raw_response=dict(row=ref['row'], candidate=ci, response=data)))
            logs = data['choices'][0]['logprobs']
            parsed = model.parse_logprobs(data, ctxlens=lengths)[0]
            errors = []
            n = len(inputs[0])
            if len(logs['tokens']) != n+1 or logs['tokens'] != ref['echo_strings']:
                errors.append('echo token strings/count/suffix')
            if any(len(logs[k]) != n+1 for k in ('token_logprobs', 'top_logprobs', 'text_offset')):
                errors.append('echo array count')
            if data['usage']['prompt_tokens'] != len(ref['model_ids']) or data['usage']['completion_tokens'] != 1:
                errors.append('BOS or generated suffix count')
            supplied = logs['token_logprobs']
            if not all(x is not None and math.isfinite(x) for x in supplied[1:]):
                report['failures'].append(dict(row=ref['row'], candidate=ci, error='nonfinite echo probabilities'))
                save(a.output, report | dict(pending_raw_response=data))
                raise ValueError('Nonfinite echo probabilities')
            k = len(cont)
            tolerance = max(0.001, k*0.0002)
            direct = ref['supplied_logprobs'][len(context):]
            observed = supplied[len(context):-1]
            delta = [x-y for x,y in zip(observed, direct, strict=True)]
            total = math.fsum(direct)
            if abs(parsed[0]-math.fsum(observed)) > 1e-8:
                errors.append('parser sum')
            if max(map(abs, delta)) > 0.0002 or abs(parsed[0]-total) > tolerance:
                errors.append('direct continuation probability')
            full_delta = [abs(supplied[i]-ref['supplied_logprobs'][i]) for i in range(1,n)]
            if max(full_delta) > 0.0002:
                errors.append('supplied echo probability alignment')
            if abs(supplied[-1]-ref['generated_logprob']) > 0.0002:
                errors.append('generated suffix probability')
            rec = dict(row=ref['row'], candidate=ci, payload=payload, response=data, parsed=parsed,
                       direct_sum=total, continuation_deltas=delta, full_echo_max_error=max(full_delta), errors=errors)
            if row['variant'] == 'plain-letter':
                shard = a.a2/runtime['arm']/f"{row['language']}-r{row['shift']}-{row['index']//5*5:03}.json"
                old = json.loads(shard.read_text())['samples']['belebele_'+row['language']]
                old = next(s for s in old if (s['doc']['link'],s['doc']['question_number']) == (row['source'],row['question']))
                rec['a2_score'] = old['filtered_resps'][ci][0]
                rec['a2_delta'] = parsed[0]-rec['a2_score']
                if abs(rec['a2_delta']) > 0.001:
                    errors.append('A2 bridge')
            report['records'].append(rec)
            report['failures'].extend(dict(row=ref['row'], candidate=ci, error=e) for e in errors)
            save(a.output, report)
            print(f"client row {ref['row']} candidate {ci}: {errors}", flush=True)
        report['oracle_sha256'] = sha(a.oracle)
        report['predictions'] = []
        for ri in report['selection']:
            group = [r for r in report['records'] if r['row']==ri]
            direct = prediction([r['direct_sum'] for r in group])
            client = prediction([r['parsed'][0] for r in group])
            tolerance = max(0.001, max(len(t)-len(rows[ri]['wire_context_ids']) for t in rows[ri]['wire_token_ids'])*0.0002)
            pred = dict(row=ri, direct=direct, client=client, inversion=direct['prediction']!=client['prediction'], tie_sensitive=direct['margin']<=2*tolerance)
            if 'a2_score' in group[0]:
                pred['a2'] = prediction([r['a2_score'] for r in group])
                pred['a2_inversion'] = pred['a2']['prediction'] != client['prediction']
            report['predictions'].append(pred)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if report['failures']:
        raise RuntimeError(f"{len(report['failures'])} validation failures; see preserved report")


if __name__ == '__main__':
    main()
