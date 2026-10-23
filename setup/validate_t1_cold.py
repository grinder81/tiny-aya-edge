"""T1 expanded cold-runtime probability gate; executing either mode performs inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / 'docs/t1-cold-validation.md'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected(rows):
    if len(rows) != 900 or [(r['language'], r['index']) for r in rows] != [
            (lang, i) for lang in ('eng_Latn', 'ben_Beng', 'hin_Deva') for i in range(300)]:
        raise ValueError('Expected exactly 300 ordered questions per language')
    for row in rows:
        if row['variant'] != 'plain-letter' or row['shift'] != 0 or row['candidates'] != [' A',' B',' C',' D']:
            raise ValueError('Original-order letter contract required')
        if len(row['model_token_ids']) != 4 or any(
                ids[:-1] != row['model_context_ids'] or len(ids) >= 8192
                for ids in row['model_token_ids']):
            raise ValueError('Single-token continuation and context limit required')
    keys = {i for i, row in enumerate(rows) if row['index'] in (0, 100, 299)}
    for language in ('eng_Latn', 'ben_Beng', 'hin_Deva'):
        subset = [i for i, row in enumerate(rows) if row['language'] == language]
        keys.add(max(subset, key=lambda i: max(map(len, rows[i]['model_token_ids']))))
    return sorted(keys)


def save(path, report):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def prediction(scores):
    order = sorted(range(4), key=lambda i: -scores[i])
    return dict(prediction=order[0], margin=scores[order[0]]-scores[order[1]])


def bind_imports(paths, frozen):
    actual = {str(Path(p).resolve()): sha(p) for p in paths}
    resolved = {}
    for path, digest in frozen.items():
        target = str(Path(path).resolve())
        if target in resolved and resolved[target] != digest:
            raise ValueError('Conflicting frozen hashes for one resolved runtime file')
        resolved[target] = digest
    if any(resolved.get(path) != digest for path, digest in actual.items()):
        raise ValueError('Actually imported runtime differs from the frozen manifest')
    return actual


def repeat_passes(error):
    return math.isfinite(error) and error <= 0.0002


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('direct', 'client'))
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--oracle', type=Path)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--expected-lock-sha256', required=True)
    p.add_argument('--protocol', type=Path, default=PROTOCOL)
    p.add_argument('--base-url', default='http://127.0.0.1:8098/v1/completions')
    p.add_argument('--order-check', action='store_true')
    a = p.parse_args()
    if not __debug__:
        raise ValueError('Optimized Python disables validation; run without -O')
    if a.order_check and a.mode != 'client':
        raise ValueError('Reverse order is a client check against the direct oracle')
    if a.output.exists() or a.output.with_suffix('.partial.json').exists():
        raise FileExistsError(a.output)
    assert sha(a.lock) == a.expected_lock_sha256
    assert sha(a.protocol) == sha(PROTOCOL)
    rows = json.loads(a.lock.read_text())['rows']
    runtime = json.loads(a.runtime.read_text())
    assert all(sha(k) == v for k, v in runtime['files'].items())
    report = dict(lock_sha256=sha(a.lock), protocol_sha256=sha(a.protocol), runtime_sha256=sha(a.runtime), script_sha256=sha(__file__),
                  protocol='T1 expanded original-order cold-prefill correctness',
                  order_check=a.order_check, selection=selected(rows), records=[], failures=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == 'direct':
        import numpy as np
        from llama_cpp.server.model import LlamaProxy
        from llama_cpp.server.settings import ModelSettings
        import sys
        import llama_cpp
        import llama_cpp.llama as wrapper
        import llama_cpp._internals as internals
        import llama_cpp.llama_cpp as native
        import llama_cpp.server.model as server_model
        import llama_cpp.server.settings as server_settings
        report['imported_runtime_files'] = bind_imports(
            [wrapper.__file__, internals.__file__, native._lib._name,
             server_model.__file__, server_settings.__file__], runtime['files'])
        report['environment'] = dict(executable=sys.executable,prefix=sys.prefix,llama_cpp_file=llama_cpp.__file__)
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
                    if not repeat_passes(rec['repeat_max_logprob_error']):
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
        command = runtime['server_command']
        expected_model = command[command.index('--model') + 1]
        models = requests.get(a.base_url.rsplit('/', 1)[0] + '/models', timeout=30)
        models.raise_for_status()
        assert [m['id'] for m in models.json()['data']] == [expected_model]
        oracle = json.loads(a.oracle.read_text())
        assert oracle['selection'] == report['selection'] and oracle['runtime_sha256'] == report['runtime_sha256']
        assert oracle['lock_sha256'] == sha(a.lock) and oracle['protocol_sha256'] == report['protocol_sha256'] and oracle['script_sha256'] == sha(__file__) and not oracle['failures']
        assert [(r['row'],r['candidate']) for r in oracle['records']] == [(i,c) for i in report['selection'] for c in range(4)]
        model = LocalCompletionsAPI(model='t1-validation', base_url=a.base_url,
            tokenizer=str(a.tokenizer), tokenizer_backend='huggingface', tokenized_requests=False,
            add_bos_token=False, max_length=8192, batch_size=1)
        report['source_files'] = {f:sha(f) for f in (api.__file__, base.__file__, inspect.getfile(TemplateLM), inspect.getfile(type(model.tokenizer)))}
        report['tokenizer_files'] = {n:sha(a.tokenizer/n) for n in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')}
        report['versions'] = {n:importlib.metadata.version(n) for n in ('lm-eval','transformers','tokenizers','datasets')}
        references = oracle['records']
        if a.order_check:
            references = list(reversed(references))
        for ref in references:
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
            report['records'].append(rec)
            report['failures'].extend(dict(row=ref['row'], candidate=ci, error=e) for e in errors)
            save(a.output, report)
            print(f"client row {ref['row']} candidate {ci}: {errors}", flush=True)
        report['oracle_sha256'] = sha(a.oracle)
        report['predictions'] = []
        for ri in sorted({r['row'] for r in report['records']}):
            group = sorted([r for r in report['records'] if r['row']==ri],key=lambda r:r['candidate'])
            direct = prediction([r['direct_sum'] for r in group])
            client = prediction([r['parsed'][0] for r in group])
            tolerance = max(0.001, max(len(t)-len(rows[ri]['wire_context_ids']) for t in rows[ri]['wire_token_ids'])*0.0002)
            pred = dict(row=ri, direct=direct, client=client, inversion=direct['prediction']!=client['prediction'], tie_sensitive=direct['margin']<=2*tolerance)
            if pred['inversion']:
                report['failures'].append(dict(row=ri, error='argmax inversion'))
            report['predictions'].append(pred)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if report['failures']:
        raise RuntimeError(f"{len(report['failures'])} validation failures; see preserved report")


if __name__ == '__main__':
    main()
