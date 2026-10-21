"""Exercise the unchanged upstream API client's A3 request path without HTTP."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.tokenizer.is_dir():
        raise ValueError('Local tokenizer directory required')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from lm_eval.models.openai_completions import LocalCompletionsAPI
    import lm_eval.models.openai_completions as api
    import lm_eval.models.api_models as base
    from lm_eval.api.model import TemplateLM
    import inspect
    data = json.loads(args.lock.read_text(encoding='utf-8'))
    if len(data['rows']) != 3600:
        raise ValueError('Expected complete frozen A3 request set')
    model_args = dict(model='offline-no-inference',
        base_url='http://127.0.0.1:1/v1/completions', tokenizer=str(args.tokenizer),
        tokenizer_backend='huggingface', tokenized_requests=False,
        add_bos_token=False, max_length=8192, batch_size=1)
    contexts = candidates = 0
    failures = []
    # Tokenization/formatting/parser checks only. Fail if any path tries HTTP.
    with patch('requests.sessions.Session.request', side_effect=RuntimeError('HTTP forbidden in offline probe')):
        model = LocalCompletionsAPI(**model_args)
        for row in data['rows']:
            if model.tok_encode(row['prompt']) == row['wire_context_ids']:
                contexts += 1
            else:
                failures.append(dict(variant=row['variant'], language=row['language'],
                                     index=row['index'], shift=row['shift'], error='context IDs'))
            for i, candidate in enumerate(row['candidates']):
                context, continuation = model._encode_pair(row['prompt'], candidate)
                inputs, lengths, _ = model.batch_loglikelihood_requests([
                    [(None, context, continuation)]])
                text = model.create_message(inputs)
                token_match = inputs == [row['wire_token_ids'][i]]
                boundary_match = context == row['wire_context_ids'] and lengths == [len(context)]
                text_match = text == row['prompt'] + candidate
                payload = model._create_payload(text, generate=False)
                payload_match = (payload['prompt'] == text and payload['echo'] is True
                                 and payload['max_tokens'] == 1)
                # The parser must sum the supplied continuation and exclude the
                # generated suffix. These artificial numbers are not model scores.
                logs = [-float(j + 1) for j in range(len(inputs[0]) + 1)]
                response = {'choices': [dict(index=0, logprobs=dict(token_logprobs=logs,
                    top_logprobs=[{'token': value} for value in logs]))]}
                parsed = model.parse_logprobs(response, ctxlens=lengths)
                parser_match = parsed == [(sum(logs[len(context):-1]), True)]
                if not all((token_match, boundary_match, text_match, payload_match, parser_match)):
                    failures.append(dict(variant=row['variant'], language=row['language'],
                        index=row['index'], shift=row['shift'], candidate=i,
                        tokens=token_match, boundary=boundary_match, text=text_match,
                        payload=payload_match, synthetic_parser=parser_match))
                else:
                    candidates += 1
    files = [__file__, api.__file__, base.__file__, inspect.getfile(TemplateLM)]
    report = dict(scope='Offline upstream client request path and synthetic parser; no endpoint or model scores',
        lock_sha256=sha(args.lock), model_args=model_args,
        versions={name: importlib.metadata.version(name) for name in
                  ('lm-eval', 'transformers', 'tokenizers', 'huggingface-hub', 'datasets')},
        files={str(path): sha(path) for path in files},
        tokenizer_files={name:sha(args.tokenizer / name) for name in
                         ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')},
        passed_contexts=contexts, passed_candidates=candidates, failures=failures,
        live_echo_logits_bridge_required=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    if failures:
        raise ValueError(f'{len(failures)} offline client failures')
    print(f'Passed {contexts} contexts and {candidates} candidate requests without HTTP')


if __name__ == '__main__':
    main()
