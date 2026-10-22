"""Compare preserved v1 HTTP responses with independent bounded same-order logits."""
import argparse
import hashlib
import json
import math
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
for key in ('oracle','http','lock','output'):
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
oracle=json.loads(a.oracle.read_text())
http=json.loads(a.http.read_text())
rows=json.loads(a.lock.read_text())['rows']
report=dict(scope='Incomplete bounded explanation only; NOT full validation gate',
    files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.oracle,a.http,a.lock,Path(__file__))},records=[])
for ref,observed in zip(oracle['records'],http['records']):
    assert (ref['row'],ref['candidate'])==(observed['row'],observed['candidate'])
    logs=observed['response']['choices'][0]['logprobs']
    values=logs['token_logprobs']
    n=len(ref['model_ids'])-1
    c=len(rows[ref['row']]['wire_context_ids'])
    deltas=[values[i]-ref['supplied_logprobs'][i] for i in range(1,n)]
    candidate_delta=math.fsum(values[c:-1])-math.fsum(ref['supplied_logprobs'][c:])
    report['records'].append(dict(row=ref['row'],candidate=ref['candidate'],
        strings_match=logs['tokens']==ref['echo_strings'],count_matches=len(values)==n+1,
        full_echo_max_error=max(map(abs,deltas)),candidate_delta=candidate_delta,
        suffix_delta=values[-1]-ref['generated_logprob']))
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report['records'],indent=2))
