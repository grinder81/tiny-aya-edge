"""Summarize complete probability/order gates without claiming accuracy results."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
report=dict(scope='Runtime probability correctness; diagnostic cases are not task accuracy estimates',arms={})
for name in ('bf16','q4'):
    files={kind:a.directory/f'{name}-{kind}.json' for kind in ('direct','client','order')}
    data={kind:json.loads(path.read_text()) for kind,path in files.items()}
    direct,client,order=(data[k] for k in ('direct','client','order'))
    assert len(direct['records'])==len(client['records'])==192 and len(order['records'])==36
    assert not any(d['failures'] for d in data.values())
    assert client['oracle_sha256']==order['oracle_sha256']==sha(files['direct'])
    arm=dict(files={str(path):sha(path) for path in files.values()},primary_contexts=len(client['predictions']),primary_candidates=192,
        reverse_order_contexts=len(order['predictions']),reverse_order_candidates=36,
        effective_bos=direct['bos_id'],effective_eos=direct['eos_id'],
        max_repeat_logprob_error=max(r.get('repeat_max_logprob_error',0) for r in direct['records']),
        max_full_echo_error=max(r['full_echo_max_error'] for r in client['records']),
        max_continuation_token_error=max(abs(x) for r in client['records'] for x in r['continuation_deltas']),
        max_continuation_sum_error=max(abs(r['parsed'][0]-r['direct_sum']) for r in client['records']),
        max_reverse_continuation_sum_error=max(abs(r['parsed'][0]-r['direct_sum']) for r in order['records']),
        argmax_inversions=[r for r in client['predictions'] if r['inversion']],
        reverse_argmax_inversions=[r for r in order['predictions'] if r['inversion']],
        legacy_a2_bound_exceedances=client['a2_bridge_failures'],
        legacy_a2_prediction_changes=[r for r in client['predictions'] if r.get('a2_inversion')],
        max_legacy_a2_absolute_difference=max(abs(r.get('a2_delta',0)) for r in client['records']))
    refs={(r['row'],r['candidate']):r for r in direct['records']}
    arm['max_suffix_error']=max(abs(r['response']['choices'][0]['logprobs']['token_logprobs'][-1]-refs[r['row'],r['candidate']]['generated_logprob']) for r in client['records'])
    arm['minimum_direct_margin']=min(r['direct']['margin'] for r in client['predictions'])
    arm['minimum_reverse_sentinel_margin']=min(r['direct']['margin'] for r in order['predictions'])
    prior=[r for r in client['records'] if 'a2_score' in r]
    arm['legacy_a2_compared_candidates']=len(prior)
    arm['legacy_a2_compared_contexts']=sum('a2' in r for r in client['predictions'])
    arm['legacy_a2_all_predictions']=[r for r in client['predictions'] if 'a2' in r]
    timing=a.directory/f'{name}-client.time'
    if timing.exists():
        seconds=float(next(line.split()[1] for line in timing.read_text().splitlines() if line.startswith('real ')))
        arm['client_total_seconds']=seconds
        arm['client_seconds_per_candidate_including_initialization_and_validation']=seconds/192
    report['arms'][name]=arm
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({name:{key:value for key,value in arm.items() if key not in ('files','legacy_a2_bound_exceedances','legacy_a2_all_predictions')} for name,arm in report['arms'].items()},indent=2))
