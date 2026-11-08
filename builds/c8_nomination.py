"""Validate a complete discovery receipt before unlocking any repetition3path."""
import json
import math
from pathlib import Path
from statistics import mean

from builds.profile_pilot_kl import sha
from builds.c8_budget import require_passed_operation

LANGS=('eng_Latn','ben_Beng','hin_Deva','pooled')
EPS=1e-5


def validate_nomination(path,plan_path):
    receipt=json.loads(Path(path).read_text())
    plan=json.loads(Path(plan_path).read_text())
    required={'schema','plan_sha256','discovery_report_sha256','discovery_report_path','implementation_sha256','pair','signs','candidates','epsilon_nats','interpretation'}
    if not required<=receipt.keys() or receipt['schema']!='c8-discovery-nomination-v1' or receipt['plan_sha256']!=sha(plan_path):
        raise ValueError('A complete frozen discovery nomination or explicit none is required')
    discovery=Path(receipt['discovery_report_path'])
    if sha(discovery)!=receipt['discovery_report_sha256']:
        raise ValueError('Discovery evidence changed')
    report=json.loads(discovery.read_text())
    identity=report['identity']
    if identity['plan_sha256']!=sha(plan_path) or identity['mode']!='diagnostics' or identity['repetitions']!=[1,2]:
        raise ValueError('Wrong frozen discovery evidence')
    require_passed_operation(plan_path,identity['budget_operation_id'],'native')
    labels={entry['tensor']+'::'+entry['native_type'] for entry in plan['diagnostics']}
    cases={case['case']['label']:case for case in report['cases']}
    if cases.keys()!=labels or len(report['cases'])!=len(labels):
        raise ValueError('Discovery cost table incomplete')
    costs={}
    for label,case in cases.items():
        rows=case['records']
        if len(rows)!=18 or {(r['repeat'],r['index'],r['language']) for r in rows}!={(rep,i,lang) for rep in (1,2) for i in range(3) for lang in LANGS[:3]}:
            raise ValueError('Discovery repetition/source coverage incomplete')
        for row in rows:
            if row['source']!=plan['probe_sources'][str(row['repeat'])][row['index']] or row['tokens']!=64 or type(row['mean_kl_nats']) not in (int,float) or not math.isfinite(row['mean_kl_nats']) or row['mean_kl_nats']<0:
                raise ValueError('Discovery source or numerical value differs')
        for rep in (1,2):
            for lang in LANGS:
                costs[rep,lang,label]=mean(r['mean_kl_nats'] for r in rows if r['repeat']==rep and
                    (r['language']==lang if lang!='pooled' else r['language']==LANGS[r['index']]))
    low,high=plan['formats']['lower'],plan['formats']['higher']
    candidates=[]
    for pair in plan['pairs']:
        first,second=pair['first'],pair['second']
        margins={str(rep):{lang:(costs[rep,lang,first+'::'+low]-costs[rep,lang,first+'::'+high])-
            (costs[rep,lang,second+'::'+low]-costs[rep,lang,second+'::'+high]) for lang in LANGS} for rep in (1,2)}
        en=[margins[str(r)]['eng_Latn'] for r in (1,2)]
        bn=[margins[str(r)]['ben_Beng'] for r in (1,2)]
        qualifies=pair['eligible'] and min(map(abs,en+bn))>EPS and en[0]*en[1]>0 and bn[0]*bn[1]>0 and en[0]*bn[0]<0
        candidates.append(dict(pair=pair,margins=margins,nominatable=bool(qualifies)))
    nominee=next((row for row in candidates if row['nominatable']),None)
    expected=dict(pair=nominee['pair'] if nominee else None,candidates=candidates,epsilon_nats=EPS,
        signs={lang:1 if nominee['margins']['1'][lang]>0 else -1 for lang in ('eng_Latn','ben_Beng')} if nominee else None)
    if any(json.dumps(receipt[k],sort_keys=True)!=json.dumps(value,sort_keys=True) for k,value in expected.items()):
        raise ValueError('Nomination does not follow the prescribed discovery selection rule')
    return receipt
