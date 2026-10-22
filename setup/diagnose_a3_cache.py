"""Preserve fresh-server repeat/candidate HTTP evidence after failed A3 v1 gate."""
import argparse
import json
from pathlib import Path
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--failed',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
failed=json.loads(a.failed.read_text())
saved=[]
for row,candidate in ((1,0),(1,0),(1,1),(0,0),(1,0)):
    record=next(r for r in failed['records'] if (r['row'],r['candidate'])==(row,candidate))
    payload=record['payload']
    request=urllib.request.Request('http://127.0.0.1:8098/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=300) as response:
        result=json.load(response)
    saved.append(dict(row=row,candidate=candidate,payload=payload,response=result))
    a.output.write_text(json.dumps(saved,ensure_ascii=False,indent=2))
    print(f'saved {row}/{candidate}',flush=True)
