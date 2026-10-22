"""Index copied A3 evidence bytes and retained raw files without modifying them."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--raw-root',type=Path,required=True)
a=p.parse_args()
output=a.directory/'evidence-index.json'
assert not output.exists()
report=dict(statuses=dict(v1='FAILED cold-vs-prefix correctness gate;192 oracle,7 completed HTTP records',
    v2='INCOMPLETE seven-candidate same-schedule explanation, not full gate',
    v3='INCOMPLETE cold-only HTTP, stopped for exact ranking optimization',
    v4='COMPLETE both-arm192-candidate primary and36-candidate reverse-order gates'),
    raw_root=str(a.raw_root),files={},retained_raw={})
for path in sorted(a.directory.rglob('*')):
    if path.is_file():
        report['files'][path.relative_to(a.directory).as_posix()]=dict(bytes=path.stat().st_size,sha256=sha(path))
for relative in ('live-validation-v1/bf16-direct.json','live-validation-v3/bf16-direct.json'):
    path=a.raw_root/relative
    report['retained_raw'][str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
report['v3_v4_bf16_oracle_records_exactly_equal']=json.loads((a.raw_root/'live-validation-v3/bf16-direct.json').read_text())['records']==json.loads((a.raw_root/'live-validation-v4/bf16-direct.json').read_text())['records']
output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(output)
