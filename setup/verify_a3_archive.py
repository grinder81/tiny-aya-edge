"""Verify original evidence bytes; optionally losslessly compress large copies."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

def sha(data):
    return hashlib.sha256(data).hexdigest()

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--compress',action='store_true')
a=p.parse_args()
root=a.directory.resolve()
index=root/'evidence-index.json'
data=json.loads(index.read_text())
for name,record in data['files'].items():
    raw=root/name
    assert raw.resolve().is_relative_to(root)
    if 'storage' in record:
        storage=record['storage']
        path=root/storage['path']
        assert path.resolve().is_relative_to(root)
        packed=path.read_bytes()
        assert sha(packed)==storage['sha256'] and len(packed)==storage['bytes']
        original=gzip.decompress(packed)
    else:
        original=raw.read_bytes()
    assert sha(original)==record['sha256'] and len(original)==record['bytes']
    if a.compress and 'storage' not in record and name.endswith('.json') and len(original)>100000:
        path=raw.with_suffix(raw.suffix+'.gz')
        assert not path.exists()
        packed=gzip.compress(original,compresslevel=9,mtime=0)
        assert gzip.decompress(packed)==original
        path.write_bytes(packed)
        record['storage']=dict(path=path.relative_to(root).as_posix(),bytes=len(packed),sha256=sha(packed),format='gzip; original decompressed bytes retain above identity')
        raw.unlink()
if a.compress:
    index.write_text(json.dumps(data,ensure_ascii=False,indent=2))
print(f"Verified {len(data['files'])} original evidence byte identities")
