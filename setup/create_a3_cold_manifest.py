"""Record an isolated patched wrapper identity without modifying prior manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser(description=__doc__)
for key in ('original','output','patch','protocol'):
    p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--additional-patch',type=Path)
a=p.parse_args()
assert not a.output.exists()
import llama_cpp.llama as wrapper
import llama_cpp._internals as internals
from llama_cpp.server import model,settings
prior=json.loads(a.original.read_text())
assert all(sha(f)==h for f,h in prior['files'].items())
new=dict(prior)
new['server_command']=list(prior['server_command'])
new['server_command'][0]=sys.executable
new['files']={f:h for f,h in prior['files'].items() if f.endswith('.gguf') or f.endswith('.so')}
new['files'].update({f:sha(f) for f in (wrapper.__file__,internals.__file__,model.__file__,settings.__file__)})
new['provenance']=dict(original_runtime_sha256=sha(a.original),patch_sha256=sha(a.patch),protocol_sha256=sha(a.protocol),executable=sys.executable,prefix=sys.prefix,scope='Cold echo likelihood prefill; saved cache restore bypass; non-scoring unchanged')
if a.additional_patch:
    new['provenance']['additional_patch_sha256']=sha(a.additional_patch)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(new,indent=2))
print(a.output)
