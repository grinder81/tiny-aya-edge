"""Clone a pinned wrapper environment and apply explicit runtime-only patches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--destination',type=Path,required=True)
p.add_argument('--patch',type=Path,action='append',required=True)
p.add_argument('--library',type=Path,required=True)
a=p.parse_args()
assert a.source.is_absolute() and a.destination.is_absolute() and not a.destination.exists()
relative=Path('lib/python3.13/site-packages/llama_cpp/llama.py')
expected='6673e5584b3c6d300a4c087690ae73505d70b93ad31cfcd5e62862c06fa85058'
assert sha(a.source/relative)==expected
assert all(f.is_file() for f in a.patch)
shutil.copytree(a.source,a.destination,symlinks=True)
for patch in a.patch:
    subprocess.run(['patch','--batch','--forward','-p1','--input',str(patch)],cwd=a.destination/relative.parent.parent,check=True)
assert sha(a.source/relative)==expected
env=dict(os.environ,LLAMA_CPP_LIB_PATH=str(a.library))
code='import json,sys,llama_cpp; print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,module=llama_cpp.__file__)))'
identity=json.loads(subprocess.check_output([str(a.destination/'bin/python'),'-c',code],env=env,text=True))
assert Path(identity['prefix']).resolve()==a.destination.resolve()
assert Path(identity['module']).resolve().is_relative_to(a.destination.resolve())
identity.update(source=str(a.source),original_wrapper_sha256=expected,wrapper_sha256=sha(a.destination/relative),patches={str(f):sha(f) for f in a.patch},invocation=' '.join(os.sys.argv))
(a.destination/'a3-preparation.json').write_text(json.dumps(identity,indent=2))
print(json.dumps(identity,indent=2))
