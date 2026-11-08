"""Sample global GPU memory and only processes owned by active C8 operations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory(status):
    result={}
    for name in ('VmRSS','VmHWM'):
        match=re.search(r'^'+name+r':\s+(\d+)\s+kB$',status,re.M)
        result[name+'_bytes']=int(match[1])*1024 if match else None
    return result


def descendants(pid):
    found=set()
    todo=[pid]
    while todo:
        current=todo.pop()
        if current in found:
            continue
        found.add(current)
        try:
            todo.extend(int(x) for x in Path(f'/proc/{current}/task/{current}/children').read_text().split())
        except (FileNotFoundError,ProcessLookupError):
            pass
    return found


def sample(plan_path,root):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
    if len(lines)!=1:
        raise ValueError('Sampler expects the recorded single-GPU host')
    used,total,utilization=map(int,lines[0].split(','))
    ledger_path=root/'resource-ledger.json'
    active=[]
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text())
        if ledger['plan_sha256']!=sha(plan_path):
            raise ValueError('Resource ledger belongs to a different plan')
        active=[row for row in ledger['operations'] if row['status']=='active']
    processes=[]
    for operation in active:
        pid=operation['pid']
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError):
            continue
        if str(plan_path.resolve()).encode() not in command:
            raise ValueError('Active operation PID does not name this plan')
        for child in sorted(descendants(pid)):
            try:
                status=Path(f'/proc/{child}/status').read_text()
            except (FileNotFoundError,ProcessLookupError):
                continue
            counters=memory(status)
            if any(value is not None for value in counters.values()):
                processes.append(dict(pid=child,operation_id=operation['id'],category=operation['category'],
                    operation_name=operation['name'],**counters))
    return dict(unix_time=time.time(),gpu_used_bytes=used*1024**2,gpu_total_bytes=total*1024**2,
        gpu_utilization_percent=utilization,processes=processes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--interval',type=float,default=1.)
    p.add_argument('--max-samples',type=int)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text())
    root=Path(plan['artifact_workspace']).resolve()
    if plan['schema']!='c8-bounded-plan-v1' or not a.directory.resolve().is_relative_to(root):
        raise ValueError('Sampler must remain inside the frozen C8 workspace')
    if a.interval<=0 or a.max_samples is not None and a.max_samples<=0:
        raise ValueError('Positive interval and sample count required')
    a.directory.mkdir(parents=True,exist_ok=False)
    identity=dict(schema='c8-resource-sampler-v1',plan_sha256=sha(a.plan),implementation_sha256=sha(__file__),
        interval_seconds=a.interval,scope='Sampled global GPU use includes unrelated desktop apps; observed process RSS/HWM is a lower bound on continuous peak. No model forward or GPU allocation.')
    (a.directory/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    samples=[]
    failure=None
    try:
        with (a.directory/'samples.jsonl').open('x') as stream:
            while not (a.directory/'STOP').exists():
                row=sample(a.plan,root)
                samples.append(row)
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                if a.max_samples is not None and len(samples)>=a.max_samples:
                    break
                time.sleep(a.interval)
    except BaseException as error:
        failure=repr(error)
        raise
    finally:
        maxima={}
        for row in samples:
            for process in row['processes']:
                key=str(process['pid'])
                previous=maxima.setdefault(key,dict(operation_id=process['operation_id'],operation_name=process['operation_name'],category=process['category'],VmRSS_bytes=None,VmHWM_bytes=None))
                for field in ('VmRSS_bytes','VmHWM_bytes'):
                    if process[field] is not None:
                        previous[field]=max(previous[field] or 0,process[field])
        summary=dict(identity=identity,samples=len(samples),error=failure,
            first_gpu_used_bytes=samples[0]['gpu_used_bytes'] if samples else None,
            max_observed_global_gpu_used_bytes=max((row['gpu_used_bytes'] for row in samples),default=None),
            max_observed_process_memory=maxima,
            samples_sha256=sha(a.directory/'samples.jsonl') if (a.directory/'samples.jsonl').exists() else None)
        (a.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(samples=len(samples),error=failure,processes=len(maxima))),flush=True)


if __name__=='__main__':
    main()
