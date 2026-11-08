"""Close a C8 ledger operation that a host failure left active, through a recorded recovery receipt.

The frozen budget refuses to start while a row is active. This closes exactly one such row as failed with the
elapsed time observed from preserved evidence, moves the operation's leftovers unchanged into a recovery folder,
removes the stale lock with its content recorded, and keeps a copy of the ledger as it was. Spent budget is
retained and nothing is deleted; caps and the plan are not touched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from builds.profile_pilot_kl import sha


def _bytes(root):
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file())


def _describe(path):
    if path.is_file():
        return dict(kind='file',bytes=path.stat().st_size,sha256=sha(path),mtime_unix=path.stat().st_mtime)
    entries={}
    for child in sorted(p for p in path.rglob('*') if p.is_file()):
        entries[str(child.relative_to(path))]=dict(bytes=child.stat().st_size,sha256=sha(child),mtime_unix=child.stat().st_mtime)
    return dict(kind='directory',entries=entries)


def recover(plan_path,operation_id,ended_unix,preserve,receipt_path,reason):
    plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text())
    root=Path(plan['artifact_workspace']).resolve()
    ledger_path=root/'resource-ledger.json'
    lock=root/'resource-operation.lock'
    receipt_path=Path(receipt_path)
    if receipt_path.exists():
        raise FileExistsError('Recovery receipt already exists; one receipt per recovery')
    ledger=json.loads(ledger_path.read_text())
    if ledger['plan_sha256']!=sha(plan_path):
        raise ValueError('Wrong C8 budget ledger')
    active=[row for row in ledger['operations'] if row['status']=='active']
    if len(active)!=1 or active[0]['id']!=operation_id:
        raise ValueError('Exactly one active operation with the given id is required')
    row=active[0]
    if not lock.exists() or lock.read_text().strip()!=operation_id:
        raise ValueError('Stale lock must name the interrupted operation')
    if ended_unix<row['started_unix']:
        raise ValueError('Observed end precedes the recorded start')
    recovery=receipt_path.parent
    recovery.mkdir(parents=True,exist_ok=True)
    before=recovery/('resource-ledger.before-recovery-'+operation_id+'.json')
    if before.exists():
        raise FileExistsError(before)
    preserved=[]
    for item in preserve:
        item=Path(item).resolve()
        if not item.exists():
            raise FileNotFoundError(item)
        if not item.is_relative_to(root):
            raise ValueError('Preserved item escapes the frozen artifact workspace')
        target=recovery/'interrupted'/operation_id/item.name
        if target.exists():
            raise FileExistsError(target)
        preserved.append(dict(original=str(item.relative_to(root)),preserved=str(target.relative_to(root)),**_describe(item)))
    shutil.copy2(ledger_path,before)
    for entry in preserved:
        target=root/entry['preserved']
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(root/entry['original']),str(target))
    seconds=ended_unix-row['started_unix']
    row.update(seconds=seconds,ended_unix=ended_unix,artifact_bytes=_bytes(root),status='failed',error=reason,
        recovery_receipt=str(receipt_path.relative_to(root)) if receipt_path.resolve().is_relative_to(root) else str(receipt_path))
    temporary=ledger_path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
    temporary.replace(ledger_path)
    lock_content=lock.read_text().strip()
    lock.unlink()
    receipt=dict(schema='c8-recovery-receipt-v1',recorded_unix=time.time(),plan_sha256=ledger['plan_sha256'],
        operation=dict(row),charged_seconds=seconds,reason=reason,
        ledger_before=dict(path=str(before.relative_to(root)),sha256=sha(before)),
        ledger_after_sha256=sha(ledger_path),lock_removed=dict(path=str(lock.relative_to(root)),content=lock_content),
        preserved=preserved,
        scope='Closes one interrupted ledger row with its observed elapsed time; spent budget retained, nothing deleted, frozen caps and plan unchanged')
    with receipt_path.open('x') as stream:
        json.dump(receipt,stream,indent=2)
        stream.write('\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget-plan',type=Path,required=True)
    p.add_argument('--operation-id',required=True)
    p.add_argument('--ended-unix',type=float,required=True,help='last observed activity of the interrupted process, from preserved evidence timestamps')
    p.add_argument('--preserve',type=Path,action='append',default=[],help='leftover file or directory of the operation to move unchanged into the recovery folder')
    p.add_argument('--receipt',type=Path,required=True,help='recovery receipt to write; its folder becomes the recovery folder')
    p.add_argument('--reason',required=True)
    a=p.parse_args()
    receipt=recover(a.budget_plan,a.operation_id,a.ended_unix,a.preserve,a.receipt,a.reason)
    print(json.dumps(dict(charged_seconds=receipt['charged_seconds'],preserved=len(receipt['preserved']),receipt=str(a.receipt))))


if __name__=='__main__':
    main()
