"""Persistent cumulative C8 deadlines, artifact cap and single-operation ownership."""
import json
import os
from pathlib import Path
import signal
import threading
import time
import uuid

from builds.profile_pilot_kl import sha


class BudgetOperation:
    def __init__(self,plan,category,name,outputs=()):
        self.plan_path=Path(plan)
        self.plan=json.loads(self.plan_path.read_text())
        self.root=Path(self.plan['artifact_workspace']).resolve()
        self.category=category
        self.name=name
        self.outputs=[Path(path) for path in outputs]
        if category not in ('native','proxy'):
            raise ValueError('Unknown C8 resource category')
        self.operation_id=uuid.uuid4().hex
        self.path=self.root/'resource-ledger.json'
        self.lock=self.root/'resource-operation.lock'
        self.limit=self.plan['cap'][category+'_minutes']*60
        self.reason=None

    def _bytes(self):
        return sum(path.stat().st_size for path in self.root.rglob('*') if path.is_file())

    def _write(self):
        temporary=self.path.with_suffix('.partial.json')
        temporary.write_text(json.dumps(self.ledger,indent=2,allow_nan=False)+'\n')
        temporary.replace(self.path)

    def __enter__(self):
        self.root.mkdir(parents=True,exist_ok=True)
        for output in self.outputs:
            if not output.resolve().is_relative_to(self.root):
                raise ValueError('C8 output escapes the frozen artifact workspace')
        with self.lock.open('x') as stream:
            stream.write(self.operation_id+'\n')
        try:
            self.ledger=json.loads(self.path.read_text()) if self.path.exists() else dict(
                schema='c8-cumulative-budget-v1',plan_sha256=sha(self.plan_path),operations=[])
            if self.ledger['plan_sha256']!=sha(self.plan_path) or any(row['status']=='active' for row in self.ledger['operations']):
                raise ValueError('Different plan or interrupted active operation; preserve and review')
            self.used=sum(row['seconds'] for row in self.ledger['operations'] if row['category']==self.category)
            if self.used>=self.limit or self._bytes()>self.plan['cap']['local_artifact_bytes']:
                raise TimeoutError('Frozen cumulative C8 budget exhausted')
            self.started=time.monotonic()
            self.deadline=self.started+(self.limit-self.used)
            self.row=dict(id=self.operation_id,category=self.category,name=self.name,
                pid=os.getpid(),started_unix=time.time(),seconds=0.,status='active')
            self.ledger['operations'].append(self.row)
            self._write()
            self.stop=threading.Event()
            def monitor():
                while not self.stop.wait(.5):
                    try:
                        if self._bytes()>self.plan['cap']['local_artifact_bytes']:
                            self.reason='Cumulative local artifact cap exceeded'
                            os.kill(os.getpid(),signal.SIGALRM)
                    except OSError as error:
                        self.reason='Artifact accounting failed: '+repr(error)
            self.thread=threading.Thread(target=monitor,daemon=True)
            self.thread.start()
            self.prior_alarm=signal.getsignal(signal.SIGALRM)
            def expired(signum,frame):
                raise TimeoutError(self.reason or 'Cumulative C8 '+self.category+' deadline reached')
            signal.signal(signal.SIGALRM,expired)
            signal.setitimer(signal.ITIMER_REAL,self.remaining())
            return self
        except BaseException:
            self.lock.unlink()
            raise

    def remaining(self):
        return max(.001,self.deadline-time.monotonic())

    def check(self,check_bytes=True):
        if self.reason or time.monotonic()>self.deadline:
            raise TimeoutError(self.reason or 'Cumulative C8 deadline exceeded')
        if check_bytes and self._bytes()>self.plan['cap']['local_artifact_bytes']:
            raise TimeoutError('Cumulative local artifact cap exceeded')

    def reserve(self,additional_bytes):
        self.check()
        if additional_bytes<0 or self._bytes()+additional_bytes>self.plan['cap']['local_artifact_bytes']:
            raise TimeoutError('Insufficient frozen C8 artifact budget for next output')

    def __exit__(self,kind,error,traceback):
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,self.prior_alarm)
        self.stop.set()
        self.thread.join(timeout=2)
        failure=error
        if failure is None:
            try:
                self.check()
            except BaseException as budget_error:
                failure=budget_error
        self.row.update(seconds=time.monotonic()-self.started,ended_unix=time.time(),
            artifact_bytes=self._bytes(),status='failed' if failure else 'passed',
            error=repr(failure) if failure else None)
        self._write()
        if self.lock.read_text().strip()!=self.operation_id:
            raise ValueError('C8 operation lock changed unexpectedly')
        self.lock.unlink()
        if error is None and failure is not None:
            raise failure
        return False


def require_passed_operation(plan_path,operation_id,category):
    plan=json.loads(Path(plan_path).read_text())
    ledger=json.loads((Path(plan['artifact_workspace'])/'resource-ledger.json').read_text())
    if ledger['plan_sha256']!=sha(plan_path):
        raise ValueError('Wrong C8 budget ledger')
    rows=[row for row in ledger['operations'] if row['id']==operation_id]
    if len(rows)!=1 or rows[0]['category']!=category or rows[0]['status']!='passed':
        raise ValueError('Required C8 operation did not pass its cumulative budget')
