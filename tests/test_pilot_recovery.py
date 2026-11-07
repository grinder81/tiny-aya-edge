import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from builds.profile_pilot_kl import sha
from builds.recover_c8_interrupted_operation import recover


class InterruptedOperationRecovery(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)/'workspace'
        self.root.mkdir()
        self.plan=self.root/'plan.json'
        self.plan.write_text(json.dumps(dict(artifact_workspace=str(self.root),cap=dict(native_minutes=90,proxy_minutes=30,local_artifact_bytes=1_000_000_000))))
        self.started=1793592016.68
        self.ledger=self.root/'resource-ledger.json'
        self.ledger.write_text(json.dumps(dict(schema='c8-cumulative-budget-v1',plan_sha256=sha(self.plan),operations=[
            dict(id='done',category='native',name='export-a',pid=1,started_unix=self.started-100,seconds=54.7,status='passed',ended_unix=self.started-45,artifact_bytes=10,error=None),
            dict(id='open',category='native',name='export-b',pid=2,started_unix=self.started,seconds=0.,status='active')])))
        self.lock=self.root/'resource-operation.lock'
        self.lock.write_text('open\n')
        self.partial=self.root/'artifacts'/'b.partial.gguf'
        self.partial.parent.mkdir()
        self.partial.write_bytes(b'x'*1000)
        self.evidence=self.root/'artifacts'/'b-evidence'
        self.evidence.mkdir()
        (self.evidence/'quantize.log').write_text('log')
        self.receipt=self.root/'recovery'/'receipt.json'

    def test_closes_row_with_observed_time_and_preserves_leftovers(self):
        receipt=recover(self.plan,'open',self.started+31.38,[self.partial,self.evidence],self.receipt,'host volume exhausted')
        ledger=json.loads(self.ledger.read_text())
        self.assertEqual([row['status'] for row in ledger['operations']],['passed','failed'])
        self.assertEqual(ledger['operations'][0]['seconds'],54.7)
        row=ledger['operations'][1]
        self.assertAlmostEqual(row['seconds'],31.38,places=5)
        self.assertEqual(row['ended_unix'],self.started+31.38)
        self.assertEqual(row['error'],'host volume exhausted')
        self.assertEqual(row['recovery_receipt'],'recovery/receipt.json')
        self.assertFalse(self.lock.exists())
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.evidence.exists())
        moved=self.root/'recovery'/'interrupted'/'open'
        self.assertEqual((moved/'b.partial.gguf').read_bytes(),b'x'*1000)
        self.assertEqual((moved/'b-evidence'/'quantize.log').read_text(),'log')
        self.assertEqual(receipt['lock_removed']['content'],'open')
        self.assertEqual(receipt['preserved'][0]['sha256'],sha(moved/'b.partial.gguf'))
        self.assertEqual(receipt['preserved'][1]['entries']['quantize.log']['bytes'],3)
        before=json.loads((self.root/receipt['ledger_before']['path']).read_text())
        self.assertEqual(before['operations'][1]['status'],'active')
        self.assertEqual(json.loads(self.receipt.read_text())['charged_seconds'],receipt['charged_seconds'])
        with self.assertRaisesRegex(ValueError,'Exactly one active'):
            recover(self.plan,'open',self.started+31.38,[],self.root/'recovery'/'second.json','again')

    def test_refuses_without_changing_anything(self):
        original=self.ledger.read_text()
        with self.assertRaisesRegex(ValueError,'Exactly one active'):
            recover(self.plan,'done',self.started+1,[self.partial],self.receipt,'wrong id')
        with self.assertRaisesRegex(ValueError,'precedes'):
            recover(self.plan,'open',self.started-1,[self.partial],self.receipt,'end before start')
        outside=self.root.parent/'outside'
        outside.write_text('not in the workspace')
        with self.assertRaisesRegex(ValueError,'escapes'):
            recover(self.plan,'open',self.started+1,[outside],self.receipt,'outside')
        self.assertTrue(outside.exists())
        self.lock.write_text('other\n')
        with self.assertRaisesRegex(ValueError,'Stale lock'):
            recover(self.plan,'open',self.started+1,[self.partial],self.receipt,'lock mismatch')
        self.assertEqual(self.ledger.read_text(),original)
        self.assertTrue(self.lock.exists())
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.receipt.exists())

    @unittest.skipUnless(hasattr(signal,'SIGALRM'),'Native execution uses POSIX WSL signals')
    def test_budget_resumes_with_spent_time_retained(self):
        from builds.c8_budget import BudgetOperation
        with patch.dict(os.environ,{'C8_HOST_STORAGE_PATH':self.temporary.name}):
            with self.assertRaises(FileExistsError):
                with BudgetOperation(self.plan,'native','blocked by the stale lock'):
                    pass
            self.lock.rename(self.root/'lock-moved-aside')
            with self.assertRaisesRegex(ValueError,'interrupted active operation'):
                with BudgetOperation(self.plan,'native','blocked by the active row'):
                    pass
            (self.root/'lock-moved-aside').rename(self.lock)
            recover(self.plan,'open',self.started+31.38,[self.partial,self.evidence],self.receipt,'host volume exhausted')
            with BudgetOperation(self.plan,'native','resumed') as operation:
                self.assertAlmostEqual(operation.used,54.7+31.38,places=6)


if __name__=='__main__':
    unittest.main()
