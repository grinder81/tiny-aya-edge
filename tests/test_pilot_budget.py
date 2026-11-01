import json
from pathlib import Path
import signal
import tempfile
import time
import unittest

from builds.c8_budget import BudgetOperation
from builds.c8_nomination import validate_nomination
from builds.profile_pilot_kl import sha


@unittest.skipUnless(hasattr(signal,'SIGALRM'),'Native execution uses POSIX WSL signals')
class CumulativeBudgets(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)/'workspace'
        self.root.mkdir()
        self.plan=self.root/'plan.json'
        self.plan.write_text(json.dumps(dict(artifact_workspace=str(self.root),cap=dict(native_minutes=1,proxy_minutes=1,local_artifact_bytes=1_000_000))))

    def test_new_invocation_retains_spent_time_and_single_owner(self):
        with BudgetOperation(self.plan,'native','first') as first:
            time.sleep(.01)
            with self.assertRaises(FileExistsError):
                with BudgetOperation(self.plan,'native','duplicate'):
                    pass
        with BudgetOperation(self.plan,'native','second') as second:
            self.assertGreater(second.used,0)
            self.assertLess(second.remaining(),60)
        ledger=json.loads((self.root/'resource-ledger.json').read_text())
        self.assertEqual([row['status'] for row in ledger['operations']],['passed','passed'])

    def test_exhaustion_cannot_be_reset_by_relaunch(self):
        (self.root/'resource-ledger.json').write_text(json.dumps(dict(plan_sha256=sha(self.plan),operations=[dict(id='old',category='native',status='passed',seconds=61)])))
        with self.assertRaises(TimeoutError):
            with BudgetOperation(self.plan,'native','retry'):
                pass
        self.assertFalse((self.root/'resource-operation.lock').exists())

    def test_artifact_excess_is_recorded_and_outside_writes_rejected(self):
        with self.assertRaises(TimeoutError):
            with BudgetOperation(self.plan,'native','oversize') as operation:
                (self.root/'oversize.bin').write_bytes(b'x'*1_100_000)
                operation.check()
        ledger=json.loads((self.root/'resource-ledger.json').read_text())
        self.assertEqual(ledger['operations'][0]['status'],'failed')
        with self.assertRaises(ValueError):
            with BudgetOperation(self.plan,'native','outside',(self.root.parent/'outside.bin',)):
                pass

    def test_placeholder_receipt_cannot_unlock_confirmation(self):
        path=self.root/'placeholder.json'
        path.write_text(json.dumps({'plan_sha256':sha(self.plan)}))
        with self.assertRaisesRegex(ValueError,'complete frozen discovery'):
            validate_nomination(path,self.plan)


if __name__=='__main__':
    unittest.main()
