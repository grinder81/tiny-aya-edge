"""Offline selection/entry guards; these tests never import a model runtime."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('validate_t1_cold.py')
spec = importlib.util.spec_from_file_location('t1_gate', SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture():
    rows = []
    for language in ('eng_Latn', 'ben_Beng', 'hin_Deva'):
        for i in range(300):
            context = [2, 7] if i != 150 else [2, 7, 8]
            rows.append(dict(language=language, index=i, variant='plain-letter',
                shift=0, candidates=[' A',' B',' C',' D'], model_context_ids=context,
                model_token_ids=[context + [token] for token in (10, 11, 12, 13)]))
    return rows


class Selection(unittest.TestCase):
    def test_import_binding_and_repeat_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            actual = Path(directory) / 'actual.py'
            other = Path(directory) / 'other.py'
            actual.write_text('same source')
            other.write_text('same source')
            pinned = {str(actual.resolve()): gate.sha(actual)}
            self.assertEqual(gate.bind_imports([actual], pinned), pinned)
            link = Path(directory) / 'alias.py'
            try:
                link.symlink_to(actual)
            except (OSError, NotImplementedError):
                pass  # Windows may not permit unprivileged symlinks; CI/Linux does.
            else:
                self.assertEqual(gate.bind_imports([actual], {str(link): gate.sha(actual)}), pinned)
                self.assertEqual(gate.bind_imports([link], pinned), pinned)
                with self.assertRaises(ValueError):
                    gate.bind_imports([link], {**pinned, str(link): '0'*64})
            with self.assertRaises(ValueError):
                gate.bind_imports([other], pinned)
            actual.write_text('changed source')
            with self.assertRaises(ValueError):
                gate.bind_imports([actual], pinned)
        for bad in (float('nan'), float('inf'), .000201):
            self.assertFalse(gate.repeat_passes(bad))
        self.assertTrue(gate.repeat_passes(.0002))

    def test_coverage_and_tie_resolution(self):
        rows = fixture()
        rows[151]['model_context_ids'] = [2, 7, 8]
        rows[151]['model_token_ids'] = [[2, 7, 8, t] for t in (10, 11, 12, 13)]
        self.assertEqual(gate.selected(rows),
            [0, 100, 150, 299, 300, 400, 450, 599, 600, 700, 750, 899])
        refs = [(i,c) for i in gate.selected(rows) for c in range(4)]
        self.assertEqual(len(refs), 48)
        self.assertEqual(list(reversed(refs))[0], (899, 3))

    def test_bad_population_and_contract_rejected(self):
        rows = fixture()
        for mutate in (
            lambda r: r.pop(),
            lambda r: r[0].update(index=1),
            lambda r: r[0].update(shift=1),
            lambda r: r[0].update(candidates=['A','B','C','D']),
            lambda r: r[0].update(model_token_ids=[[2,7,10,11]]*4),
        ):
            bad = copy.deepcopy(rows)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                gate.selected(bad)

    def test_optimized_entry_rejected_before_input_read(self):
        result = subprocess.run([sys.executable, '-O', str(SCRIPT), 'direct',
            '--lock', 'absent', '--runtime', 'absent', '--output', 'absent',
            '--expected-lock-sha256', '0'*64], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Optimized Python disables validation', result.stderr)


if __name__ == '__main__':
    unittest.main()
