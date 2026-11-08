"""Reject misleading native dry-run parsers before byte-budget decisions."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class NativeLayoutTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'builds/profile_pilot_bytes.py'
        spec=importlib.util.spec_from_file_location('pilot_bytes',path)
        self.module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'gguf':types.ModuleType('gguf')}):
            spec.loader.exec_module(self.module)

    def test_exact_types_not_rounded_sizes(self):
        log='[   1/   2] token_embd.weight - [2048, 262144], type = bf16, size = 1024.00 MiB -> 420.00 MiB (q6_K)\n[   2/   2] output_norm.weight - [2048], type = f32, size = 0.008 MiB'
        self.assertEqual(self.module.parse_layout(log,{'token_embd.weight':None,'output_norm.weight':None}),
                         {'token_embd.weight':'Q6_K','output_norm.weight':'F32'})

    def test_incomplete_duplicate_and_unknown_stop(self):
        line='[1/2] a - [256], type = bf16, size = 1 MiB -> 0.1 MiB (q3_K)'
        for log,tensors in [(line,{'a':None,'b':None}),(line+'\n'+line,{'a':None}),(line,{'b':None})]:
            with self.assertRaises(ValueError):
                self.module.parse_layout(log,tensors)


if __name__=='__main__':
    unittest.main()
