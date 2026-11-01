import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class MechanicalFormatRule(unittest.TestCase):
    def test_native_format_choice_uses_inventory_only(self):
        path=Path(__file__).resolve().parents[1]/'builds/prepare_c8_plan.py'
        spec=importlib.util.spec_from_file_location('c8_plan',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'gguf':types.ModuleType('gguf')}):
            spec.loader.exec_module(module)
        self.assertEqual(module.choose_formats('Q4_K',144),('Q3_K','Q4_K'))
        self.assertEqual(module.choose_formats('Q6_K',210),('Q3_K','Q4_K'))
        self.assertEqual(module.choose_formats('Q3_K',110),('Q2_K','Q3_K'))
        self.assertEqual(module.choose_formats('IQ3_S',110),('Q2_K','Q3_K'))


if __name__=='__main__':
    unittest.main()
