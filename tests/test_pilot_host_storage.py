import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from builds.host_storage_guard import HostStorageGuard, RESERVE_BYTES


class HostStorage(unittest.TestCase):
    def test_wsl_requires_explicit_backing_volume(self):
        with patch.dict(os.environ, {}, clear=True), patch('platform.release', return_value='microsoft-standard-WSL2'):
            with self.assertRaisesRegex(ValueError, 'Windows volume'):
                HostStorageGuard()

    def test_virtual_free_space_cannot_mask_host_exhaustion(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'C8_HOST_STORAGE_PATH': root}):
            guard = HostStorageGuard()
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=RESERVE_BYTES + 10)) as usage:
                guard.check(10)
                with self.assertRaises(TimeoutError):
                    guard.check(11)
                usage.assert_called_with(Path(root))
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=11)):
                with self.assertRaises(TimeoutError):
                    guard.check()
            self.assertEqual(guard.receipt()['minimum_observed_free_bytes'], 11)

    def test_unreadable_volume_fails_closed(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'C8_HOST_STORAGE_PATH': root}):
            guard = HostStorageGuard()
            with patch('shutil.disk_usage', side_effect=OSError('unavailable')):
                with self.assertRaises(OSError):
                    guard.check()


if __name__ == '__main__':
    unittest.main()
