"""Check host-volume headroom as well as a WSL virtual disk's logical space."""
import os
from pathlib import Path
import platform
import shutil


RESERVE_BYTES = 20 * 1024**3


class HostStorageGuard:
    def __init__(self):
        configured = os.environ.get('C8_HOST_STORAGE_PATH')
        wsl = 'microsoft' in platform.release().lower()
        if wsl and not configured:
            raise ValueError('Set C8_HOST_STORAGE_PATH to the Windows volume backing the WSL VHD')
        self.path = Path(configured) if configured else None
        self.minimum_free_bytes = None
        if self.path and (not self.path.is_absolute() or not self.path.is_dir()):
            raise ValueError('Host storage monitor needs an existing absolute directory')

    def check(self, additional_bytes=0):
        if additional_bytes < 0:
            raise ValueError('Negative storage reservation')
        if self.path is None:
            return
        free = shutil.disk_usage(self.path).free
        self.minimum_free_bytes = free if self.minimum_free_bytes is None else min(free, self.minimum_free_bytes)
        if free < RESERVE_BYTES + additional_bytes:
            raise TimeoutError('Host storage headroom below reserve plus next output: '
                               f'{free} < {RESERVE_BYTES} + {additional_bytes} bytes')

    def receipt(self):
        return dict(path=str(self.path) if self.path else None,
                    reserve_bytes=RESERVE_BYTES,
                    minimum_observed_free_bytes=self.minimum_free_bytes)
