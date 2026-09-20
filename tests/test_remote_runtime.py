"""Fresh subprocesses catch runtime failures hidden by import order."""
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest


class TestRemoteRuntime(unittest.TestCase):
    @unittest.skipUnless(platform.system() == 'Linux', 'Remote Linux runtime only')
    def test_both_import_orders_without_activated_shell(self):
        root = Path(__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items()
               if k not in ('LD_LIBRARY_PATH', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV')}
        env['EXPERIMENT_PYTHON'] = sys.executable
        for imports in ('import torch; import numpy', 'import numpy; import torch'):
            with self.subTest(imports=imports):
                result = subprocess.run(['bash', str(root / 'run_remote_python.sh'), '-c',
                    imports + '; print(torch.__version__, numpy.__version__)'],
                    env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
