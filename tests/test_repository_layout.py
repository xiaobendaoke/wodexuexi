"""Lightweight layout guard regression tests; no PyTorch imports."""
import importlib.util
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/maintenance/check_layout.py'
spec = importlib.util.spec_from_file_location('check_layout', SCRIPT)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class LayoutTests(unittest.TestCase):
    def test_allowed_entries(self):
        allowed = runpy.run_path(str(layout.ALLOWLIST))['ROOT_ALLOWLIST']
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'README.md').write_text('allowed')
            (root / 'docs').mkdir()
            self.assertEqual(layout.check_layout(root, allowed), [])

    def test_unknown_file_and_directory_are_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'README.md').write_text('keep this')
            (root / 'surprise.json').write_text('{}')
            (root / 'new_outputs').mkdir()
            before = sorted(p.name for p in root.iterdir())
            allowed = {'files': ['README.md'], 'directories': []}
            errors = layout.check_layout(root, allowed)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any('surprise.json' in e for e in errors))
            self.assertTrue(any('new_outputs' in e for e in errors))
            self.assertEqual(before, sorted(p.name for p in root.iterdir()))
            self.assertEqual((root / 'README.md').read_text(), 'keep this')
            self.assertEqual((root / 'surprise.json').read_text(), '{}')

    def test_type_mismatch_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'README.md').mkdir()
            (root / 'docs').symlink_to(root / 'README.md', target_is_directory=True)
            self.assertEqual(len(layout.check_layout(root, {'files': ['README.md'], 'directories': ['docs']})), 2)

    def test_worktree_git_file_and_missing_optional_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / '.git').write_text('gitdir: /some/worktree')
            self.assertEqual(layout.check_layout(root, {'files': ['README.md'], 'directories': ['.git']}), [])

    def test_cli_from_other_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(SCRIPT), '--root', temp], cwd=temp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (Path(temp) / 'unexpected.txt').write_text('preserve')
            result = subprocess.run([sys.executable, str(SCRIPT), '--root', temp], cwd=temp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('unexpected.txt', result.stdout)
            result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(Path(temp)/'missing')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
