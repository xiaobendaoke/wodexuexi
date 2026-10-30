"""Standard-library guards for targeted experiment recovery."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from scripts.analysis import recover_dynamic_service_joint as recovery


class RecoveryGuards(unittest.TestCase):
    def test_publication_never_replaces_existing_different_result(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'source.json';b=Path(folder)/'done.json'
            a.write_text('new');b.write_text('old')
            with self.assertRaises(RuntimeError):recovery.copy_new(a,b)
            self.assertEqual(b.read_text(),'old')
            c=Path(folder)/'new.json';recovery.copy_new(a,c)
            self.assertEqual(c.read_text(),'new');recovery.copy_new(a,c)

    def test_signal_requires_matching_command_line(self):
        process=Mock();process.exists.return_value=True;process.read_bytes.return_value=b'python\0another-task.py\0'
        with patch.object(recovery,'Path',return_value=process),patch.object(recovery.os,'kill') as kill:
            with self.assertRaises(RuntimeError):recovery.stop_owned(9624,['the-specific-frozen-source'])
            kill.assert_not_called()

    def test_pid_change_prevents_forced_kill(self):
        command=Mock();command.exists.return_value=True
        command.read_bytes.side_effect=[b'python\0target.py\0',b'python\0someone-else.py\0']
        status=Mock();status.read_text.return_value='9624 (python) R '
        def path(name):return status if str(name).endswith('/stat') else command
        with patch.object(recovery,'Path',side_effect=path),patch.object(recovery.os,'kill') as kill,patch.object(recovery.time,'sleep'):
            with self.assertRaises(RuntimeError):recovery.stop_owned(9624,['target.py'])
            self.assertEqual(kill.call_count,1)
            self.assertEqual(kill.call_args.args[1],recovery.signal.SIGTERM)


if __name__=='__main__':unittest.main()
