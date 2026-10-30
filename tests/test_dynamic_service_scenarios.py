"""Standard-library workload contract checks; safe on the writing host."""
import unittest
from utils.dynamic_service_scenarios import Scenario, seed_for


class ScenarioContract(unittest.TestCase):
    def test_balanced_paired_workloads(self):
        tapes = [Scenario(seed_for('test', 0), kind) for kind in 'ABCD']
        for tape in tapes:
            self.assertEqual(sum(tape.counts), 50000)
            self.assertEqual(tape.tasks, tapes[0].tasks)
            self.assertEqual(tape.ue_positions, tapes[0].ue_positions)
            for t in (0, 19, 500, 999):
                slot = tape.slot(t)
                self.assertEqual(len(slot), len({r.ue for r in slot}))
                self.assertTrue(all(.4 <= r.deadline <= .95 for r in slot))
        self.assertEqual(tapes[2].counts.count(25), 500)
        self.assertEqual(tapes[2].counts.count(75), 500)

    def test_replay_and_no_rng_consumption(self):
        tape = Scenario(1, 'D')
        first = tape.slot(2)
        tape.slot(99)
        self.assertEqual(first, tape.slot(2))
        self.assertEqual(first, Scenario(1, 'D').slot(2))
        self.assertNotEqual(seed_for('dev', 0), seed_for('confirm', 0))

    def test_explicit_validation(self):
        for kind in ('', 'E'):
            with self.assertRaises(ValueError):
                Scenario(1, kind)
        with self.assertRaises(IndexError):
            Scenario(1, 'A').slot(1000)

    def test_cli_entrypoint_is_present(self):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/'scripts/analysis/dynamic_service_experiment.py'), '--help'],
                                capture_output=True, text=True, cwd=root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('evaluate', result.stdout)

    def test_seed_audit_checks_fields_and_hidden_smoke(self):
        import tempfile
        import json
        from pathlib import Path
        from scripts.analysis.dynamic_service_pipeline import audit_seeds
        candidate = seed_for('confirm/train', 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'metrics.json').write_text(json.dumps({'cycles': candidate}))
            self.assertEqual(audit_seeds([root])['status'], 'PASS')
            hidden = root/'.tmp/smoke/actual.json'
            hidden.parent.mkdir(parents=True)
            hidden.write_text(json.dumps({'training_seed': candidate}))
            result = audit_seeds([root])
            self.assertEqual(result['status'], 'REQUIRES_REVIEW')
            self.assertEqual(result['collisions'][0]['seeds'], [candidate])

    def test_seed_audit_handles_chunk_boundaries_and_lists(self):
        import tempfile
        import json
        from pathlib import Path
        from scripts.analysis.dynamic_service_pipeline import audit_seeds
        candidate = seed_for('confirm/train', 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Place the key across the 1 MiB read boundary; also exercise
            # uppercase keys and a list rather than a single seed value.
            raw = b'{"padding":"' + b'x'*(1024*1024-19) + b'","TRAINING_SEEDS":[' + str(candidate).encode() + b']}'
            json.loads(raw)
            (root/'large.json').write_bytes(raw)
            result = audit_seeds([root])
            self.assertEqual(result['status'], 'REQUIRES_REVIEW')
            self.assertEqual(result['collisions'][0]['seeds'], [candidate])


if __name__ == '__main__':
    unittest.main()
