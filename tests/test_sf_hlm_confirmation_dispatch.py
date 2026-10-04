"""Pure scheduler regression; no simulation or Torch is imported."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paths import REPO_ROOT

ENTRY = Path(os.environ.get('SF_HLM_CONFIRMATION_DISPATCHER',
    REPO_ROOT / 'scripts/analysis/sf_hlm_confirmation_dispatch.py'))
spec = importlib.util.spec_from_file_location('confirmation_dispatch', ENTRY)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def protocol():
    tasks = [['dev_eval', 'learned', 1, 9, 0]]
    for b in (11, 12):
        tasks += [['foundation_training', 'shared', b, b + 100, -1],
            ['confirmation_original_training', 'original_adapted', b, b + 200, -1]]
        tasks += [['component_dataset', 'greedy', b, b + 200, e] for e in range(2)]
        tasks += [['component_merge', 'greedy', b, b + 200, -1],
            ['component_lower', 'learned', b, 0, -1]]
        tasks += [['confirmation_budget', 'frozen_capture', b, b + 200, e] for e in range(2)]
        tasks += [[phase, method, b, 300, 0] for phase in
            ('confirmation_eval', 'confirmation_timing') for method in
            (*d.BASE, 'bounded_feedback')]
        tasks += [['confirmation_residual_training', 'bounded_feedback', b, b + 100, -1]]
    return dict(tasks=tasks, confirmation_blocks=[11, 12], dataset_episodes=2,
        source_hash='source', smoke=False)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.p = protocol()
        self.graph = d.dependency_graph(self.p)

    def key(self, phase, block=11, method=None):
        return next(t for t in self.graph if t[0] == phase and t[2] == block
            and (method is None or t[1] == method))

    def test_registered_population_excludes_feedback_and_development(self):
        self.assertEqual(len(self.graph), 40)
        self.assertTrue(set(self.graph) <= {tuple(t) for t in self.p['tasks']})
        self.assertFalse(any(t[0].startswith('dev') or 'bounded' in t[1] for t in self.graph))

    def test_teacher_does_not_wait_for_original_adaptation(self):
        merge = self.key('component_merge')
        self.assertEqual(len(self.graph[merge]), 2)
        self.assertTrue(all(t[0] == 'component_dataset' for t in self.graph[merge]))
        self.assertNotIn(self.key('confirmation_original_training'), self.graph[merge])

    def test_learning_consumes_only_completed_teacher(self):
        lower = self.key('component_lower')
        self.assertEqual(self.graph[lower], frozenset([self.key('component_merge')]))
        ready = d.eligible(self.graph, set(), set())
        self.assertNotIn(lower, ready)
        self.assertEqual({t[0] for t in ready}, {'foundation_training'})

    def test_evaluation_uses_own_required_final_checkpoint(self):
        foundation = self.key('foundation_training')
        learned = self.key('confirmation_eval', method='learned')
        original = self.key('confirmation_eval', method='original_adapted')
        self.assertEqual(self.graph[learned], frozenset([foundation, self.key('component_lower')]))
        self.assertEqual(self.graph[original], frozenset([foundation, self.key('confirmation_original_training')]))
        greedy = self.key('confirmation_eval', method='greedy')
        self.assertEqual(self.graph[greedy], frozenset([foundation]))

    def test_timing_waits_for_all_shared_work_in_all_blocks(self):
        timing = self.key('confirmation_timing')
        shared = {t for t in self.graph if t[0] != 'confirmation_timing'}
        self.assertEqual(self.graph[timing], frozenset(shared))
        missing = self.key('confirmation_budget', block=12)
        self.assertNotIn(timing, d.eligible(self.graph, shared - {missing}, set()))
        self.assertIn(timing, d.eligible(self.graph, shared, set()))

    def test_timing_submissions_cannot_queue_beyond_isolated_slots(self):
        timing = [t for t in self.graph if t[0] == 'confirmation_timing']
        selected = d.submissions(timing[6:], timing[:6], 192, 8)
        self.assertEqual(selected, timing[6:8])
        self.assertEqual(d.submissions(timing[8:], timing[:8], 192, 8), [])

    def test_shared_submissions_retain_full_calibrated_capacity(self):
        shared = [('confirmation_eval', 'learned', 1, 2, e) for e in range(240)]
        self.assertEqual(d.submissions(shared, [], 192, 8), shared[:192])
        self.assertEqual(d.submissions(shared[190:], shared[:190], 192, 8), shared[190:192])

    def test_submission_limits_still_enforce_global_capacity(self):
        timing = [t for t in self.graph if t[0] == 'confirmation_timing']
        self.assertEqual(len(d.submissions(timing, [], 4, 8)), 4)
        with self.assertRaises(ValueError):
            d.submissions(timing, [], 192, 0)

    def test_active_and_failed_jobs_cannot_be_reissued(self):
        task = self.key('foundation_training')
        self.assertNotIn(task, d.eligible(self.graph, set(), {task}))
        self.assertEqual(d.eligible(self.graph, set(), set(), failed=True), [])

    def test_all_dependencies_are_registered(self):
        for deps in self.graph.values():
            self.assertTrue(deps <= self.graph.keys())

    def test_duplicate_task_is_rejected(self):
        self.p['tasks'].append(list(self.key('foundation_training')))
        with self.assertRaises(ValueError):
            d.dependency_graph(self.p)

    def test_receipt_paths_include_component_layout(self):
        a = argparse.Namespace(results_root=Path('/external/run'))
        dataset = self.key('component_dataset')
        self.assertEqual(d.result_folder(a, dataset), Path('/external/run/components/dataset_episode/complete/greedy/block_11_workload_211/episode_00000'))
        self.assertEqual(d.result_folder(a, self.key('component_lower')), Path('/external/run/components/lower_training/complete/learned/block_11'))
        self.assertEqual(d.result_folder(a, self.key('component_merge')), Path('/external/run/components/dataset/block_11'))

    def test_worker_arguments_preserve_registered_identity(self):
        a = argparse.Namespace(**{k: Path('/external') / k for k in d.PATH_ARGS})
        task = self.key('component_dataset')
        cmd = d.worker_command(a, task)
        self.assertEqual(cmd[2], 'worker')
        for k, v in zip(('phase', 'method', 'block', 'workload', 'episode'), task):
            self.assertEqual(cmd[cmd.index('--' + k) + 1], str(v))
        self.assertNotIn('--workers', cmd)

    def test_success_requires_valid_manifest_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = argparse.Namespace(results_root=Path(tmp))
            (a.results_root / 'protocol.json').write_text('{}')
            task = self.key('confirmation_eval', method='greedy')
            folder = d.result_folder(a, task)
            folder.mkdir(parents=True)
            (folder / 'data').write_text('original')
            contract = dict(zip(('phase', 'method', 'block', 'workload', 'episode'), task))
            contract.update(source_hash='source', followup_protocol_sha256=d.sha(a.results_root / 'protocol.json'))
            receipt = dict(status='PASS', contract=contract, files={'data': d.sha(folder / 'data')})
            (folder / 'result.json').write_text(json.dumps(receipt))
            self.assertTrue(d.verify_result(a, self.p, task))
            (folder / 'data').write_text('changed')
            with self.assertRaises(ValueError):
                d.verify_result(a, self.p, task)

    def test_partial_training_receipt_cannot_release_consumers(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = argparse.Namespace(results_root=Path(tmp))
            (a.results_root / 'protocol.json').write_text('{}')
            task = self.key('foundation_training')
            folder = d.result_folder(a, task)
            (folder / 'canonical/episodes').mkdir(parents=True)
            (folder / 'canonical/latest.json').write_text('{"completed_episodes": 2}')
            (folder / 'canonical/episodes/00000.json').write_text('{}')
            contract = dict(zip(('phase', 'method', 'block', 'workload', 'episode'), task))
            contract.update(source_hash='source', followup_protocol_sha256=d.sha(a.results_root / 'protocol.json'))
            (folder / 'result.json').write_text(json.dumps(dict(status='PASS', contract=contract,
                episodes=2, files={})))
            self.p['foundation_episodes'] = 2
            with self.assertRaises(ValueError):
                d.verify_result(a, self.p, task)

    def test_missing_receipt_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = argparse.Namespace(results_root=Path(tmp))
            self.assertFalse(d.verify_result(a, self.p, self.key('foundation_training')))

    def test_zero_exit_without_receipt_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = argparse.Namespace(**{k: Path(tmp) / k for k in d.PATH_ARGS})
            with patch.object(d.subprocess, 'run'):
                with self.assertRaises(RuntimeError):
                    d.execute(a, self.p, self.key('foundation_training'))

    def fake_process(self, root, pid, words):
        folder = root / str(pid)
        folder.mkdir()
        (folder / 'cmdline').write_bytes(('\0'.join(words) + '\0').encode())
        fields = ['S'] + ['0'] * 18 + [str(pid * 10)]
        (folder / 'stat').write_text(str(pid) + ' (test process) ' + ' '.join(fields))

    def test_duplicate_live_workers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = argparse.Namespace(**{k: Path(tmp) / k for k in d.PATH_ARGS})
            proc = Path(tmp) / 'proc'
            proc.mkdir()
            task = self.key('foundation_training')
            words = d.worker_command(a, task)
            self.fake_process(proc, 31, words)
            self.fake_process(proc, 32, words)
            with self.assertRaises(ValueError):
                d.discover_workers(a, self.graph, proc)

    def test_orphan_canonical_child_is_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = argparse.Namespace(**{k: Path(tmp) / k for k in d.PATH_ARGS})
            proc = Path(tmp) / 'proc'
            proc.mkdir()
            task = self.key('foundation_training')
            self.fake_process(proc, 33, ['python', str(a.source_root / 'run_canonical_experiment.py'),
                '--output', str(d.result_folder(a, task) / 'canonical')])
            found = d.discover_workers(a, self.graph, proc)
            self.assertEqual(found[task]['pid'], 33)
            self.assertTrue(found[task]['orphan_canonical'])
            self.fake_process(proc, 34, d.worker_command(a, task))
            found = d.discover_workers(a, self.graph, proc)
            self.assertEqual(found[task]['pid'], 34)
            self.assertFalse(found[task]['orphan_canonical'])


if __name__ == '__main__':
    unittest.main()
