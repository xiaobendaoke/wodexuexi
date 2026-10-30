"""Standard-library regression checks for experiment scheduling dependencies."""
import unittest
from pathlib import Path

from scripts.analysis.dynamic_service_resource_dispatch import build_jobs


METHODS = ('full', 'no_memory', 'no_feedback', 'parallel', 'independent',
           'greedy_lower', 'matching_lower', 'history', 'mpc')


class DispatchDependencies(unittest.TestCase):
    def test_full_waits_for_both_snapshot_controls(self):
        jobs = build_jobs(Path('results'), METHODS, primary_only=True)
        full = next(j for j in jobs if j['id'] == 'eval_dev_0_full_D_1.0_0_0')
        finished = {(0, 'full'), (0, 'independent')}
        self.assertFalse(all(dep in finished for dep in full['dependencies']))
        finished.add((0, 'parallel'))
        self.assertTrue(all(dep in finished for dep in full['dependencies']))

    def test_primary_scope_covers_every_method_block_and_workload(self):
        jobs = build_jobs(Path('results'), METHODS, primary_only=True)
        self.assertEqual(len(jobs), 1135)
        self.assertEqual(len({j['id'] for j in jobs}), len(jobs))
        evaluations = [j for j in jobs if j['command'] == 'evaluate']
        self.assertEqual(len(evaluations), 1110)
        self.assertTrue(all(j['kind'] == 'D' and j['scale'] == 1. for j in evaluations))
        for method in METHODS:
            for block in range(1 if method in ('history', 'mpc') else 5):
                actual = {(j['workload'], j['repeat']) for j in evaluations
                          if j['method'] == method and j['block'] == block}
                self.assertEqual(actual, {(w, e) for w in range(5) for e in range(6)})

    def test_full_grid_is_preserved_and_primary_evaluations_go_first(self):
        jobs = build_jobs(Path('results'), METHODS)
        self.assertEqual(len(jobs), 13345)
        evaluations = [j for j in jobs if j['command'] == 'evaluate']
        self.assertTrue(all(j['kind'] == 'D' and j['scale'] == 1.
                            for j in evaluations[:1110]))
        lookup = {j['id']: j for j in jobs}
        for job in jobs:
            for block, method in job['dependencies']:
                self.assertIn(f'train_dev_{block}_{method}', lookup)
        for block in range(5):
            parallel = lookup[f'train_dev_{block}_parallel']
            self.assertEqual(parallel['dependencies'], [(block, 'full')])

    def test_nonlearning_and_other_lower_controls_have_no_false_dependency(self):
        jobs = build_jobs(Path('results'), METHODS, primary_only=True)
        for job in jobs:
            if job['command'] != 'evaluate':
                continue
            if job['method'] in ('history', 'mpc'):
                self.assertEqual(job['dependencies'], [])
            elif job['method'] in ('greedy_lower', 'matching_lower'):
                self.assertEqual(job['dependencies'], [(job['block'], 'full')])


if __name__ == '__main__':
    unittest.main()
