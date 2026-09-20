import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_canonical_suite import paired_effect
from run_canonical_experiment import parse_args, train
from run_formal_suite import run_suite
from utils.formal_protocol import matrix, verify_release
from utils.canonical_experiment import source_manifest


class TestFormalProtocol(unittest.TestCase):
    def test_frozen_matrix_has_full_seeds_and_retraining_sensitivity(self):
        plan = matrix()
        self.assertEqual(plan['training_seeds'], [42, 84, 126, 168, 210])
        self.assertEqual(len(plan['workload_seeds']), 10)
        self.assertEqual(plan['eval_episodes'], 6)
        self.assertNotIn(plan['training_workload_seed'], plan['workload_seeds'])
        self.assertEqual(len(plan['variants']), 11)
        self.assertEqual({v['ue_count'] for v in plan['variants']}, {60, 80, 100, 120, 140})

    def test_formal_run_rejects_missing_release(self):
        with self.assertRaises(ValueError):
            verify_release(None, None, source_manifest())

    def test_uniform_counts_physical_execution_destinations(self):
        import config
        from marl_models.uniform_baseline.uniform_baseline import UniformBaseline
        model = UniformBaseline('uniform_baseline', config.NUM_UAVS, config.OBS_DIM_SINGLE, 2, 'cpu')
        masks = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV, config.OFFLOAD_NUM_ACTIONS))
        masks[0, 0, 0] = 1
        masks[1, 0, [0, 2]] = 1
        actions = model.get_offload_action(np.zeros((config.NUM_UAVS, config.OFFLOAD_OBS_DIM_SINGLE)), masks)
        self.assertEqual(actions[0, 0], 0)
        self.assertEqual(actions[1, 0], 0)

    def test_ci_uses_training_seed_units(self):
        effect = paired_effect(np.array([[1, 2, 3], [2, 3, 4], [0, 1, 2], [1, 2, 3], [2, 3, 4]]))
        self.assertEqual(effect['n_training_seeds'], 5)
        self.assertEqual(effect['n_workloads'], 3)
        self.assertLess(effect['ci95'][0], effect['mean_difference'])
        self.assertGreater(effect['ci95'][1], effect['mean_difference'])
        self.assertIsNone(paired_effect([[1, 2]])['ci95'])

    def test_sensitivity_capacity_140_trains_without_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--method', 'hierarchical', '--training-seed', '42', '--workload-seed', '294',
                               '--episodes', '1', '--steps', '2', '--rollout', '2', '--ppo-epochs', '1',
                               '--batch-size', '2', '--ue-count', '140', '--compute-scale', '0.6',
                               '--purpose', 'verification', '--output', tmp])
            self.assertEqual(train(args)['status'], 'PASS')

    def test_suite_trains_evaluates_aggregates_and_resumes(self):
        plan = matrix('main', smoke=True)
        plan['variants'][0]['methods'] = ['hierarchical', 'local']
        plan['workload_seeds'] = [294, 336]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'suite'
            run_suite(plan, output)
            first = (output / 'analysis.json').read_bytes()
            self.assertEqual(len(json.loads(first)['units']), 4)
            run_suite(plan, output)
            self.assertEqual(first, (output / 'analysis.json').read_bytes())
            result = next(output.glob('main/local/workload_*/attempt_*/episodes/*.json'))
            result.write_text('{}')
            with self.assertRaises(ValueError):
                run_suite(plan, output)
