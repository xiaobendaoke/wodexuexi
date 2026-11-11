from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import config
from environment.env import Env
from utils.snapshot_evaluator import capture, evaluate


class TestSnapshotEvaluator(unittest.TestCase):
    def test_purity_and_live_equivalence(self):
        env = Env(scenario_seed=19)
        env.reset()
        before = deepcopy(env.uavs[0].cache)
        snapshot = capture(env)
        a = tuple(0 for _ in snapshot.requests)
        b = tuple(1 for _ in snapshot.requests)
        first = evaluate(snapshot, a)
        evaluate(snapshot, b)
        self.assertEqual(first, evaluate(snapshot, a))
        np.testing.assert_array_equal(before, env.uavs[0].cache)
        actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=int)
        with patch.object(env, '_get_obs', return_value=[]):
            _, _, metrics = env.step(np.zeros((config.NUM_UAVS, 2)), offloading_actions=actions)
        for name in ('lower_service_latency', 'service_requests_processed', 'deadline_satisfied_count'):
            self.assertAlmostEqual(first['metrics'][name], metrics[name], places=9)
        self.assertEqual(first['cache_after'], [u.cache.tolist() for u in env.uavs])

    def test_invalid_assignment_rejected(self):
        env = Env(scenario_seed=19)
        env.reset()
        snapshot = capture(env)
        with self.assertRaises(ValueError):
            evaluate(snapshot, [-1] * max(1, len(snapshot.requests)))
