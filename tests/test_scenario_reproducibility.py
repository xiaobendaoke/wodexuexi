import importlib
import unittest
import numpy as np
import config
from environment.env import Env


class TestScenarioReproducibility(unittest.TestCase):
    def test_configuration_reload_preserves_global_rng(self):
        np.random.seed(100)
        before = np.random.get_state()
        importlib.reload(config)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_interleaved_environments_ignore_policy_rng_and_other_populations(self):
        first = Env(scenario_seed=713)
        a = first.reset()
        # Another scenario must not overwrite the first environment's Zipf map.
        unrelated = Env(scenario_seed=999)
        unrelated.reset()
        second = Env(scenario_seed=713)
        b = second.reset()
        np.testing.assert_array_equal(a, b)
        for _ in range(10):
            a, ra, ma = first.step(np.zeros((config.NUM_UAVS, 2)))
            np.random.uniform(size=1000)
            unrelated.step(np.zeros((config.NUM_UAVS, 2)))
            b, rb, mb = second.step(np.zeros((config.NUM_UAVS, 2)))
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(ra, rb)
            self.assertEqual(ma, mb)

    def test_trajectory_differences_do_not_shift_latent_requests_or_mobility(self):
        first = Env(scenario_seed=27)
        second = Env(scenario_seed=27)
        first.reset()
        second.reset()
        for _ in range(8):
            first.step(np.zeros((config.NUM_UAVS, 2)))
            second.step(np.ones((config.NUM_UAVS, 2)))
            np.testing.assert_array_equal([u.pos for u in first.ues], [u.pos for u in second.ues])
            self.assertEqual([u.latent_request for u in first.ues], [u.latent_request for u in second.ues])
            self.assertEqual([u.current_request.is_service for u in first.ues], [u.current_request.is_service for u in second.ues])

    def test_hotspot_layouts_respect_configured_separation(self):
        for seed in range(20):
            env = Env(scenario_seed=seed)
            env.reset()
            centers = env._population["hotspot_centers"]
            for i, left in enumerate(centers):
                for right in centers[i + 1:]:
                    self.assertGreaterEqual(np.linalg.norm(left-right), config.HOTSPOT_SEPARATION)
