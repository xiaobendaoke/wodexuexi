import unittest
import numpy as np
import config
from environment.env import Env


class TestEnvTemporalAlignment(unittest.TestCase):
    """TEST-01: Environment Temporal Alignment
    
    Verifies that transition (s_t, a_t, r_t, s_{t+1}) applies current action a_t's
    displacement and collision/boundary resolution in the SAME step r_t and metrics_t.
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_first_step_displacement_accounted(self):
        actions = np.ones((config.NUM_UAVS, 2), dtype=np.float32)
        self.env.step(actions)
        for uav in self.env.uavs:
            self.assertGreater(
                uav._dist_moved, 0.0,
                f"UAV {uav.id} _dist_moved is 0.0 in step 0; current action a_0 displacement was not applied"
            )

    def test_zero_vs_max_movement_separation_at_step_0(self):
        np.random.seed(123)
        env_hover = Env()
        env_hover.reset()
        zero_actions = np.zeros((config.NUM_UAVS, 2), dtype=np.float32)
        env_hover.step(zero_actions)

        np.random.seed(123)
        env_fly = Env()
        env_fly.reset()
        max_actions = np.ones((config.NUM_UAVS, 2), dtype=np.float32)
        env_fly.step(max_actions)

        dist_hover = sum(u._dist_moved for u in env_hover.uavs)
        dist_fly = sum(u._dist_moved for u in env_fly.uavs)
        self.assertEqual(dist_hover, 0.0)
        self.assertGreater(dist_fly, 0.0)


if __name__ == "__main__":
    unittest.main()
