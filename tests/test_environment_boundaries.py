"""Cache ownership and current-slot observation regressions."""
import unittest
from unittest.mock import patch

import numpy as np

import config
from environment import comm_model as comms
from environment.env import Env
from environment.request_types import Request


class TestEnvironmentBoundaries(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset(initial_positions=[
            [100, 100], [310, 100], [520, 100], [100, 500], [520, 500],
        ])
        for uav in self.env.uavs:
            uav._current_covered_ues = []
            uav.cache[:] = False
            uav.set_neighbors(self.env.uavs)
        ue = self.env.ues[0]
        ue.current_request = Request.service(1000, 0, 10.0, 1)
        ue.pos[:2] = self.env.uavs[0].pos[:2] + [10, 0]
        ue.assigned = True
        self.env.uavs[0]._current_covered_ues = [ue]

    def test_service_offload_does_not_create_untransferred_source_cache(self):
        for action in (config.OFFLOAD_ACTION_MBS, config.OFFLOAD_ACTION_COOP_BASE + 1):
            with self.subTest(action=action):
                self.setUp()
                for uav in self.env.uavs:
                    uav.cache[:] = False
                actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
                actions[0, 0] = action
                # Prevent next-slot request generation from replacing the fixture.
                with patch.object(self.env, "_get_obs", return_value=[]):
                    self.env.step(np.zeros((config.NUM_UAVS, 2)), offloading_actions=actions)
                self.assertFalse(self.env.uavs[0].cache[0], "Source never received this service image")
                if action != config.OFFLOAD_ACTION_MBS:
                    self.assertTrue(self.env.uavs[1].cache[0])

    def test_lower_context_uses_current_backhaul_after_movement(self):
        source = self.env.uavs[0]
        source._uav_mbs_rate = 1.0  # Deliberately stale previous-slot value.
        expected = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(source.pos, config.MBS_POS))
        captured = []
        original = source._build_service_offload_context

        def capture(*args, **kwargs):
            context, neighbor = original(*args, **kwargs)
            captured.append(context)
            return context, neighbor

        with patch.object(source, "_build_service_offload_context", side_effect=capture):
            obs, masks = self.env.get_offloading_obs_and_masks()
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(np.isfinite(masks).all())
        self.assertEqual(len(captured), 1)
        self.assertAlmostEqual(captured[0].uav_mbs_rate, expected, places=6)

    def test_periodic_cache_ties_use_file_id(self):
        source = self.env.uavs[0]
        source._ema_scores[:] = 0
        sizes = np.ones(config.NUM_FILES, dtype=np.int64)
        capacity = np.full(config.NUM_UAVS, 4, dtype=np.int64)
        with patch.object(config, "FILE_SIZES", sizes), patch.object(config, "UAV_STORAGE_CAPACITY", capacity):
            source.gdsf_cache_update()
        self.assertEqual(np.flatnonzero(source.cache).tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
