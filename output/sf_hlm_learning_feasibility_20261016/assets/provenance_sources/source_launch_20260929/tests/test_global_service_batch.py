"""Cross-UAV final-load regressions through the production Env.step path."""
import unittest
from unittest.mock import patch

import numpy as np

import config
from environment.env import Env
from environment.request_types import Request


class TestGlobalServiceBatch(unittest.TestCase):
    def run_batch(self, target_local=True, reverse=False):
        np.random.seed(42)
        env = Env()
        env.reset(initial_positions=[
            [100, 100], [310, 100], [520, 100], [100, 500], [520, 500],
        ])
        for uav in env.uavs:
            uav._current_covered_ues = []
            uav.cache[:] = True
            uav.set_neighbors(env.uavs)
        for ue in env.ues:
            ue.assigned = False
        for i in range(3):
            ue = env.ues[i]
            ue.pos[:2] = env.uavs[i].pos[:2] + [10, 0]
            ue.current_request = Request.service(50000, 0, 100.0, 1)
            ue.assigned = True
            env.uavs[i]._current_covered_ues = [ue]
        target = env.uavs[1]
        if reverse:
            env._uavs = [env.uavs[i] for i in (1, 0, 2, 3, 4)]
        actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        for index, uav in enumerate(env.uavs):
            if uav.id in (0, 2):
                actions[index, 0] = config.OFFLOAD_ACTION_COOP_BASE + 1
            elif uav.id == 1:
                actions[index, 0] = config.OFFLOAD_ACTION_LOCAL if target_local else config.OFFLOAD_ACTION_MBS
        # Keep the just-executed requests available for assertions; next-slot
        # request generation is covered independently by lifecycle tests.
        with patch.object(env, "_get_obs", return_value=[]):
            _, _, metrics = env.step(np.zeros((config.NUM_UAVS, 2)), offloading_actions=actions)
        return env, target, metrics

    def test_local_and_two_incoming_tasks_share_global_final_load(self):
        _, target, _ = self.run_batch()
        cycles = float(config.CPU_CYCLES_PER_BYTE[0]) * 50000
        expected = 3 * config.K_CPU * cycles * (float(config.UAV_COMPUTING_CAPACITY[1]) / 3) ** 2
        self.assertAlmostEqual(target.service_compute_energy, expected, places=7)

    def test_target_mbs_task_excluded_but_incoming_tasks_retained(self):
        _, target, _ = self.run_batch(target_local=False)
        cycles = float(config.CPU_CYCLES_PER_BYTE[0]) * 50000
        expected = 2 * config.K_CPU * cycles * (float(config.UAV_COMPUTING_CAPACITY[1]) / 2) ** 2
        self.assertAlmostEqual(target.service_compute_energy, expected, places=7)

    def test_global_uav_order_preserves_latency_and_energy(self):
        forward, _, first = self.run_batch()
        reverse, _, second = self.run_batch(reverse=True)
        np.testing.assert_allclose(
            [ue.latency_current_request for ue in forward.ues[:3]],
            [ue.latency_current_request for ue in reverse.ues[:3]],
            rtol=0, atol=1e-10,
        )
        for name in ("service_compute_energy", "service_comm_energy", "energy"):
            self.assertAlmostEqual(first[name], second[name], places=7)


if __name__ == "__main__":
    unittest.main()
