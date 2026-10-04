"""An upper observation pilot may expose nearby unadmitted UEs without admitting them."""
import unittest

import numpy as np

import config
from environment.env import Env


class UpperUncoveredObservationTest(unittest.TestCase):
    def test_candidate_preserves_model_architectures(self):
        from run_canonical_experiment import configure_method

        baseline = configure_method("lower_no_attention")
        self.assertEqual(config.UPPER_UE_OBSERVATION_MODE, "covered")
        candidate = configure_method("lower_mlp_upper_uncovered_obs")
        try:
            self.assertEqual(config.UPPER_UE_OBSERVATION_MODE, "covered_plus_uncovered")
            self.assertEqual(type(baseline["upper"]), type(candidate["upper"]))
            self.assertEqual(type(baseline["lower"]), type(candidate["lower"]))
            self.assertEqual(config.OBS_DIM_SINGLE, baseline["upper"].obs_dim)
        finally:
            config.UPPER_UE_OBSERVATION_MODE = "covered"

    def test_fixed_actions_leave_environment_metrics_unchanged(self):
        original = config.UPPER_UE_OBSERVATION_MODE
        try:
            observed = []
            for mode in ("covered", "covered_plus_uncovered"):
                config.UPPER_UE_OBSERVATION_MODE = mode
                env = Env(scenario_seed=911)
                env.reset()
                _, _, metrics = env.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))
                observed.append({key: metrics[key] for key in (
                    "generated_service_count", "admitted_service_count",
                    "deadline_satisfied_count", "uav_fleet_total_energy",
                    "service_requests_uncovered", "service_offloads_mbs",
                )})
            self.assertEqual(*observed)
        finally:
            config.UPPER_UE_OBSERVATION_MODE = original

    def test_candidate_exposes_outside_coverage_target_only_to_upper_actor(self):
        env = Env(scenario_seed=907)
        env.reset()
        positions = [(50, 50), (650, 50), (50, 650), (650, 650), (350, 350)]
        for uav, xy in zip(env.uavs, positions):
            uav.pos[:2] = xy
        for ue in env.ues:
            ue.pos[:2] = (0, 0)
        env.ues[0].pos[:2] = (500, 350)  # 150 m from UAV 4: observable, not admitted
        env.ues[1].pos[:2] = (400, 350)  # 50 m from UAV 4: admitted
        for uav in env.uavs:
            uav.current_covered_ues.clear()

        original = config.UPPER_UE_OBSERVATION_MODE
        try:
            start = config.SELF_OBS_DIM + config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM
            config.UPPER_UE_OBSERVATION_MODE = "covered"
            baseline = env._get_obs()[4][start:].reshape(-1, config.UE_OBS_DIM)
            self.assertTrue(np.allclose(baseline[0, :2], [50 / 700, 0]))
            self.assertTrue(np.allclose(baseline[1], 0))
            self.assertFalse(env.ues[0].assigned)

            config.UPPER_UE_OBSERVATION_MODE = "covered_plus_uncovered"
            for uav in env.uavs:
                uav.current_covered_ues.clear()
            candidate = env._get_obs()[4][start:].reshape(-1, config.UE_OBS_DIM)
            self.assertTrue(np.allclose(candidate[0, :2], baseline[0, :2]))
            self.assertTrue(np.allclose(candidate[1, :2], [150 / 700, 0]))
            self.assertFalse(env.ues[0].assigned)
            self.assertEqual(len(env.uavs[4].current_covered_ues), 1)
            self.assertEqual(candidate.shape, baseline.shape)
        finally:
            config.UPPER_UE_OBSERVATION_MODE = original


if __name__ == "__main__":
    unittest.main()
