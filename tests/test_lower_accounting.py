import unittest
from unittest.mock import patch
import numpy as np
import config
from environment.env import Env
from environment.request_types import Request
from run_hierarchical_mappo_experiment import lower_rewards_from_metrics


class TestLowerAccounting(unittest.TestCase):
    def test_config_reload_does_not_restore_obsolete_capacity(self):
        from utils.logger import refresh_derived_config_fields
        refresh_derived_config_fields()
        self.assertEqual(config.MAX_OFFLOAD_REQUESTS_PER_UAV, config.NUM_UES)
        self.assertEqual(config.OFFLOAD_OBS_DIM_SINGLE, 5 + config.NUM_UES * config.OFFLOAD_REQUEST_FEATURE_DIM)

    def test_external_metrics_cannot_change_lower_reward_or_penalties(self):
        metrics = {"lower_admitted_service_count": 2., "lower_deadline_satisfaction_rate": .5,
                   "lower_service_latency": 3., "lower_assignment_energy": 10., "lower_mbs_load_ratio": .5,
                   "energy": 100., "latency": 100., "deadline_satisfaction_rate": .1,
                   "mbs_load_ratio": .1, "offloading_ratio_cooperative": 0.}
        first = lower_rewards_from_metrics(metrics, [0.] * config.NUM_UAVS, 2., 3.)
        metrics.update(energy=1e9, latency=1e9, deadline_satisfaction_rate=0., mbs_load_ratio=0., offloading_ratio_cooperative=1.)
        second = lower_rewards_from_metrics(metrics, [999.] * config.NUM_UAVS, 2., 3.)
        self.assertEqual(first, second)

    def test_empty_lower_domain_has_no_reward_or_constraint_violation(self):
        metrics = {"lower_admitted_service_count": 0., "lower_deadline_satisfaction_rate": 0.,
                   "lower_service_latency": 0., "lower_assignment_energy": 0., "lower_mbs_load_ratio": 0.}
        rewards, diag = lower_rewards_from_metrics(metrics, [0.] * config.NUM_UAVS, 3., 3.)
        self.assertEqual(rewards, [0.] * config.NUM_UAVS)
        self.assertTrue(all(value == 0 for value in diag.values()))

    def test_full_capacity_is_visible_and_executed_without_fallback(self):
        for count in (0, 1, 30, 50, 100):
            with self.subTest(count=count):
                np.random.seed(42)
                env = Env()
                env.reset()
                for uav in env.uavs:
                    uav._current_covered_ues = []
                for i, ue in enumerate(env.ues):
                    ue.current_request = Request.service(1000, 0, 1., 1) if i < count else Request.energy()
                    ue.assigned = i < count
                    if i < count:
                        ue.pos[:2] = env.uavs[0].pos[:2] + [i / 10 + 1, 0]
                        env.uavs[0]._current_covered_ues.append(ue)
                obs, masks = env.get_offloading_obs_and_masks()
                self.assertEqual(int((masks.sum(-1) > 0).sum()), count)
                _, _, metrics = env.step(np.zeros((config.NUM_UAVS, 2)), offloading_actions=np.zeros(masks.shape[:2], dtype=np.int64))
                self.assertEqual(metrics["service_learned_decision_count"], count)
                self.assertEqual(metrics["capacity_fallback_count"], 0)
                self.assertEqual(metrics["service_heuristic_decision_count"], 0)
                self.assertEqual(metrics["lower_admitted_service_count"], count)

    def test_no_quality_mask_preserves_hard_constraints_and_padding(self):
        np.random.seed(42)
        env = Env()
        env.reset()
        with patch.object(config, "OFFLOAD_MASK_MODE", "none"):
            obs, masks = env.get_offloading_obs_and_masks()
        components = env.last_offload_mask_components
        np.testing.assert_array_equal(masks, components["hard"])
        for i, uav in enumerate(env.uavs):
            self.assertFalse(masks[i, :, config.OFFLOAD_ACTION_COOP_BASE + uav.id].any())
            for target in set(range(config.NUM_UAVS)) - {n.id for n in uav.neighbors}:
                self.assertFalse(masks[i, :, config.OFFLOAD_ACTION_COOP_BASE + target].any())
        self.assertFalse(masks[~components["padding_valid"]].any())
