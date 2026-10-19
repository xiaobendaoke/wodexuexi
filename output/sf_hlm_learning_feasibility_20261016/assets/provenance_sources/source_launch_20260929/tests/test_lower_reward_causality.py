import unittest
import copy
import numpy as np
import config
from environment.env import Env
from run_hierarchical_mappo_experiment import lower_rewards_from_metrics


class TestLowerRewardCausality(unittest.TestCase):
    """TEST-10: Lower Reward Causality (SSOT Section 7.2)
    
    Verifies that lower-layer policy reward is strictly invariant to upper-layer
    flight and hover motion energy:
        LOWER_REWARD_ENERGY = ASSIGNMENT_CAUSAL_NONFLIGHT
    Lower reward must only penalize/reward causal computing and communication
    energy resulting from offloading decisions, not flight or hover energy.
    """

    def setUp(self):
        np.random.seed(42)

    def test_lower_reward_invariant_to_uav_movement(self):
        """Execute identical service offloading decisions in two environments:
        Env 1: UAVs hover (displacement = 0)
        Env 2: UAVs fly at maximum speed (displacement = max_dist)
        
        Under canonical semantics, the lower offload reward must be identical.
        Legacy bug LOWER-02: lower_rewards_from_metrics includes metrics['energy']
        which contains flight and hover energy, penalizing offloading policy for
        trajectory flight movement.
        """
        # Env 1: Hover
        env_hover = Env()
        env_hover.reset()
        zero_actions = np.zeros((config.NUM_UAVS, 2), dtype=np.float32)
        offload_actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)

        # Clone the same pre-transition state, not next-slot requests or UE
        # references belonging to another environment.
        env_fly = copy.deepcopy(env_hover)
        _, system_rewards_hover, metrics_hover = env_hover.step(zero_actions, offloading_actions=offload_actions)
        lower_rewards_hover, _ = lower_rewards_from_metrics(metrics_hover, system_rewards_hover)

        max_actions = np.ones((config.NUM_UAVS, 2), dtype=np.float32)
        _, system_rewards_fly, metrics_fly = env_fly.step(max_actions, offloading_actions=offload_actions)
        lower_rewards_fly, _ = lower_rewards_from_metrics(metrics_fly, system_rewards_fly)

        self.assertNotEqual(metrics_hover["flight_energy"], metrics_fly["flight_energy"])

        # In canonical semantics:
        # Lower rewards evaluate strictly causal non-flight energy.
        # Flight motion from max_actions must NOT change lower_rewards.
        # Under legacy code, metrics_fly['energy'] is different due to flight/hover energy,
        # so lower_rewards_hover != lower_rewards_fly (EXPECTED_RED).
        self.assertAlmostEqual(
            lower_rewards_hover[0], lower_rewards_fly[0], places=3,
            msg=f"Lower reward polluted by flight motion! Hover reward: {lower_rewards_hover[0]}, Fly reward: {lower_rewards_fly[0]}"
        )


if __name__ == "__main__":
    unittest.main()
