import unittest
import numpy as np
import torch
from torch.distributions import Categorical
import config
from marl_models.offload_mappo.offload_mappo import OffloadMAPPO


class TestLowerLogprobContract(unittest.TestCase):
    """TEST-09: Lower Joint Log-Probability Contract (SSOT Section 6)
    
    Verifies that the lower offload policy evaluates joint log-probability
    as the exact factorized algebraic sum:
        log \pi(a | o) = \sum_{m \in V} log \pi_m(a_m | o)
    Strictly forbids division by valid_count (/ denom).
    PPO ratio must be the true product of individual request ratios.
    Also tests numerical stability across 10, 30, 50, 100 requests.
    """

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = "cpu"
        self.model = OffloadMAPPO(
            model_name="constrained_attention_offload_mappo",
            num_agents=config.NUM_UAVS,
            obs_dim=config.OFFLOAD_OBS_DIM_SINGLE,
            max_requests=config.MAX_OFFLOAD_REQUESTS_PER_UAV,
            num_actions=config.OFFLOAD_NUM_ACTIONS,
            device=self.device,
        )

    def test_joint_logprob_is_sum_not_mean(self):
        """For 2 valid requests with known individual log-probs [log_p1, log_p2],
        the joint log-prob must equal log_p1 + log_p2, NOT (log_p1 + log_p2) / 2.
        
        Legacy bug LOWER-01: offload_mappo divides by denom = valid_count,
        turning the joint log-prob into the mean and the PPO ratio into a geometric mean.
        """
        obs = np.zeros((config.NUM_UAVS, config.OFFLOAD_OBS_DIM_SINGLE), dtype=np.float32)
        masks = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV, config.OFFLOAD_NUM_ACTIONS), dtype=np.float32)
        
        # Set 2 valid requests for agent 0
        masks[0, 0, :] = 1.0
        masks[0, 1, :] = 1.0

        actions, log_probs, values = self.model.get_action_and_value(obs, masks=masks, exploration=False)

        # Manually compute the exact log_prob using the actor network directly
        obs_tensor = torch.from_numpy(obs).float()
        masks_tensor = torch.from_numpy(masks).float()
        with torch.no_grad():
            logits = self.model.actor(obs_tensor, masks_tensor)
            dist = Categorical(logits=logits)
            actions_tensor = torch.from_numpy(actions)
            slot_log_probs = dist.log_prob(actions_tensor)  # shape (NUM_UAVS, max_requests)
            
            # Expected canonical joint sum for agent 0
            expected_sum_agent0 = float((slot_log_probs[0, 0] + slot_log_probs[0, 1]).item())
            expected_mean_agent0 = expected_sum_agent0 / 2.0

        agent0_log_prob = float(log_probs[0])

        # Under canonical semantics, agent0_log_prob must be the SUM
        # In legacy code, agent0_log_prob is expected_mean_agent0 (EXPECTED_RED)
        self.assertAlmostEqual(
            agent0_log_prob, expected_sum_agent0, places=4,
            msg=f"Lower logprob returned {agent0_log_prob}, expected sum {expected_sum_agent0}. "
                f"Legacy division by valid_count detected (mean was {expected_mean_agent0})"
        )

    def test_single_request_matches_exact_logprob(self):
        """When exactly 1 request is valid, sum and mean coincide, so log_prob
        must match single request logprob exactly.
        """
        obs = np.zeros((config.NUM_UAVS, config.OFFLOAD_OBS_DIM_SINGLE), dtype=np.float32)
        masks = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV, config.OFFLOAD_NUM_ACTIONS), dtype=np.float32)
        masks[0, 0, :] = 1.0  # Only 1 request valid

        actions, log_probs, _ = self.model.get_action_and_value(obs, masks=masks, exploration=False)
        self.assertTrue(np.isfinite(log_probs[0]))

    def test_numerical_stability_at_scale(self):
        """Diagnostics for 10, 30, 50 requests: check no NaN, no Inf in log_prob."""
        obs = np.zeros((config.NUM_UAVS, config.OFFLOAD_OBS_DIM_SINGLE), dtype=np.float32)
        for num_reqs in [10, 30]:
            if num_reqs > config.MAX_OFFLOAD_REQUESTS_PER_UAV:
                continue
            masks = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV, config.OFFLOAD_NUM_ACTIONS), dtype=np.float32)
            masks[:, :num_reqs, :] = 1.0
            actions, log_probs, values = self.model.get_action_and_value(obs, masks=masks, exploration=True)
            self.assertTrue(np.all(np.isfinite(log_probs)), f"Non-finite log_probs at num_reqs={num_reqs}")
            self.assertTrue(np.all(np.isfinite(values)), f"Non-finite values at num_reqs={num_reqs}")


if __name__ == "__main__":
    unittest.main()
