"""Collection/update agreement, padding exclusion, and checkpoint contracts."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Categorical

import config
from marl_models.offload_mappo.offload_mappo import OffloadMAPPO
from marl_models.joint_mappo.joint_mappo import JointMAPPO


class TestPolicyUpdateContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(4)

    def make_case(self, count, joint=False):
        dim = config.OFFLOAD_OBS_DIM_SINGLE + (config.OBS_DIM_SINGLE if joint else 0)
        cls = JointMAPPO if joint else OffloadMAPPO
        model = cls("joint_mappo" if joint else "offload_mappo", config.NUM_UAVS, dim, config.ACTION_DIM, "cpu")
        obs = np.zeros((config.NUM_UAVS, dim), dtype=np.float32)
        offset = config.OBS_DIM_SINGLE if joint else 0
        obs[:, offset + 5:].reshape(config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV, -1)[:, :count, 0] = 1
        masks = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV, config.OFFLOAD_NUM_ACTIONS), dtype=np.float32)
        masks[:, :count, :2] = 1
        return model, obs, masks

    def test_lower_collection_update_ratio_and_entropy_at_all_capacities(self):
        for count in (0, 1, 30, 50, 100):
            with self.subTest(count=count):
                model, obs, masks = self.make_case(count)
                actions, lp, values = model.get_action_and_value(obs, masks=masks)
                with torch.no_grad():
                    distribution = Categorical(logits=model.actor(torch.tensor(obs), torch.tensor(masks)))
                    valid = torch.tensor(masks.sum(-1) > 0)
                    expected = (distribution.log_prob(torch.tensor(actions)) * valid).sum(-1)
                    entropy = (distribution.entropy() * valid).sum(-1).mean().item()
                np.testing.assert_allclose(lp, expected.numpy(), rtol=1e-6, atol=1e-5)
                batch = {"obs": torch.tensor(obs[None]), "actions": torch.tensor(actions[None]),
                         "masks": torch.tensor(masks[None]), "valid_slots": valid[None].float(),
                         "old_log_probs": torch.tensor(lp[None]), "old_values": torch.tensor(values[None]),
                         "advantages": torch.ones((1, config.NUM_UAVS)), "returns": torch.tensor(values[None]) + 0.1}
                with patch.object(model.actor_optimizer, "step", wraps=model.actor_optimizer.step) as step:
                    losses = model.update(batch)
                    self.assertEqual(step.call_count, int(count > 0))
                self.assertTrue(all(np.isfinite(v) for v in losses.values()))
                self.assertAlmostEqual(losses["entropy"], entropy, places=4)
                if count:
                    self.assertAlmostEqual(losses["ratio_mean"], 1.0, places=5)

    def test_joint_returns_the_same_raw_action_used_for_logprob(self):
        model, obs, masks = self.make_case(50, joint=True)
        actions, offload, lp, values = model.get_action_and_value(obs, masks=masks)
        with torch.no_grad():
            motion, logits = model.actor(torch.tensor(obs), torch.tensor(masks))
            expected = motion.log_prob(torch.tensor(actions)).sum(-1)
            valid = torch.tensor(masks.sum(-1) > 0)
            expected += (Categorical(logits=logits).log_prob(torch.tensor(offload)) * valid).sum(-1)
        np.testing.assert_allclose(lp, expected.numpy(), atol=1e-5)
        batch = {"obs": torch.tensor(obs[None]), "trajectory_actions": torch.tensor(actions[None]),
                 "offload_actions": torch.tensor(offload[None]), "offload_masks": torch.tensor(masks[None]),
                 "valid_slots": valid[None].float(), "old_log_probs": torch.tensor(lp[None]),
                 "old_values": torch.tensor(values[None]), "advantages": torch.ones((1, config.NUM_UAVS)),
                 "returns": torch.tensor(values[None]) + .1}
        losses = model.update(batch)
        self.assertAlmostEqual(losses["ratio_mean"], 1.0, places=5)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))

    def test_legacy_metadata_is_rejected_even_if_shapes_match(self):
        model, _, _ = self.make_case(1)
        with tempfile.TemporaryDirectory() as folder:
            model.save(folder)
            path = folder + "/offload_mappo.pth"
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["metadata"].pop("probability_contract", None)
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                model.load(folder)
