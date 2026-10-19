"""Architecture isolation for canonical lower-layer attention ablations."""
import unittest

import config
from marl_models.offload_mappo.offload_mappo import (
    OffloadActor,
    OffloadCritic,
    OffloadMLPActor,
    OffloadMLPCritic,
)
from run_canonical_experiment import configure_method


class TestLowerAttentionVariants(unittest.TestCase):
    def test_actor_and_critic_attention_are_independent(self):
        expected = {
            "hierarchical": (OffloadActor, OffloadCritic),
            "lower_actor_attention_only": (OffloadActor, OffloadMLPCritic),
            "lower_critic_attention_only": (OffloadMLPActor, OffloadCritic),
            "lower_no_attention": (OffloadMLPActor, OffloadMLPCritic),
        }
        for method, (actor_type, critic_type) in expected.items():
            with self.subTest(method=method):
                lower = configure_method(method)["lower"]
                self.assertIsInstance(lower.actor, actor_type)
                self.assertIsInstance(lower.critic, critic_type)
                self.assertEqual(config.OFFLOAD_MASK_MODE, "quality")
                self.assertEqual(config.OFFLOAD_CONSTRAINT_MODE, "adaptive_penalty")

    def test_lower_attention_dimensions_do_not_change_upper_dimensions(self):
        original_hidden = config.OFFLOAD_ATTN_HIDDEN_DIM
        original_heads = config.OFFLOAD_ATTN_NUM_HEADS
        upper_hidden = config.ATTN_HIDDEN_DIM
        upper_heads = config.ATTN_NUM_HEADS
        try:
            config.OFFLOAD_ATTN_HIDDEN_DIM = 32
            config.OFFLOAD_ATTN_NUM_HEADS = 4
            lower = configure_method("hierarchical")["lower"]
            self.assertEqual(lower.actor.request_attention.embed_dim, 32)
            self.assertEqual(lower.actor.request_attention.num_heads, 4)
            self.assertEqual(lower.critic.attn.embed_dim, 32)
            self.assertEqual(config.ATTN_HIDDEN_DIM, upper_hidden)
            self.assertEqual(config.ATTN_NUM_HEADS, upper_heads)
        finally:
            config.OFFLOAD_ATTN_HIDDEN_DIM = original_hidden
            config.OFFLOAD_ATTN_NUM_HEADS = original_heads

    def test_checkpoint_rejects_a_different_critic_architecture(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            full = configure_method("hierarchical")["lower"]
            full.save(directory)
            same = configure_method("hierarchical")["lower"]
            same.load(directory)
            actor_only = configure_method("lower_actor_attention_only")["lower"]
            with self.assertRaisesRegex(ValueError, "Incompatible or historical lower checkpoint"):
                actor_only.load(directory)


if __name__ == "__main__":
    unittest.main()
