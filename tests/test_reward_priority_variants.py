"""Reward-scale pilots change one upper reward setting, not the architecture."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import config
from run_canonical_experiment import METHODS, configure_method, parse_args, train
from utils.canonical_experiment import config_snapshot


class TestRewardPriorityVariants(unittest.TestCase):
    def tearDown(self):
        importlib.reload(config)

    def snapshot(self, method):
        importlib.reload(config)
        models = configure_method(method)
        return config_snapshot(), type(models["upper"]), type(models["lower"].actor), type(models["lower"].critic)

    def test_pilot_variants_change_exactly_one_reward_constant(self):
        baseline, upper, actor, critic = self.snapshot("lower_no_attention")
        cases = {
            "lower_mlp_upper_dsr_w5": ("REWARD_W_DSR", 5.0),
            "lower_mlp_upper_uncovered_obs": ("UPPER_UE_OBSERVATION_MODE", "covered_plus_sensed_uncovered"),
        }
        for method, (changed, value) in cases.items():
            with self.subTest(method=method):
                self.assertIn(method, METHODS)
                settings, candidate_upper, candidate_actor, candidate_critic = self.snapshot(method)
                self.assertEqual({key for key in settings if settings[key] != baseline[key]}, {changed})
                self.assertEqual(settings[changed], value)
                self.assertIs(candidate_upper, upper)
                self.assertIs(candidate_actor, actor)
                self.assertIs(candidate_critic, critic)

    def test_first_rollout_changes_only_the_expected_upper_reward_term(self):
        methods = ("lower_no_attention", "lower_mlp_upper_dsr_w5")
        with tempfile.TemporaryDirectory() as temporary:
            episodes = {}
            for method in methods:
                output = Path(temporary) / method
                args = parse_args(["--method", method, "--training-seed", "31", "--workload-seed", "37",
                                   "--episodes", "1", "--steps", "8", "--rollout", "8",
                                   "--ppo-epochs", "1", "--batch-size", "2", "--threads", "2",
                                   "--purpose", "verification", "--output", str(output)])
                train(args)
                episodes[method] = json.loads((output / "episodes/00000.json").read_text())
            baseline = episodes[methods[0]]["steps"]
            for method in methods[1:]:
                candidate = episodes[method]["steps"]
                for original, changed in zip(baseline, candidate):
                    self.assertEqual(original["metrics"], changed["metrics"])
                    metrics = original["metrics"]
                    expected = 4.0 * metrics["deadline_satisfaction_rate"]
                    self.assertAlmostEqual(changed["team_reward"] - original["team_reward"],
                                           expected * config.REWARD_SCALING_FACTOR, places=8)


if __name__ == "__main__":
    unittest.main()
