"""Real rollout/GAE/update/checkpoint/resume tests on the remote server."""
import json
from pathlib import Path
import tempfile
import unittest
import torch

from run_canonical_experiment import parse_args, train


class TestCanonicalRunner(unittest.TestCase):
    def args(self, output, method="hierarchical", episodes=1, extra=()):
        return parse_args(["--method", method, "--training-seed", "7", "--workload-seed", "19",
                           "--episodes", str(episodes), "--steps", "5", "--rollout", "3",
                           "--ppo-epochs", "1", "--batch-size", "2", "--output", str(output),
                           "--purpose", "verification", *extra])

    def test_episode_boundary_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            full, resumed = Path(tmp) / "full", Path(tmp) / "resumed"
            train(self.args(full, episodes=2))
            train(self.args(resumed))
            train(self.args(resumed, episodes=2, extra=["--resume"]))
            for episode in range(2):
                left = json.loads((full / "episodes" / f"{episode:05d}.json").read_text())
                right = json.loads((resumed / "episodes" / f"{episode:05d}.json").read_text())
                self.assertEqual(left, right)
            for model, filename in (("upper", "attention_mappo.pth"), ("lower", "offload_mappo.pth")):
                left = torch.load(full / "checkpoints/episode_00001" / model / filename, weights_only=True)
                right = torch.load(resumed / "checkpoints/episode_00001" / model / filename, weights_only=True)
                for key in left["actor"]:
                    self.assertTrue(torch.equal(left["actor"][key], right["actor"][key]), key)

    def test_joint_training_and_separate_workload_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, evaluation = Path(tmp) / "joint", Path(tmp) / "evaluation"
            train(self.args(run, method="joint"))
            args = self.args(evaluation, method="joint", extra=["--evaluate-checkpoint", str(run / "checkpoints/episode_00000")])
            args.workload_seed = 20
            result = train(args)
            self.assertTrue(result["evaluation"])
            self.assertFalse((evaluation / "checkpoints").exists())
            with self.assertRaises(FileExistsError):
                train(args)
