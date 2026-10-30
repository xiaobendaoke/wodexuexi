"""Run on the experiment server only (NumPy/SciPy/Torch)."""
import unittest
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
import numpy as np
import torch
from dataclasses import fields
from environment.dynamic_service import DynamicServiceEnv, evaluate, project_positions
from utils.dynamic_service_scenarios import Scenario


class RuntimeContract(unittest.TestCase):
    def test_checkpoint_resume_is_identical(self):
        import tempfile
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            def run(episodes, output):
                result = subprocess.run([sys.executable, str(root/'scripts/analysis/dynamic_service_experiment.py'),
                                'train', '--split', 'smoke', '--episodes', str(episodes), '--steps', '4',
                                '--output', str(output)], capture_output=True, text=True, cwd=root)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            run(2, base/'continuous')
            run(1, base/'resumed')
            run(2, base/'resumed')
            a = torch.load(base/'continuous/models/smoke/0/full.pt', weights_only=False)
            b = torch.load(base/'resumed/models/smoke/0/full.pt', weights_only=False)
            for layer in ('upper', 'lower'):
                for key in a[layer]:
                    self.assertTrue(torch.equal(a[layer][key], b[layer][key]), (layer, key))
            self.assertTrue(torch.equal(a['rng'], b['rng']))

    def test_policy_probability_replay_and_updates(self):
        from marl_models.dynamic_service_hmarl import UpperPolicy, LowerPolicy, update_lower, update_upper
        env = DynamicServiceEnv(Scenario(81, 'D'))
        upper, lower = UpperPolicy(), LowerPolicy()
        hidden = upper.initial_state()
        rollout = []
        for step_index in range(40):
            obs, batch = env.observe(), env.batch()
            u = upper.act(obs, hidden, stochastic=True)
            dist, _, replay_hidden = upper(obs, hidden)
            np.testing.assert_allclose(dist.log_prob(torch.tensor(u['raw'])).sum(-1).detach(), u['logprob'], atol=1e-5)
            l = lower.act(batch, stochastic=True)
            if len(batch):
                logs, _, _ = lower.replay(batch, l['assignment'])
                np.testing.assert_allclose(logs.detach(), l['logprob'], atol=1e-5)
                if step_index == 0:
                    loss = update_lower(lower, torch.optim.Adam(lower.parameters(), lr=3e-4),
                                        [dict(batch=batch, **l, reward=.5)])
                    self.assertTrue(np.isfinite(loss))
            stats = env.step(u['actions'], l['assignment'])
            rollout.append(dict(observation=obs, hidden_before=hidden.numpy(), raw=u['raw'],
                                logprob=u['logprob'], value=u['value'], reward=stats['reward']))
            hidden = u['hidden']
        self.assertTrue(np.isfinite(update_upper(upper, torch.optim.Adam(upper.parameters(), lr=3e-4), rollout)))
        self.assertEqual(float(upper.initial_state().abs().sum()), 0.)

    def test_empty_and_enumerated_assignment(self):
        from environment.dynamic_service import Batch, prefix_batch
        from utils.dynamic_service_baselines import conditional_match, matching_search, objective
        from marl_models.dynamic_service_hmarl import LowerPolicy
        import itertools
        env = DynamicServiceEnv(Scenario(7, 'A'))
        small = prefix_batch(env.batch(), 3)
        choices = [np.flatnonzero(row) for row in small.legal]
        candidates = list(itertools.product(*choices))
        best = min(objective(small, x) for x in candidates)
        matched = []
        for assignment in candidates:
            quotas = np.bincount(assignment, minlength=6)
            result = conditional_match(small, quotas)
            if result is not None:
                matched.append(objective(small, result))
        self.assertEqual(min(matched), best)
        empty = prefix_batch(small, 0)
        self.assertEqual(evaluate(empty, [])['success'], 0)
        self.assertEqual(len(matching_search(empty)), 0)
        self.assertEqual(len(LowerPolicy().act(empty)['assignment']), 0)
        perm = np.array([2, 0, 1])
        changed = Batch(*(getattr(small, f.name)[perm] if f.name in
                         ('ids', 'ue', 'source', 'size', 'cycles', 'deadline', 'tx', 'legal')
                         else getattr(small, f.name) for f in fields(Batch)))
        np.testing.assert_allclose(evaluate(small, small.source)['latencies'][perm],
                                   evaluate(changed, changed.source)['latencies'])

    def test_mbs_shares_final_batch(self):
        env = DynamicServiceEnv(Scenario(3, 'D'))
        batch = env.batch()
        if len(batch):
            assignment = np.full(len(batch), 5)
            result = evaluate(batch, assignment)
            expected = batch.tx[:, 5] + batch.cycles * len(batch) / batch.capacity[5]
            np.testing.assert_allclose(result['latencies'], expected)

    def test_statistics_and_rejection(self):
        from utils.dynamic_service_analysis import contrast, holm, analyze
        import tempfile
        import json
        from pathlib import Path
        self.assertEqual(holm([.01, .04, .03]), [.03, .06, .06])
        result = contrast([.01, .02, .03, .04, .05])
        self.assertAlmostEqual(result['mean'], .03)
        self.assertGreater(result['ci95'][0], 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root/'evaluation/dev/0/full/D_1.0_0_0.json'
            file.parent.mkdir(parents=True)
            file.write_text(json.dumps(dict(smoke=True, rows=[])))
            with self.assertRaises(ValueError):
                analyze(root, 'dev', primary_only=True)

    def test_baseline_information_and_soft_budget(self):
        from utils.dynamic_service_baselines import HistoryController, matching_search
        env = DynamicServiceEnv(Scenario(20, 'D'))
        controller = HistoryController()
        controller.update(env.observe(), env.positions)
        action = controller.mpc(env, env.batch(), budget=1.)
        self.assertTrue(np.isfinite(action).all())
        self.assertGreater(controller.last_mpc_candidates, 0)
        assignment = matching_search(env.batch())
        self.assertEqual(len(assignment), len(env.batch()))

    def test_request_order_and_validation(self):
        env = DynamicServiceEnv(Scenario(7, 'A'))
        batch = env.batch()
        result = evaluate(batch, batch.source)
        self.assertLessEqual(result['success'], len(batch))
        with self.assertRaises(ValueError):
            evaluate(batch, np.full(len(batch), 9))
        with self.assertRaises(ValueError):
            evaluate(batch, batch.source.astype(float) + .5)

    def test_causal_movement_and_tape(self):
        a, b = [DynamicServiceEnv(Scenario(19, 'D')) for _ in range(2)]
        before = a.positions.copy()
        ra = a.step(np.zeros((5, 2)), a.batch().source)
        rb = b.step(np.ones((5, 2)), b.batch().source)
        self.assertEqual(ra['success'], rb['success'])
        np.testing.assert_array_equal(before, a.positions)
        self.assertEqual(a.current_requests, b.current_requests)
        self.assertEqual(a.tape.fingerprint(), b.tape.fingerprint())

    def test_motion_is_safe(self):
        env = DynamicServiceEnv(Scenario(9, 'A'))
        old = env.positions.copy()
        new = project_positions(old, np.ones((5, 2)))
        self.assertTrue(np.all(np.linalg.norm(new-old, axis=1) <= 15.001))
        for i in range(5):
            for j in range(i):
                self.assertGreaterEqual(np.linalg.norm(new[i]-new[j]), 199.999)


if __name__ == '__main__':
    unittest.main()
