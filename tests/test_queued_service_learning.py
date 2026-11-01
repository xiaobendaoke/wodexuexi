"""Queue-policy contracts; Torch checks run only on the experiment server."""
import copy
import math
import os
import unittest

from environment.queued_service import Engine, Protocol, Task, safe_path
from scripts.analysis.queued_service_validation import START
from utils.queued_service_learning import LearningAdapter, VirtualQueue, state_features, finite_gae


def observation():
    engine = Engine(Protocol(), START, [(140., 140.), (390., 370.)])
    engine.add([Task(1, 0, 0., 2.5, 16e6, 12e9), Task(2, 1, 0., 2.8, 20e6, 18e9)])
    engine.admit()
    return engine.observe()


class LearningContract(unittest.TestCase):
    def test_prefix_does_not_mutate_observation(self):
        obs = observation(); before = copy.deepcopy(obs)
        prefix = VirtualQueue(obs); prefix.assign(1, 2)
        self.assertEqual(obs, before)
        self.assertNotEqual(state_features(obs, {}), state_features(prefix.obs, {}))

    def test_virtual_cooperation_records_future_transmission(self):
        obs = observation(); local = VirtualQueue(obs); coop = VirtualQueue(obs)
        local.assign(1, 0); coop.assign(1, 2)
        self.assertEqual(local.obs['tasks'][0]['remaining_transfer'], 0)
        self.assertEqual(coop.obs['tasks'][0]['remaining_transfer'], 16e6)
        self.assertEqual(coop.obs['tasks'][0]['stage'], 'upload')

    def test_allocation_masks_and_dimensions(self):
        prefix = VirtualQueue(observation())
        state, features, mask, prior = prefix.features(1, {})
        self.assertEqual(len(state), 80)
        self.assertEqual((len(features), len(features[0])), (6, 30))
        self.assertEqual(mask, [True, False, True, False, False, True])
        self.assertTrue(all(math.isfinite(x) for x in prior))
        self.assertTrue(prior[0] > prior[5])

    def test_illegal_and_repeated_prefix_are_rejected(self):
        prefix = VirtualQueue(observation())
        with self.assertRaises(ValueError): prefix.assign(1, 1)
        prefix.assign(1, 0)
        with self.assertRaises(ValueError): prefix.assign(1, 0)

    def test_candidates_are_safe_and_have_hover(self):
        obs = observation(); adapter = LearningAdapter(); adapter.prepare(obs)
        bank = adapter.motion(obs)
        self.assertEqual(len(bank['features']), 16)
        self.assertEqual(len(bank['features'][0]), 44)
        self.assertIn([(0., 0.)]*5, bank['actions'])
        for action in bank['actions']: self.assertTrue(safe_path(obs['positions'], action, obs['protocol']))

    def test_plan_changes_motion_features_but_preserves_bank(self):
        obs = observation(); adapter = LearningAdapter(); adapter.prepare(obs)
        prefix = VirtualQueue(obs); prefix.assign(1, 2); prefix.assign(2, 2)
        before = adapter.motion(obs); after = adapter.motion(prefix.obs)
        self.assertEqual(before['actions'], after['actions'])
        self.assertNotEqual(before['features'], after['features'])
        self.assertNotEqual(before['state'], after['state'])

    def test_future_completion_credits_earlier_allocation(self):
        adv, returns = finite_gae([0., 0., 1.], [0., 0., 0.])
        self.assertAlmostEqual(adv[0], .95**2)
        self.assertEqual(adv, returns)
        self.assertEqual(finite_gae([1.], [5.]), ([-4.], [1.]))

    def test_credit_does_not_cross_episodes(self):
        self.assertEqual(finite_gae([0.], [0.]), ([0.], [0.]))
        with self.assertRaises(ValueError): finite_gae([0.], [])

    def test_history_updates_once_per_physical_slot(self):
        obs = observation(); adapter = LearningAdapter(); adapter.prepare(obs)
        weights = dict(adapter.history.weights)
        with self.assertRaises(ValueError): adapter.prepare(obs)
        self.assertEqual(adapter.history.weights, weights)


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS') == '1', 'server-only Torch validation')
class TorchContract(unittest.TestCase):
    def test_sample_replay_and_both_future_gradients(self):
        import torch
        from marl_models.queued_service_joint import JointPolicy
        torch.manual_seed(17001)
        for method in ('joint', 'decomposed'):
            model = JointPolicy(method); adapter = LearningAdapter()
            row, plan, motion = model.act(observation(), adapter, 42)
            out = model.replay([row])
            self.assertAlmostEqual(out['allocation_log'][0].item(), row['allocation_log'], places=5)
            self.assertAlmostEqual(out['motion_log'][0].item(), row['motion_log'], places=5)
            loss = -(out['allocation_log']+out['motion_log']).sum()
            loss.backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.allocation.parameters()), 0)
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.motion.parameters()), 0)
            engine = Engine.from_observation(observation()); engine.assign(plan); engine.advance(motion); engine.audit()

    def test_motion_conditioning_and_critic_are_explicit(self):
        import torch
        from marl_models.queued_service_joint import JointPolicy
        torch.manual_seed(17001)
        for method in ('joint', 'decomposed'):
            model = JointPolicy(method); obs = observation()
            row, plan, _ = model.act(obs, LearningAdapter(), 42)
            prefix = VirtualQueue(obs)
            for uid, node in plan.items(): prefix.assign(uid, node)
            expected = prefix.obs if method == 'joint' else obs
            weights = row['history_weights']
            self.assertEqual(row['state'], state_features(obs, weights))
            self.assertEqual(row['motion_state'], state_features(expected, weights))

    def test_ppo_update_and_empty_allocation(self):
        import torch
        from marl_models.queued_service_joint import JointPolicy, update
        torch.manual_seed(17001)
        model = JointPolicy('joint'); obs = observation(); obs['tasks'] = []
        row, plan, _ = model.act(obs, LearningAdapter(), 42)
        self.assertEqual(plan, {})
        replay = model.replay([row]); self.assertEqual(replay['allocation_log'][0].item(), 0)
        rows = []
        for seed in range(4):
            r, _, _ = model.act(observation(), LearningAdapter(), seed)
            r.update(advantage=float(seed-1), returns=float(seed+1)); rows.append(r)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        before = {k: p.clone() for k, p in model.state_dict().items()}
        metrics = update(model, optimizer, rows)
        self.assertGreater(metrics['optimizer_steps'], 0)
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(p, before[k]) for k, p in model.state_dict().items() if k.startswith('allocation.')))
        self.assertTrue(any(not torch.equal(p, before[k]) for k, p in model.state_dict().items() if k.startswith('motion.')))


if __name__ == '__main__': unittest.main()
