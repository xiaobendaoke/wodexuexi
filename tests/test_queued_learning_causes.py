import copy
import unittest
from types import SimpleNamespace
import numpy as np
from marl_models.queued_planner_distillation import initialize
from utils.queued_learning_causes import prepared, loss_gradient, node_features, fit
from utils.queued_learning_validation import train_curve


class CauseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.row = dict(a=rng.normal(size=(2, 3)), m=rng.normal(size=(2, 3, 4)), context=np.array([.2]),
                        prior=np.array([.4, .6]), q=np.arange(6.).reshape(2, 3), time=60)
        self.forecast = np.tile(self.row['q'], (16, 1, 1))

    def test_rank_gradient(self):
        for mode in ('joint', 'decomposed'):
            data = prepared([self.row], [self.forecast], mode)
            model = initialize(self.row, hidden=3, seed=2)
            model['parameters']['motion']['v'][:] = [.1, -.2, .3]
            model['parameters']['allocation']['v'][:] = [.2, .1, -.1]
            for objective in ('hard', 'vote', 'rank'):
                _, grad = loss_gradient(data, model, objective)
                for branch in model['parameters']:
                    for name, values in model['parameters'][branch].items():
                        index = tuple(0 for _ in values.shape)
                        before = values[index];epsilon = 1e-5
                        values[index] = before+epsilon;a = loss_gradient(data, model, objective)[0]
                        values[index] = before-epsilon;b = loss_gradient(data, model, objective)[0]
                        values[index] = before
                        self.assertAlmostEqual(grad[branch][name][index], (a-b)/(2*epsilon), places=6)

    def test_shift_invariance_and_flat_rank(self):
        a = prepared([self.row], [self.forecast], 'joint')
        b = prepared([self.row], [self.forecast+100], 'joint')
        np.testing.assert_allclose(a['weights'], b['weights'])
        np.testing.assert_allclose(a['preferences'], b['preferences'])
        flat = prepared([self.row], [np.ones_like(self.forecast)], 'joint')
        loss, grad = loss_gradient(flat, initialize(self.row, hidden=3), 'rank')
        self.assertEqual(loss, 0.)
        self.assertTrue(all(np.all(g == 0) for branch in grad.values() for g in branch.values()))

    def test_forecasts_reject_nonfinite(self):
        bad = self.forecast.copy();bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):prepared([self.row], [bad], 'joint')

    def test_node_representation_complete_and_permutation_invariant(self):
        p = SimpleNamespace(capacity=(1., 1., 1., 1., 1., 2.), coverage=100., inter_range=460., mbs=(350., 350., 30.))
        tasks = [dict(uid=i, ue=0, source=0, target=None, deadline=62., remaining_cycles=1.,
                      remaining_upload=1., remaining_transfer=0., bits=1., stage='upload') for i in range(40)]
        obs = dict(protocol=p, positions=[(0., 0.)]*5, ues={0: (0., 0.)}, tasks=tasks, time=60)
        plan = {i: 0 for i in range(40)};motion = [(0., 0.)]*5
        first = node_features(obs, plan, motion)
        self.assertEqual(first.shape, (64,))
        obs['tasks'] = tasks[::-1]
        np.testing.assert_allclose(first, node_features(obs, plan, motion))
        self.assertAlmostEqual(np.sinh(first[-2]), .4)
        obs['tasks'] = tasks[:32]
        self.assertFalse(np.allclose(first, node_features(obs, plan, motion)))

    def test_vote_preserves_prediction_disagreement(self):
        qs = self.forecast.copy();qs[:8] = qs[:8, :, ::-1]
        data = prepared([self.row], [qs], 'joint')
        self.assertGreater(np.count_nonzero(data['votes']), 1)
        self.assertAlmostEqual(data['votes'].sum(), 1.)

    def test_hard_fit_exactly_preserves_existing_adam(self):
        row = dict(self.row, split='training', workload=1)
        hold = dict(self.row, split='holdout', workload=2)
        initial = initialize(row, hidden=3, seed=2)
        old, _ = train_curve([row], [hold], initial, 'joint', 16, (16,), 5)
        new, _ = fit([row], [self.forecast], [hold], [self.forecast], initial, 'joint', 'hard', 16, 5)
        for branch in old['parameters']:
            for key in old['parameters'][branch]:
                np.testing.assert_array_equal(old['parameters'][branch][key], new['parameters'][branch][key])


if __name__ == '__main__':unittest.main()
