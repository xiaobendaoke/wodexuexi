"""Numerical/protocol checks; no local physical runs or Torch."""
import unittest
import numpy as np
from marl_models.queued_planner_distillation import initialize, loss_gradient
from utils.queued_planner_distillation import planner_target
from utils.queued_learning_validation import pack, statistics, case_grid, projection, phase, tiny_indices, label_stability


class ValidationTests(unittest.TestCase):
    def rows(self):
        rng = np.random.default_rng(12)
        return [dict(a=rng.normal(size=(k, 3)), m=rng.normal(size=(k, m, 2)), prior=np.ones(k)/k,
                     context=rng.normal(size=4), q=rng.normal(size=(k, m)), time=t,
                     split='training', workload=940000) for k, m, t in [(2, 3, 10), (1, 2, 60), (3, 1, 80)]]

    def test_vector_gradient_matches_scalar_ragged_both_modes(self):
        rows = self.rows();model = initialize(rows[0], hidden=4, seed=2)
        model['parameters']['allocation']['v'][:] = .2
        model['parameters']['motion']['v'][:] = -.1
        for mode in ('joint', 'decomposed'):
            batch = pack(rows, mode)
            stat, gradient = statistics(batch, model)
            ref = [loss_gradient(r, model, planner_target(r['q'], r['prior']), mode) for r in rows]
            self.assertAlmostEqual(stat['loss'], np.mean([r[0] for r in ref]))
            for name, params in gradient.items():
                for key, value in params.items():
                    np.testing.assert_allclose(value, np.mean([r[1][name][key] for r in ref], axis=0), atol=1e-12)

    def test_grid_predefined_and_balanced(self):
        grid = case_grid()
        self.assertEqual(len(grid), 27)
        self.assertEqual({g['range'] for g in grid}, {260., 360., 460.})
        self.assertEqual({g['sla'] for g in grid}, {'tight', 'normal', 'relaxed'})
        self.assertEqual({g['peak'] for g in grid}, {.45, .6, .75})

    def test_feature_projection_reproducible_no_global_rng_mutation(self):
        x = np.arange(12).reshape(3, 4)/12
        np.testing.assert_array_equal(projection(x, 8), projection(x, 8))
        self.assertEqual(projection(x, 8).shape, (3, 8))

    def test_tiny_subset_balances_phases_without_reading_scores(self):
        rows = [dict(time=t, workload=940000, q=object()) for t in range(0, 200, 2)]
        ids = tiny_indices(rows)
        self.assertEqual(len(ids), 32)
        self.assertEqual(sum(phase(rows[i]['time']) == 'peak' for i in ids), 16)

    def test_label_stability_detects_noisy_reversal_and_constant_labels(self):
        qs = np.array([[[3., 0.]], [[0., 3.]], [[0., 3.]], [[3., 0.]]])
        z = label_stability(qs, [1.])
        self.assertGreater(z['single_prediction_variance'], 0.)
        flat = label_stability(np.ones((4, 2, 3)), [.5, .5])
        self.assertEqual(flat['single_prediction_variance'], 0.)
        self.assertAlmostEqual(flat['split_selection_regret'], 0.)


if __name__ == '__main__':
    unittest.main()
