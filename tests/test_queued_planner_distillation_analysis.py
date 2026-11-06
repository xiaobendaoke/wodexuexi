import unittest
import numpy as np
from scripts.analysis.analyze_queued_planner_distillation import conditional_opportunity, teacher_projection


class OpportunityTests(unittest.TestCase):
    def test_correlated_optimal_ties_have_independent_projection_loss(self):
        z = teacher_projection([[3., 0.], [0., 3.]], [.5, .5])
        self.assertAlmostEqual(z['teacher_mutual_information'], np.log(2))
        self.assertAlmostEqual(z['independent_projection_score_regret'], 1.5)
        self.assertTrue(z['correlated_teacher'])

    def test_consistent_optimum_and_all_tied_teacher_need_no_correlation(self):
        for q in ([[4., 1.], [2., 0.]], [[1., 1.], [1., 1.]]):
            z = teacher_projection(q, [.3, .7])
            self.assertAlmostEqual(z['teacher_mutual_information'], 0.)
            self.assertAlmostEqual(z['independent_projection_score_regret'], 0.)

    def test_different_best_motions_have_positive_conditional_opportunity(self):
        z = conditional_opportunity([[3., 0.], [0., 3.]], [.5, .5])
        self.assertAlmostEqual(z['opportunity'], 1.5)
        self.assertAlmostEqual(z['interaction_variance_fraction'], 1.)

    def test_nonzero_interaction_need_not_change_best_motion(self):
        z = conditional_opportunity([[4., 0.], [2., 1.]], [.5, .5])
        self.assertGreater(z['interaction_variance_fraction'], 0.)
        self.assertEqual(z['opportunity'], 0.)

    def test_one_plan_or_additive_scores_do_not_need_conditioning(self):
        self.assertEqual(conditional_opportunity([[0., 4.]], [1.])['opportunity'], 0.)
        z = conditional_opportunity([[0., 1.], [2., 3.]], [.2, .8])
        self.assertAlmostEqual(z['opportunity'], 0.)
        self.assertAlmostEqual(z['interaction_variance_fraction'], 0.)

    def test_translation_and_candidate_permutations_do_not_change_opportunity(self):
        q = np.array([[3., 1.], [0., 5.]])
        a = conditional_opportunity(q, [.2, .8])
        b = conditional_opportunity((q+100)[::-1, ::-1], [.8, .2])
        self.assertAlmostEqual(a['opportunity'], b['opportunity'])
        with self.assertRaises(ValueError):
            conditional_opportunity(q, [.2, .2])


if __name__ == '__main__':
    unittest.main()
