import unittest
from utils.assignment_search import best_improvement, exact_small


class TestAssignmentSearch(unittest.TestCase):
    def test_local_optimum_is_not_mislabeled_global(self):
        costs = {(0, 0): 2, (0, 1): 3, (1, 0): 3, (1, 1): 0}
        local = best_improvement([[0, 1], [0, 1]], [0, 0], costs.__getitem__)
        exact = exact_small([[0, 1], [0, 1]], costs.__getitem__)
        self.assertEqual(local['objective'], 2)
        self.assertTrue(local['local_optimality_verified'])
        self.assertEqual(exact['objective'], 0)
        self.assertEqual(exact['evaluations'], 4)

    def test_monotonicity_tie_break_and_budget(self):
        objective = lambda a: sum(a)
        result = best_improvement([[1, 0], [0, 1]], [1, 1], objective)
        self.assertEqual(result['trace'], [2, 1, 0])
        limited = best_improvement([[0, 1]], [1], objective, max_rounds=0)
        self.assertFalse(limited['local_optimality_verified'])
        with self.assertRaises(ValueError):
            exact_small([[0, 1]] * 20, objective, max_assignments=10)

    def test_nonfinite_cost_rejected(self):
        with self.assertRaises(ValueError):
            exact_small([[0]], lambda _: float('nan'))
