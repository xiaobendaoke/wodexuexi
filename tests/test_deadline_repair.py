"""Pure-Python tests; production environment tests run on remote Linux."""
from copy import deepcopy
import unittest
from utils.deadline_repair import assess, repair


def row(deadline=1., work=.4, source=0):
    return {'deadline': deadline, 'source': source, 'options': {
        0: {'target': source, 'base': 0., 'work': work, 'energy': 0., 'compute_energy': 1.},
        1: {'target': 'mbs', 'base': .9, 'work': 0., 'energy': 1., 'compute_energy': 0.}}}


class TestDeadlineRepair(unittest.TestCase):
    def test_empty_single_and_zero_budget(self):
        self.assertEqual(repair([], (), 'deadline', 32)['assignment'], ())
        self.assertEqual(repair([row()], (0,), 'ordinary', 0)['assignment'], (0,))
        self.assertEqual(assess([row()], (0,))['misses'], 0)

    def test_batch_externality_and_budget(self):
        rows = [row() for _ in range(3)]
        before = deepcopy(rows)
        self.assertEqual(assess(rows, (0, 0, 0))['misses'], 3)
        for order in ('ordinary', 'deadline'):
            result = repair(rows, (0, 0, 0), order, 32, energy_ratio=10.)
            self.assertEqual(result['predicted']['misses'], 0)
            self.assertLessEqual(result['evaluations'], 32)
            self.assertEqual(len(result['assignment']), 3)
        self.assertEqual(rows, before)

    def test_energy_guard_and_illegal_actions(self):
        rows = [row() for _ in range(3)]
        result = repair(rows, (0, 0, 0), 'deadline', 32, energy_ratio=1.)
        self.assertEqual(result['assignment'], (0, 0, 0))
        for a in [(0,), (0, 0, 5), (0, 0, True)]:
            with self.assertRaises(ValueError): assess(rows, a)

    def test_cross_source_shared_target(self):
        a, b = row(work=.6), row(work=.6, source=1)
        b['options'][2] = dict(b['options'][0], target=0)
        self.assertEqual(assess([a,b], (0,2))['misses'], 2)
        self.assertEqual(assess([a,b], (0,0))['misses'], 0)

    def test_full_capacity_deterministic(self):
        rows = [row(work=.04) for _ in range(30)]
        for mode in ('ordinary', 'deadline'):
            a = repair(rows, (0,)*30, mode, 32, energy_ratio=10.)
            b = repair(rows, (0,)*30, mode, 32, energy_ratio=10.)
            self.assertEqual(a, b)
            self.assertLessEqual(a['evaluations'], 32)


if __name__ == '__main__': unittest.main()
