"""Light contracts locally; frozen-model and matching checks only on Linux."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('feasibility_core', ROOT / 'utils/sf_hlm_feasibility.py')
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


class ScoringContracts(unittest.TestCase):
    def test_deadline_boundary_and_unadmitted_denominator(self):
        rows = [{'latency': .75, 'deadline': 1.}, {'latency': .8, 'deadline': 1.}]
        self.assertEqual(core.fee_count(rows, .25), 1)
        self.assertEqual(core.fee_count(rows, .26), 0)
        self.assertEqual(core.block_rate([{'generated': 5, 'success': 1}]), .2)

    def test_score_each_timing_before_averaging(self):
        rows = [{'latency': .9, 'deadline': 1.}]
        self.assertEqual(core.mean_fee_count(rows, [0., .3]), .5)
        self.assertEqual(core.fee_count(rows, .15), 0)

    def test_invalid_fees_and_request_times_are_rejected(self):
        for fee in [-1., math.nan, math.inf]:
            with self.assertRaises(ValueError): core.fee_count([], fee)
        for row in [{'latency': -1., 'deadline': 1.}, {'latency': .1, 'deadline': math.nan}]:
            with self.assertRaises(ValueError): core.fee_count([row], 0.)

    def test_empty_batch_is_retained(self):
        self.assertEqual(core.fee_count([], .1), 0)
        self.assertEqual(core.block_rate([{'generated': 2, 'success': 0}]), 0.)
        with self.assertRaises(ValueError): core.block_rate([{'generated': 0, 'success': 1}])


class DesignContracts(unittest.TestCase):
    def test_population_and_random_prefix_seed_pairing(self):
        identity = ['learned', 620001, 780001, 0, 250]
        specs = core.branch_specs(identity, 5)
        self.assertEqual(len(specs), 15)
        for repetition in range(5):
            pair = [q for q in specs if q['kind'] == 'random' and q['repeat'] == repetition]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]['seed'], pair[1]['seed'])
        self.assertEqual({q['method'] for q in specs}, {'G','L16','L8','R16','R8','H16','H8'})

    def test_partial_duplicate_or_foreign_population_is_rejected(self):
        expected = [('a', 1), ('b', 2)]
        core.require_population(expected, expected)
        for observed in [expected[:1], [expected[0], expected[0]], expected + [('c', 3)]]:
            with self.assertRaises(ValueError): core.require_population(observed, expected)

    def test_random_repetitions_are_averaged_not_selected(self):
        values = [0., 1., 0., 0., 0.]
        self.assertEqual(core.average_repeats(values), .2)
        with self.assertRaises(ValueError): core.average_repeats([])

    def test_block_rates_sum_counts_before_dividing(self):
        self.assertAlmostEqual(core.block_rate([{'generated': 1, 'success': 1},
                                               {'generated': 9, 'success': 0}]), .1)


if __name__ == '__main__': unittest.main()
