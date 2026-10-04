"""Regression checks for the prespecified whole-system paired analysis."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis.whole_system_comparison import METRICS, analyze, holm_adjust


class WholeSystemComparisonTest(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        self.assertEqual(holm_adjust([.04, .01, .03]), [.06, .03, .06])

    def test_learned_and_nonlearned_use_distinct_replicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {'protocol': 'test', 'candidate_method': 'candidate',
                    'primary_comparators': ['other'], 'training_seeds': [11, 13, 17],
                    'workload_seeds': [101, 103],
                    'variants': [{'name': 'main', 'methods': ['candidate', 'other', 'greedy']}]}
            (root / 'suite.json').write_text(json.dumps({'plan': plan, 'source_hash': 'source'}))
            (root / 'status.json').write_text(json.dumps({'status': 'PASS'}))
            units = []
            for method in plan['variants'][0]['methods']:
                for seed in ([None] if method == 'greedy' else plan['training_seeds']):
                    for workload in plan['workload_seeds']:
                        dsr = {'candidate': .4, 'other': .2, 'greedy': .1}[method]
                        if method == 'candidate':
                            dsr += {11: 0.0, 13: .01, 17: -.01}[seed]
                        metrics = {metric: dsr for metric in METRICS}
                        units.append({'variant': 'main', 'method': method,
                                      'training_seed': seed, 'workload_seed': workload, **metrics})
            (root / 'analysis.json').write_text(json.dumps(
                {'status': 'PASS', 'source_hash': 'source', 'units': units}))
            result = analyze(root)
            self.assertTrue(result['decision_rule_met'])
            self.assertAlmostEqual(result['comparisons']['other']['metrics']['DSR']['mean_difference'], .2)
            self.assertAlmostEqual(result['comparisons']['greedy']['metrics']['DSR']['mean_difference'], .3)
            self.assertEqual(result['comparisons']['greedy']['kind'], 'nonlearned')
            self.assertTrue((root / 'whole_system_paired_comparison.json').exists())
            result['comparisons']['greedy']['metrics']['DSR']['holm_adjusted_p'] = 1
            self.assertNotIn('holm_adjusted_p', json.loads(
                (root / 'whole_system_paired_comparison.json').read_text())['comparisons']['greedy']['metrics']['DSR'])


if __name__ == '__main__':
    unittest.main()
