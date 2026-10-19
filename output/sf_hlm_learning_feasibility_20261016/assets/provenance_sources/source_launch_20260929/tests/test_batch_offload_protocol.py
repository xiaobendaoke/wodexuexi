"""Lightweight, dependency-free protocol and promotion checks."""
import copy
import unittest
from scripts.analysis.batch_offload_protocol import A,B,C,D,make_plan,validate_plan,select_candidate


class ProtocolTests(unittest.TestCase):
    def test_sizes_and_disjoint_roles(self):
        pilot = make_plan(); confirmation = make_plan('confirmation',B)
        self.assertEqual(len(pilot['training_seeds'])*len(pilot['variants'][0]['methods']),20)
        self.assertEqual(len(confirmation['training_seeds'])*len(confirmation['variants'][0]['methods']),40)
        roles = lambda p: set(p['training_seeds']+p['workload_seeds']+list(p['training_workload_seeds'].values()))
        self.assertFalse(roles(pilot)&roles(confirmation))
        validate_plan(pilot)
        with self.assertRaises(ValueError): validate_plan(confirmation)
        confirmation['pilot_provenance'] = {'pilot_report_hash':'test'}
        validate_plan(confirmation)
        pilot['steps'] = 9
        with self.assertRaises(ValueError): validate_plan(pilot)

    def test_promotion_thresholds_and_tie(self):
        def row(delta):
            return {'metrics':{'DSR':{'mean_difference':delta,'positive_training_seeds':4},
                               'UAV Fleet Total Energy':{'candidate_mean':104.,'reference_mean':100.}}}
        data = {B:row(.011),C:row(.014)}; timing = {m:{'p95_seconds':.5} for m in (B,C)}
        self.assertEqual(select_candidate(data,timing),B)
        data[C]=row(.017)
        self.assertEqual(select_candidate(data,timing),C)
        timing[C]['p95_seconds']=1.
        self.assertEqual(select_candidate(data,timing),B)
        data[B]=row(.009)
        self.assertIsNone(select_candidate(data,timing))
        data[C]=row(.03); timing[C]['p95_seconds']=.5
        data[C]['metrics']['DSR']['positive_training_seeds']=3
        self.assertIsNone(select_candidate(data,timing))
