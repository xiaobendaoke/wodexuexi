"""Remote production-adapter regression checks."""
import pickle
import unittest
import numpy as np
from environment.env import Env
from utils.snapshot_evaluator import capture, evaluate, AssignmentSnapshot
from utils.deadline_repair import snapshot_inputs, compare_snapshot


class TestRepairSnapshot(unittest.TestCase):
    def setUp(self):
        self.env=Env(scenario_seed=19);self.env.reset();self.snap=capture(self.env)

    def test_mask_and_purity_and_full_assignment(self):
        before=pickle.dumps(self.env,protocol=5)
        rng=pickle.dumps(np.random.get_state())
        rows=snapshot_inputs(self.snap)
        self.assertEqual(len(rows),len(self.snap.requests))
        initial=tuple(0 for _ in rows)
        baseline=evaluate(self.snap,initial)
        result=compare_snapshot(self.snap,initial,budgets=(4,))
        self.assertEqual(result['baseline_actual']['misses'],sum(not r['success'] for r in baseline['requests']))
        for proposal in result['comparisons'].values():
            self.assertEqual(len(proposal['assignment']),len(rows))
            for row,a in zip(rows,proposal['assignment']):self.assertIn(a,row['options'])
            self.assertLessEqual(proposal['evaluations'],4)
        self.assertEqual(before,pickle.dumps(self.env,protocol=5))
        self.assertEqual(rng,pickle.dumps(np.random.get_state()))

    def test_coop_prediction_ignores_neighbor_cache_truth(self):
        before=snapshot_inputs(self.snap)
        env=pickle.loads(self.snap.payload)
        for uav in env.uavs:
            uav.cache[:]=~uav.cache
        alt=AssignmentSnapshot(pickle.dumps(env,protocol=5),self.snap.config_hash,self.snap.requests,self.snap.candidates,self.snap.motion)
        after=snapshot_inputs(alt)
        checked=0
        for a,b in zip(before,after):
            for action,option in a['options'].items():
                # Compare common quality-legal targets; own-cache changes may
                # alter the quality mask, but may not leak target cache truth.
                if action>=2 and action in b['options']:
                    self.assertEqual(option,b['options'][action]);checked+=1
        self.assertGreater(checked,0,'Fixture must exercise cooperative predictions')


if __name__=='__main__':unittest.main()
