"""Finite-return and frozen-allocation regressions for motion-only PPO."""
import copy
import math
import os
import unittest


class CreditTest(unittest.TestCase):
    def test_full_return_includes_drained_tail(self):
        from scripts.analysis.queued_service_motion_ppo import complete_credit
        rows=[dict(reward=x,motion_value=1.) for x in (.1,.2,.3)]
        complete_credit(rows)
        for row,want in zip(rows,(.6,.5,.3)):
            self.assertAlmostEqual(row['returns'],want)
            self.assertAlmostEqual(row['motion_advantage'],want-1.)


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch')
class UpdateTest(unittest.TestCase):
    def test_replay_and_motion_update_preserve_allocation(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy
        from utils.queued_service_shared_budget import BudgetAdapter
        from tests.test_queued_service_learning import observation
        from scripts.analysis.queued_service_motion_ppo import replay_motion,update
        torch.manual_seed(7);model=ResidualPolicy('joint')
        row,_,_=model.act(observation(),BudgetAdapter(),17)
        old=replay_motion(model,[row])
        self.assertAlmostEqual(float(old['log'][0]),row['motion_log'],places=5)
        n=sum(row['motion_mask']);self.assertGreater(n,1)
        rows=[]
        for i in range(8):
            r=copy.deepcopy(row);a=i%n;r.update(motion_action=a,motion_log=math.log(r['motion_probs'][a]),returns=float(a+1));rows.append(r)
        before={k:v.clone() for k,v in model.allocation.state_dict().items()}
        metrics=update(model,rows,epochs=2,batch=4)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.allocation.state_dict().items()))
        self.assertGreater(metrics['motion_change'],0)
        self.assertEqual(metrics['allocation_change'],0)
        self.assertGreater(metrics['actor_steps'],0)
