"""Pre-input bypass, legal pooling and action-independent conditional credit."""
import copy
import os
import unittest
from utils.queued_service_plan_residual import pooled_features,phase_keys,branch_advantages


class ResidualLogic(unittest.TestCase):
    def test_pool_excludes_padding_and_retains_absolute_information(self):
        self.assertEqual(pooled_features([[1,4],[3,2],[999,999]],[True,True,False]),[2.,3.,1,2,3,4])
        with self.assertRaises(ValueError):pooled_features([[1]],[False])

    def test_phase_grouping_uses_public_time_not_episode_identity(self):
        rows=[{'state':[0.]*72+[x]+[0.]*7} for x in (.5,.5,.1)]
        self.assertEqual(phase_keys(rows),[100,100,180])

    def test_branches_use_separate_valid_baselines(self):
        a,m=branch_advantages([5,7],[1,2],[3,4])
        self.assertEqual(a,[4,5]);self.assertEqual(m,[2,3])
        with self.assertRaises(ValueError):branch_advantages([1],[1],[])


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch checks')
class ResidualTorch(unittest.TestCase):
    def test_bypass_and_native_pre_input_are_exact(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy,tensor
        from marl_models.queued_service_shared_budget import BudgetPolicy
        from utils.queued_service_shared_budget import BudgetAdapter
        from tests.test_queued_service_learning import observation
        torch.manual_seed(17001);old=BudgetPolicy('decomposed')
        torch.manual_seed(17001);new=ResidualPolicy('joint')
        obs=observation();a,_,_=old.act(obs,BudgetAdapter(),42);b,_,_=new.act(obs,BudgetAdapter(),42)
        self.assertTrue(torch.allclose(torch.tensor(a['motion_probs']),torch.tensor(b['motion_probs']),atol=1e-7,rtol=0));self.assertEqual(a['allocations'],b['allocations'])
        new.motion.gate.data.zero_()
        with torch.no_grad():new.motion.residual.score[-1].bias.fill_(1.)
        c,_,_=new.act(obs,BudgetAdapter(),42);self.assertTrue(torch.allclose(torch.tensor(a['motion_probs']),torch.tensor(c['motion_probs']),atol=1e-7,rtol=0))

    def test_decomposed_ignores_post_actor_but_value_can_condition(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy
        from utils.queued_service_shared_budget import BudgetAdapter
        from tests.test_queued_service_learning import observation
        torch.manual_seed(1);model=ResidualPolicy('decomposed');r,_,_=model.act(observation(),BudgetAdapter(),4)
        q=copy.deepcopy(r);q['post_state']=[v+.5 for v in q['post_state']]
        q['post_features']=[[v+.3 for v in x] for x in q['post_features']]
        a=model.replay([r]);b=model.replay([q]);self.assertTrue(torch.equal(a['motion_log'],b['motion_log']))
        with torch.no_grad():model.critic.plan[-1].weight.fill_(.1)
        a=model.replay([r]);b=model.replay([q]);self.assertTrue(torch.equal(a['value'],b['value']))
        self.assertFalse(torch.equal(a['motion_value'],b['motion_value']))

    def test_values_do_not_depend_on_chosen_motion_or_allocation_for_pre(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy
        from utils.queued_service_shared_budget import BudgetAdapter
        from tests.test_queued_service_learning import observation
        model=ResidualPolicy('joint');r,_,_=model.act(observation(),BudgetAdapter(),4);q=copy.deepcopy(r)
        q['motion_action']=(r['motion_action']+1)%sum(r['motion_mask']);a=model.replay([r]);b=model.replay([q])
        self.assertTrue(torch.equal(a['value'],b['value']));self.assertTrue(torch.equal(a['motion_value'],b['motion_value']))

    def test_phase_loss_removes_time_offset_and_replay_updates_both_branches(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy,residual_update,centered_loss
        from utils.queued_service_shared_budget import BudgetAdapter
        from tests.test_queued_service_learning import observation
        x=torch.tensor([1.,2.,10.,12.]);y=torch.tensor([2.,4.,20.,21.]);keys=[0,0,1,1]
        self.assertAlmostEqual(float(centered_loss(x,y,keys)),float(centered_loss(x+torch.tensor([9.,9.,-7.,-7.]),y,keys)))
        torch.manual_seed(17001);model=ResidualPolicy('joint');rows=[]
        for i in range(8):
            r,_,_=model.act(observation(),BudgetAdapter(),i);z=model.replay([r])
            self.assertAlmostEqual(float(z['motion_log'][0]),r['motion_log'],places=5)
            self.assertAlmostEqual(float(z['allocation_log'][0]),r['allocation_log'],places=5)
            r.update(advantage=10+i-r['value'],returns=10.+i);rows.append(r)
        metrics=residual_update(model,torch.optim.Adam(model.parameters(),lr=.0003),rows)
        self.assertGreater(metrics['motion_parameter_change'],0);self.assertGreater(metrics['allocation_parameter_change'],0)
        self.assertGreater(metrics['plan_parameter_change'],0)


if __name__=='__main__':unittest.main()
