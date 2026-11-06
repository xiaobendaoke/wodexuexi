"""Rollback includes Adam state; accepted proposals obey full-batch KL limits."""
import copy
import os
import unittest


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch')
class TrustTest(unittest.TestCase):
    def test_backtracking_commits_one_adam_step(self):
        import torch
        from marl_models.queued_motion_trust import guarded_step
        m=torch.nn.Linear(1,1,bias=False);m.weight.data.zero_();o=torch.optim.Adam(m.parameters(),lr=1.)
        (-m.weight.sum()).backward()
        d=guarded_step(m,o,lambda:float(m.weight.item())<=.3)
        self.assertTrue(d['accepted']);self.assertEqual(d['backtracks'],2)
        self.assertAlmostEqual(float(m.weight.item()),.25,places=5)
        self.assertEqual(int(o.state[m.weight]['step']),1);self.assertEqual(o.param_groups[0]['lr'],1.)

    def test_rejection_restores_parameters_and_adam(self):
        import torch
        from marl_models.queued_motion_trust import guarded_step
        m=torch.nn.Linear(1,1,bias=False);o=torch.optim.Adam(m.parameters(),lr=.1)
        m.weight.sum().backward();o.step();o.zero_grad();before=copy.deepcopy(o.state_dict());p=m.weight.detach().clone()
        m.weight.sum().backward();d=guarded_step(m,o,lambda:False,max_backtracks=3)
        self.assertFalse(d['accepted']);self.assertTrue(torch.equal(p,m.weight))
        after=o.state_dict();self.assertEqual(before['param_groups'],after['param_groups'])
        for k,v in before['state'][0].items():self.assertTrue(torch.equal(v,after['state'][0][k]))

    def test_full_rollout_limits_and_allocation_freeze(self):
        import torch
        from tests.test_queued_long_unfreeze import UnfreezeTest
        from marl_models.queued_motion_trust import update_motion
        f=UnfreezeTest();m=f.setup_model();rows=f.rows(m);before=copy.deepcopy(m.allocation.state_dict())
        d=update_motion(m,rows,actor_epochs=2,critic_epochs=1)
        self.assertGreater(d['actor_steps'],0);self.assertLessEqual(d['final_full_kl'],.020001)
        self.assertTrue(all(x['full_kl']<=.020001 and x['step_kl']<=.0002001 for x in d['accepted_trace']))
        self.assertTrue(all(torch.equal(v,m.allocation.state_dict()[k]) for k,v in before.items()))
        self.assertAlmostEqual(d['accepted_trace'][-1]['full_kl'],d['final_full_kl'],places=6)


if __name__=='__main__':unittest.main()
