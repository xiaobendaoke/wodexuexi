"""Expanded bank, absolute features, zero-residual and branch-freeze contracts."""
import copy
import os
import unittest


class AdapterTest(unittest.TestCase):
    def test_bank_fixed_across_allocation_and_fixed_scale_features(self):
        from tests.test_queued_service_learning import observation
        from utils.queued_long_value import LongAdapter,FEATURES
        from utils.queued_service_learning import VirtualQueue
        obs=observation();old=copy.deepcopy(obs);adapter=LongAdapter(True);adapter.prepare(obs)
        pre=adapter.motion(obs);v=VirtualQueue(obs)
        for t in obs['tasks']:
            if t['source'] is not None:v.assign(t['uid'],t['source'])
        post=adapter.motion(v.obs)
        self.assertEqual(pre['actions'],post['actions']);self.assertEqual(pre['mask'],post['mask'])
        self.assertEqual(len(pre['features'][0]),FEATURES);self.assertEqual(obs,old)
        self.assertTrue(all(x==0 for row,ok in zip(pre['features'],pre['mask']) if not ok for x in row))


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch')
class ModelTest(unittest.TestCase):
    def test_zero_residual_and_masked_q(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy
        from marl_models.queued_long_value import wrap,tensor
        from utils.queued_long_value import LongAdapter
        from tests.test_queued_service_learning import observation
        torch.manual_seed(11);old=ResidualPolicy('joint');new=wrap(copy.deepcopy(old));adapter=LongAdapter(True);obs=observation()
        row,_,_=new.act(obs,adapter,81)
        s=tensor([row['motion_state']]);f=tensor([row['motion_features']]);p=tensor([row['post_state']]);pf=tensor([row['post_features']]);mask=tensor([row['motion_mask']],torch.bool)
        with torch.no_grad():
            a=old.motion(s,f[:,:,:86],mask,p,pf[:,:,:86],True)
            b=new.motion(s,f,mask,p,pf,True)
        self.assertTrue(torch.allclose(a.probs,b.probs,atol=1e-7,rtol=0))
        self.assertEqual(float(b.probs.masked_select(~mask).sum()),0.)

    def test_motion_fit_preserves_allocation(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy
        from marl_models.queued_long_value import wrap,fit_values
        from utils.queued_long_value import LongAdapter
        from tests.test_queued_service_learning import observation
        m=wrap(ResidualPolicy('joint'));a=LongAdapter(True);obs=observation();row,_,_=m.act(obs,a,9)
        sample=dict(state=row['motion_state'],features=row['motion_features'],post_state=row['post_state'],post_features=row['post_features'],mask=row['motion_mask'],values=list(range(sum(row['motion_mask']))))
        before={k:v.clone() for k,v in m.allocation.state_dict().items()}
        fit_values(m,[sample],epochs=2)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.allocation.state_dict().items()))

    def test_branch_updates_obey_freezes_and_prefix_baseline(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy
        from marl_models.queued_long_value import wrap,prefix_values,update_branches
        from utils.queued_long_value import LongAdapter
        from tests.test_queued_service_learning import observation
        torch.manual_seed(31);m=wrap(ResidualPolicy('joint'));rows=[]
        for i in range(8):
            row,_,_=m.act(observation(),LongAdapter(True),i+19)
            with torch.no_grad():v,_,tokens=prefix_values(m,[row])
            for t,value in zip(tokens,v.tolist()):t['prefix_value']=value
            row['returns']=float(i+1);rows.append(row)
        for arm,frozen in (('motion','allocation'),('allocation','motion'),('joint',None)):
            z=update_branches(copy.deepcopy(m),rows,arm,actor_epochs=1,critic_epochs=1)
            if frozen:self.assertEqual(z['parameter_changes'][frozen],0)
            self.assertGreater(sum(z['actor_steps'].values()),0)
