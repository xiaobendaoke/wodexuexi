"""Post-freeze adversarial checks; does not change the experiment contract."""
import copy
import os
import unittest


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch checks')
class ResidualAudit(unittest.TestCase):
    def test_nonconstant_residual_is_disabled_by_zero_gate(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy,tensor
        from tests.test_queued_service_learning import observation
        from utils.queued_service_shared_budget import BudgetAdapter
        torch.manual_seed(17001);model=ResidualPolicy('joint');r,_,_=model.act(observation(),BudgetAdapter(),4)
        s=tensor([r['motion_state']]);f=tensor([r['motion_features']]);mask=tensor([r['motion_mask']],torch.bool)
        ps=tensor([r['post_state']]);pf=tensor([r['post_features']])
        with torch.no_grad():
            model.motion.residual.score[-1].weight.fill_(2.)
            base=model.motion.base(s,f,mask).probs
            changed=model.motion(s,f,mask,ps,pf).probs
            self.assertGreater(float((base-changed).abs().max()),1e-6)
            model.motion.gate.zero_();bypass=model.motion(s,f,mask,ps,pf).probs
            self.assertTrue(torch.allclose(base,bypass,atol=1e-7,rtol=0))

    def test_tensor_pool_matches_public_summary_and_excludes_padding(self):
        import torch
        from marl_models.queued_service_plan_residual import pool
        from utils.queued_service_plan_residual import pooled_features
        rows=[[1.,4.],[3.,2.],[999.,999.]];mask=[True,True,False]
        self.assertEqual(pool(torch.tensor([rows]),torch.tensor([mask])).tolist()[0],pooled_features(rows,mask))

    def test_motion_baseline_changes_cannot_enter_allocation_gradient(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy,residual_update
        from tests.test_queued_service_learning import observation
        from utils.queued_service_shared_budget import BudgetAdapter
        torch.manual_seed(17001);a=ResidualPolicy('joint');b=copy.deepcopy(a);rows=[]
        for i in range(8):
            r,_,_=a.act(observation(),BudgetAdapter(),i);r.update(returns=10.+i,advantage=10.+i-r['value']);rows.append(r)
        altered=copy.deepcopy(rows)
        for i,r in enumerate(altered):r['motion_value']+=(-1 if i%2 else 1)*10
        for model,batch in ((a,rows),(b,altered)):
            torch.manual_seed(23);residual_update(model,torch.optim.SGD(model.parameters(),lr=.01),batch,epochs=1,minibatch=8)
        for p,q in zip(a.allocation.parameters(),b.allocation.parameters()):self.assertTrue(torch.equal(p,q))
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(a.motion.parameters(),b.motion.parameters())))


if __name__=='__main__':unittest.main()
