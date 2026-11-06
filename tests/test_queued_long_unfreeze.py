"""Action-independent prefix credit and optimizer continuity across active branches."""
import copy
import os
import unittest


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch')
class UnfreezeTest(unittest.TestCase):
    def setup_model(self):
        import torch
        from marl_models.queued_service_plan_residual import ResidualPolicy
        from marl_models.queued_long_value import wrap
        torch.manual_seed(114);return wrap(ResidualPolicy('joint'))

    def rows(self,m):
        import torch
        from utils.queued_long_value import LongAdapter
        from tests.test_queued_service_learning import observation
        from marl_models.queued_long_value import prefix_values
        result=[]
        for i in range(8):
            row,_,_=m.act(observation(),LongAdapter(True),100+i)
            with torch.no_grad():v,_,tokens=prefix_values(m,[row])
            for t,value in zip(tokens,v.tolist()):t['prefix_value']=value
            row['returns']=float(i+1);result.append(row)
        return result

    def test_prefix_baseline_does_not_read_selected_action(self):
        import torch
        from marl_models.queued_long_value import prefix_values
        m=self.setup_model();rows=self.rows(m);other=copy.deepcopy(rows)
        for r in other:
            for t in r['allocations']:t['action']=5;t['probs']=[1/6]*6
        with torch.no_grad():a,_,_=prefix_values(m,rows);b,_,_=prefix_values(m,other)
        self.assertTrue(torch.equal(a,b))

    def test_optimizer_states_continue_after_fresh_branch_sampling(self):
        import torch
        from marl_models.queued_long_value import update_branches
        m=self.setup_model();a=update_branches(m,self.rows(m),'allocation',actor_epochs=1,critic_epochs=1)
        frozen={k:v.clone() for k,v in m.allocation.state_dict().items()}
        first=max(float(s['step']) for s in a['_optimizer_states']['critic']['state'].values())
        b=update_branches(m,self.rows(m),'motion',actor_epochs=1,critic_epochs=1,optimizer_states=a['_optimizer_states'])
        self.assertTrue(all(torch.equal(v,frozen[k]) for k,v in m.allocation.state_dict().items()))
        self.assertIn('allocation',b['_optimizer_states'])
        self.assertGreater(max(float(s['step']) for s in b['_optimizer_states']['critic']['state'].values()),first)
