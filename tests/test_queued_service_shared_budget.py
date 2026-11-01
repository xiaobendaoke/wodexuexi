"""Resource-aware features and value scaling; Torch tests are server-only."""
import copy
import math
import os
import unittest
from environment.queued_service import Engine,Protocol,Task,safe_path
from scripts.analysis.queued_service_validation import START
from tests.test_queued_service_learning import observation
from utils.queued_service_learning import VirtualQueue
from utils.queued_service_shared_budget import BudgetAdapter,evaluate_candidate,mc_credit,MOTION_DIM


def transfer_observation(two=True):
    p=Protocol(capacity=(1e6,)*6,inter_bps=10e6,uplink_bps=100e6)
    e=Engine(p,START,[(100.,100.),(350.,350.)])
    e.add([Task(1,0,0.,3.,100e6,1e9)]+([Task(2,1,0.,3.,100e6,1e9)] if two else []));e.admit()
    e.assign({1:2,**({2:0} if two else {})})
    for task in e.tasks.values():task.served_upload=task.bits;task.remaining_upload=0.
    e._settle();return e.observe()


class BudgetFeatures(unittest.TestCase):
    def test_inter_pool_is_shared_not_repeated_per_link(self):
        one=evaluate_candidate(transfer_observation(False),[(0.,0.)]*5)
        two=evaluate_candidate(transfer_observation(True),[(0.,0.)]*5)
        self.assertAlmostEqual(one['transfer_by_source'][0],10e6)
        self.assertAlmostEqual(two['transfer_by_source'][0],5e6)
        self.assertAlmostEqual(sum(two['transfer_by_source']),10e6)
        self.assertLessEqual(two['airtime']['inter'],1.+1e-9)

    def test_forecast_does_not_mutate_public_observation(self):
        obs=transfer_observation();old=copy.deepcopy(obs)
        result=evaluate_candidate(obs,[(0.,0.)]*5)
        self.assertEqual(obs,old);self.assertEqual(result['generated'],2)
        self.assertEqual(result['audit']['status'],'PASS')

    def test_margin_retains_candidate_geometry_after_assignment(self):
        obs=transfer_observation();a=[(0.,0.)]*5;b=[(10.,0.)]+[(0.,0.)]*4
        x=evaluate_candidate(obs,a);y=evaluate_candidate(obs,b)
        self.assertNotEqual(x['inter_margin'],y['inter_margin'])

    def test_bank_is_same_safe_bank_and_new_dimensions(self):
        obs=observation();prefix=VirtualQueue(obs)
        prefix.assign(1,2);prefix.assign(2,2)
        adapter=BudgetAdapter();adapter.prepare(obs);bank=adapter.motion(prefix.obs)
        self.assertEqual(len(bank['features']),16);self.assertEqual(len(bank['features'][0]),MOTION_DIM)
        self.assertEqual(MOTION_DIM,86)
        self.assertEqual(adapter.forecast_calls,len(bank['actions']))
        self.assertTrue(all(safe_path(obs['positions'],a,obs['protocol']) for a in bank['actions']))

    def test_preallocation_defaults_are_explicit_source_targets(self):
        result=evaluate_candidate(observation(),[(0.,0.)]*5)
        self.assertEqual(result['default_source_assignments'],2)

    def test_contact_features_retain_initial_obligations_after_fast_completion(self):
        obs=observation();prefix=VirtualQueue(obs);prefix.assign(1,2);prefix.assign(2,2)
        result=evaluate_candidate(prefix.obs,[(0.,0.)]*5)
        self.assertGreater(result['inter_margin'],0)
        self.assertGreater(result['upload_margin'][0],0)

    def test_mc_has_no_bootstrap_tail_or_cross_episode(self):
        self.assertEqual(mc_credit([0.,0.,1.],[2.,2.,2.]),([-1.,-1.,-1.],[1.,1.,1.]))
        self.assertEqual(mc_credit([0.],[0.]),([0.],[0.]))
        with self.assertRaises(ValueError):mc_credit([1.],[])


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch checks')
class BudgetTorch(unittest.TestCase):
    def test_sample_replay_value_scale_and_both_gradients(self):
        import torch
        from marl_models.queued_service_shared_budget import BudgetPolicy,VALUE_SCALE
        for method in ('joint','decomposed'):
            torch.manual_seed(17001);model=BudgetPolicy(method)
            row,plan,motion=model.act(observation(),BudgetAdapter(),42);z=model.replay([row])
            self.assertAlmostEqual(float(z['allocation_log'][0]),row['allocation_log'],places=5)
            self.assertAlmostEqual(float(z['motion_log'][0]),row['motion_log'],places=5)
            self.assertAlmostEqual(float(z['value'][0])*VALUE_SCALE,row['value'],places=4)
            (-(z['allocation_log']+z['motion_log']).sum()).backward()
            for module in (model.allocation,model.motion):self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()),0)
            e=Engine.from_observation(observation());e.assign(plan);e.advance(motion);e.audit()

    def test_normalization_keeps_padding_out(self):
        import torch
        from marl_models.queued_service_shared_budget import normalized_candidates
        x=torch.tensor([[[1.,2.],[3.,4.],[999.,999.]]]);mask=torch.tensor([[True,True,False]])
        result=normalized_candidates(x,mask)
        self.assertTrue(torch.allclose(result[0,:2].mean(0),torch.zeros(2)))
        self.assertTrue(torch.equal(result[0,2],torch.zeros(2)))

    def test_normalized_critic_update_has_correct_raw_target_units(self):
        import torch
        from marl_models.queued_service_shared_budget import BudgetPolicy,budget_update
        torch.manual_seed(17001);model=BudgetPolicy('joint');rows=[]
        for i in range(4):
            row,_,_=model.act(observation(),BudgetAdapter(),i)
            row.update(advantage=float(i-1),returns=float((i+1)*20));rows.append(row)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        metrics=budget_update(model,optimizer,rows)
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.assertGreater(metrics['allocation_parameter_change'],0);self.assertGreater(metrics['motion_parameter_change'],0)
        self.assertEqual([r['returns'] for r in rows],[20.,40.,60.,80.])


if __name__=='__main__':unittest.main()
