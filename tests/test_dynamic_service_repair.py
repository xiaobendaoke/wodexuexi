"""Server numerical regression for the feasible-motion/current-credit repair."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
import unittest
import numpy as np
import torch
from environment.dynamic_service_v2 import DynamicServiceEnvV2
from environment.dynamic_service import project_positions,prefix_batch,evaluate
from utils.dynamic_service_scenarios import Scenario
from utils.dynamic_service_feasible_motion import motion_library
from marl_models.dynamic_service_repair import RepairedPolicy,RepairedCritic,update_repaired


class FeasibleMotionRegression(unittest.TestCase):
    def test_every_unmasked_plan_executes_without_projection_loss(self):
        env=DynamicServiceEnvV2(Scenario(81,'D'))
        positions=env.positions.copy()
        for step in range(8):
            env.positions=positions;bank=motion_library(env.observe())
            self.assertEqual(bank['actions'].shape,(257,5,2))
            self.assertTrue(bank['mask'][0]);self.assertTrue(np.isfinite(bank['features']).all())
            for index in np.flatnonzero(bank['mask']):
                action=bank['actions'][index];after=project_positions(positions,action)
                np.testing.assert_allclose(after-positions,action*15,atol=1e-3,rtol=0)
                self.assertLessEqual(np.linalg.norm(after-positions,axis=1).max(),15.001)
                self.assertTrue((after>=49.999).all() and (after<=650.001).all())
                distances=np.linalg.norm(after[:,None]-after[None],axis=-1)+np.eye(5)*1000
                self.assertGreaterEqual(distances.min(),199.999)
            indices=np.flatnonzero(bank['mask']);positions=project_positions(positions,bank['actions'][indices[(step*13+3)%len(indices)]])

    def test_unseen_request_features_do_not_enter_candidate_bank(self):
        env=DynamicServiceEnvV2(Scenario(81,'D'));obs=env.observe()
        obs['visible'][:,-1]=False;obs['ue'][:,-1]=0
        a=motion_library(obs);obs['ue'][:,-1]=999;b=motion_library(obs)
        np.testing.assert_array_equal(a['features'],b['features'])
        np.testing.assert_array_equal(a['mask'],b['mask'])


class RepairPolicyRegression(unittest.TestCase):
    def test_sampling_probability_replay_empty_batch_and_shared_initialization(self):
        env=DynamicServiceEnvV2(Scenario(81,'D'));obs=env.observe();batch=env.batch()
        for method in ('repair_joint','repair_full_credit','repair_no_plan','feasible_trajectory','local_allocation'):
            torch.manual_seed(123);actor=RepairedPolicy(method)
            fixed=batch.source if method=='feasible_trajectory' else None
            sample=actor.act(obs,batch,actor.initial_state(),seed=19,fixed_assignment=fixed)
            replay=actor.replay(obs,batch,actor.initial_state(),sample)
            np.testing.assert_allclose(replay['allocation_log'].detach(),sample['allocation_log'],atol=3e-6)
            np.testing.assert_allclose(replay['motion_log'].detach(),sample['motion_log'],atol=3e-6)
            self.assertTrue(batch.legal[np.arange(len(batch)),sample['assignment']].all())
            self.assertTrue(sample['bank']['mask'][sample['motion_index']])
            a=actor.act(obs,prefix_batch(batch,0),actor.initial_state(),seed=19,fixed_assignment=np.empty(0,np.int64) if fixed is not None else None)
            self.assertEqual(len(a['assignment']),0)
            b=actor.act(obs,batch,actor.initial_state(),seed=19,fixed_assignment=fixed)
            np.testing.assert_array_equal(sample['assignment'],b['assignment'])
            np.testing.assert_array_equal(sample['actions'],b['actions'])
        torch.manual_seed(9);a=RepairedPolicy('repair_joint')
        torch.manual_seed(9);b=RepairedPolicy('repair_full_credit')
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)

    def test_one_actor_optimizer_independent_critic_and_current_credit(self):
        actor=RepairedPolicy('repair_joint');critic=RepairedCritic()
        self.assertFalse({id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()})
        rows=[];env=DynamicServiceEnvV2(Scenario(81,'D'));ah=actor.initial_state();ch=critic.initial_state()
        for t in range(8):
            obs=env.observe();batch=env.batch();sample=actor.act(obs,batch,ah,seed=19+t)
            values=critic.predict(obs,batch,ch,sample,t)
            result=env.step(sample['actions'],sample['assignment'])
            rows.append(dict(observation=obs,batch=batch,hidden_before=ah.numpy().copy(),critic_hidden_before=ch.numpy().copy(),
                **sample,**values,reward=result['reward'],time=t))
            ah=torch.tensor(sample['hidden']);ch=torch.tensor(values['critic_hidden'])
        metrics=update_repaired(actor,critic,torch.optim.Adam(actor.parameters(),lr=1e-4),
            torch.optim.Adam(critic.parameters(),lr=3e-4),rows,sequence=8,epochs=1)
        self.assertGreater(metrics['actor_steps'],0);self.assertGreater(metrics['critic_steps'],0)
        self.assertTrue(np.isfinite(metrics['actor_loss']));self.assertEqual(metrics['allocation_credit'],'current_batch')
        self.assertLessEqual(metrics['max_accepted_conditional_kl'],.05+1e-6)


class RunnerRegression(unittest.TestCase):
    def test_resume_preserves_parameters_optimizers_and_rng(self):
        import tempfile
        import subprocess
        import sys
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        def equal(a,b):
            if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for key in a:equal(a[key],b[key])
            elif isinstance(a,(tuple,list)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):equal(x,y)
            else:self.assertEqual(a,b)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            def run(out,stop):
                p=subprocess.run([sys.executable,str(root/'scripts/analysis/dynamic_service_repair_experiment.py'),
                    'train','--split','smoke','--episodes','2','--stop-after',str(stop),'--steps','4',
                    '--method','repair_joint','--output',str(out)],capture_output=True,text=True,cwd=root)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            run(base/'continuous',2);run(base/'resumed',1);run(base/'resumed',2)
            a=torch.load(base/'continuous/models/smoke/0/repair_joint/latest.pt',weights_only=False)
            b=torch.load(base/'resumed/models/smoke/0/repair_joint/latest.pt',weights_only=False)
            for key in ('actor','critic','aopt','copt','rng'):equal(a[key],b[key])


if __name__=='__main__':unittest.main()
