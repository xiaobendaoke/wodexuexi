"""Causal joint-policy regressions. Numerical cases run on the server."""
import copy
import os
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
import unittest
import numpy as np
import torch
from environment.dynamic_service import evaluate, prefix_batch
from environment.dynamic_service_v2 import DynamicServiceEnvV2
from utils.dynamic_service_scenarios import Scenario
from utils.dynamic_service_joint_rollout import causal_returns, slot_surrogate
from marl_models.dynamic_service_joint import JointPolicy, JointCritic, update_joint


class CausalReturns(unittest.TestCase):
    def test_only_physical_transitions_apply_lambda(self):
        for n in (0, 1, 25, 75):
            rows=[dict(reward=.2, prefix_values=np.zeros(n), motion_value=0.),
                  dict(reward=.3, prefix_values=np.zeros(2), motion_value=0.)]
            if not n: rows[0]['reward']=0.
            result=causal_returns(rows,lam=.5)
            self.assertAlmostEqual(result[0]['motion_target'],.15)
            np.testing.assert_allclose(result[0]['prefix_targets'],rows[0]['reward']+.15)
            self.assertEqual(result[1]['motion_target'],0.)
            np.testing.assert_allclose(result[1]['prefix_targets'],.3)

    def test_reward_before_movement_and_true_terminal(self):
        rows=[dict(reward=.7,prefix_values=np.zeros(3),motion_value=0.)]
        result=causal_returns(rows,lam=1.)
        np.testing.assert_allclose(result[0]['prefix_targets'],.7,atol=2e-8,rtol=0)
        self.assertEqual(result[0]['motion_target'],0.)
        with self.assertRaises(ValueError): causal_returns(rows,terminal=False)

    def test_nonzero_prefix_baselines_telescope_without_token_discount(self):
        for n in (1,25,75):
            rows=[dict(reward=.2,prefix_values=np.arange(n),motion_value=10.),
                  dict(reward=.3,prefix_values=np.array([3.,4.]),motion_value=8.)]
            result=causal_returns(rows,lam=.5)
            self.assertAlmostEqual(result[0]['motion_target'],1.65)
            np.testing.assert_allclose(result[0]['prefix_targets'],1.85)
            np.testing.assert_allclose(causal_returns(rows,allocation_only=True)[0]['prefix_targets'],.2)

    def test_surrogate_sums_requests_then_averages_slots(self):
        logs=torch.zeros(4,requires_grad=True)
        a=slot_surrogate(logs,torch.zeros(4),torch.ones(4))
        b=slot_surrogate(logs[:1],torch.zeros(1),torch.ones(1))
        self.assertAlmostEqual(float(a),4*float(b))


class JointRegression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.env=DynamicServiceEnvV2(Scenario(81,'D'))

    def test_prefix_replay_masks_empty_and_gru_once(self):
        for method in ('joint','joint_no_condition','joint_mlp'):
            actor=JointPolicy(method);obs=self.env.observe();batch=self.env.batch()
            hidden=actor.initial_state();sample=actor.act(obs,batch,hidden,True)
            replay=actor.replay(obs,batch,hidden,sample)
            np.testing.assert_allclose(replay['allocation_log'].detach(),sample['allocation_log'],atol=3e-6)
            np.testing.assert_allclose(replay['motion_log'].detach(),sample['motion_log'],atol=3e-6)
            np.testing.assert_allclose(replay['hidden'].detach(),sample['hidden'],atol=1e-6)
            self.assertTrue(batch.legal[np.arange(len(batch)),sample['assignment']].all())
            self.assertEqual(len(actor.act(obs,prefix_batch(batch,0),hidden)['assignment']),0)
            forced=copy.deepcopy(batch);forced.legal[:]=False
            forced.legal[np.arange(len(batch)),batch.source]=True
            np.testing.assert_array_equal(actor.act(obs,forced,hidden,True)['assignment'],batch.source)

    def test_no_condition_is_invariant_to_assignment(self):
        actor=JointPolicy('joint_no_condition');batch=self.env.batch()
        state=actor.encoder(self.env.observe(),actor.initial_state())
        a=actor.motion_distribution(state,batch,batch.source)
        b=actor.motion_distribution(state,batch,np.full(len(batch),5))
        torch.testing.assert_close(a.mean,b.mean)

    def test_both_branches_reach_shared_encoder_without_critic_overlap(self):
        actor=JointPolicy('joint');critic=JointCritic('joint')
        sample=actor.act(self.env.observe(),self.env.batch(),actor.initial_state(),True)
        replay=actor.replay(self.env.observe(),self.env.batch(),actor.initial_state(),sample)
        params=list(actor.encoder.parameters())
        for key in ('allocation_log','motion_log'):
            grads=torch.autograd.grad(replay[key].sum(),params,retain_graph=True,allow_unused=True)
            self.assertTrue(any(g is not None and g.abs().sum()>0 for g in grads))
        self.assertFalse({id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()})

    def test_safe_movement_and_current_reward_independence(self):
        actor=JointPolicy('joint');batch=self.env.batch();old=self.env.positions.copy()
        sample=actor.act(self.env.observe(),batch,actor.initial_state(),True)
        row=self.env.step(sample['actions'],sample['assignment'])
        self.assertEqual(row['success'],evaluate(batch,sample['assignment'])['success'])
        self.assertLessEqual(np.linalg.norm(self.env.positions-old,axis=1).max(),15.001)
        pair=np.linalg.norm(self.env.positions[:,None]-self.env.positions[None],axis=-1)+np.eye(5)*1000
        self.assertGreaterEqual(pair.min(),199.999)

    def test_update_and_optimizer_rng_resume(self):
        import io
        actor=JointPolicy('joint');critic=JointCritic('joint')
        ao=torch.optim.Adam(actor.parameters(),lr=3e-4);co=torch.optim.Adam(critic.parameters(),lr=3e-4)
        ah=actor.initial_state();ch=critic.initial_state();rows=[]
        for t in range(4):
            obs=self.env.observe();batch=self.env.batch();before=ah.numpy().copy();cb=ch.numpy().copy()
            sample=actor.act(obs,batch,ah,True);values=critic.predict(obs,batch,ch,sample['assignment'],t)
            row=self.env.step(sample['actions'],sample['assignment'])
            rows.append(dict(observation=obs,batch=batch,hidden_before=before,critic_hidden_before=cb,
                             **sample,**values,reward=row['reward'],time=t))
            ah=torch.tensor(sample['hidden']);ch=torch.tensor(values['critic_hidden'])
        metrics=update_joint(actor,critic,ao,co,rows,sequence=4,burn_in=0,epochs=1)
        self.assertGreater(metrics['actor_steps'],0);self.assertGreater(metrics['critic_steps'],0)
        self.assertTrue(np.isfinite(metrics['actor_loss']))
        stream=io.BytesIO();torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),
            ao=ao.state_dict(),co=co.state_dict(),rng=torch.get_rng_state()),stream);stream.seek(0)
        saved=torch.load(stream,weights_only=False)
        other=JointPolicy('joint');other.load_state_dict(saved['actor'])
        np.testing.assert_array_equal(actor.act(self.env.observe(),self.env.batch(),ah)['actions'],
                                      other.act(self.env.observe(),self.env.batch(),ah)['actions'])


class RunnerRegression(unittest.TestCase):
    def test_resume_preserves_actor_critic_optimizers_and_rng(self):
        import tempfile
        import subprocess
        import sys
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        def equal(a,b):
            if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in a:equal(a[k],b[k])
            elif isinstance(a,(tuple,list)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):equal(x,y)
            else:self.assertEqual(a,b)
        with tempfile.TemporaryDirectory() as directory:
            for method in ('joint','history_allocation','trajectory_matching'):
                base=Path(directory)/method
                def run(out,stop):
                    p=subprocess.run([sys.executable,str(root/'scripts/analysis/dynamic_service_joint_experiment.py'),
                        'train','--split','smoke','--episodes','2','--stop-after',str(stop),'--steps','4',
                        '--method',method,'--output',str(out)],capture_output=True,text=True,cwd=root)
                    self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                run(base/'continuous',2);run(base/'resumed',1);run(base/'resumed',2)
                a=torch.load(base/'continuous/models/smoke/0'/method/'latest.pt',weights_only=False)
                b=torch.load(base/'resumed/models/smoke/0'/method/'latest.pt',weights_only=False)
                for key in ('actor','critic','aopt','copt','rng'):equal(a[key],b[key])


if __name__=='__main__':unittest.main()
