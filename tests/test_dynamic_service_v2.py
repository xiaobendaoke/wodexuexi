"""Second-release numerical regression checks; run on the server only."""
import unittest
import numpy as np
import torch

from environment.dynamic_service import evaluate, project_positions, prefix_batch
from environment.dynamic_service_v2 import DynamicServiceEnvV2, aggregate_demand, boundary_actions
from marl_models.dynamic_service_hmarl_v2 import UpperPolicyV2, LowerPolicyV2, update_lower_v2, update_upper_v2
from utils.dynamic_service_scenarios import Scenario


class UpperRevision(unittest.TestCase):
    def test_union_uses_only_public_records(self):
        env=DynamicServiceEnvV2(Scenario(81,'D'))
        raw=env.raw_observation()
        raw['visible'][:,-1]=False  # exercise a genuinely unavailable record
        shared=aggregate_demand(raw)
        union=raw['visible'].any(axis=0)
        np.testing.assert_array_equal(shared['visible'],np.broadcast_to(union,(5,100)))
        self.assertTrue((shared['ue'][:,~union]==0).all())
        for u in np.flatnonzero(union):
            witness=int(np.flatnonzero(raw['visible'][:,u])[0])
            for i in range(5):
                np.testing.assert_allclose(shared['ue'][i,u,2:],raw['ue'][witness,u,2:])
                np.testing.assert_allclose(shared['ue'][i,u,:2]+raw['nodes'][i,:2],
                                           raw['ue'][witness,u,:2]+raw['nodes'][witness,:2],atol=1e-6)
        corrupted={k:v.copy() for k,v in raw.items()}
        corrupted['ue'][:,~union]=999
        np.testing.assert_array_equal(aggregate_demand(corrupted)['ue'],shared['ue'])

    def test_boundary_mapping_moves_inward_and_replays_latent_probability(self):
        env=DynamicServiceEnvV2(Scenario(1,'D'))
        env.positions[0]=[50,50]
        policy=UpperPolicyV2();obs=env.observe();h=policy.initial_state()
        action=boundary_actions(np.zeros((5,2)),env.positions)
        self.assertTrue((action[0]>0).all())
        self.assertTrue((project_positions(env.positions,action)[0]>50).all())
        torch.manual_seed(3);sample=policy.act(obs,h,stochastic=True)
        dist,_,_=policy(obs,h)
        np.testing.assert_allclose(dist.log_prob(torch.tensor(sample['raw'])).sum(-1).detach(),sample['logprob'],atol=1e-5)
        np.testing.assert_allclose(sample['actions'],boundary_actions(sample['raw'],env.positions),atol=1e-6)
        for _ in range(30):
            a=boundary_actions(np.random.default_rng(_).normal(size=(5,2))*20,env.positions)
            next_pos=project_positions(env.positions,a)
            self.assertTrue((next_pos>=50-1e-3).all() and (next_pos<=650+1e-3).all())
            self.assertLessEqual(np.linalg.norm(next_pos-env.positions,axis=1).max(),15+1e-3)
            pair=np.linalg.norm(next_pos[:,None]-next_pos[None],axis=-1)+np.eye(5)*1000
            self.assertGreaterEqual(pair.min(),200-1e-3)

    def test_motion_feedback_and_current_service_causality(self):
        env=DynamicServiceEnvV2(Scenario(7,'D'));old=env.positions.copy();batch=env.batch()
        expected=evaluate(batch,batch.source)['success']
        row=env.step(np.ones((5,2)),batch.source)
        self.assertEqual(row['success'],expected)
        np.testing.assert_allclose(env.observe()['motion'][:,:2],(env.positions-old)/15,atol=1e-6)
        env.step(np.ones((5,2)),env.batch().source,decision_timeout=True)
        np.testing.assert_array_equal(env.observe()['motion'],0)
        self.assertFalse(DynamicServiceEnvV2(Scenario(7,'D')).observe()['motion'].any())


class LowerRevision(unittest.TestCase):
    def test_candidate_deadline_features_match_final_prefix_evaluator(self):
        env=DynamicServiceEnvV2(Scenario(3,'D'));batch=env.batch();policy=LowerPolicyV2()
        chosen=[]
        for i in range(min(8,len(batch))):
            features=policy.dynamic_features(batch,chosen,i).detach().numpy()
            for node in range(6):
                if not batch.legal[i,node]:continue
                result=evaluate(prefix_batch(batch,i+1),chosen+[node])
                self.assertAlmostEqual(float(features[node,4]),min(20,result['latencies'][i]/batch.deadline[i]),places=5)
                self.assertAlmostEqual(float(features[node,5]),np.clip(1-result['latencies'][i]/batch.deadline[i],-19,1),places=5)
                self.assertEqual(bool(features[node,10]),result['latencies'][i]>batch.deadline[i])
            chosen.append(int(batch.source[i]))
        parallel=LowerPolicyV2(mode='parallel')
        if len(batch)>2:
            np.testing.assert_array_equal(parallel.dynamic_features(batch,[0,0],2),parallel.dynamic_features(batch,[5,5],2))

    def test_masks_empty_batch_and_probability_replay(self):
        env=DynamicServiceEnvV2(Scenario(7,'D'));batch=env.batch();policy=LowerPolicyV2()
        batch.legal[:]=False
        batch.legal[np.arange(len(batch)),batch.source]=True
        sample=policy.act(batch,stochastic=True)
        np.testing.assert_array_equal(sample['assignment'],batch.source)
        log,value,_=policy.replay(batch,sample['assignment'])
        np.testing.assert_allclose(log.detach(),sample['logprob'],atol=1e-5)
        self.assertTrue(((value>=0)&(value<=1)).all())
        self.assertEqual(len(policy.act(prefix_batch(batch,0))['assignment']),0)

    def test_kl_guard_stops_before_an_excessive_update(self):
        env=DynamicServiceEnvV2(Scenario(81,'D'));batch=env.batch();policy=LowerPolicyV2()
        sample=policy.act(batch,stochastic=True);sample['logprob']-=10
        initial={k:v.clone() for k,v in policy.state_dict().items()}
        metrics=update_lower_v2(policy,torch.optim.Adam(policy.parameters(),lr=3e-4),
                    [dict(batch=batch,**sample,reward=.6)])
        self.assertTrue(metrics['stopped_kl']);self.assertEqual(metrics['optimizer_steps'],0)
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in policy.state_dict().items()))

    def test_vectorized_decoder_replays_the_same_prefix_states(self):
        env=DynamicServiceEnvV2(Scenario(81,'D'));batch=env.batch();policy=LowerPolicyV2()
        sample=policy.act(batch,True);encoded=policy.encode(batch);chosen=[];logs=[];values=[]
        for i,target in enumerate(sample['assignment']):
            dist,value=policy.distribution(batch,encoded,chosen,i)
            logs.append(dist.log_prob(torch.tensor(target)));values.append(value);chosen.append(int(target))
        log,value,_=policy.replay(batch,sample['assignment'])
        np.testing.assert_allclose(log.detach(),torch.stack(logs).detach(),atol=2e-6)
        np.testing.assert_allclose(value.detach(),torch.stack(values).detach(),atol=2e-6)

    def test_upper_and_lower_updates_are_finite_and_replay_after_save(self):
        import io
        env=DynamicServiceEnvV2(Scenario(81,'D'));upper=UpperPolicyV2();lower=LowerPolicyV2();hidden=upper.initial_state()
        rows=[];lower_rows=[]
        for _ in range(40):
            obs=env.observe();before=hidden.numpy().copy();batch=env.batch()
            a=upper.act(obs,hidden,True);b=lower.act(batch,True);hidden=a['hidden']
            out=env.step(a['actions'],b['assignment'])
            rows.append(dict(observation=obs,hidden_before=before,raw=a['raw'],logprob=a['logprob'],value=a['value'],reward=out['reward']))
            lower_rows.append(dict(batch=batch,**b,reward=out['success']/max(1,len(batch))))
        lm=update_lower_v2(lower,torch.optim.Adam(lower.parameters(),lr=3e-4),lower_rows[:8])
        um=update_upper_v2(upper,torch.optim.Adam(upper.parameters(),lr=3e-4),rows)
        self.assertGreater(lm['optimizer_steps'],0);self.assertGreater(um['optimizer_steps'],0)
        self.assertTrue(np.isfinite(lm['loss']) and np.isfinite(um['loss']))
        stream=io.BytesIO();torch.save(dict(upper=upper.state_dict(),lower=lower.state_dict(),rng=torch.get_rng_state()),stream);stream.seek(0)
        saved=torch.load(stream,weights_only=False);copy=UpperPolicyV2();copy.load_state_dict(saved['upper'])
        np.testing.assert_array_equal(upper.act(env.observe(),hidden)['actions'],copy.act(env.observe(),hidden)['actions'])


class RunnerRevision(unittest.TestCase):
    def test_resume_preserves_models_optimizers_and_rng_for_both_layers(self):
        import tempfile
        import subprocess
        import sys
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        def equal(a,b):
            if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in a:equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):equal(x,y)
            else:self.assertEqual(a,b)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            for phase in ('upper_probe','lower_probe'):
                def run(n,out):
                    p=subprocess.run([sys.executable,str(root/'scripts/analysis/dynamic_service_v2_experiment.py'),
                        'train','--split','smoke','--phase',phase,'--episodes',str(n),'--steps','8','--output',str(out)],
                        capture_output=True,text=True,cwd=root)
                    self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                run(2,base/phase/'continuous');run(1,base/phase/'resumed');run(2,base/phase/'resumed')
                a=torch.load(base/phase/'continuous/models/smoke/0'/f'{phase}.pt',weights_only=False)
                b=torch.load(base/phase/'resumed/models/smoke/0'/f'{phase}.pt',weights_only=False)
                for key in ('upper','lower','uopt','lopt','rng'):equal(a[key],b[key])


if __name__=='__main__':
    unittest.main()
