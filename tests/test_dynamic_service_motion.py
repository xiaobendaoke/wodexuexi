"""Server numerical contracts for the isolated motion-learning short trial."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
import unittest
import numpy as np
import torch
from utils.dynamic_service_motion_credit import motion_gae,public_service_scores,label_indices
from marl_models.dynamic_service_motion import MotionPolicy,MotionValue,update_motion
from environment.dynamic_service_v2 import DynamicServiceEnvV2
from utils.dynamic_service_scenarios import Scenario
from utils.dynamic_service_feasible_motion import motion_library


class MotionCredit(unittest.TestCase):
    def test_reward_is_shifted_and_terminal_motion_has_no_credit(self):
        a,r=motion_gae([99,2,3],[0,0,0],gamma=1,lam=1)
        np.testing.assert_array_equal(a,[5,3]);np.testing.assert_array_equal(r,[5,3])
        b,_=motion_gae([-99,2,3],[0,0,0],gamma=1,lam=1)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(motion_gae([1],[0])[0]),0)

    def test_labels_use_only_public_records_and_are_reproducible(self):
        obs=DynamicServiceEnvV2(Scenario(81,'D')).observe();obs['visible'][:,-1]=False;obs['ue'][:,-1]=0
        bank=motion_library(obs);ix=label_indices(bank,0,19)
        a=public_service_scores(obs,bank,ix)
        obs['ue'][:,-1]=999;b=public_service_scores(obs,bank,ix)
        np.testing.assert_array_equal(a,b);self.assertTrue(bank['mask'][ix].all())
        np.testing.assert_array_equal(ix,label_indices(bank,0,19))
        self.assertTrue((a>=0).all())


class MotionLearning(unittest.TestCase):
    def test_sampling_replay_mask_empty_and_identical_initialization(self):
        env=DynamicServiceEnvV2(Scenario(81,'D'));obs=env.observe()
        torch.manual_seed(19);a=MotionPolicy();torch.manual_seed(19);b=MotionPolicy()
        for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],atol=0,rtol=0)
        s=a.act(obs,a.initial_state(),19);z=a.replay(obs,a.initial_state(),s)
        np.testing.assert_allclose(z['log'].detach(),s['log'],atol=3e-6)
        self.assertTrue(s['bank']['mask'][s['index']]);self.assertTrue(np.isfinite(s['probs']).all())
        obs['ue'][:,:,2:]=0
        self.assertTrue(np.isfinite(a.act(obs,a.initial_state(),19)['probs']).all())

    def test_ppo_and_auxiliary_update_reach_motion_and_encoder(self):
        from utils.dynamic_service_baselines import matching_search
        for weight in (0.,.5):
            torch.manual_seed(19);actor=MotionPolicy();critic=MotionValue();env=DynamicServiceEnvV2(Scenario(81,'D'))
            ah=actor.initial_state();ch=critic.initial_state();rows=[]
            for t in range(8):
                obs=env.observe();s=actor.act(obs,ah,19+t);v,newch=critic.predict(obs,ch,t)
                ix=label_indices(s['bank'],s['index'],19+t);scores=public_service_scores(obs,s['bank'],ix)
                r=env.step(s['actions'],matching_search(env.batch()))
                rows.append(dict(obs=obs,hidden_before=ah.numpy().copy(),critic_hidden_before=ch.numpy().copy(),
                    **s,value=v,reward=r['reward'],time=t,label_indices=ix,label_scores=scores))
                ah=torch.tensor(s['hidden']);ch=torch.tensor(newch)
            before={k:v.clone() for k,v in actor.state_dict().items()}
            m=update_motion(actor,critic,torch.optim.Adam(actor.parameters(),lr=3e-4),torch.optim.Adam(critic.parameters(),lr=3e-4),
                rows,aux_weight=weight,epochs=1,sequence=8)
            self.assertGreater(m['actor_steps'],0);self.assertGreater(m['critic_steps'],0)
            self.assertTrue(np.isfinite(m['actor_loss']));self.assertLessEqual(m['max_accepted_chunk_kl'],.03+1e-6)
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in actor.state_dict().items() if k.startswith('encoder.')))
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in actor.state_dict().items() if k.startswith('score.')))


class MotionResume(unittest.TestCase):
    def test_resume_preserves_models_optimizers_rng_for_both_methods(self):
        import tempfile
        import subprocess
        import sys
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        def equal(a,b):
            if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in a:equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):equal(x,y)
            else:self.assertEqual(a,b)
        with tempfile.TemporaryDirectory() as directory:
            for method in ('motion_ppo','motion_aux'):
                base=Path(directory)/method
                def run(out,stop):
                    p=subprocess.run([sys.executable,str(root/'scripts/analysis/dynamic_service_motion_experiment.py'),
                        'train','--smoke','--episodes','2','--steps','4','--stop-after',str(stop),'--method',method,'--output',str(out)],
                        cwd=root,capture_output=True,text=True)
                    self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                run(base/'continuous',2);run(base/'resumed',1);run(base/'resumed',2)
                a=torch.load(base/'continuous/models/0'/method/'latest.pt',weights_only=False)
                b=torch.load(base/'resumed/models/0'/method/'latest.pt',weights_only=False)
                for k in ('actor','critic','aopt','copt','rng'):equal(a[k],b[k])


if __name__=='__main__':unittest.main()
