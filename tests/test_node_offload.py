"""Remote-only tests for action-aligned lower proposals under frozen uppers."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
import config
from utils.node_offload_observation import NODE_FEATURE_NAMES, pack_observation
from marl_models.node_offload_mappo import NodeOffloadMAPPO


def fixture(n=3):
    d=5+n*config.OFFLOAD_REQUEST_FEATURE_DIM
    obs=np.zeros((2,d),np.float32)
    obs[0,5::config.OFFLOAD_REQUEST_FEATURE_DIM]=1
    masks=np.zeros((2,n,7),np.float32);masks[0,:,0:2]=1
    requests=[(0,i,i,0,1.) for i in range(n)]
    descriptions=[{0:dict(tx=.1,compute=.2,capacity=1.,belief=.8,energy=2.,compute_energy=3.),
                   1:dict(tx=.3,compute=.1,capacity=10.,belief=1.,energy=4.,compute_energy=0.)} for _ in range(n)]
    packed=pack_observation(obs,masks,requests,descriptions,capacity_reference=1.,energy_reference=1.)
    return obs,masks,packed


class TestNodeOffload(unittest.TestCase):
    def model(self,kind,n=3):
        return NodeOffloadMAPPO(kind,2,5+n*config.OFFLOAD_REQUEST_FEATURE_DIM,n,'cpu',max_requests=n)

    def test_action_alignment_and_padding(self):
        obs,masks,packed=fixture()
        nodes=packed[:,obs.shape[-1]:].reshape(2,3,7,len(NODE_FEATURE_NAMES))
        self.assertAlmostEqual(nodes[0,0,0,4],.1,places=6)
        self.assertAlmostEqual(nodes[0,0,1,4],.3,places=6)
        self.assertTrue((nodes[:,:,2:]==0).all());self.assertTrue((nodes[1]==0).all())
        for kind in ['flat','shared']:
            model=self.model(kind)
            actions,lp,_=model.get_action_and_value(packed,masks=masks,exploration=True)
            self.assertTrue((actions[0]<2).all());self.assertTrue((actions[1]==0).all());self.assertEqual(lp[1],0.)
            before=model.actor(torch.tensor(packed),torch.tensor(masks))
            dirty=packed.copy();dirty[1]=999
            after=model.actor(torch.tensor(dirty),torch.tensor(masks))
            self.assertTrue(torch.equal(before[1],after[1]))

    def test_shared_node_permutation_equivariance(self):
        obs,masks,packed=fixture();model=self.model('shared')
        t=torch.tensor(packed);m=torch.tensor(masks);logits=model.actor(t,m)
        nodes=t[:,obs.shape[-1]:].reshape(2,3,7,-1)
        perm=torch.tensor([1,0,2,3,4,5,6])
        rotated=torch.cat([t[:,:obs.shape[-1]],nodes[:,:,perm].reshape(2,-1)],-1)
        self.assertTrue(torch.allclose(model.actor(rotated,m[:,:,perm]),logits[:,:,perm],atol=1e-6))

    def test_sample_probability_replays_and_critic_ignores_nodes(self):
        from torch.distributions import Categorical
        obs,masks,packed=fixture()
        for kind in ['flat','shared']:
            model=self.model(kind);actions,lp,values=model.get_action_and_value(packed,masks=masks,exploration=True)
            logits=model.actor(torch.tensor(packed),torch.tensor(masks));dist=Categorical(logits=logits)
            new=(dist.log_prob(torch.tensor(actions))*torch.tensor(masks.sum(-1)>0)).sum(-1)
            self.assertTrue(np.allclose(lp,new.detach().numpy(),atol=1e-6))
            changed=packed.copy();changed[:,obs.shape[-1]:]=123
            self.assertTrue(torch.equal(model.critic(torch.tensor(packed)),model.critic(torch.tensor(changed))))
            batch=dict(obs=torch.tensor(packed),actions=torch.tensor(actions),masks=torch.tensor(masks),
                       valid_slots=torch.tensor(masks.sum(-1)>0,dtype=torch.float32),old_log_probs=torch.tensor(lp),
                       advantages=torch.tensor([1.,0.]),returns=torch.tensor(values)+.1,old_values=torch.tensor(values))
            loss=model.update(batch);self.assertTrue(all(np.isfinite(v) for v in loss.values()))
            self.assertAlmostEqual(loss['ratio_mean'],1.,places=5)

    def test_empty_update_and_full_capacity(self):
        for n in [1,30]:
            obs,masks,packed=fixture(n)
            for kind in ['flat','shared']:
                model=self.model(kind,n);masks[:]=0;packed[:]=0
                actions,lp,v=model.get_action_and_value(packed,masks=masks)
                self.assertTrue((actions==0).all());self.assertTrue((lp==0).all())
                batch=dict(obs=torch.tensor(packed),actions=torch.tensor(actions),masks=torch.tensor(masks),
                           valid_slots=torch.zeros(2,n),old_log_probs=torch.tensor(lp),advantages=torch.zeros(2),
                           returns=torch.zeros(2),old_values=torch.tensor(v))
                self.assertTrue(all(np.isfinite(x) for x in model.update(batch).values()))

    def test_checkpoint_contract_and_replay(self):
        obs,masks,packed=fixture();flat=self.model('flat');shared=self.model('shared')
        with tempfile.TemporaryDirectory() as folder:
            flat.save(folder)
            with self.assertRaises(ValueError):shared.load(folder)
            restored=self.model('flat');restored.load(folder)
            self.assertTrue(np.array_equal(flat.get_action_and_value(packed,masks=masks,exploration=False)[0],
                                          restored.get_action_and_value(packed,masks=masks,exploration=False)[0]))

    def test_duplicate_or_missing_description_rejected(self):
        obs,masks,_=fixture()
        with self.assertRaises(ValueError):pack_observation(obs,masks,[],[],capacity_reference=1.,energy_reference=1.)

    def test_live_features_are_pure_and_match_predictor(self):
        import pickle,random
        from environment.env import Env
        from utils.deadline_repair_online import prepare
        from utils.node_offload_observation import describe_candidates,prepare_observation
        env=Env(scenario_seed=19);env.reset();obs,masks=env.get_offloading_obs_and_masks()
        requests,rows=prepare(env,masks)
        before=pickle.dumps(env,protocol=5)
        rng=pickle.dumps((np.random.get_state(),random.getstate(),torch.get_rng_state().tolist()))
        descriptions=describe_candidates(env,requests,rows);packed=prepare_observation(env,obs,masks,requests,rows)
        self.assertTrue(np.isfinite(packed).all())
        for row,options in zip(rows,descriptions):
            for a,d in options.items():
                expected=row['options'][a]['base']+row['options'][a]['work']
                self.assertAlmostEqual(d['tx']+d['compute'],expected,places=10)
        self.assertEqual(before,pickle.dumps(env,protocol=5))
        self.assertEqual(rng,pickle.dumps((np.random.get_state(),random.getstate(),torch.get_rng_state().tolist())))

    def test_runner_writes_numpy_metrics(self):
        import importlib.util,json,os
        ext=Path(os.environ['NODE_OFFLOAD_EXTENSION'])
        spec=importlib.util.spec_from_file_location('node_runner_test',ext/'scripts/analysis/node_offload_experiment.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'metrics.json';module.write(p,dict(x=np.float32(.25),n=np.int64(2)))
            self.assertEqual(json.loads(p.read_text()),dict(x=.25,n=2))


if __name__=='__main__':unittest.main()
