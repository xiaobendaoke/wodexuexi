"""Remote-only live adapter and closed-loop execution regression."""
import pickle
import random
import unittest
import numpy as np
import torch
from environment.env import Env
from utils.snapshot_evaluator import capture, evaluate
from utils.deadline_repair import snapshot_inputs, repair
from utils.deadline_repair_online import prepare, apply


class TestOnlineRepair(unittest.TestCase):
    def test_live_adapter_matches_frozen_and_is_pure(self):
        env=Env(scenario_seed=19);env.reset()
        _,masks=env.get_offloading_obs_and_masks()
        before=pickle.dumps(env,protocol=5)
        rng=pickle.dumps((np.random.get_state(),random.getstate(),torch.get_rng_state().tolist()))
        requests,rows=prepare(env,masks)
        self.assertEqual(rows,snapshot_inputs(capture(env)))
        self.assertEqual(requests,capture(env).requests)
        self.assertEqual(before,pickle.dumps(env,protocol=5))
        self.assertEqual(rng,pickle.dumps((np.random.get_state(),random.getstate(),torch.get_rng_state().tolist())))
        actions=np.zeros(masks.shape[:2],dtype=np.int64)
        for request,row in zip(requests,rows):actions[request[0],request[1]]=next(iter(row['options']))
        for mode in ['baseline','ordinary','deadline']:
            revised,diag=apply(requests,rows,actions,mode)
            self.assertEqual(actions.shape,revised.shape)
            for request,row in zip(requests,rows):self.assertIn(int(revised[request[0],request[1]]),row['options'])
            initial=tuple(int(actions[r[0],r[1]]) for r in requests)
            expected=initial if mode=='baseline' else repair(rows,initial,mode,128)['assignment']
            self.assertEqual(tuple(int(revised[r[0],r[1]]) for r in requests),expected)
            if mode=='baseline':self.assertTrue(np.array_equal(actions,revised))
            snap=capture(env);actual=evaluate(snap,expected)
            clone=pickle.loads(snap.payload)
            _,_,metrics=clone.step(np.zeros((len(env.uavs),2)),offloading_actions=revised)
            self.assertEqual(metrics['deadline_satisfied_count'],actual['metrics']['deadline_satisfied_count'])
            self.assertAlmostEqual(metrics['lower_assignment_energy'],actual['metrics']['lower_assignment_energy'])
        self.assertEqual(before,pickle.dumps(env,protocol=5))

    def test_empty_padding_and_illegal_action(self):
        a=np.zeros((5,30),dtype=np.int64)
        result,diag=apply((),[],a,'deadline')
        self.assertTrue(np.array_equal(a,result));self.assertEqual(diag['changed_actions'],0)
        row={'source':0,'ue_id':1,'deadline':1.,'options':{0:{'target':0,'base':0.,'work':.1,'energy':1.,'compute_energy':0.}}}
        req=((0,0,1,0,1.),)
        a[0,0]=6
        with self.assertRaises(ValueError):apply(req,[row],a,'ordinary')
        a[0,0]=0;a[0,1]=6
        result,_=apply(req,[row],a,'deadline')
        self.assertEqual(result[0,1],6)

if __name__=='__main__':unittest.main()
