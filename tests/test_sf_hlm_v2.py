"""Remote regression tests for versioned graphs, archive RL and service attention."""
import copy
import importlib.util
import os
from pathlib import Path
import pickle
import platform
import sys
import tempfile
import json
import shutil
import subprocess
from types import SimpleNamespace
import unittest

if platform.system() != 'Linux':
    raise unittest.SkipTest('Torch, matching and environment tests require the remote server')

SOURCE=Path(os.environ['SF_HLM_SOURCE']);EXTENSION=Path(os.environ['SF_HLM_EXTENSION'])
sys.path.insert(0,str(SOURCE))
for name in ('utils.deadline_repair','utils.deadline_repair_online','utils.node_offload_observation',
             'utils.sf_hlm_matching','utils.sf_hlm_search','marl_models.sf_hlm','utils.sf_hlm_resources',
             'utils.sf_hlm_v2','marl_models.sf_hlm_v2'):
    spec=importlib.util.spec_from_file_location(name,EXTENSION/(name.replace('.','/')+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
sys.path.insert(0,str(EXTENSION/'utils'))
import sf_hlm_rollouts as rollouts
import numpy as np
import torch
from utils.sf_hlm_v2 import CompleteQuotaSearch,CriticalArchiveSearch,graph,critical_count,search,validate_metadata
from utils.sf_hlm_matching import allocation_counts,feedback
from marl_models.sf_hlm_v2 import CompleteQuotaPPO,ArchivePolicy,ServiceMAPPO,service_view
from utils.sf_hlm_resources import initialize,CpuLease,topology


def fixture(n=8):
    return [dict(deadline=1.,source=i%2,ue_id=i,size=1000,cycles=200000,options={
        0:dict(target=i%2,base=.1,work=.22,energy=2.,compute_energy=2.,cache_belief=.8,capacity=1e9),
        1:dict(target='mbs',base=1.1,work=0.,energy=1.,compute_energy=0.,cache_belief=1.,capacity=2e10),
        2+(1-i%2):dict(target=1-i%2,base=.2,work=.3,energy=2.,compute_energy=2.,cache_belief=.3,capacity=1e9)
    }) for i in range(n)]


class V2Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(17)

    def test_strict_observed_metadata(self):
        rows=fixture();validate_metadata(rows)
        for key in ('size','cycles'):
            bad=copy.deepcopy(rows);del bad[0][key]
            with self.assertRaises(ValueError):CompleteQuotaSearch(bad,num_uavs=2)
        for key in ('capacity','cache_belief'):
            bad=copy.deepcopy(rows);del bad[0]['options'][0][key]
            with self.assertRaises(ValueError):CompleteQuotaSearch(bad,num_uavs=2)

    def test_current_best_and_request_assignment_are_distinct(self):
        engine=CompleteQuotaSearch(fixture(),num_uavs=2,base_repair_budget=0)
        for action in np.flatnonzero(engine.action_mask()[1:])+1:
            clone=copy.deepcopy(engine);clone.step(int(action))
            if clone.current_assignment!=clone.assignment:
                state=graph(clone)
                self.assertFalse(np.array_equal(state['edges'][...,-2],state['edges'][...,-1]))
                self.assertFalse(np.array_equal(state['nodes'][:,:6],state['nodes'][:,6:12]))
                return
        self.fail('Fixture must exercise a worse feasible incumbent')

    def test_critical_boundaries_mbs_budget_and_empty(self):
        for base in (.1,.3,1.1):
            for work in (0.,.1,.3):
                for deadline in (1.,base+work*3):
                    expected=max([0]+[n for n in range(1,11) if base+work*n<=deadline])
                    self.assertEqual(critical_count(base,work,deadline,10),expected)
        for rows in ([],fixture(1),fixture(30)):
            engine=CriticalArchiveSearch(rows,num_uavs=2,matching_budget=16)
            previous=engine.best['predicted']['objective'][:2]
            while not engine.done:
                parents=engine.parents();self.assertLessEqual(len(parents),3)
                engine.step(0)
                self.assertLessEqual(engine.best['predicted']['objective'][:2],previous)
                self.assertLessEqual(engine.best['predicted']['energy'],engine.cap+1e-9)
                previous=engine.best['predicted']['objective'][:2]
            self.assertLessEqual(engine.calls,16)
            self.assertEqual(sum(allocation_counts(rows,engine.assignment,2)),len(rows))

    def test_ppo_probability_checkpoint_and_legacy_rejection(self):
        state=graph(CompleteQuotaSearch(fixture(),num_uavs=2));model=CompleteQuotaPPO(2)
        a,lp,v=model.act(state,exploration=True)
        newer,_,_=model.evaluate([state],torch.tensor([a]));self.assertAlmostEqual(lp,float(newer[0]),places=5)
        loss=model.update([dict(state=state,action=a,log_prob=lp,value=v,reward=1.,done=True)],epochs=1)
        self.assertAlmostEqual(loss['initial_ratio_mean'],1.,places=5)
        with tempfile.TemporaryDirectory() as d:
            model.save(d,{'v':2});restored=CompleteQuotaPPO(2);restored.load(d,{'v':2})
            self.assertEqual(model.act(state),restored.act(state))
            with self.assertRaises(ValueError):restored.load(d,{'v':1})

    def test_grouped_leave_one_out_probability_recompute(self):
        model=ArchivePolicy(2);group=[]
        for seed in range(8):
            torch.manual_seed(seed)
            _,d,records=search(fixture(),num_uavs=2,family='critical',model=model,exploration=True,budget=4)
            group.append(dict(records=records,**{'return':d['terminal_return']}))
        wrong=copy.deepcopy(group);wrong[0]['records'][0]['log_prob']+=.1
        with self.assertRaises(ValueError):model.update_groups([wrong])
        loss=model.update_groups([group]);self.assertAlmostEqual(loss['initial_ratio_mean'],1.,places=5)
        self.assertTrue(np.isfinite(loss['pg_loss']))

    def test_service_identity_purity_warm_start_and_zero_feedback(self):
        import config
        from environment.env import Env
        from run_canonical_experiment import configure_method
        from marl_models.utils import get_model
        configure_method('lower_mlp_upper_uncovered_obs');env=Env(scenario_seed=19);obs=np.asarray(env.reset(),np.float32)
        summary=np.arange((config.NUM_UAVS+1)*6,dtype=np.float32).reshape(-1,6)/100
        before=pickle.dumps(env,protocol=5);view=service_view(env,summary)
        self.assertEqual(before,pickle.dumps(env,protocol=5))
        for i,u in enumerate(env.uavs):
            for j in range(config.NUM_UAVS):
                if view[i,j,9]:np.testing.assert_array_equal(view[i,j,:6],summary[j])
                else:np.testing.assert_array_equal(view[i,j],np.zeros(10))
        base=get_model('attention_mappo')
        model=ServiceMAPPO(config.OBS_DIM_SINGLE,config.NUM_UAVS);model.warm_start(base)
        raw=base.actor(torch.tensor(obs));new,_=model.distribution(torch.tensor(obs)[None],torch.tensor(view)[None])
        torch.testing.assert_close(new.mean[0],raw.mean,atol=1e-6,rtol=1e-6)
        disabled=ServiceMAPPO(config.OBS_DIM_SINGLE,config.NUM_UAVS,enabled=False)
        # Exercise trained residual as well as its zero-initialized starting state.
        torch.nn.init.normal_(disabled.actor_feedback.residual.weight,std=.01)
        changed=view.copy();changed[...,:6]+=100
        np.testing.assert_array_equal(disabled.act(obs,view)[0],disabled.act(obs,changed)[0])

    def test_serial_parallel_samples_and_gradients(self):
        with tempfile.TemporaryDirectory() as d:
            initialize(d,4);args=SimpleNamespace(source_root=SOURCE,extension_root=EXTENSION,resources=Path(d))
            model=ArchivePolicy(2);tasks=rollouts.rollout_tasks(model,[fixture()],[0],protocol_hash='test',seed=27,iteration=1,budget=4)
            collectors=[rollouts.ParallelCollector(args,workers=n) for n in (0,2)]
            values=[]
            try:
                for collector in collectors:values.append(collector.map(tasks))
            finally:
                for collector in collectors:collector.close()
            for a,b in zip(*values):
                self.assertEqual(a['id'],b['id']);self.assertEqual(a['assignment'],b['assignment']);self.assertEqual(a['return'],b['return'])
                for x,y in zip(a['records'],b['records']):
                    self.assertEqual(x['action'],y['action']);self.assertAlmostEqual(x['log_prob'],y['log_prob'],places=5)
            twin=ArchivePolicy(2);twin.network.load_state_dict(model.network.state_dict())
            model.update_groups([values[0]]);twin.update_groups([values[1]])
            for x,y in zip(model.network.parameters(),twin.network.parameters()):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)

    def test_episode_prefix_and_pre_movement_execution(self):
        import config
        from environment.env import Env
        from utils.canonical_experiment import digest,exogenous_frame
        from utils.snapshot_evaluator import capture,evaluate
        from utils.sf_hlm_search import common_repair
        from utils.deadline_repair_online import prepare,greedy_assignment
        serial=Env(scenario_seed=890001);serial.reset()
        for _ in range(6):serial.step(np.zeros((config.NUM_UAVS,2),np.float32))
        obs=serial.reset();parallel=Env(scenario_seed=890001);parallel.reset();other=parallel.reset()
        np.testing.assert_array_equal(obs,other)
        self.assertEqual(digest(exogenous_frame(serial)),digest(exogenous_frame(parallel)))
        _,mask=serial.get_offloading_obs_and_masks();requests,rows=prepare(serial,mask)
        greedy,_=greedy_assignment(rows);assignment,_=common_repair(rows,greedy)
        first=evaluate(capture(serial,np.zeros((config.NUM_UAVS,2))),assignment)
        second=evaluate(capture(serial,np.ones((config.NUM_UAVS,2))),assignment)
        self.assertEqual(first['requests'],second['requests'])


@unittest.skipUnless(os.environ.get('SF_HLM_V2_SMOKE'),'Requires completed v2 verification pipeline')
class RecoveryTests(unittest.TestCase):
    def assert_nested(self,a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for k in a:self.assert_nested(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_nested(x,y)
        else:self.assertEqual(a,b)

    def test_interrupted_lower_upper_and_original_resume(self):
        original=Path(os.environ['SF_HLM_V2_SMOKE']);p=json.loads((original/'protocol.json').read_text())
        self.assertTrue(p['smoke'])
        with tempfile.TemporaryDirectory(dir=original.parent) as tmp:
            clone=Path(tmp)/'resume';shutil.copytree(original,clone)
            jobs=[t for t in p['tasks'] if (t[:3] in (['lower_training','complete','learned'],['lower_training','critical','learned']) and t[3] in p['historical_blocks']) or (t[0]=='system_training' and t[1]=='critical' and t[2] in ('full','original'))]
            for phase,family,method,block,workload,episode in jobs:
                with self.subTest(phase=phase,family=family,method=method):
                    rel=Path(phase)/family/method/(f'block_{block}' if phase=='lower_training' else f'block_{block}_workload_{workload}')
                    out=clone/rel;baseline=original/rel;(out/'result.json').unlink()
                    recovery=out/'recovery'/('iteration_00000' if phase=='lower_training' else 'episode_00002')
                    import hashlib
                    files={str(f.relative_to(recovery)):hashlib.sha256(f.read_bytes()).hexdigest() for f in recovery.rglob('*') if f.is_file()}
                    (out/'latest.json').write_text(json.dumps(dict(checkpoint=str(recovery.relative_to(out)),files=files)))
                    cmd=[sys.executable,str(EXTENSION/'scripts/analysis/sf_hlm_v2_experiment.py'),'worker']
                    for flag,value in [('source-root',SOURCE),('suite-root',os.environ['SF_HLM_SUITE']),('extension-root',EXTENSION),('output',clone),('legacy-results',os.environ['SF_HLM_LEGACY_RESULTS']),('resources',os.environ['SF_HLM_RESOURCES']),('phase',phase),('family',family),('method',method),('block',block),('workload',workload),('episode',episode),('seed',p['lower_seeds'][str(block)])]:cmd+=['--'+flag,str(value)]
                    done=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(done.returncode,0,done.stdout+done.stderr)
                    for f in (baseline/'checkpoint').rglob('*.pt'):
                        self.assert_nested(torch.load(f,weights_only=False),torch.load(out/'checkpoint'/f.relative_to(baseline/'checkpoint'),weights_only=False))
                    if phase=='system_training':
                        for f in (baseline/'episodes').glob('*.json'):
                            self.assertEqual(json.loads(f.read_text()),json.loads((out/'episodes'/f.name).read_text()))


if __name__=='__main__':unittest.main()
