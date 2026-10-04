"""Regression for the registered follow-up; numerical work is Linux-only."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import json
import shutil
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
import unittest

if sys.platform != 'linux':
    raise unittest.SkipTest('Torch, matching and environment validation runs remotely')

SOURCE = Path(os.environ['SF_HLM_SOURCE'])
EXTENSION = Path(os.environ['SF_HLM_EXTENSION'])
ADAPTATION = Path(os.environ['SF_HLM_ADAPTATION'])
FOLLOWUP = Path(os.environ['SF_HLM_FOLLOWUP'])
sys.path.insert(0, str(SOURCE))
for name, root in [
    *[(n, EXTENSION) for n in ('utils.deadline_repair', 'utils.deadline_repair_online',
      'utils.node_offload_observation', 'utils.sf_hlm_matching', 'utils.sf_hlm_search',
      'utils.sf_hlm_v2', 'marl_models.sf_hlm', 'marl_models.sf_hlm_v2')],
    ('marl_models.sf_hlm_upper_adaptation', ADAPTATION),
    ('utils.sf_hlm_followup', FOLLOWUP),
    ('marl_models.sf_hlm_bounded_residual', FOLLOWUP),
]:
    spec = importlib.util.spec_from_file_location(name, root/(name.replace('.', '/')+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

import config
import numpy as np
import torch
from torch.distributions import Normal
from utils.sf_hlm_followup import FixedMBSSearch, fixed_mbs_search, common_mbs_initial
from utils.sf_hlm_matching import allocation_counts
from utils.sf_hlm_v2 import graph
from marl_models.sf_hlm_v2 import CompleteQuotaPPO
from marl_models.sf_hlm_bounded_residual import BoundedResidualMAPPO


def fixture(n=8):
    return [dict(deadline=1., source=i % 2, ue_id=i, size=1000, cycles=200000, options={
        0: dict(target=i % 2, base=.1, work=.22, energy=2., compute_energy=2., cache_belief=.8, capacity=1e9),
        1: dict(target='mbs', base=.55, work=0., energy=1., compute_energy=0., cache_belief=1., capacity=2e10),
        2+(1-i % 2): dict(target=1-i % 2, base=.2, work=.3, energy=2., compute_energy=2., cache_belief=.3, capacity=1e9)
    }) for i in range(n)]


def records(model, n=8):
    rng = np.random.default_rng(18)
    output = []
    for i in range(n):
        obs = rng.normal(0, .2, (config.NUM_UAVS, config.OBS_DIM_SINGLE)).astype(np.float32)
        view = rng.normal(0, .1, (config.NUM_UAVS, config.NUM_UAVS+1, 10)).astype(np.float32)
        view[..., 9] = 1
        action, lp, value = model.act(obs, view, exploration=True)
        output.append(dict(obs=obs, feedback=view, action=model.last_latent.copy(), executed_action=action,
            log_prob=lp, value=value, old_mean=model.last_mean.copy(), old_std=model.last_std.copy(),
            reward=np.full(config.NUM_UAVS, .1+i*.01, np.float32), done=i == n-1))
    return output


class FollowupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_fixed_mbs_including_initialization_and_fallback(self):
        for rows in ([], fixture(1), fixture(30)):
            _, d = common_mbs_initial(rows, num_uavs=2)
            target = d['fixed_mbs_count']
            for budget in (0, 1, 16):
                e = FixedMBSSearch(rows, num_uavs=2, matching_budget=budget)
                while not e.done:
                    mask = e.action_mask()
                    for k in np.flatnonzero(mask[1:])+1:
                        source, destination, _ = e.actions[k]
                        self.assertNotIn(2, (source, destination))
                    choices = np.flatnonzero(mask[1:])+1
                    e.step(int(choices[0]) if len(choices) else 0)
                    self.assertEqual(allocation_counts(rows, e.current_assignment, 2)[-1], target)
                self.assertEqual(allocation_counts(rows, e.assignment, 2)[-1], target)
                self.assertLessEqual(e.calls, budget)
                self.assertLessEqual(e.best['energy'], e.cap+1e-7)

    def test_mask_probability_renormalization(self):
        state = graph(FixedMBSSearch(fixture(), num_uavs=2))
        m = CompleteQuotaPPO(2)
        logits, _ = m.forward_states([state])
        dist = torch.distributions.Categorical(logits=logits)
        self.assertAlmostEqual(float(dist.probs.sum()), 1., places=6)
        self.assertTrue(torch.all(dist.probs[0, ~torch.tensor(state['mask'])] == 0))
        action, lp, _ = m.act(state, exploration=True)
        recomputed, _, _ = m.evaluate([state], torch.tensor([action]))
        self.assertAlmostEqual(lp, float(recomputed[0]), places=5)

    def test_restricted_search_all_policies_and_bad_actions(self):
        rows = fixture(); m = CompleteQuotaPPO(2)
        for kind in ('greedy', 'random', 'heuristic', 'learned'):
            a, d, _ = fixed_mbs_search(rows, num_uavs=2, kind=kind, model=m,
                                       rng=np.random.default_rng(42), budget=16)
            self.assertEqual(len(a), len(rows))
            self.assertEqual(allocation_counts(rows, a, 2)[-1], d['fixed_mbs_count'])
        e = FixedMBSSearch(rows, num_uavs=2)
        illegal = next(k for k, a in enumerate(e.actions[1:], 1) if 2 in a[:2])
        with self.assertRaises(ValueError):e.step(illegal)
        with self.assertRaises(ValueError):e.step(True)
        for wall in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):FixedMBSSearch(rows,num_uavs=2,wall_seconds=wall)

    def test_zero_initialization_and_latent_probability(self):
        for kind in ('bounded_feedback', 'bounded_zero', 'unbounded_feedback'):
            m = BoundedResidualMAPPO(config.OBS_DIM_SINGLE, kind=kind)
            row = records(m, 1)[0]
            a, _, _ = m.act(row['obs'], row['feedback'])
            raw = m.actor(torch.tensor(row['obs'])).mean.detach().numpy()
            np.testing.assert_allclose(a, raw, atol=1e-7)
            dist, _ = m.distribution(torch.tensor(row['obs'])[None], torch.tensor(row['feedback'])[None])
            lp = dist.log_prob(torch.tensor(row['action'])[None]).sum(-1)[0].detach().numpy()
            np.testing.assert_allclose(lp, row['log_prob'], atol=1e-6)
            self.assertFalse(np.allclose(row['action'], row['executed_action']))

    def test_training_and_evaluation_action_bounds(self):
        m = BoundedResidualMAPPO(config.OBS_DIM_SINGLE, kind='bounded_feedback')
        torch.nn.init.normal_(m.actor_feedback.residual.weight, std=.4)
        for row in records(m, 12):
            base = m.actor(torch.tensor(row['obs'])).mean.detach().numpy()
            self.assertLessEqual(float(np.abs(row['executed_action']-base).max()), .150001)
            a, _, _ = m.act(row['obs'], row['feedback'])
            self.assertLessEqual(float(np.abs(a-base).max()), .150001)

    def test_update_freeze_zero_pressure_and_checkpoint_resume(self):
        for kind in ('bounded_feedback', 'bounded_zero', 'unbounded_feedback'):
            m = BoundedResidualMAPPO(config.OBS_DIM_SINGLE, kind=kind)
            before = {k:copy.deepcopy(getattr(m,k).state_dict()) for k in ('actor','critic')}
            loss = m.update_rollout(records(m), epochs=1, batch_size=4)
            self.assertAlmostEqual(loss['initial_ratio_mean'], 1., places=5)
            self.assertGreaterEqual(loss['anchor_kl'], 0.)
            for k, values in before.items():
                for n,v in values.items():self.assertTrue(torch.equal(v,getattr(m,k).state_dict()[n]))
            with tempfile.TemporaryDirectory() as d:
                m.save(d, {'test':kind}); twin=BoundedResidualMAPPO(config.OBS_DIM_SINGLE,kind=kind)
                twin.load(d, {'test':kind})
                torch.manual_seed(29); batch=records(m)
                torch.manual_seed(29); other=records(twin)
                for x,y in zip(batch,other):np.testing.assert_array_equal(x['action'],y['action'])
                torch.manual_seed(40);m.update_rollout(batch,epochs=1,batch_size=4)
                torch.manual_seed(40);twin.update_rollout(other,epochs=1,batch_size=4)
                for k in ('actor','critic','actor_feedback','critic_feedback'):
                    for n,v in getattr(m,k).state_dict().items():self.assertTrue(torch.equal(v,getattr(twin,k).state_dict()[n]))
                with self.assertRaises(ValueError):twin.load(d, {'test':'wrong'})
            if kind=='bounded_zero':
                x=batch[0];v=x['feedback'].copy();v[...,:6]+=100
                np.testing.assert_array_equal(m.act(x['obs'],v)[0],m.act(x['obs'],x['feedback'])[0])


@unittest.skipUnless(os.environ.get('SF_HLM_FOLLOWUP_SMOKE'),'Requires completed follow-up smoke')
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original=Path(os.environ['SF_HLM_FOLLOWUP_SMOKE'])
        self.protocol=json.loads((self.original/'protocol.json').read_text())
        self.assertTrue(self.protocol['smoke'])

    def command(self,root,task):
        cmd=[sys.executable,str(FOLLOWUP/'scripts/analysis/sf_hlm_followup.py'),'worker']
        values={'source-root':SOURCE,'suite-root':os.environ['SF_HLM_SUITE'],'extension-root':EXTENSION,
                'output':os.environ['SF_HLM_OUTPUT'],'legacy-results':os.environ['SF_HLM_LEGACY_RESULTS'],
                'resources':os.environ['SF_HLM_RESOURCES'],'adaptation-root':ADAPTATION,
                'followup-root':FOLLOWUP,'results-root':root}
        values.update(zip(('phase','method','block','workload','episode'),task))
        for k,v in values.items():cmd+=['--'+k,str(v)]
        return cmd

    def execute(self,root,task):
        result=subprocess.run(self.command(root,task),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def folder(self,root,task):
        phase,method,block,workload,episode=task
        out=root/phase/method/f'block_{block}_workload_{workload}'
        return out if episode<0 else out/f'episode_{episode:05d}'

    def nested(self,a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for k in a:self.nested(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.nested(x,y)
        else:self.assertEqual(a,b)

    def test_process_recovery_original_optimizer_constraints_and_residual(self):
        tasks=[t for t in self.protocol['tasks'] if t[0] in ('dev_original_training','dev_residual_training')]
        with tempfile.TemporaryDirectory(dir=self.original.parent) as tmp:
            root=Path(tmp);shutil.copy(self.original/'protocol.json',root/'protocol.json')
            for task in tasks:
                with self.subTest(kind=task[1]):
                    baseline=self.folder(self.original,task);target=self.folder(root,task)
                    recovery=baseline/'recovery/episode_00000'
                    shutil.copytree(recovery,target/'recovery/episode_00000')
                    files={str(f.relative_to(recovery)):hashlib.sha256(f.read_bytes()).hexdigest() for f in recovery.rglob('*') if f.is_file()}
                    (target/'latest.json').write_text(json.dumps(dict(checkpoint='recovery/episode_00000',files=files)))
                    (target/'episodes').mkdir()
                    shutil.copy(baseline/'episodes/00000.json',target/'episodes/00000.json')
                    self.execute(root,task)
                    for f in (baseline/'checkpoint').rglob('*.pt'):
                        self.nested(torch.load(f,weights_only=False),torch.load(target/'checkpoint'/f.relative_to(baseline/'checkpoint'),weights_only=False))
                    for f in (baseline/'episodes').glob('*.json'):
                        self.assertEqual(json.loads(f.read_text()),json.loads((target/'episodes'/f.name).read_text()))

    def test_serial_parallel_episode_actions_latents_probabilities_and_counts(self):
        tasks=[t for t in self.protocol['tasks'] if t[0]=='dev_eval']
        with tempfile.TemporaryDirectory(dir=self.original.parent) as tmp:
            root=Path(tmp)
            for execution in ('serial','parallel'):
                out=root/execution;out.mkdir();shutil.copy(self.original/'protocol.json',out/'protocol.json')
                for phase in ('dev_original_training','dev_residual_training'):
                    shutil.copytree(self.original/phase,out/phase)
                if execution=='serial':
                    for task in tasks:self.execute(out,task)
                else:
                    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda t:self.execute(out,t),tasks))
            for task in tasks:
                a=json.loads((self.folder(root/'serial',task)/'episode.json').read_text())
                b=json.loads((self.folder(root/'parallel',task)/'episode.json').read_text())
                for doc in (a,b):
                    for step in doc['steps']:
                        step.pop('policy_seconds')
                        step['diagnostic'].pop('search_seconds',None)
                self.assertEqual(a,b)


if __name__ == '__main__':unittest.main()
