"""Remote regression for upper-update controls and frozen residual learning."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

if sys.platform != 'linux':
    raise unittest.SkipTest('Model and simulation validation runs on remote Linux')

SOURCE = Path(os.environ['SF_HLM_SOURCE'])
EXTENSION = Path(os.environ['SF_HLM_EXTENSION'])
ADAPTATION = Path(os.environ['SF_HLM_ADAPTATION'])
sys.path.insert(0, str(SOURCE))
for name, path in [
    ('utils.deadline_repair', EXTENSION/'utils/deadline_repair.py'),
    ('utils.deadline_repair_online', EXTENSION/'utils/deadline_repair_online.py'),
    ('utils.sf_hlm_matching', EXTENSION/'utils/sf_hlm_matching.py'),
    ('utils.sf_hlm_search', EXTENSION/'utils/sf_hlm_search.py'),
    ('utils.sf_hlm_v2', EXTENSION/'utils/sf_hlm_v2.py'),
    ('marl_models.sf_hlm', EXTENSION/'marl_models/sf_hlm.py'),
    ('marl_models.sf_hlm_v2', EXTENSION/'marl_models/sf_hlm_v2.py'),
    ('marl_models.sf_hlm_upper_adaptation', ADAPTATION/'marl_models/sf_hlm_upper_adaptation.py'),
]:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)

import config
import numpy as np
import torch
from marl_models.sf_hlm_upper_adaptation import UpperAdaptation
from marl_models.sf_hlm_v2 import PlainMAPPO
from marl_models.attention_mappo.attention_mappo import AttentionMAPPO


def batch(model, n=8):
    rng = np.random.default_rng(18)
    records = []
    for i in range(n):
        obs = rng.normal(0, .2, (config.NUM_UAVS, config.OBS_DIM_SINGLE)).astype(np.float32)
        view = rng.normal(0, .1, (config.NUM_UAVS, config.NUM_UAVS+1, 10)).astype(np.float32)
        view[..., 9] = 1
        action, lp, value = model.act(obs, view, exploration=True)
        records.append(dict(obs=obs, feedback=view, action=action, log_prob=lp, value=value,
                            old_mean=model.last_mean.copy(), old_std=model.last_std.copy(),
                            reward=np.full(config.NUM_UAVS, .1+i*.01, np.float32), done=i==n-1))
    return records


class UpperTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(12)

    def test_original_update_matches_reference_parameters(self):
        model = UpperAdaptation(config.OBS_DIM_SINGLE, kind='original_update')
        reference = AttentionMAPPO('reference', config.NUM_UAVS, config.OBS_DIM_SINGLE, 2, 'cpu')
        reference.actor._orig_mod.load_state_dict(model.actor.state_dict())
        reference.critic._orig_mod.load_state_dict(model.critic.state_dict())
        records = batch(model)
        batches = model.rollout_batches(records, epochs=1, batch_size=4)
        for b in batches:
            reference.update(b); model.update_batch(b)
        for module, original in [(model.actor, reference.actor._orig_mod), (model.critic, reference.critic._orig_mod)]:
            for k, v in module.state_dict().items():
                self.assertTrue(torch.allclose(v, original.state_dict()[k], atol=1e-7, rtol=1e-6), k)

    def test_current_update_matches_frozen_wrapper(self):
        model = UpperAdaptation(config.OBS_DIM_SINGLE, kind='current_update')
        reference = PlainMAPPO(config.OBS_DIM_SINGLE, config.NUM_UAVS, enabled=False)
        for k in ('actor', 'critic', 'actor_feedback', 'critic_feedback'):
            getattr(reference, k).load_state_dict(getattr(model, k).state_dict())
        records = batch(model)
        torch.manual_seed(22); reference.update(records, epochs=1, batch_size=4)
        torch.manual_seed(22); model.update_rollout(records, epochs=1, batch_size=4)
        for k in ('actor', 'critic'):
            for name, value in getattr(model, k).state_dict().items():
                self.assertTrue(torch.allclose(value, getattr(reference, k).state_dict()[name], atol=1e-7, rtol=1e-6), name)

    def test_residual_freeze_probability_and_recovery(self):
        for kind in ('residual_feedback', 'residual_zero'):
            model = UpperAdaptation(config.OBS_DIM_SINGLE, kind=kind)
            frozen = {k:{n:v.clone() for n,v in getattr(model,k).state_dict().items()} for k in ('actor','critic')}
            records = batch(model)
            loss = model.update_rollout(records, epochs=1, batch_size=4)
            self.assertAlmostEqual(loss['initial_ratio_mean'], 1., places=5)
            for k in frozen:
                for n,v in frozen[k].items():self.assertTrue(torch.equal(v,getattr(model,k).state_dict()[n]),n)
            self.assertGreater(float(model.actor_feedback.residual.weight.abs().sum()),0)
            with tempfile.TemporaryDirectory() as d:
                contract = {'test':kind}; model.save(d,contract)
                restored = UpperAdaptation(config.OBS_DIM_SINGLE,kind=kind); restored.load(d,contract)
                torch.manual_seed(29); model.update_rollout(records:=batch(model),epochs=1,batch_size=4)
                torch.manual_seed(29); batch(restored)
                restored.update_rollout(records,epochs=1,batch_size=4)
                for k in ('actor','critic','actor_feedback','critic_feedback'):
                    for n,v in getattr(model,k).state_dict().items():self.assertTrue(torch.equal(v,getattr(restored,k).state_dict()[n]),n)
                with self.assertRaises(ValueError):restored.load(d,{'test':'wrong'})

    def test_zero_initialization_and_pressure_ablation(self):
        full=UpperAdaptation(config.OBS_DIM_SINGLE,kind='residual_feedback')
        zero=UpperAdaptation(config.OBS_DIM_SINGLE,kind='residual_zero')
        for k in ('actor','critic','actor_feedback','critic_feedback'):getattr(zero,k).load_state_dict(getattr(full,k).state_dict())
        obs=np.zeros((config.NUM_UAVS,config.OBS_DIM_SINGLE),np.float32)
        view=np.zeros((config.NUM_UAVS,config.NUM_UAVS+1,10),np.float32);view[...,9]=1;view[...,:6]=1
        a=full.act(obs,view)[0];b=zero.act(obs,view)[0]
        self.assertTrue(np.allclose(a,b,atol=1e-7))
        raw=full.actor(torch.tensor(obs)).mean.detach().numpy();self.assertTrue(np.allclose(a,raw,atol=1e-7))
        records=batch(zero);zero.update_rollout(records,epochs=1,batch_size=4)
        x=zero.act(obs,view)[0];view[...,:6]=9;y=zero.act(obs,view)[0];self.assertTrue(np.array_equal(x,y))


if __name__=='__main__':unittest.main()
