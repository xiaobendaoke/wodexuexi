"""Remote-only regression tests for batch and autoregressive offloading."""
import importlib
import numpy as np
import unittest
import torch
import config
from marl_models.offload_mappo.offload_mappo import BatchOffloadActor, OffloadMAPPO


def inputs(n=4):
    d = config.OFFLOAD_REQUEST_FEATURE_DIM
    obs = torch.zeros(2, 5 + n*d)
    rows = obs[:, 5:].reshape(2, n, d)
    rows[0, :, 0] = 1
    rows[0, :, 1] = torch.arange(1, n+1)/n
    masks = torch.zeros(2, n, config.OFFLOAD_NUM_ACTIONS)
    masks[0, :, :2] = 1
    return obs, masks


def check_sample_and_teacher_forcing_agree(sequential, n):
    torch.manual_seed(71)
    obs, masks = inputs(n)
    actor = BatchOffloadActor(obs.shape[-1], n, masks.shape[-1], sequential=sequential)
    actions, logp, entropy = actor.decide(obs, masks, exploration=True)
    replay, replay_logp, replay_entropy = actor.decide(obs, masks, actions=actions)
    assert torch.equal(actions, replay)
    assert torch.allclose(logp, replay_logp, atol=1e-6)
    assert torch.allclose(entropy, replay_entropy, atol=1e-6)
    assert (actions[0] < 2).all() and (actions[1] == 0).all()
    assert logp[1] == 0 and entropy[1] == 0
    (-replay_logp[0]).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())


def test_prefix_is_causal_and_batch_channels_are_zero():
    obs, masks = inputs()
    actions = torch.zeros(2, 4, dtype=torch.long)
    actor = BatchOffloadActor(obs.shape[-1], 4, masks.shape[-1], sequential=True)
    seen = []
    handle = actor.net.register_forward_pre_hook(lambda m, x: seen.append(x[0].detach().clone()))
    actor.decide(obs, masks, actions=actions)
    first = seen.copy(); seen.clear()
    actions[0, 2] = 1
    actor.decide(obs, masks, actions=actions)
    assert all(torch.equal(first[i], seen[i]) for i in range(3))
    assert not torch.equal(first[3], seen[3])
    handle.remove()
    actor.sequential = False
    seen.clear()
    actor.net.register_forward_pre_hook(lambda m, x: seen.append(x[0].detach().clone()))
    actor.decide(obs, masks, actions=actions)
    assert all((x[..., -2*masks.shape[-1]:] == 0).all() for x in seen)


def test_masked_teacher_action_rejected():
    obs, masks = inputs()
    actor = BatchOffloadActor(obs.shape[-1], 4, masks.shape[-1], sequential=True)
    actions = torch.zeros(2, 4, dtype=torch.long); actions[0, 0] = 2
    with unittest.TestCase().assertRaises(ValueError):
        actor.decide(obs, masks, actions=actions)


def test_matched_architecture_and_padding_ignored():
    obs, masks = inputs()
    b = BatchOffloadActor(obs.shape[-1], 4, masks.shape[-1], sequential=False)
    c = BatchOffloadActor(obs.shape[-1], 4, masks.shape[-1], sequential=True)
    assert [p.shape for p in b.parameters()] == [p.shape for p in c.parameters()]
    masks[0, 3] = 0; obs[0, 5+3*config.OFFLOAD_REQUEST_FEATURE_DIM] = 0
    before = b.decide(obs, masks, exploration=False)
    obs[0, 5+3*config.OFFLOAD_REQUEST_FEATURE_DIM+1:] = 999
    after = b.decide(obs, masks, exploration=False)
    assert torch.equal(before[0], after[0]) and torch.allclose(before[1], after[1])


def test_method_configuration():
    from run_canonical_experiment import configure_method
    importlib.reload(config)
    model = configure_method('vanilla_lower_mlp_upper_uncovered_obs')
    assert model['upper'].model_name == 'vanilla_mappo'
    assert not model['lower'].actor_attention and not model['lower'].critic_attention
    assert config.UPPER_UE_OBSERVATION_MODE == 'covered_plus_uncovered'
    for method in ('lower_batch_upper_uncovered_obs', 'lower_sequential_upper_uncovered_obs',
                   'vanilla_lower_batch_upper_uncovered_obs', 'vanilla_lower_sequential_upper_uncovered_obs'):
        model = configure_method(method)
        assert isinstance(model['lower'].actor, BatchOffloadActor)
        assert not model['lower'].critic_attention
    importlib.reload(config)


def test_sample_and_teacher_forcing_agree():
    for sequential in (False, True):
        for n in (1, 4, 30):
            check_sample_and_teacher_forcing_agree(sequential, n)



def test_sequential_checkpoint_resume_and_empty_update():
    import json
    import tempfile
    from pathlib import Path
    from run_canonical_experiment import parse_args, train
    def args(path, episodes=1, extra=()):
        return parse_args(['--method','lower_sequential_upper_uncovered_obs','--training-seed','9',
                           '--workload-seed','21','--episodes',str(episodes),'--steps','4','--rollout','2',
                           '--ppo-epochs','1','--batch-size','2','--threads','1',
                           '--purpose','verification','--output',str(path),*extra])
    with tempfile.TemporaryDirectory() as folder:
        left,right = Path(folder)/'full',Path(folder)/'resume'
        train(args(left,2)); train(args(right)); train(args(right,2,('--resume',)))
        for i in range(2):
            assert json.loads((left/'episodes'/f'{i:05d}.json').read_text()) == json.loads((right/'episodes'/f'{i:05d}.json').read_text())
        evaluation = args(Path(folder)/'eval',extra=('--evaluate-checkpoint',str(left/'checkpoints/episode_00001')))
        train(evaluation)
        raw = json.loads((Path(folder)/'eval/episodes/00000.json').read_text())
        assert all(r['policy_inference_seconds'] >= 0 for r in raw['steps'])
        wrong = args(Path(folder)/'wrong',extra=('--evaluate-checkpoint',str(left/'checkpoints/episode_00001')))
        wrong.method = 'lower_batch_upper_uncovered_obs'
        with unittest.TestCase().assertRaises(ValueError): train(wrong)
    importlib.reload(config)
    model = OffloadMAPPO('sequential_offload_mappo',2,5+4*config.OFFLOAD_REQUEST_FEATURE_DIM,4,'cpu',max_requests=4)
    obs,masks = inputs(); masks.zero_(); obs.zero_()
    actions,logp,_ = model.actor.decide(obs,masks)
    batch = {'obs':obs,'actions':actions,'masks':masks,'valid_slots':torch.zeros(2,4),
             'old_log_probs':logp.detach(),'advantages':torch.ones(2),'returns':torch.zeros(2),'old_values':torch.zeros(2)}
    loss = model.update(batch)
    assert all(np.isfinite(v) for v in loss.values())


def test_direct_checkpoint_rejects_batch_sequential_swap():
    import tempfile
    importlib.reload(config)
    d=5+4*config.OFFLOAD_REQUEST_FEATURE_DIM
    batch=OffloadMAPPO('batch_offload_mappo',2,d,4,'cpu',max_requests=4)
    sequential=OffloadMAPPO('sequential_offload_mappo',2,d,4,'cpu',max_requests=4)
    with tempfile.TemporaryDirectory() as folder:
        batch.save(folder)
        with unittest.TestCase().assertRaises(ValueError): sequential.load(folder)
        batch.load(folder)

def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))
