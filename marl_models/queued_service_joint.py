"""Central team PPO prototype: autoregressive queues followed by feasible motion.

Both methods use future team returns. Their only structural difference is whether
motion sees the new allocation prefix. This is not a decentralized MAPPO claim.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from utils.queued_service_learning import STATE_DIM, ALLOCATION_DIM, MOTION_DIM, VirtualQueue, state_features


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


class CandidateScore(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(STATE_DIM, 64), nn.LayerNorm(64), nn.Tanh())
        self.candidate = nn.Sequential(nn.Linear(dimension, 64), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.score[-1].weight, gain=.05); nn.init.zeros_(self.score[-1].bias)

    def forward(self, state, features, mask, prior=None):
        context = self.context(state).unsqueeze(1).expand(-1, features.shape[1], -1)
        candidate = self.candidate(features)
        logits = self.score(torch.cat((context, candidate, context*candidate), -1)).squeeze(-1)
        if prior is not None: logits = logits+prior
        assert bool(mask.any(-1).all())
        return Categorical(logits=logits.masked_fill(~mask, -1e9))


class JointPolicy(nn.Module):
    def __init__(self, method):
        super().__init__()
        if method not in ('joint', 'decomposed'): raise ValueError('unknown queue learning method')
        self.method = method
        self.allocation = CandidateScore(ALLOCATION_DIM)
        self.motion = CandidateScore(MOTION_DIM)
        self.critic = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

    @torch.no_grad()
    def act(self, obs, adapter, seed):
        adapter.prepare(obs); weights = adapter.history.weights
        before = state_features(obs, weights); prefix = VirtualQueue(obs); plan = {}; tokens = []
        generator = torch.Generator().manual_seed(int(seed)); allocation_log = 0.
        for task in obs['tasks']:
            if task['source'] is None or task['target'] is not None: continue
            state, features, mask, prior = prefix.features(task['uid'], weights)
            dist = self.allocation(tensor([state]), tensor([features]), tensor([mask], torch.bool), tensor([prior]))
            action = int(torch.multinomial(dist.probs[0], 1, generator=generator))
            log = float(dist.log_prob(torch.tensor([action]))[0]); allocation_log += log
            tokens.append(dict(state=state, features=features, mask=mask, prior=prior, action=action, probs=dist.probs[0].tolist()))
            plan[task['uid']] = action; prefix.assign(task['uid'], action)
        bank = adapter.motion(prefix.obs if self.method == 'joint' else obs)
        dist = self.motion(tensor([bank['state']]), tensor([bank['features']]), tensor([bank['mask']], torch.bool))
        action = int(torch.multinomial(dist.probs[0], 1, generator=generator))
        row = dict(state=before, allocations=tokens, allocation_log=allocation_log,
            motion_state=bank['state'], motion_features=bank['features'], motion_mask=bank['mask'], motion_action=action,
            motion_probs=dist.probs[0].tolist(), motion_log=float(dist.log_prob(torch.tensor([action]))[0]),
            value=float(self.critic(tensor([before]))[0, 0]), history_weights=dict(weights))
        return row, plan, bank['actions'][action]

    def replay(self, rows):
        states = tensor([r['state'] for r in rows]); n = len(rows)
        owners = []; tokens = []
        for i, row in enumerate(rows):
            owners += [i]*len(row['allocations']); tokens += row['allocations']
        logs = torch.zeros(n); entropy = torch.zeros(n); kl = torch.zeros(n); counts = torch.zeros(n)
        if tokens:
            owner = torch.tensor(owners)
            dist = self.allocation(tensor([x['state'] for x in tokens]), tensor([x['features'] for x in tokens]),
                tensor([x['mask'] for x in tokens], torch.bool), tensor([x['prior'] for x in tokens]))
            old = tensor([x['probs'] for x in tokens]); oldlog = old.clamp_min(1e-30).log()
            token_kl = (old*(oldlog-dist.probs.clamp_min(1e-30).log())).sum(-1)
            logs = logs.index_add(0, owner, dist.log_prob(torch.tensor([x['action'] for x in tokens])))
            entropy = entropy.index_add(0, owner, dist.entropy())
            kl = kl.index_add(0, owner, token_kl)
            counts = counts.index_add(0, owner, torch.ones(len(owner)))
        motion = self.motion(tensor([x['motion_state'] for x in rows]), tensor([x['motion_features'] for x in rows]),
            tensor([x['motion_mask'] for x in rows], torch.bool))
        old = tensor([x['motion_probs'] for x in rows])
        return dict(allocation_log=logs, allocation_entropy=entropy/counts.clamp_min(1), allocation_count=counts,
            allocation_kl=kl, motion_log=motion.log_prob(torch.tensor([x['motion_action'] for x in rows])),
            motion_entropy=motion.entropy(), motion_kl=(old*(old.clamp_min(1e-30).log()-motion.probs.clamp_min(1e-30).log())).sum(-1),
            value=self.critic(states).squeeze(-1))


def update(model, optimizer, rows, epochs=2, minibatch=128):
    advantages = tensor([r['advantage'] for r in rows]); returns = tensor([r['returns'] for r in rows])
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(.05)
    old_a = tensor([r['allocation_log'] for r in rows]); old_m = tensor([r['motion_log'] for r in rows])
    before = {name: {k: v.clone() for k, v in getattr(model, name).state_dict().items()} for name in ('allocation', 'motion')}
    metrics = dict(optimizer_steps=0, actor_steps=0, actor_kl_skips=0, allocation_grad=0., motion_grad=0., critic_loss=0.,
        allocation_loss=0., motion_loss=0., allocation_entropy=0., motion_entropy=0.)
    for _ in range(epochs):
        for ids in torch.randperm(len(rows)).split(minibatch):
            selected = [rows[i] for i in ids.tolist()]; z = model.replay(selected); adv = advantages[ids]
            valid = z['allocation_count'] > 0
            def surrogate(log, old):
                ratio = (log-old).clamp(-20, 20).exp()
                return -torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv)
            la = surrogate(z['allocation_log'], old_a[ids])[valid].mean() if bool(valid.any()) else z['value'].sum()*0
            lm = surrogate(z['motion_log'], old_m[ids]).mean()
            critic = (z['value']-returns[ids]).square().mean()
            accepted = float((z['allocation_kl']+z['motion_kl']).mean().detach()) <= .05
            actor = la+lm-.001*(z['allocation_entropy'].mean()+z['motion_entropy'].mean())
            loss = .5*critic+(actor if accepted else actor*0)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite queue PPO loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not accepted:
                for module in (model.allocation, model.motion):
                    for parameter in module.parameters(): parameter.grad = None
            norms = {}
            for name in ('allocation', 'motion', 'critic'):
                norms[name] = float(nn.utils.clip_grad_norm_(getattr(model, name).parameters(), .5, error_if_nonfinite=True))
            optimizer.step(); metrics['optimizer_steps'] += 1
            metrics['actor_steps'] += int(accepted); metrics['actor_kl_skips'] += int(not accepted)
            metrics['allocation_grad'] += norms['allocation']; metrics['motion_grad'] += norms['motion']
            for key, value in (('critic_loss', critic), ('allocation_loss', la), ('motion_loss', lm),
                    ('allocation_entropy', z['allocation_entropy'].mean()), ('motion_entropy', z['motion_entropy'].mean())):
                metrics[key] += float(value.detach())
    for key in ('allocation_grad', 'motion_grad', 'critic_loss', 'allocation_loss', 'motion_loss', 'allocation_entropy', 'motion_entropy'):
        metrics[key] /= max(metrics['optimizer_steps'], 1)
    kls = []
    with torch.no_grad():
        for start in range(0, len(rows), minibatch):
            z = model.replay(rows[start:start+minibatch]); kls += (z['allocation_kl']+z['motion_kl']).tolist()
    metrics['joint_conditional_kl'] = float(np.mean(kls))
    for name in ('allocation', 'motion'):
        metrics[name+'_parameter_change'] = math.sqrt(sum(float((v-before[name][k]).square().sum()) for k, v in getattr(model, name).state_dict().items()))
    if not all(math.isfinite(v) for v in metrics.values()): raise FloatingPointError('nonfinite queue PPO metric')
    return metrics
