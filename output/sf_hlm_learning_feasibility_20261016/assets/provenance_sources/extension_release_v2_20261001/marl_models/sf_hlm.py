"""SF-HLM policies: central graph quota PPO and feedback-aware Attention MAPPO.

Quota PPO probabilities belong to sampled count transfers, never matching
outputs. Supervised teacher updates are separate from on-policy PPO updates.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from utils.sf_hlm_search import REQUEST_DIM, EDGE_DIM, NODE_DIM, transfer_actions, SEARCH_CONTRACT
from utils.sf_hlm_matching import MATCHING_CONTRACT, FEEDBACK_CONTRACT


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, output_dim))


def collate(states):
    if not states: raise ValueError('No graph states')
    b = len(states); r = max(1, max(len(s['requests']) for s in states)); j = len(states[0]['nodes'])
    arrays = dict(requests=np.zeros((b, r, REQUEST_DIM), np.float32),
                  edges=np.zeros((b, r, j, EDGE_DIM), np.float32),
                  legal=np.zeros((b, r, j), bool), nodes=np.zeros((b, j, NODE_DIM), np.float32),
                  mask=np.zeros((b, len(states[0]['mask'])), bool))
    for i, s in enumerate(states):
        n = len(s['requests'])
        for key in ('requests', 'edges', 'legal'): arrays[key][i, :n] = s[key]
        arrays['nodes'][i] = s['nodes']; arrays['mask'][i] = s['mask']
        if not s['mask'][0]: raise ValueError('Stop must always be available')
    if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError('Nonfinite graph')
    return {k: torch.from_numpy(v) for k, v in arrays.items()}


class QuotaNetwork(nn.Module):
    def __init__(self, num_uavs, hidden=128):
        super().__init__(); self.num_uavs = num_uavs; self.hidden = hidden
        self.request = mlp(REQUEST_DIM, hidden, hidden)
        self.node = mlp(NODE_DIM, hidden, hidden)
        self.edge = mlp(EDGE_DIM, hidden, hidden)
        self.messages = nn.ModuleList([mlp(hidden * 3, hidden, hidden) for _ in range(2)])
        self.req_updates = nn.ModuleList([mlp(hidden * 2, hidden, hidden) for _ in range(2)])
        self.node_updates = nn.ModuleList([mlp(hidden * 2, hidden, hidden) for _ in range(2)])
        self.transfer = mlp(hidden * 3 + 3, hidden, 1)
        self.stop = mlp(hidden * 2, hidden, 1); self.value = mlp(hidden * 2, hidden, 1)
        actions = transfer_actions(num_uavs + 1)[1:]
        self.register_buffer('sources', torch.tensor([a[0] for a in actions], dtype=torch.long))
        self.register_buffer('destinations', torch.tensor([a[1] for a in actions], dtype=torch.long))
        self.register_buffer('amounts', torch.eye(3)[[(1, 2, 4).index(a[2]) for a in actions]])

    def forward(self, graph):
        legal = graph['legal']; weight = legal[..., None].float()
        valid = legal.any(-1)[..., None].float()
        req = self.request(graph['requests']) * valid; node = self.node(graph['nodes']); edge = self.edge(graph['edges']) * weight
        b, r, j, h = edge.shape
        for message, ru, nu in zip(self.messages, self.req_updates, self.node_updates):
            pair = torch.cat([req[:, :, None].expand(-1, -1, j, -1),
                              node[:, None].expand(-1, r, -1, -1), edge], -1)
            msg = message(pair) * weight
            req_msg = msg.sum(2) / weight.sum(2).clamp_min(1.)
            node_msg = msg.sum(1) / weight.sum(1).clamp_min(1.)
            req = ru(torch.cat([req, req_msg], -1)) * valid
            node = nu(torch.cat([node, node_msg], -1))
        pooled_req = req.sum(1) / valid.sum(1).clamp_min(1.)
        pooled_node = node.mean(1); pooled = torch.cat([pooled_req, pooled_node], -1)
        n_actions = len(self.sources)
        action_features = torch.cat([node[:, self.sources], node[:, self.destinations],
                                    pooled_req[:, None].expand(-1, n_actions, -1),
                                    self.amounts[None].expand(b, -1, -1)], -1)
        logits = torch.cat([self.stop(pooled), self.transfer(action_features).squeeze(-1)], -1)
        logits = logits.masked_fill(~graph['mask'], -1e9)
        return logits, self.value(pooled).squeeze(-1)


def advantages(records, gamma, lam):
    if not records: return np.empty(0), np.empty(0)
    rewards = np.asarray([r['reward'] for r in records], dtype=np.float32)
    values = np.asarray([r['value'] for r in records], dtype=np.float32)
    result = np.zeros_like(values); running = np.zeros_like(values[-1])
    for i in range(len(records) - 1, -1, -1):
        cont = 1. - float(records[i]['done'])
        next_value = values[i + 1] if i + 1 < len(records) else np.zeros_like(values[i])
        delta = rewards[i] + gamma * cont * next_value - values[i]
        running = delta + gamma * lam * cont * running; result[i] = running
    return result, result + values


class QuotaPPO:
    def __init__(self, num_uavs=5, hidden=128, learning_rate=3e-4):
        self.num_uavs = num_uavs; self.hidden = hidden; self.network = QuotaNetwork(num_uavs, hidden)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def forward_states(self, states): return self.network(collate(states))

    @torch.no_grad()
    def act(self, state, exploration=False):
        logits, value = self.forward_states([state]); distribution = Categorical(logits=logits)
        action = distribution.sample() if exploration else logits.argmax(-1)
        return int(action[0]), float(distribution.log_prob(action)[0]), float(value[0])

    def evaluate(self, states, actions):
        logits, values = self.forward_states(states); distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), values

    def update(self, records, epochs=8, batch_size=64):
        if not records: return dict(ppo_loss=0., value_loss=0., initial_ratio_mean=1.)
        adv, ret = advantages(records, 1., .95)
        adv = torch.tensor((adv - adv.mean()) / max(float(adv.std()), 1e-6)); ret = torch.tensor(ret)
        old = torch.tensor([r['log_prob'] for r in records]); actions = torch.tensor([r['action'] for r in records])
        initial = []
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                ids = list(range(start, min(start + batch_size, len(records))))
                lp, _, _ = self.evaluate([records[i]['state'] for i in ids], actions[ids])
                initial.extend(torch.exp(lp - old[ids]).tolist())
        if not np.allclose(initial, 1., atol=2e-5): raise ValueError('Quota PPO is not on-policy')
        losses = []
        for _ in range(epochs):
            for ids in torch.randperm(len(records)).split(batch_size):
                lp, entropy, value = self.evaluate([records[int(i)]['state'] for i in ids], actions[ids])
                ratio = torch.exp(lp - old[ids]); surrogate = torch.minimum(ratio * adv[ids], ratio.clamp(.8, 1.2) * adv[ids])
                actor = -surrogate.mean() - .005 * entropy.mean(); critic = .5 * (value - ret[ids]).square().mean()
                self.optimizer.zero_grad(set_to_none=True); (actor + critic).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), .5); self.optimizer.step()
                losses.append((float(actor.detach()), float(critic.detach())))
        if not np.isfinite(losses).all(): raise FloatingPointError('Nonfinite quota PPO')
        return dict(ppo_loss=float(np.mean(losses, 0)[0]), value_loss=float(np.mean(losses, 0)[1]), initial_ratio_mean=float(np.mean(initial)))

    def pretrain(self, records, epochs=5, batch_size=64):
        losses = []
        for _ in range(epochs):
            for ids in torch.randperm(len(records)).split(batch_size):
                if not len(ids): continue
                sample = [records[int(i)] for i in ids]
                logits, _ = self.forward_states([r['state'] for r in sample])
                loss = nn.functional.cross_entropy(logits, torch.tensor([r['action'] for r in sample]))
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), .5); self.optimizer.step(); losses.append(float(loss.detach()))
        return dict(supervised_loss=float(np.mean(losses)) if losses else 0., teacher_samples=len(records))

    def save(self, folder, contract):
        path = Path(folder); path.mkdir(parents=True, exist_ok=True)
        torch.save(dict(network=self.network.state_dict(), optimizer=self.optimizer.state_dict(), contract=contract,
                        search=SEARCH_CONTRACT, matching=MATCHING_CONTRACT, hidden=self.hidden, num_uavs=self.num_uavs), path / 'quota_ppo.pt')

    def load(self, folder, contract):
        data = torch.load(Path(folder) / 'quota_ppo.pt', weights_only=False, map_location='cpu')
        if data['contract'] != contract or data['search'] != SEARCH_CONTRACT or data['matching'] != MATCHING_CONTRACT or data['hidden'] != self.hidden or data['num_uavs'] != self.num_uavs:
            raise ValueError('Quota checkpoint incompatible')
        self.network.load_state_dict(data['network']); self.optimizer.load_state_dict(data['optimizer'])


class FeedbackMAPPO:
    """Keep original base-observation attention; fuse separate coordinator input.

    Raw sampled Gaussian actions retain their probabilities; clipping is the
    existing deterministic execution mapping. Zero feedback is an ablation.
    """
    def __init__(self, obs_dim, num_uavs=5, enabled=True):
        from marl_models.attention_mappo.agents import ActorNetwork, CriticNetwork
        self.num_uavs = num_uavs; self.enabled = enabled; self.obs_dim = obs_dim
        self.actor = ActorNetwork(obs_dim, 2); self.critic = CriticNetwork(obs_dim)
        self.actor_feedback = mlp((num_uavs + 1) * 6 + 8, 64, 2)
        self.critic_feedback = mlp((num_uavs + 1) * 6, 64, num_uavs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.actor_feedback.parameters()) + list(self.critic_feedback.parameters())

    def distribution(self, obs, summary):
        b = obs.shape[0]; flat = summary.reshape(b, -1)
        if not self.enabled: flat = flat * 0
        raw = self.actor(obs.reshape(-1, obs.shape[-1]))
        own = summary[:, :self.num_uavs] if self.enabled else summary[:, :self.num_uavs] * 0
        conditioning = torch.cat([raw.mean, flat[:, None].expand(-1, self.num_uavs, -1).reshape(-1, flat.shape[-1]), own.reshape(-1, 6)], -1)
        shift = self.actor_feedback(conditioning)
        mean = torch.tanh(torch.atanh(raw.mean.clamp(-.999999, .999999)) + shift)
        values = self.critic(obs) + self.critic_feedback(flat)
        return Normal(mean.reshape(b, self.num_uavs, 2), raw.stddev.reshape(b, self.num_uavs, 2)), values

    @torch.no_grad()
    def act(self, obs, summary, exploration=False):
        dist, value = self.distribution(torch.as_tensor(obs, dtype=torch.float32)[None], torch.as_tensor(summary, dtype=torch.float32)[None])
        action = dist.sample() if exploration else dist.mean
        return action[0].numpy(), dist.log_prob(action).sum(-1)[0].numpy(), value[0].numpy()

    def evaluate(self, obs, summary, actions):
        dist, value = self.distribution(obs, summary)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value

    def update(self, records, epochs=8, batch_size=256):
        if not records: return dict(actor_loss=0., critic_loss=0.)
        import config
        gamma = float(getattr(config, 'DISCOUNT_FACTOR', .99))
        adv, ret = advantages(records, gamma, .95)
        adv = torch.tensor((adv - adv.mean()) / max(float(adv.std()), 1e-6)); ret = torch.tensor(ret)
        obs = torch.tensor(np.stack([r['obs'] for r in records])); summary = torch.tensor(np.stack([r['feedback'] for r in records]))
        actions = torch.tensor(np.stack([r['action'] for r in records])); old = torch.tensor(np.stack([r['log_prob'] for r in records]))
        initial = []
        with torch.no_grad():
            for ids in torch.arange(len(records)).split(batch_size):
                lp, _, _ = self.evaluate(obs[ids], summary[ids], actions[ids]); initial.extend(torch.exp(lp - old[ids]).numpy().ravel())
        if not np.allclose(initial, 1., atol=2e-5): raise ValueError('Feedback MAPPO is not on-policy')
        losses = []
        for _ in range(epochs):
            for ids in torch.randperm(len(records)).split(batch_size):
                lp, entropy, value = self.evaluate(obs[ids], summary[ids], actions[ids]); ratio = torch.exp(lp - old[ids])
                actor = -torch.minimum(ratio * adv[ids], ratio.clamp(.8, 1.2) * adv[ids]).mean() - .005 * entropy.mean()
                critic = .5 * (value - ret[ids]).square().mean()
                self.optimizer.zero_grad(set_to_none=True); (actor + critic).backward()
                nn.utils.clip_grad_norm_(self.parameters(), .5); self.optimizer.step()
                losses.append((float(actor.detach()), float(critic.detach())))
        if not np.isfinite(losses).all(): raise FloatingPointError('Nonfinite feedback MAPPO')
        return dict(actor_loss=float(np.mean(losses, 0)[0]), critic_loss=float(np.mean(losses, 0)[1]), initial_ratio_mean=float(np.mean(initial)))

    def save(self, folder, contract):
        path = Path(folder); path.mkdir(parents=True, exist_ok=True)
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        actor_feedback=self.actor_feedback.state_dict(), critic_feedback=self.critic_feedback.state_dict(),
                        optimizer=self.optimizer.state_dict(), enabled=self.enabled, num_uavs=self.num_uavs, obs_dim=self.obs_dim,
                        feedback=FEEDBACK_CONTRACT, contract=contract), path / 'feedback_mappo.pt')

    def load(self, folder, contract):
        data = torch.load(Path(folder) / 'feedback_mappo.pt', weights_only=False, map_location='cpu')
        if data['contract'] != contract or data['enabled'] != self.enabled or data['num_uavs'] != self.num_uavs or data['obs_dim'] != self.obs_dim or data['feedback'] != FEEDBACK_CONTRACT:
            raise ValueError('Upper feedback checkpoint incompatible')
        for name in ('actor', 'critic', 'actor_feedback', 'critic_feedback'): getattr(self, name).load_state_dict(data[name])
        self.optimizer.load_state_dict(data['optimizer'])
