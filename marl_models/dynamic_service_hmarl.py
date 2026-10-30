"""Recurrent spatial MAPPO and direct graph-autoregressive PPO policies."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class UpperPolicy(nn.Module):
    def __init__(self, memory=True, feedback=True):
        super().__init__()
        self.memory, self.feedback_enabled = memory, feedback
        self.ue_embed = nn.Sequential(nn.Linear(6, 128), nn.Tanh())
        self.self_embed = nn.Sequential(nn.Linear(6, 128), nn.Tanh())
        self.ue_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.neighbor_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(384, 128), nn.Tanh())
        self.gru = nn.GRUCell(128, 128)
        self.actor = nn.Linear(128, 2)
        self.log_std = nn.Parameter(torch.full((2,), -.7))
        self.critic = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def initial_state(self):
        return torch.zeros(5, 128, device=self.device)

    def forward(self, observation, hidden):
        ue = tensor(observation['ue'], self.device)
        visible = torch.as_tensor(observation['visible'], dtype=torch.bool, device=self.device)
        nodes = tensor(observation['nodes'], self.device)
        feedback = tensor(observation['feedback'], self.device)
        if not self.feedback_enabled:
            feedback = feedback * 0
        query = self.self_embed(torch.cat((nodes, feedback), -1))
        tokens = self.ue_embed(ue)
        # A zero dummy token prevents all-masked attention NaNs.
        tokens = torch.cat((tokens, torch.zeros(5, 1, 128, device=self.device)), 1)
        padding = torch.cat((~visible, torch.zeros(5, 1, dtype=torch.bool, device=self.device)), 1)
        demand = self.ue_attention(query[:, None], tokens, tokens, key_padding_mask=padding, need_weights=False)[0][:, 0]
        neighbor_mask = torch.cdist(nodes[:, :2], nodes[:, :2]) > 460/700
        neighbors = self.neighbor_attention(query[None], query[None], query[None], attn_mask=neighbor_mask, need_weights=False)[0][0]
        x = self.fuse(torch.cat((query, demand, neighbors), -1))
        new_hidden = self.gru(x, hidden) if self.memory else x
        distribution = Normal(self.actor(new_hidden), self.log_std.clamp(-3, 1).exp())
        global_hidden = new_hidden.mean(0, keepdim=True).expand(5, -1)
        values = self.critic(torch.cat((new_hidden, global_hidden), -1)).squeeze(-1)
        return distribution, values, new_hidden

    @torch.no_grad()
    def act(self, observation, hidden, stochastic=False):
        dist, value, new_hidden = self(observation, hidden)
        raw = dist.sample() if stochastic else dist.mean
        return dict(raw=raw.cpu().numpy(), actions=raw.clamp(-1, 1).cpu().numpy(),
                    logprob=dist.log_prob(raw).sum(-1).cpu().numpy(), value=value.cpu().numpy(), hidden=new_hidden)


class LowerPolicy(nn.Module):
    def __init__(self, mode='autoregressive'):
        super().__init__()
        if mode not in ('autoregressive', 'parallel', 'independent'):
            raise ValueError(mode)
        self.mode = mode
        self.request_embed = nn.Sequential(nn.Linear(4, 128), nn.Tanh())
        self.node_embed = nn.Sequential(nn.Linear(5, 128), nn.Tanh())
        self.edge_embed = nn.Sequential(nn.Linear(4, 128), nn.Tanh())
        self.messages = nn.ModuleList([nn.Linear(256, 128) for _ in range(2)])
        self.node_updates = nn.ModuleList([nn.Linear(256, 128) for _ in range(2)])
        self.request_updates = nn.ModuleList([nn.Linear(256, 128) for _ in range(2)])
        self.score = nn.Sequential(nn.Linear(128*3+4, 128), nn.Tanh(), nn.Linear(128, 1))
        self.value = nn.Sequential(nn.Linear(128*2+4, 128), nn.Tanh(), nn.Linear(128, 1))

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, batch):
        r = tensor(np.c_[batch.size/5e6, batch.cycles/14e9, batch.deadline, batch.source/5], self.device)
        n = tensor(np.c_[batch.node_positions/700, batch.capacity/2e11, np.arange(6) == 5], self.device)
        single = batch.cycles[:, None]/batch.capacity
        edge = tensor(np.stack([np.minimum(batch.tx, 10)/10, single,
                                np.clip(batch.deadline[:, None]-batch.tx-single, -10, 1), batch.legal], -1), self.device)
        request, node, edge = self.request_embed(r), self.node_embed(n), self.edge_embed(edge)
        mask = tensor(batch.legal, self.device)[..., None]
        if self.mode != 'independent':
            for message, nupdate, rupdate in zip(self.messages, self.node_updates, self.request_updates):
                expanded = request[:, None].expand(-1, 6, -1)
                msg = torch.tanh(message(torch.cat((expanded, edge), -1))) * mask
                pooled = msg.sum(0)/mask.sum(0).clamp_min(1)
                node = torch.tanh(nupdate(torch.cat((node, pooled), -1)))
                pooled = (node[None]*mask).sum(1)/mask.sum(1).clamp_min(1)
                request = torch.tanh(rupdate(torch.cat((request, pooled), -1)))
        return request, node, edge

    def dynamic_features(self, batch, chosen, i):
        features = np.zeros((6, 4), np.float32)
        if self.mode != 'autoregressive':
            return tensor(features, self.device)
        assignment = np.array(chosen, dtype=int)
        counts = np.bincount(assignment, minlength=6)
        features[:, 0] = counts/100
        for j in range(6):
            members = np.flatnonzero(assignment == j)
            features[j, 1] = batch.cycles[members].sum()/1e12
            if len(members):
                old = batch.tx[members, j] + batch.cycles[members]*counts[j]/batch.capacity[j]
                new = batch.tx[members, j] + batch.cycles[members]*(counts[j]+1)/batch.capacity[j]
                features[j, 2] = (old > batch.deadline[members]).sum()/100
                features[j, 3] = ((new > batch.deadline[members]) & (old <= batch.deadline[members])).sum()/100
        return tensor(features, self.device)

    def distribution(self, batch, encoded, chosen, i):
        request, nodes, edges = encoded
        dynamic = self.dynamic_features(batch, chosen, i)
        r = request[i].expand(6, -1)
        logits = self.score(torch.cat((r, nodes, edges[i], dynamic), -1)).squeeze(-1)
        mask = torch.as_tensor(batch.legal[i], device=self.device, dtype=torch.bool)
        distribution = Categorical(logits=logits.masked_fill(~mask, -torch.inf))
        value = self.value(torch.cat((request.mean(0), nodes.mean(0), dynamic.mean(0)))).squeeze(-1)
        return distribution, value

    @torch.no_grad()
    def act(self, batch, stochastic=False):
        if not len(batch):
            return dict(assignment=np.empty(0, np.int64), logprob=np.empty(0), values=np.empty(0))
        encoded = self.encode(batch)
        chosen, logs, values = [], [], []
        for i in range(len(batch)):
            dist, value = self.distribution(batch, encoded, chosen, i)
            action = dist.sample() if stochastic else dist.logits.argmax()
            chosen.append(int(action))
            logs.append(float(dist.log_prob(action)))
            values.append(float(value))
        return dict(assignment=np.array(chosen, np.int64), logprob=np.array(logs), values=np.array(values))

    def replay(self, batch, assignment):
        encoded = self.encode(batch)
        chosen, logs, values, entropies = [], [], [], []
        for i, action in enumerate(assignment):
            dist, value = self.distribution(batch, encoded, chosen, i)
            logs.append(dist.log_prob(torch.as_tensor(action, device=self.device)))
            values.append(value)
            entropies.append(dist.entropy())
            chosen.append(int(action))
        return torch.stack(logs), torch.stack(values), torch.stack(entropies)


def ppo_loss(new_log, old_log, advantages, values, targets, entropy):
    ratio = (new_log-old_log).exp()
    actor = -torch.minimum(ratio*advantages, ratio.clamp(.8, 1.2)*advantages).mean()
    return actor + .5*(values-targets).square().mean() - .005*entropy.mean()


def update_lower(policy, optimizer, transitions):
    losses = []
    # Each graph is a decoder trajectory with terminal batch reward, gamma=1.
    for _ in range(4):
        pending = []
        for item in transitions:
            if not len(item['batch']):
                continue
            log, values, entropy = policy.replay(item['batch'], item['assignment'])
            target = torch.full_like(values, item['reward'])
            old_value = tensor(item['values'], policy.device)
            advantage = target-old_value
            loss = ppo_loss(log, tensor(item['logprob'], policy.device), advantage,
                            values, target, entropy)
            pending.append(loss)
            losses.append(float(loss.detach()))
            if len(pending) == 8:
                optimizer.zero_grad(set_to_none=True)
                torch.stack(pending).mean().backward()
                nn.utils.clip_grad_norm_(policy.parameters(), .5)
                optimizer.step()
                pending = []
        if pending:
            optimizer.zero_grad(set_to_none=True)
            torch.stack(pending).mean().backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
    return float(np.mean(losses)) if losses else 0.


def update_upper(policy, optimizer, rollout):
    rewards = np.array([x['reward'] for x in rollout])[:, None]
    old_values = np.stack([x['value'] for x in rollout])
    advantages = np.zeros_like(old_values)
    running, next_value = np.zeros(5), np.zeros(5)
    for t in reversed(range(len(rollout))):
        delta = rewards[t] + .99*next_value - old_values[t]
        running = delta + .99*.95*running
        advantages[t], next_value = running, old_values[t]
    targets = advantages + old_values
    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
    losses = []
    for _ in range(4):
        for start in range(0, len(rollout), 32):
            burn = max(0, start-16)
            hidden = tensor(rollout[burn]['hidden_before'], policy.device)
            with torch.no_grad():
                for t in range(burn, start):
                    _, _, hidden = policy(rollout[t]['observation'], hidden)
            log, vals, entropy = [], [], []
            end = min(start+32, len(rollout))
            for t in range(start, end):
                dist, value, hidden = policy(rollout[t]['observation'], hidden)
                log.append(dist.log_prob(tensor(rollout[t]['raw'], policy.device)).sum(-1))
                vals.append(value)
                entropy.append(dist.entropy().sum(-1))
            loss = ppo_loss(torch.stack(log), tensor(np.stack([x['logprob'] for x in rollout[start:end]]), policy.device),
                            tensor(advantages[start:end], policy.device), torch.stack(vals),
                            tensor(targets[start:end], policy.device), torch.stack(entropy))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))
