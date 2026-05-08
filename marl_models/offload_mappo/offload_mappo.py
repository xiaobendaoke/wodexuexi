from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import config
from marl_models.base_model import ExperienceBatch, MARLModel
from marl_models.buffer_and_helpers import get_state_dict, layer_init, load_safe


class OffloadActor(nn.Module):
    def __init__(self, obs_dim: int, max_requests: int, num_actions: int) -> None:
        super().__init__()
        self.max_requests = max_requests
        self.num_actions = num_actions
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
        )
        self.head = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, max_requests * num_actions), std=0.01)

    def forward(self, obs: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        features = self.net(obs)
        logits = self.head(features).view(*obs.shape[:-1], self.max_requests, self.num_actions)
        if masks is not None:
            logits = logits.masked_fill(masks <= 0.0, -1.0e9)
        return logits


class OffloadCritic(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.ATTN_HIDDEN_DIM)),
            nn.LayerNorm(config.ATTN_HIDDEN_DIM),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=config.ATTN_HIDDEN_DIM,
            num_heads=config.ATTN_NUM_HEADS,
            batch_first=True,
        )
        self.value = nn.Sequential(
            layer_init(nn.Linear(config.ATTN_HIDDEN_DIM * 2, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(obs)
        context, _ = self.attn(emb, emb, emb, need_weights=False)
        fused = torch.cat([emb, context], dim=-1)
        return self.value(fused).squeeze(-1)


class OffloadMAPPO(MARLModel):
    """Discrete lower-layer MAPPO for request-level service offloading."""

    def __init__(
        self,
        model_name: str,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        max_requests: int | None = None,
        num_actions: int | None = None,
    ) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.max_requests = int(max_requests or config.MAX_OFFLOAD_REQUESTS_PER_UAV)
        self.num_actions = int(num_actions or config.OFFLOAD_NUM_ACTIONS)
        self.actor = OffloadActor(obs_dim, self.max_requests, self.num_actions).to(device)
        self.critic = OffloadCritic(obs_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        masks = np.ones((self.num_agents, self.max_requests, self.num_actions), dtype=np.float32)
        actions, _, _ = self.get_action_and_value(observations, masks=masks, exploration=exploration)
        return actions

    def get_action_and_value(
        self,
        obs: np.ndarray,
        state: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        exploration: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del state
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        if masks is None:
            masks_tensor = torch.ones(
                (obs_tensor.shape[0], self.max_requests, self.num_actions),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            masks_tensor = torch.from_numpy(masks).float().to(self.device)
        with torch.no_grad():
            logits = self.actor(obs_tensor, masks_tensor)
            dist = Categorical(logits=logits)
            if exploration:
                actions = dist.sample()
            else:
                actions = torch.argmax(logits, dim=-1)
            slot_valid = (masks_tensor.sum(dim=-1) > 0.0).float()
            action_log_probs = dist.log_prob(actions) * slot_valid
            denom = slot_valid.sum(dim=-1).clamp_min(1.0)
            log_probs = action_log_probs.sum(dim=-1) / denom
            values = self.critic(obs_tensor.unsqueeze(0)).squeeze(0)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, batch: ExperienceBatch) -> dict[str, float]:
        assert isinstance(batch, dict), "OffloadMAPPO expects an on-policy dict batch"
        obs_batch: torch.Tensor = batch["obs"]
        actions_batch: torch.Tensor = batch["actions"].long()
        masks_batch: torch.Tensor = batch["masks"]
        valid_slots: torch.Tensor = batch["valid_slots"]
        old_log_probs: torch.Tensor = batch["old_log_probs"]
        advantages: torch.Tensor = batch["advantages"]
        returns: torch.Tensor = batch["returns"]
        old_values: torch.Tensor = batch["old_values"]

        values = self.critic(obs_batch)
        values_clipped = old_values + torch.clamp(values - old_values, -config.PPO_CLIP_EPS, config.PPO_CLIP_EPS)
        value_loss_1 = (values - returns).pow(2)
        value_loss_2 = (values_clipped - returns).pow(2)
        critic_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        logits = self.actor(obs_batch, masks_batch)
        dist = Categorical(logits=logits)
        slot_log_probs = dist.log_prob(actions_batch) * valid_slots
        denom = valid_slots.sum(dim=-1).clamp_min(1.0)
        new_log_probs = slot_log_probs.sum(dim=-1) / denom
        entropy = (dist.entropy() * valid_slots).sum(dim=-1) / denom

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - config.PPO_ENTROPY_COEF * entropy.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        return {
            "actor": float(actor_loss.item()),
            "critic": float(critic_loss.item()),
            "entropy": float(entropy.mean().item()),
        }

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        torch.save(
            {
                "actor": get_state_dict(self.actor),
                "critic": get_state_dict(self.critic),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "metadata": {
                    "max_requests": self.max_requests,
                    "num_actions": self.num_actions,
                    "obs_dim": self.obs_dim,
                },
            },
            os.path.join(directory, "offload_mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path = os.path.join(directory, "offload_mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        checkpoint: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=True)
        load_safe(self.actor, checkpoint["actor"])
        load_safe(self.critic, checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
