"""
IPPO (Independent PPO) baseline。

与 VanillaMAPPO 的关键区别：
- Critic 使用单 agent 局部观测，而非全局状态
- 体现 independent learning：每个 agent 独立评估自己的 value
- 消融 CTDE 和多智能体协作的必要性
"""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import torch
from torch.distributions import Normal

import config
from marl_models.base_model import ExperienceBatch, MARLModel
from marl_models.buffer_and_helpers import get_state_dict, load_safe
from marl_models.ippo_baseline.agents import IPPOActorNetwork, IPPOCriticNetwork


class IPPOBaseline(MARLModel):
    """IPPO / Self-Interested PPO：独立学习，无 centralized critic。"""

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.actor = IPPOActorNetwork(obs_dim, action_dim).to(device)
        self.critic = IPPOCriticNetwork(obs_dim).to(device)
        self.actor = cast(IPPOActorNetwork, torch.compile(self.actor, backend="eager"))
        self.critic = cast(IPPOCriticNetwork, torch.compile(self.critic, backend="eager"))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observations).float().to(self.device)
            dist: Normal = self.actor(obs_tensor)
            actions = dist.sample() if exploration else dist.mean
        return np.clip(actions.cpu().numpy(), -1.0, 1.0)

    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取动作、log_prob 和 value。

        关键区别：critic 使用局部 obs 而非全局 state。
        state 参数保留以兼容接口，但不被使用。
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            dist: Normal = self.actor(obs_tensor)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
            # IPPO: critic 使用局部 obs，每个 agent 独立评估
            values = self.critic(obs_tensor)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, batch: ExperienceBatch) -> dict[str, float]:
        assert isinstance(batch, dict), "IPPOBaseline expects OnPolicyExperienceBatch (dict)"

        obs_batch: torch.Tensor = batch["obs"]
        actions_batch: torch.Tensor = batch["actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        old_values_batch: torch.Tensor = batch["old_values"]

        # --- Critic update ---
        # IPPO: critic 使用局部 obs，展平为 (batch * num_agents, obs_dim)
        flat_obs_for_critic = obs_batch.reshape(-1, self.obs_dim)
        values = self.critic(flat_obs_for_critic)
        old_values_flat = old_values_batch.reshape(-1)
        returns_flat = returns_batch.reshape(-1)

        values_clipped = old_values_flat + torch.clamp(
            values - old_values_flat, -config.PPO_CLIP_EPS, config.PPO_CLIP_EPS
        )
        value_loss_1 = (values - returns_flat).pow(2)
        value_loss_2 = (values_clipped - returns_flat).pow(2)
        critic_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        # --- Actor update ---
        flat_obs = obs_batch.reshape(-1, self.obs_dim)
        flat_actions = actions_batch.reshape(-1, self.action_dim)
        flat_old_log_probs = old_log_probs_batch.reshape(-1)
        flat_advantages = advantages_batch.reshape(-1)

        dist: Normal = self.actor(flat_obs)
        new_log_probs = dist.log_prob(flat_actions).sum(dim=-1)
        ratio = torch.exp(new_log_probs - flat_old_log_probs)
        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * flat_advantages
        entropy = dist.entropy().mean()
        actor_loss = -torch.min(surr1, surr2).mean() - config.PPO_ENTROPY_COEF * entropy

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        return {
            "actor": float(actor_loss.item()),
            "critic": float(critic_loss.item()),
            "entropy": float(entropy.item()),
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
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "num_agents": self.num_agents,
                },
            },
            os.path.join(directory, "ippo_baseline.pth"),
        )

    def load(self, directory: str) -> None:
        path = os.path.join(directory, "ippo_baseline.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        load_safe(self.actor, checkpoint["actor"])
        load_safe(self.critic, checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
