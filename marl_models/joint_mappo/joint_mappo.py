from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
import torch
from torch.distributions import Categorical, Normal

import config
from marl_models.base_model import ExperienceBatch, MARLModel
from marl_models.buffer_and_helpers import get_state_dict, load_safe
from marl_models.joint_mappo.agents import JointActorNetwork, JointCriticNetwork


class JointMAPPO(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.joint_obs_dim = obs_dim
        self.trajectory_action_dim = config.ACTION_DIM
        self.max_requests = config.MAX_OFFLOAD_REQUESTS_PER_UAV
        self.num_offload_actions = config.OFFLOAD_NUM_ACTIONS
        self.joint_state_dim = self.joint_obs_dim * num_agents
        self.actor = JointActorNetwork(self.joint_obs_dim, self.trajectory_action_dim, self.max_requests, self.num_offload_actions).to(device)
        self.critic = JointCriticNetwork(self.joint_state_dim, num_agents).to(device)
        self.actor = cast(JointActorNetwork, torch.compile(self.actor, backend="eager"))
        self.critic = cast(JointCriticNetwork, torch.compile(self.critic, backend="eager"))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        masks = np.ones((observations.shape[0], self.max_requests, self.num_offload_actions), dtype=np.float32)
        trajectory_actions, _, _, _ = self.get_action_and_value(observations, masks=masks, exploration=exploration)
        return trajectory_actions

    def get_action_and_value(
        self,
        obs: np.ndarray,
        state: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        exploration: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del state
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        if masks is None:
            masks_tensor = torch.ones((obs_tensor.shape[0], self.max_requests, self.num_offload_actions), dtype=torch.float32, device=self.device)
        else:
            masks_tensor = torch.from_numpy(masks).float().to(self.device)
        with torch.no_grad():
            trajectory_dist, offload_logits = self.actor(obs_tensor, masks_tensor)
            if exploration:
                trajectory_actions = trajectory_dist.sample()
                offload_actions = Categorical(logits=offload_logits).sample()
            else:
                trajectory_actions = trajectory_dist.mean
                offload_actions = torch.argmax(offload_logits, dim=-1)
            offload_dist = Categorical(logits=offload_logits)
            trajectory_log_probs = trajectory_dist.log_prob(trajectory_actions).sum(dim=-1)
            valid_slots = (masks_tensor.sum(dim=-1) > 0.0).float()
            offload_slot_log_probs = offload_dist.log_prob(offload_actions) * valid_slots
            denom = valid_slots.sum(dim=-1).clamp_min(1.0)
            offload_log_probs = offload_slot_log_probs.sum(dim=-1) / denom
            log_probs = trajectory_log_probs + offload_log_probs
            values = self.critic(obs_tensor.unsqueeze(0)).squeeze(0)
        return (
            torch.clamp(trajectory_actions, -1.0, 1.0).cpu().numpy(),
            offload_actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def update(self, batch: ExperienceBatch) -> dict[str, float]:
        assert isinstance(batch, dict), "JointMAPPO expects an on-policy dict batch"
        obs_batch: torch.Tensor = batch["obs"]
        trajectory_actions_batch: torch.Tensor = batch["trajectory_actions"]
        offload_actions_batch: torch.Tensor = batch["offload_actions"].long()
        offload_masks_batch: torch.Tensor = batch["offload_masks"]
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

        trajectory_dist, offload_logits = self.actor(obs_batch, offload_masks_batch)
        offload_dist = Categorical(logits=offload_logits)
        trajectory_log_probs = trajectory_dist.log_prob(trajectory_actions_batch).sum(dim=-1)
        offload_slot_log_probs = offload_dist.log_prob(offload_actions_batch) * valid_slots
        denom = valid_slots.sum(dim=-1).clamp_min(1.0)
        offload_log_probs = offload_slot_log_probs.sum(dim=-1) / denom
        new_log_probs = trajectory_log_probs + offload_log_probs
        trajectory_entropy = trajectory_dist.entropy().sum(dim=-1)
        offload_entropy = (offload_dist.entropy() * valid_slots).sum(dim=-1) / denom
        entropy = trajectory_entropy + offload_entropy

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
                    "joint_obs_dim": self.joint_obs_dim,
                    "trajectory_action_dim": self.trajectory_action_dim,
                    "max_requests": self.max_requests,
                    "num_offload_actions": self.num_offload_actions,
                    "num_agents": self.num_agents,
                },
            },
            os.path.join(directory, "joint_mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path = os.path.join(directory, "joint_mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        checkpoint: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=True)
        load_safe(self.actor, checkpoint["actor"])
        load_safe(self.critic, checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
