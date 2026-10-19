from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import config
from marl_models.buffer_and_helpers import layer_init


class JointActorNetwork(nn.Module):
    def __init__(self, joint_obs_dim: int, trajectory_action_dim: int, max_requests: int, num_offload_actions: int) -> None:
        super().__init__()
        self.max_requests = max_requests
        self.num_offload_actions = num_offload_actions
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(joint_obs_dim, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
        )
        self.trajectory_mean = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, trajectory_action_dim), std=0.01)
        self.trajectory_log_std = nn.Parameter(torch.zeros(1, trajectory_action_dim))
        self.offload_head = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, max_requests * num_offload_actions), std=0.01)

    def forward(self, obs: torch.Tensor, masks: torch.Tensor | None = None) -> tuple[Normal, torch.Tensor]:
        original_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        features = self.backbone(flat_obs)
        mean = torch.tanh(self.trajectory_mean(features)).view(*original_shape, -1)
        log_std = torch.clamp(self.trajectory_log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        trajectory_dist = Normal(mean, std)
        offload_logits = self.offload_head(features).view(*original_shape, self.max_requests, self.num_offload_actions)
        if masks is not None:
            offload_logits = offload_logits.masked_fill(masks <= 0.0, -1.0e9)
        return trajectory_dist, offload_logits


class JointCriticNetwork(nn.Module):
    def __init__(self, joint_state_dim: int, num_agents: int) -> None:
        super().__init__()
        self.value = nn.Sequential(
            layer_init(nn.Linear(joint_state_dim, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, num_agents), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.reshape(obs.shape[0], -1)
        return self.value(obs)
