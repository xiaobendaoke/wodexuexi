from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import config
from marl_models.buffer_and_helpers import layer_init


class VanillaActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
        )
        self.mean = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        features = self.net(obs)
        mean = torch.tanh(self.mean(features))
        log_std = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)


class VanillaCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, num_agents: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.value = nn.Sequential(
            layer_init(nn.Linear(state_dim, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, num_agents), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 3:
            state = state.reshape(state.shape[0], -1)
        return self.value(state)
