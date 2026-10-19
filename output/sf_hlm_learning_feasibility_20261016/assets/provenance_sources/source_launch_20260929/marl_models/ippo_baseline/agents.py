"""
IPPO baseline 的 Actor 和 Critic 网络。

与 VanillaMAPPO 的关键区别：
- Critic 输入为单 agent 局部观测 (obs_dim)，而非全局状态 (num_agents * obs_dim)
- 体现 independent learning 范式：每个 agent 用局部信息评估价值
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import config
from marl_models.buffer_and_helpers import layer_init


class IPPOActorNetwork(nn.Module):
    """IPPO Actor：与 VanillaActorNetwork 结构一致，独立处理每个 agent。"""

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


class IPPOCriticNetwork(nn.Module):
    """IPPO Critic：输入为单 agent 局部观测，输出标量 value。

    与 VanillaCriticNetwork 的关键区别：
    - 输入维度: obs_dim (局部) vs num_agents * obs_dim (全局)
    - 输出维度: 1 (单 agent) vs num_agents (所有 agent)
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.value = nn.Sequential(
            layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)),
            nn.LayerNorm(config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """输入 (batch, obs_dim)，输出 (batch,)。"""
        return self.value(obs).squeeze(-1)
