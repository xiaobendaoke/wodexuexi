"""
中文注释说明：marl_models/masac/agents.py

文件作用：
    定义MASAC 多智能体软演员评论家算法中的智能体、Actor/Critic 网络和参数更新逻辑，是该算法训练的执行单元。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - ActorNetwork: 核心类，封装本模块中的主要状态和行为。
    - CriticNetwork: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    config, marl_models, torch

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

import config
from marl_models.buffer_and_helpers import layer_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# 类 ActorNetwork，继承自 ：核心类，封装本模块中的主要状态和行为。
class ActorNetwork(nn.Module):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_dim, action_dim。
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.mean: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim))
        self.log_std: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim))

    # 函数 forward：定义神经网络前向传播计算，主要参数：obs。
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))
        mean: torch.Tensor = self.mean(x)
        log_std: torch.Tensor = torch.clamp(self.log_std(x), min=config.LOG_STD_MIN, max=config.LOG_STD_MAX)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return mean, log_std

    # 函数 sample：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs。
    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std: torch.Tensor = log_std.exp()
        dist: Normal = Normal(mean, std)

        # Reparameterization for backpropagation
        x_t: torch.Tensor = dist.rsample()
        y_t: torch.Tensor = torch.tanh(x_t)  # Squash action to be in [-1, 1]
        action: torch.Tensor = y_t

        # Calculate log probability, correcting for the tanh squashing
        # This correction is a key part of the SAC algorithm
        log_prob: torch.Tensor = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + config.EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)

        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return action, log_prob


# 类 CriticNetwork，继承自 ：核心类，封装本模块中的主要状态和行为。
class CriticNetwork(nn.Module):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：total_obs_dim, total_action_dim。
    def __init__(self, total_obs_dim: int, total_action_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(total_obs_dim + total_action_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

    # 函数 forward：定义神经网络前向传播计算，主要参数：joint_obs, joint_action。
    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.cat([joint_obs, joint_action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.out(x)
