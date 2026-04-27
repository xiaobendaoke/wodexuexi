"""
中文注释说明：marl_models/attention_mappo/agents.py

文件作用：
    定义MAPPO 多智能体近端策略优化算法中的智能体、Actor/Critic 网络和参数更新逻辑，是该算法训练的执行单元。

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
from marl_models.attention import AttentionActorBase, AttentionCriticBase
import torch
import torch.nn as nn
from torch.distributions import Normal


# 类 ActorNetwork，继承自 AttentionActorBase：核心类，封装本模块中的主要状态和行为。
class ActorNetwork(AttentionActorBase):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_dim, action_dim。
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(obs_dim)
        self.mean: nn.Linear = layer_init(nn.Linear(self.hidden_dim, action_dim))
        self.log_std: nn.Parameter = nn.Parameter(torch.zeros(1, action_dim))

    # 函数 forward：定义神经网络前向传播计算，主要参数：obs。
    def forward(self, obs: torch.Tensor) -> Normal:
        x: torch.Tensor = self.get_feature_embedding(obs)
        mean: torch.Tensor = torch.tanh(self.mean(x))
        log_std: torch.Tensor = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std: torch.Tensor = torch.exp(log_std).expand_as(mean)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return Normal(mean, std)


# 类 CriticNetwork，继承自 AttentionCriticBase：核心类，封装本模块中的主要状态和行为。
class CriticNetwork(AttentionCriticBase):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_dim。
    def __init__(self, obs_dim: int) -> None:
        # action_dim=0 because this is V(s), not Q(s,a)
        super().__init__(obs_dim, action_dim=0)
        self.v_head: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.fusion_dim, self.mlp_dim)), nn.LayerNorm(self.mlp_dim), nn.ReLU(), layer_init(nn.Linear(self.mlp_dim, 1)))

    # 函数 forward：定义神经网络前向传播计算，主要参数：obs_tensor。
    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes observations once for the whole batch, then runs vectorized attention simultaneously for all agents.
        Args:
            obs_tensor: (Batch, Num_Agents, Obs_Dim)
        Returns:
            values: (Batch, Num_Agents)
        """
        # Run the heavy encoder once for all agents
        # (Batch, Num_Agents, Obs) -> (Batch, Num_Agents, Hidden)
        all_embeddings: torch.Tensor = self.get_all_embeddings(obs_tensor)

        # Vectorized attention for all agents simultaneously
        # (Batch, Num_Agents, Hidden) -> (Batch, Num_Agents, Fusion_Dim)
        combined: torch.Tensor = self.vectorized_attend_to_others(all_embeddings)

        # Pass the fused embeddings through the value head
        # (Batch, Num_Agents, Fusion_Dim) -> (Batch, Num_Agents, 1) -> (Batch, Num_Agents)
        values: torch.Tensor = self.v_head(combined).squeeze(-1)

        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return values
