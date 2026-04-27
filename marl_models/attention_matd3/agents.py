"""
中文注释说明：marl_models/attention_matd3/agents.py

文件作用：
    定义MATD3 多智能体双延迟深度确定性策略梯度算法中的智能体、Actor/Critic 网络和参数更新逻辑，是该算法训练的执行单元。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - ActorNetwork: 核心类，封装本模块中的主要状态和行为。
    - CriticNetwork: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    marl_models, torch

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.buffer_and_helpers import layer_init
from marl_models.attention import AttentionActorBase, AttentionCriticBase
import torch
import torch.nn as nn


# 类 ActorNetwork，继承自 AttentionActorBase：核心类，封装本模块中的主要状态和行为。
class ActorNetwork(AttentionActorBase):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_dim, action_dim。
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(obs_dim)
        self.out: nn.Linear = layer_init(nn.Linear(self.hidden_dim, action_dim), std=0.01)  # Small std for output

    # 函数 forward：定义神经网络前向传播计算，主要参数：obs。
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.get_feature_embedding(obs)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return torch.tanh(self.out(x))


# 类 CriticNetwork，继承自 AttentionCriticBase：核心类，封装本模块中的主要状态和行为。
class CriticNetwork(AttentionCriticBase):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_dim, action_dim。
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(obs_dim, action_dim)

        # Final Head: Output 1 value
        self.q_head: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.fusion_dim, self.mlp_dim)), nn.LayerNorm(self.mlp_dim), nn.ReLU(), layer_init(nn.Linear(self.mlp_dim, 1)))

    # 函数 forward：定义神经网络前向传播计算，主要参数：obs_tensor, action_tensor, agent_index。
    def forward(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor, agent_index: int) -> torch.Tensor:
        embedding: torch.Tensor = self.get_q_embedding(obs_tensor, action_tensor, agent_index)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.q_head(embedding)
