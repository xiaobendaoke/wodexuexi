from __future__ import annotations

"""
中文注释说明：marl_models/base_model.py

文件作用：
    定义多智能体强化学习模型的公共接口，统一训练、动作选择、保存和加载等能力。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - OffPolicyExperienceBatch: 模块级变量，在本文件后续流程中被复用。
    - OnPolicyExperienceBatch: 模块级变量，在本文件后续流程中被复用。
    - ExperienceBatch: 模块级变量，在本文件后续流程中被复用。
    - MARLModel: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    abc, numpy, torch, typing

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Any, Dict, Optional, Union

# 关键变量 OffPolicyExperienceBatch：模块级变量，在本文件后续流程中被复用。
OffPolicyExperienceBatch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# 关键变量 OnPolicyExperienceBatch：模块级变量，在本文件后续流程中被复用。
OnPolicyExperienceBatch = dict[str, torch.Tensor]
# 关键变量 ExperienceBatch：模块级变量，在本文件后续流程中被复用。
ExperienceBatch = Union[OffPolicyExperienceBatch, OnPolicyExperienceBatch]


# 类 MARLModel，继承自 ABC：核心类，封装本模块中的主要状态和行为。
class MARLModel(ABC):
    """
    Abstract Base Class for Multi-Agent Reinforcement Learning models.
    This class defines the essential methods that any MARL algorithm implementation
    must have to be compatible with the training framework.
    """

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：model_name, num_agents, obs_dim, action_dim, device。
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        self.model_name = model_name
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    # 函数 get_action_and_value：智能体输出的动作，通常包含移动方向、服务缓存或任务卸载相关决策，主要参数：obs, state。
    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets actions, log probabilities, and state values.
        Essential for on-policy algorithms like PPO.
        """
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise NotImplementedError("This method is required for on-policy algorithms.")

    # 函数 select_actions：所有智能体在当前时间步的联合动作，主要参数：observations, exploration。
    @abstractmethod
    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        """
        Selects actions for all agents based on their observations.
        """
        pass

    # 函数 update：更新模型、环境或统计量的状态，主要参数：batch。
    @abstractmethod
    def update(self, batch: ExperienceBatch) -> Optional[Dict[str, Any]]:
        """
        Performs a learning update on the model's networks using a batch of experiences.

        Args:
            batch (ExperienceBatch): A dictionary (for on-policy) or a tuple (for off-policy).
        """
        pass

    # 函数 reset：重置环境或对象状态，开始新的回合。
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the model's internal state (if any) for a new episode.
        """
        pass

    # 函数 save：保存模型参数或实验结果，主要参数：directory。
    @abstractmethod
    def save(self, directory: str) -> None:
        pass

    # 函数 load：加载模型参数或实验数据，主要参数：directory。
    @abstractmethod
    def load(self, directory: str) -> None:
        pass
