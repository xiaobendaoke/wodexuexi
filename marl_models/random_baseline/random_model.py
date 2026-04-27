"""
中文注释说明：marl_models/random_baseline/random_model.py

文件作用：
    实现随机动作基线模型的模型封装，负责动作选择、经验存储、网络更新、模型保存和加载。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - RandomModel: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    marl_models, numpy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import MARLModel, ExperienceBatch
import numpy as np


# 类 RandomModel，继承自 MARLModel：核心类，封装本模块中的主要状态和行为。
class RandomModel(MARLModel):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：model_name, num_agents, obs_dim, action_dim, device。
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)

    # 函数 select_actions：所有智能体在当前时间步的联合动作，主要参数：observations, exploration。
    def select_actions(self, observations: np.ndarray, exploration: bool = True) -> np.ndarray:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return np.random.uniform(-1.0, 1.0, (self.num_agents, self.action_dim))

    # 函数 update：更新模型、环境或统计量的状态，主要参数：batch。
    def update(self, batch: ExperienceBatch) -> dict:
        """Random baseline does not learn, return empty losses dict."""
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {}

    # 函数 reset：重置环境或对象状态，开始新的回合。
    def reset(self) -> None:
        pass

    # 函数 save：保存模型参数或实验结果，主要参数：directory。
    def save(self, directory: str) -> None:
        pass

    # 函数 load：加载模型参数或实验数据，主要参数：directory。
    def load(self, directory: str) -> None:
        pass
