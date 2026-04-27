"""
中文注释说明：marl_models/nearest_greedy_baseline/nearest_greedy_model.py

文件作用：
    实现最近距离贪心基线模型的模型封装，负责动作选择、经验存储、网络更新、模型保存和加载。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - NearestGreedyModel: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    marl_models, config, numpy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import MARLModel, ExperienceBatch
import config
import numpy as np


# 类 NearestGreedyModel，继承自 MARLModel：核心类，封装本模块中的主要状态和行为。
class NearestGreedyModel(MARLModel):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：model_name, num_agents, obs_dim, action_dim, device。
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)

        # Calculate exactly where the UE observations start in the flattened array
        self.ue_start_idx: int = 2 + config.NUM_FILES + config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM

    # 函数 select_actions：所有智能体在当前时间步的联合动作，主要参数：observations, exploration。
    def select_actions(self, observations: np.ndarray, exploration: bool = True) -> np.ndarray:
        # In obs, UEs are pre-sorted by distance. Hence, the very first UE block in the observation is the nearest.
        dx: np.ndarray = observations[:, self.ue_start_idx] * config.AREA_WIDTH
        dy: np.ndarray = observations[:, self.ue_start_idx + 1] * config.AREA_HEIGHT
        distances: np.ndarray = np.sqrt(dx**2 + dy**2)

        # Default action: If no UEs are in range (distance is 0 due to zero-padding), do a moderate random walk.
        actions: np.ndarray = np.random.uniform(-0.2, 0.2, size=(self.num_agents, self.action_dim)).astype(np.float32)

        # Masking: Only apply the greedy vector to UAVs that actually see a UE
        mask: np.ndarray = distances > 0.1
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if np.any(mask):
            actions[mask, 0] = dx[mask] / distances[mask]
            actions[mask, 1] = dy[mask] / distances[mask]
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return actions

    # 函数 update：更新模型、环境或统计量的状态，主要参数：batch。
    def update(self, batch: ExperienceBatch) -> dict:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {}  # Does not learn, return empty losses dict.

    # 函数 reset：重置环境或对象状态，开始新的回合。
    def reset(self) -> None:
        pass

    # 函数 save：保存模型参数或实验结果，主要参数：directory。
    def save(self, directory: str) -> None:
        pass

    # 函数 load：加载模型参数或实验数据，主要参数：directory。
    def load(self, directory: str) -> None:
        pass
