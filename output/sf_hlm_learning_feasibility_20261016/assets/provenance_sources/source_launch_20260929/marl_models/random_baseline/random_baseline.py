"""
Random Policy baseline。

轨迹动作：每架 UAV 随机采样 [-1, 1] 的 2D 连续动作。
卸载动作：对每个有效请求，在 mask 允许的动作中随机选择。
不执行任何学习，update() 返回空字典。
"""

from __future__ import annotations

import numpy as np

import config
from marl_models.base_model import MARLModel, ExperienceBatch


class RandomBaseline(MARLModel):
    """Random Policy：绝对性能下界，验证学习是否有效。"""

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.num_offload_actions: int = config.OFFLOAD_NUM_ACTIONS
        self.max_requests: int = config.MAX_OFFLOAD_REQUESTS_PER_UAV

    def select_actions(self, observations: np.ndarray, exploration: bool = True) -> np.ndarray:
        """随机采样轨迹动作 [-1, 1]。"""
        return np.random.uniform(-1.0, 1.0, size=(self.num_agents, self.action_dim)).astype(np.float32)

    def get_offload_action(self, offload_obs: np.ndarray, action_masks: np.ndarray | None = None) -> np.ndarray:
        """对每个请求在 mask 允许的动作中随机选择。

        Args:
            offload_obs: shape (NUM_UAVS, OFFLOAD_OBS_DIM_SINGLE)
            action_masks: shape (NUM_UAVS, MAX_OFFLOAD_REQUESTS_PER_UAV, OFFLOAD_NUM_ACTIONS)
                          1.0 = legal, 0.0 = masked

        Returns:
            offload_actions: shape (NUM_UAVS, MAX_OFFLOAD_REQUESTS_PER_UAV) int64
        """
        offload_actions = np.zeros((self.num_agents, self.max_requests), dtype=np.int64)

        for uav_idx in range(self.num_agents):
            for req_idx in range(self.max_requests):
                if action_masks is not None:
                    legal = np.where(action_masks[uav_idx, req_idx] > 0.5)[0]
                else:
                    legal = np.arange(self.num_offload_actions)

                if len(legal) == 0:
                    # fallback: 选择 local
                    offload_actions[uav_idx, req_idx] = config.OFFLOAD_ACTION_LOCAL
                else:
                    offload_actions[uav_idx, req_idx] = np.random.choice(legal)

        return offload_actions

    def update(self, batch: ExperienceBatch) -> dict:
        """不执行任何学习。"""
        return {}

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        pass

    def load(self, directory: str) -> None:
        pass
