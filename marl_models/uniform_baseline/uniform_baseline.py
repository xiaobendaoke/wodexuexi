"""
Uniform / Load-Balanced Heuristic baseline。

轨迹策略：复用 uncoordinated_greedy 的向地图中心飞行逻辑。
卸载策略：对每个请求选择当前负载最小的合法目标（mask-aware load-balanced）。
不执行任何学习，update() 返回空字典。
"""

from __future__ import annotations

import numpy as np

import config
from marl_models.base_model import MARLModel, ExperienceBatch


class UniformBaseline(MARLModel):
    """Uniform Policy：负载均衡启发式，对齐 MAHHV 的 Uniform Policy。"""

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.num_offload_actions: int = config.OFFLOAD_NUM_ACTIONS
        self.max_requests: int = config.MAX_OFFLOAD_REQUESTS_PER_UAV

        # 轨迹策略参数（复用 uncoordinated_greedy）
        self.target_pos: np.ndarray = np.array([[config.AREA_WIDTH / 2.0, config.AREA_HEIGHT / 2.0]], dtype=np.float32)
        self.area_dims: np.ndarray = np.array([[config.AREA_WIDTH, config.AREA_HEIGHT]], dtype=np.float32)

        # 负载计数器（每个 episode reset）
        self._load_counts: np.ndarray = np.zeros(self.num_offload_actions, dtype=np.float32)

    def select_actions(self, observations: np.ndarray, exploration: bool = True) -> np.ndarray:
        """向地图中心飞行，复用 uncoordinated_greedy 逻辑。"""
        current_pos: np.ndarray = observations[:, :2] * self.area_dims
        delta: np.ndarray = self.target_pos - current_pos
        distances: np.ndarray = np.linalg.norm(delta, axis=1, keepdims=True)

        # 默认动作：小幅随机抖动
        actions: np.ndarray = np.random.uniform(-0.1, 0.1, size=(self.num_agents, self.action_dim)).astype(np.float32)

        # 距离 > 10m 的 UAV 向中心飞
        mask: np.ndarray = (distances > 10.0).flatten()
        if np.any(mask):
            actions[mask] = delta[mask] / distances[mask]

        return actions

    def get_offload_action(self, offload_obs: np.ndarray, action_masks: np.ndarray | None = None) -> np.ndarray:
        """Mask-aware load-balanced 卸载。

        对每个请求，在合法目标中选择当前负载最小的目标。
        优先级：local → cooperative UAV → MBS（负载相同时）。
        """
        # 每次调用重置负载计数（按 time slot 粒度）
        self._load_counts = np.zeros(self.num_offload_actions, dtype=np.float32)

        offload_actions = np.zeros((self.num_agents, self.max_requests), dtype=np.int64)

        for uav_idx in range(self.num_agents):
            for req_idx in range(self.max_requests):
                if action_masks is not None:
                    legal = np.where(action_masks[uav_idx, req_idx] > 0.5)[0]
                else:
                    legal = np.arange(self.num_offload_actions)

                if len(legal) == 0:
                    offload_actions[uav_idx, req_idx] = config.OFFLOAD_ACTION_LOCAL
                    continue

                # 选择负载最小的合法目标
                legal_loads = self._load_counts[legal]
                min_load = np.min(legal_loads)
                min_load_indices = legal[legal_loads <= min_load + 1e-6]

                # 负载相同时按优先级排序：local(0) < cooperative(2..N) < MBS(1)
                if len(min_load_indices) > 1:
                    priority_order = self._get_priority_order(min_load_indices)
                    chosen = priority_order[0]
                else:
                    chosen = min_load_indices[0]

                offload_actions[uav_idx, req_idx] = chosen
                self._load_counts[chosen] += 1

        return offload_actions

    @staticmethod
    def _get_priority_order(indices: np.ndarray) -> np.ndarray:
        """负载相同时的优先级：local(0) < cooperative(2..N) < MBS(1)。"""
        # 排序 key: local=0, coop=1, mbs=2
        def _sort_key(idx):
            if idx == config.OFFLOAD_ACTION_LOCAL:
                return 0
            elif idx == config.OFFLOAD_ACTION_MBS:
                return 2
            else:
                return 1
        return indices[np.argsort([_sort_key(i) for i in indices])]

    def update(self, batch: ExperienceBatch) -> dict:
        """不执行任何学习。"""
        return {}

    def reset(self) -> None:
        """重置负载计数器。"""
        self._load_counts = np.zeros(self.num_offload_actions, dtype=np.float32)

    def save(self, directory: str) -> None:
        pass

    def load(self, directory: str) -> None:
        pass
