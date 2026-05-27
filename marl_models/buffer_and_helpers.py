from __future__ import annotations

"""
中文注释说明：marl_models/buffer_and_helpers.py

文件作用：
    提供经验回放缓冲区、轨迹缓存和训练辅助函数，支撑多种 MARL 算法复用。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - ReplayBuffer: 核心类，封装本模块中的主要状态和行为。
    - RolloutBuffer: 核心类，封装本模块中的主要状态和行为。
    - AttentionRolloutBuffer: 核心类，封装本模块中的主要状态和行为。
    - GaussianNoise: 核心类，封装本模块中的主要状态和行为。
    - soft_update(): 更新模型、环境或统计量的状态。
    - layer_init(): 关键函数，承载本模块的一段可复用实验逻辑。
    - get_state_dict(): 环境或智能体观测状态，用于生成动作或训练样本。
    - load_safe(): 加载模型参数或实验数据。

主要依赖：
    marl_models, config, torch, numpy, collections

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import OffPolicyExperienceBatch
import config
import torch
import torch.nn as nn
import numpy as np
from collections.abc import Generator


# FIX: Replace deque-based ReplayBuffer with numpy ring-buffer.
# The original deque caused O(n) random access: `self.buffer[i]` on a deque
# is O(n) per index, making each sample() call O(batch_size * buffer_size).
# A numpy ring-buffer gives O(1) indexing and enables vectorised batch slicing.
class ReplayBuffer:
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：max_size。
    def __init__(self, max_size: int) -> None:
        self.max_size: int = max_size
        self.ptr: int = 0  # Points to next write position
        self.size: int = 0  # Current number of valid entries

        # Pre-allocate numpy arrays.
        self._obs: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)  # Shape: (max_size, num_uavs, obs_dim)
        self._next_obs: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)
        self._actions: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
        self._rewards: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS), dtype=np.float32)
        self._dones: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS), dtype=np.float32)

    # 函数 add：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_arr, actions, rewards, next_obs_arr, done。
    def add(self, obs_arr: np.ndarray, actions: np.ndarray, rewards: list[float], next_obs_arr: np.ndarray, done: bool) -> None:
        rewards_arr: np.ndarray = np.array(rewards, dtype=np.float32)
        dones_arr: np.ndarray = np.full(config.NUM_UAVS, float(done), dtype=np.float32)

        self._obs[self.ptr] = obs_arr
        self._actions[self.ptr] = actions
        self._rewards[self.ptr] = rewards_arr
        self._next_obs[self.ptr] = next_obs_arr
        self._dones[self.ptr] = dones_arr

        # Ring-buffer write pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # 函数 sample：关键函数，承载本模块的一段可复用实验逻辑，主要参数：batch_size。
    def sample(self, batch_size: int) -> OffPolicyExperienceBatch:
        """Sample a batch of experiences. O(1) random batch sampling via numpy fancy-indexing."""
        indices: np.ndarray = np.random.randint(0, self.size, size=batch_size)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return (
            self._obs[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_obs[indices],
            self._dones[indices],
        )

    # 函数 __len__：关键函数，承载本模块的一段可复用实验逻辑。
    def __len__(self) -> int:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.size


# 类 RolloutBuffer：核心类，封装本模块中的主要状态和行为。
class RolloutBuffer:
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：num_agents, obs_dim, action_dim, buffer_size, device。
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, buffer_size: int, device: str) -> None:
        self.num_agents: int = num_agents
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.state_dim: int = obs_dim * num_agents
        self.buffer_size: int = buffer_size
        self.device: str = device

        # Initialize storage
        self.states: np.ndarray = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.observations: np.ndarray = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.log_probs: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)

        # For GAE calculation
        self.advantages: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)

        self.step: int = 0

    # 函数 add：关键函数，承载本模块的一段可复用实验逻辑，主要参数：state, obs, actions, log_probs, rewards, done。
    def add(self, state: np.ndarray, obs: np.ndarray, actions: np.ndarray, log_probs: np.ndarray, rewards: list[float], done: bool, values: np.ndarray) -> None:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if self.step >= self.buffer_size:
            # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
            raise ValueError("Rollout buffer overflow")
        self.states[self.step] = state
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = np.array(rewards, dtype=np.float32)
        self.dones[self.step] = np.full(config.NUM_UAVS, float(done), dtype=np.float32)
        self.values[self.step] = values

        self.step += 1

    # 函数 compute_returns_and_advantages：关键函数，承载本模块的一段可复用实验逻辑，主要参数：last_values, gamma, gae_lambda。
    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Computes the advantages and returns for the collected trajectories using GAE."""
        last_gae_lam: float = 0.0
        # 循环处理：遍历 t 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for t in reversed(range(self.buffer_size)):
            next_values: np.ndarray = last_values if t == self.buffer_size - 1 else self.values[t + 1]
            delta: np.ndarray = self.rewards[t] + gamma * next_values * (1.0 - self.dones[t]) - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * last_gae_lam

        self.returns = self.advantages + self.values

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # 函数 get_batches：关键函数，承载本模块的一段可复用实验逻辑，主要参数：batch_size。
    def get_batches(self, batch_size: int) -> Generator[dict[str, torch.Tensor], None, None]:
        """A generator that yields mini-batches from the buffer."""
        num_samples: int = self.buffer_size * self.num_agents

        states: np.ndarray = np.repeat(self.states, self.num_agents, axis=0)
        agent_ids: np.ndarray = np.tile(np.arange(self.num_agents), self.buffer_size)
        obs: np.ndarray = self.observations.reshape(-1, self.obs_dim)
        actions: np.ndarray = self.actions.reshape(-1, self.action_dim)  # Reshape to (N, action_dim)
        log_probs: np.ndarray = self.log_probs.reshape(-1)
        advantages: np.ndarray = self.advantages.reshape(-1)
        returns: np.ndarray = self.returns.reshape(-1)
        values: np.ndarray = self.values.reshape(-1)

        # FIX: Convert entire epoch's data to GPU tensors ONCE before looping,
        # instead of calling torch.as_tensor(..., device=...) per mini-batch.
        # This eliminates repeated CPU→GPU transfers inside the hot path.
        t_states: torch.Tensor = torch.from_numpy(states).to(self.device)
        t_agent_ids: torch.Tensor = torch.from_numpy(agent_ids).to(self.device)
        t_obs: torch.Tensor = torch.from_numpy(obs).to(self.device)
        t_actions: torch.Tensor = torch.from_numpy(actions).to(self.device)
        t_log_probs: torch.Tensor = torch.from_numpy(log_probs).to(self.device)
        t_advantages: torch.Tensor = torch.from_numpy(advantages).to(self.device)
        t_returns: torch.Tensor = torch.from_numpy(returns).to(self.device)
        t_values: torch.Tensor = torch.from_numpy(values).to(self.device)

        indices: torch.Tensor = torch.randperm(num_samples, device=self.device)

        # 循环处理：遍历 start 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for start in range(0, num_samples, batch_size):
            idx: torch.Tensor = indices[start : start + batch_size]

            yield {
                "states": t_states[idx],
                "agent_ids": t_agent_ids[idx],
                "obs": t_obs[idx],
                "actions": t_actions[idx],
                "old_log_probs": t_log_probs[idx],
                "advantages": t_advantages[idx],
                "returns": t_returns[idx],
                "old_values": t_values[idx],
            }

    # 函数 clear：关键函数，承载本模块的一段可复用实验逻辑。
    def clear(self) -> None:
        self.step = 0


# 类 AttentionRolloutBuffer，继承自 RolloutBuffer：核心类，封装本模块中的主要状态和行为。
class AttentionRolloutBuffer(RolloutBuffer):
    """Preserves (Batch, Num_Agents, Dim) structure required for Graph Attention."""

    # 函数 get_batches：关键函数，承载本模块的一段可复用实验逻辑，主要参数：batch_size。
    def get_batches(self, batch_size: int):
        num_time_steps: int = self.buffer_size

        # FIX (same as above): upload all data to GPU once per epoch
        t_states: torch.Tensor = torch.from_numpy(self.states).to(self.device)
        t_obs: torch.Tensor = torch.from_numpy(self.observations).to(self.device)
        t_actions: torch.Tensor = torch.from_numpy(self.actions).to(self.device)
        t_log_probs: torch.Tensor = torch.from_numpy(self.log_probs).to(self.device)
        t_advantages: torch.Tensor = torch.from_numpy(self.advantages).to(self.device)
        t_returns: torch.Tensor = torch.from_numpy(self.returns).to(self.device)
        t_values: torch.Tensor = torch.from_numpy(self.values).to(self.device)

        indices: torch.Tensor = torch.randperm(num_time_steps, device=self.device)

        # 循环处理：遍历 start 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for start in range(0, num_time_steps, batch_size):
            idx: torch.Tensor = indices[start : start + batch_size]

            yield {
                "states": t_states[idx],
                "obs": t_obs[idx],
                "actions": t_actions[idx],
                "old_log_probs": t_log_probs[idx],
                "advantages": t_advantages[idx],
                "returns": t_returns[idx],
                "old_values": t_values[idx],
            }


class DiscreteOffloadRolloutBuffer:
    """Rollout buffer for lower-layer request-level discrete MAPPO."""

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        max_requests: int,
        num_actions: int,
        buffer_size: int,
        device: str,
    ) -> None:
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.max_requests = max_requests
        self.num_actions = num_actions
        self.buffer_size = buffer_size
        self.device = device

        self.observations = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, max_requests), dtype=np.int64)
        self.masks = np.zeros((buffer_size, num_agents, max_requests, num_actions), dtype=np.float32)
        self.valid_slots = np.zeros((buffer_size, num_agents, max_requests), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.step = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        masks: np.ndarray,
        log_probs: np.ndarray,
        rewards: list[float] | np.ndarray,
        done: bool,
        values: np.ndarray,
    ) -> None:
        if self.step >= self.buffer_size:
            raise ValueError("Discrete offload rollout buffer overflow")
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.masks[self.step] = masks
        self.valid_slots[self.step] = (np.sum(masks, axis=-1) > 0.0).astype(np.float32)
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = np.asarray(rewards, dtype=np.float32)
        self.dones[self.step] = np.full(self.num_agents, float(done), dtype=np.float32)
        self.values[self.step] = values
        self.step += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        last_gae_lam: float = 0.0
        for t in reversed(range(self.buffer_size)):
            next_values = last_values if t == self.buffer_size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * (1.0 - self.dones[t]) - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * last_gae_lam
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int):
        t_obs = torch.from_numpy(self.observations).to(self.device)
        t_actions = torch.from_numpy(self.actions).to(self.device)
        t_masks = torch.from_numpy(self.masks).to(self.device)
        t_valid_slots = torch.from_numpy(self.valid_slots).to(self.device)
        t_log_probs = torch.from_numpy(self.log_probs).to(self.device)
        t_advantages = torch.from_numpy(self.advantages).to(self.device)
        t_returns = torch.from_numpy(self.returns).to(self.device)
        t_values = torch.from_numpy(self.values).to(self.device)

        indices = torch.randperm(self.buffer_size, device=self.device)
        for start in range(0, self.buffer_size, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": t_obs[idx],
                "actions": t_actions[idx],
                "masks": t_masks[idx],
                "valid_slots": t_valid_slots[idx],
                "old_log_probs": t_log_probs[idx],
                "advantages": t_advantages[idx],
                "returns": t_returns[idx],
                "old_values": t_values[idx],
            }

    def clear(self) -> None:
        self.step = 0


class JointMAPPOBuffer:
    def __init__(
        self,
        num_agents: int,
        joint_obs_dim: int,
        trajectory_action_dim: int,
        max_requests: int,
        num_offload_actions: int,
        buffer_size: int,
        device: str,
    ) -> None:
        self.num_agents = num_agents
        self.joint_obs_dim = joint_obs_dim
        self.trajectory_action_dim = trajectory_action_dim
        self.max_requests = max_requests
        self.num_offload_actions = num_offload_actions
        self.buffer_size = buffer_size
        self.device = device
        self.observations = np.zeros((buffer_size, num_agents, joint_obs_dim), dtype=np.float32)
        self.trajectory_actions = np.zeros((buffer_size, num_agents, trajectory_action_dim), dtype=np.float32)
        self.offload_actions = np.zeros((buffer_size, num_agents, max_requests), dtype=np.int64)
        self.offload_masks = np.zeros((buffer_size, num_agents, max_requests, num_offload_actions), dtype=np.float32)
        self.valid_slots = np.zeros((buffer_size, num_agents, max_requests), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.step = 0

    def add(
        self,
        joint_obs: np.ndarray,
        trajectory_actions: np.ndarray,
        offload_actions: np.ndarray,
        offload_masks: np.ndarray,
        log_probs: np.ndarray,
        rewards: list[float] | np.ndarray,
        done: bool,
        values: np.ndarray,
    ) -> None:
        if self.step >= self.buffer_size:
            raise ValueError("Joint MAPPO rollout buffer overflow")
        self.observations[self.step] = joint_obs
        self.trajectory_actions[self.step] = trajectory_actions
        self.offload_actions[self.step] = offload_actions
        self.offload_masks[self.step] = offload_masks
        self.valid_slots[self.step] = (np.sum(offload_masks, axis=-1) > 0.0).astype(np.float32)
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = np.asarray(rewards, dtype=np.float32)
        self.dones[self.step] = np.full(self.num_agents, float(done), dtype=np.float32)
        self.values[self.step] = values
        self.step += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        last_gae_lam: float = 0.0
        for t in reversed(range(self.buffer_size)):
            next_values = last_values if t == self.buffer_size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * (1.0 - self.dones[t]) - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * last_gae_lam
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int):
        t_obs = torch.from_numpy(self.observations).to(self.device)
        t_trajectory_actions = torch.from_numpy(self.trajectory_actions).to(self.device)
        t_offload_actions = torch.from_numpy(self.offload_actions).to(self.device)
        t_offload_masks = torch.from_numpy(self.offload_masks).to(self.device)
        t_valid_slots = torch.from_numpy(self.valid_slots).to(self.device)
        t_log_probs = torch.from_numpy(self.log_probs).to(self.device)
        t_advantages = torch.from_numpy(self.advantages).to(self.device)
        t_returns = torch.from_numpy(self.returns).to(self.device)
        t_values = torch.from_numpy(self.values).to(self.device)
        indices = torch.randperm(self.buffer_size, device=self.device)
        for start in range(0, self.buffer_size, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": t_obs[idx],
                "trajectory_actions": t_trajectory_actions[idx],
                "offload_actions": t_offload_actions[idx],
                "offload_masks": t_offload_masks[idx],
                "valid_slots": t_valid_slots[idx],
                "old_log_probs": t_log_probs[idx],
                "advantages": t_advantages[idx],
                "returns": t_returns[idx],
                "old_values": t_values[idx],
            }

    def clear(self) -> None:
        self.step = 0


# 函数 soft_update：更新模型、环境或统计量的状态，主要参数：target_net, source_net, tau。
def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Performs a soft update of the target network's parameters."""
    # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
    with torch.no_grad():
        # 循环处理：遍历 (target_param, param) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.copy_(tau * param + (1.0 - tau) * target_param)


# 类 GaussianNoise：核心类，封装本模块中的主要状态和行为。
class GaussianNoise:
    """Gaussian noise with decay for exploration."""

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑。
    def __init__(self) -> None:
        self.scale: float = config.INITIAL_NOISE_SCALE

    # 函数 sample：关键函数，承载本模块的一段可复用实验逻辑。
    def sample(self) -> np.ndarray:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return np.random.normal(0, self.scale, config.ACTION_DIM)

    # 函数 decay：关键函数，承载本模块的一段可复用实验逻辑。
    def decay(self) -> None:
        self.scale = max(config.MIN_NOISE_SCALE, self.scale * config.NOISE_DECAY_RATE)

    # 函数 reset：重置环境或对象状态，开始新的回合。
    def reset(self) -> None:
        self.scale = config.INITIAL_NOISE_SCALE


# 函数 layer_init：关键函数，承载本模块的一段可复用实验逻辑，主要参数：layer, std, bias_const。
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Added orthogonal initialization for better training stability"""
    nn.init.orthogonal_(layer.weight, std)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return layer


# 函数 get_state_dict：环境或智能体观测状态，用于生成动作或训练样本，主要参数：model。
def get_state_dict(model):  # Helper to strip the compile wrapper
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if hasattr(model, "_orig_mod"):
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return model._orig_mod.state_dict()
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return model.state_dict()


# 函数 load_safe：加载模型参数或实验数据，主要参数：model, state_dict。
def load_safe(model, state_dict):  # Helper to load into potentially compiled models
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if hasattr(model, "_orig_mod"):  # If compiled, try loading into _orig_mod first
        # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
        try:
            model._orig_mod.load_state_dict(state_dict)
            # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
            return
        except Exception:
            pass  # Fallback to loading directly
    model.load_state_dict(state_dict)
