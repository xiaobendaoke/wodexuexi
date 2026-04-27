"""
中文注释说明：marl_models/attention_mappo/attention_mappo.py

文件作用：
    实现MAPPO 多智能体近端策略优化算法的模型封装，负责动作选择、经验存储、网络更新、模型保存和加载。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - AttentionMAPPO: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    marl_models, config, numpy, os, torch, typing

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.attention_mappo.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import get_state_dict, load_safe
import config
import numpy as np
import os
import torch
from torch.distributions import Normal
from typing import cast


# 类 AttentionMAPPO，继承自 MARLModel：核心类，封装本模块中的主要状态和行为。
class AttentionMAPPO(MARLModel):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：model_name, num_agents, obs_dim, action_dim, device。
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)

        self.actor: ActorNetwork = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic: CriticNetwork = CriticNetwork(obs_dim).to(device)

        self.actor = cast(ActorNetwork, torch.compile(self.actor, backend="eager"))
        self.critic = cast(CriticNetwork, torch.compile(self.critic, backend="eager"))

        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

    # 函数 select_actions：所有智能体在当前时间步的联合动作，主要参数：observations, exploration。
    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with torch.no_grad():
            obs_tensor: torch.Tensor = torch.from_numpy(observations).to(self.device)
            dist: Normal = self.actor(obs_tensor)
            actions: torch.Tensor = dist.sample() if exploration else dist.mean

        # For training, we clip in train.py but for testing we clip here itself
        return np.clip(actions.cpu().numpy(), -1.0, 1.0)

    # 函数 get_action_and_value：智能体输出的动作，通常包含移动方向、服务缓存或任务卸载相关决策，主要参数：obs, state。
    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The actor treats each agent as an independent sample in the batch: Needs (Num_Agents, Obs_Dim)
        with torch.no_grad():
            obs_tensor_actor: torch.Tensor = torch.from_numpy(obs).to(self.device)

            # The critic needs to see the "whole swarm" at once to compute attention between agents: Needs (1, Num_Agents, Obs_Dim)
            obs_tensor_critic: torch.Tensor = obs_tensor_actor.unsqueeze(0)

            dist: Normal = self.actor(obs_tensor_actor)
            actions: torch.Tensor = dist.sample()
            log_probs: torch.Tensor = dist.log_prob(actions).sum(dim=-1)

            values: torch.Tensor = self.critic(obs_tensor_critic).view(-1)  # (1, Num_Agents) -> (Num_Agents,)

        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    # 函数 update：更新模型、环境或统计量的状态，主要参数：batch。
    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, dict), "MAPPO expects OnPolicyExperienceBatch (dict)"
        obs_batch: torch.Tensor = batch["obs"]
        actions_batch: torch.Tensor = batch["actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        old_values_batch: torch.Tensor = batch["old_values"]

        # Critic Update
        values: torch.Tensor = self.critic(obs_batch)  # (Batch, Num_Agents)

        # Value Clipping
        values_clipped: torch.Tensor = old_values_batch + torch.clamp(values - old_values_batch, -config.PPO_CLIP_EPS, config.PPO_CLIP_EPS)
        vf_loss1: torch.Tensor = (values - returns_batch).pow(2)
        vf_loss2: torch.Tensor = (values_clipped - returns_batch).pow(2)
        critic_loss: torch.Tensor = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        # Actor Update
        # Flatten for Actor: The Actor processes agents independently (local attention).
        flat_obs: torch.Tensor = obs_batch.view(-1, self.obs_dim)
        flat_actions: torch.Tensor = actions_batch.view(-1, config.ACTION_DIM)
        flat_old_log_probs: torch.Tensor = old_log_probs_batch.view(-1)
        flat_advantages: torch.Tensor = advantages_batch.view(-1)

        dist: Normal = self.actor(flat_obs)
        new_log_probs: torch.Tensor = dist.log_prob(flat_actions).sum(dim=-1)
        ratio: torch.Tensor = torch.exp(new_log_probs - flat_old_log_probs)
        surr1: torch.Tensor = ratio * flat_advantages
        surr2: torch.Tensor = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * flat_advantages
        actor_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

        entropy: torch.Tensor = dist.entropy().mean()
        actor_loss -= config.PPO_ENTROPY_COEF * entropy

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        # Return losses for logging
        return {
            "actor": float(actor_loss.item()),
            "critic": float(critic_loss.item()),
            "entropy": float(entropy.item()),
        }

    # 函数 reset：重置环境或对象状态，开始新的回合。
    def reset(self) -> None:
        pass

    # 函数 save：保存模型参数或实验结果，主要参数：directory。
    def save(self, directory: str) -> None:
        torch.save(
            {
                "actor": get_state_dict(self.actor),
                "critic": get_state_dict(self.critic),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(directory, "attention_mappo.pth"),
        )

    # 函数 load：加载模型参数或实验数据，主要参数：directory。
    def load(self, directory: str) -> None:
        path = os.path.join(directory, "attention_mappo.pth")
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if not os.path.exists(path):
            # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        load_safe(self.actor, checkpoint["actor"])
        load_safe(self.critic, checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

