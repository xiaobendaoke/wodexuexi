"""
中文注释说明：marl_models/attention_maddpg/attention_maddpg.py

文件作用：
    实现MADDPG 多智能体深度确定性策略梯度算法的模型封装，负责动作选择、经验存储、网络更新、模型保存和加载。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - AttentionMADDPG: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    marl_models, config, torch, numpy, typing, os

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.attention_maddpg.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise, get_state_dict, load_safe
import config
import torch
import torch.nn.functional as F
import numpy as np
from typing import cast
import os


# 类 AttentionMADDPG，继承自 MARLModel：核心类，封装本模块中的主要状态和行为。
class AttentionMADDPG(MARLModel):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：model_name, num_agents, obs_dim, action_dim, device。
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)

        self.actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]

        self.target_actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self._init_target_networks()

        self.actors = [cast(ActorNetwork, torch.compile(actor, backend="eager")) for actor in self.actors]
        self.critics = [cast(CriticNetwork, torch.compile(critic, backend="eager")) for critic in self.critics]

        self.actor_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(actor.parameters(), lr=config.ACTOR_LR) for actor in self.actors]
        self.critic_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics]

        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

    # 函数 select_actions：所有智能体在当前时间步的联合动作，主要参数：observations, exploration。
    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with torch.no_grad():
            obs_tensor: torch.Tensor = torch.from_numpy(observations).to(self.device)
            actions: np.ndarray = np.empty_like(observations[:, : config.ACTION_DIM])

            # 循环处理：遍历 i 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for i in range(self.num_agents):
                action: np.ndarray = self.actors[i](obs_tensor[i].unsqueeze(0)).squeeze(0).cpu().numpy()

                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if exploration:
                    action += self.noise[i].sample()
                actions[i] = np.clip(action, -1.0, 1.0)

        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return actions

    # 函数 update：更新模型、环境或统计量的状态，主要参数：batch。
    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, tuple) and len(batch) == 5, "MADDPG expects OffPolicyExperienceBatch (tuple of 5 elements)"
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch

        obs_tensor: torch.Tensor = torch.from_numpy(obs_batch).to(self.device, non_blocking=True)
        actions_tensor: torch.Tensor = torch.from_numpy(actions_batch).to(self.device, non_blocking=True)
        rewards_tensor: torch.Tensor = torch.from_numpy(rewards_batch).to(self.device, non_blocking=True)
        next_obs_tensor: torch.Tensor = torch.from_numpy(next_obs_batch).to(self.device, non_blocking=True)
        dones_tensor: torch.Tensor = torch.from_numpy(dones_batch).to(self.device, non_blocking=True)

        agent_losses: list[float] = []
        agent_critic_losses: list[float] = []

        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with torch.no_grad():
            next_actions_list: list[torch.Tensor] = [self.target_actors[i](next_obs_tensor[:, i, :]) for i in range(self.num_agents)]
            next_actions_tensor: torch.Tensor = torch.stack(next_actions_list, dim=1)

        # 循环处理：遍历 agent_idx 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for agent_idx in range(self.num_agents):
            # Update Critic
            with torch.no_grad():
                target_q_value: torch.Tensor = self.target_critics[agent_idx](next_obs_tensor, next_actions_tensor, agent_idx)
                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_done: torch.Tensor = dones_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_value * (1 - agent_done)

            self.critic_optimizers[agent_idx].zero_grad(set_to_none=True)
            current_q_value: torch.Tensor = self.critics[agent_idx](obs_tensor, actions_tensor, agent_idx)

            critic_loss: torch.Tensor = F.mse_loss(current_q_value, y)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_optimizers[agent_idx].step()
            agent_critic_losses.append(float(critic_loss.item()))

            # Update Actor
            pred_actions_tensor: torch.Tensor = actions_tensor.detach().clone()
            pred_actions_tensor[:, agent_idx, :] = self.actors[agent_idx](obs_tensor[:, agent_idx, :])

            self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
            actor_loss: torch.Tensor = -self.critics[agent_idx](obs_tensor, pred_actions_tensor, agent_idx).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.actor_optimizers[agent_idx].step()
            agent_losses.append(float(actor_loss.item()))

            soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
            soft_update(self.target_critics[agent_idx], self.critics[agent_idx], config.UPDATE_FACTOR)

        # 循环处理：遍历 n 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for n in self.noise:
            n.decay()

        # Return averaged losses across all agents
        return {
            "actor": float(np.mean(agent_losses)) if agent_losses else 0.0,
            "critic": float(np.mean(agent_critic_losses)),
        }

    # 函数 _init_target_networks：关键函数，承载本模块的一段可复用实验逻辑。
    def _init_target_networks(self) -> None:
        # 循环处理：遍历 (actor, target_actor) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        # 循环处理：遍历 (critic, target_critic) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

    # 函数 reset：重置环境或对象状态，开始新的回合。
    def reset(self) -> None:
        # 循环处理：遍历 n 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for n in self.noise:
            n.reset()

    # 函数 save：保存模型参数或实验结果，主要参数：directory。
    def save(self, directory: str) -> None:
        # 循环处理：遍历 i 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for i in range(self.num_agents):
            torch.save(
                {
                    "actor": get_state_dict(self.actors[i]),
                    "critic": get_state_dict(self.critics[i]),
                    "target_actor": self.target_actors[i].state_dict(),
                    "target_critic": self.target_critics[i].state_dict(),
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic_optimizer": self.critic_optimizers[i].state_dict(),
                    "noise_scale": self.noise[i].scale,
                },
                os.path.join(directory, f"agent_{i}.pth"),
            )

    # 函数 load：加载模型参数或实验数据，主要参数：directory。
    def load(self, directory: str) -> None:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if not os.path.exists(directory):
            # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
            raise FileNotFoundError(f"❌ Model directory not found: {directory}")

        # 循环处理：遍历 i 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for i in range(self.num_agents):
            agent_path: str = os.path.join(directory, f"agent_{i}.pth")
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if not os.path.exists(agent_path):
                # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
                raise FileNotFoundError(f"❌ Model file not found: {agent_path}")
            checkpoint: dict = torch.load(agent_path, map_location=self.device, weights_only=True)

            load_safe(self.actors[i], checkpoint["actor"])
            load_safe(self.critics[i], checkpoint["critic"])
            self.target_actors[i].load_state_dict(checkpoint["target_actor"])
            self.target_critics[i].load_state_dict(checkpoint["target_critic"])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizers[i].load_state_dict(checkpoint["critic_optimizer"])
            self.noise[i].scale = checkpoint["noise_scale"]

