"""
中文注释说明：marl_models/matd3/matd3.py

文件作用：
    实现MATD3 多智能体双延迟深度确定性策略梯度算法的模型封装，负责动作选择、经验存储、网络更新、模型保存和加载。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - MATD3: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    marl_models, config, torch, numpy, typing, os

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.matd3.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise, get_state_dict, load_safe
import config
import torch
import torch.nn.functional as F
import numpy as np
from typing import cast
import os


# 类 MATD3，继承自 MARLModel：核心类，封装本模块中的主要状态和行为。
class MATD3(MARLModel):
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：model_name, num_agents, obs_dim, action_dim, device。
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim

        self.actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics_1: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.critics_2: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]

        self.target_actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_1: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_2: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self._init_target_networks()

        self.actors = [cast(ActorNetwork, torch.compile(actor, backend="eager")) for actor in self.actors]
        self.critics_1 = [cast(CriticNetwork, torch.compile(critic, backend="eager")) for critic in self.critics_1]
        self.critics_2 = [cast(CriticNetwork, torch.compile(critic, backend="eager")) for critic in self.critics_2]

        self.actor_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(actor.parameters(), lr=config.ACTOR_LR) for actor in self.actors]
        self.critic_1_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics_1]
        self.critic_2_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics_2]

        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

        # Delayed Updates Counter
        self.update_counter: int = 0

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
        assert isinstance(batch, tuple) and len(batch) == 5, "MATD3 expects OffPolicyExperienceBatch (tuple of 5 elements)"
        self.update_counter += 1
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch

        obs_tensor: torch.Tensor = torch.from_numpy(obs_batch).to(self.device, non_blocking=True)
        actions_tensor: torch.Tensor = torch.from_numpy(actions_batch).to(self.device, non_blocking=True)
        rewards_tensor: torch.Tensor = torch.from_numpy(rewards_batch).to(self.device, non_blocking=True)
        next_obs_tensor: torch.Tensor = torch.from_numpy(next_obs_batch).to(self.device, non_blocking=True)
        dones_tensor: torch.Tensor = torch.from_numpy(dones_batch).to(self.device, non_blocking=True)

        batch_size: int = obs_tensor.shape[0]
        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        agent_losses: list[float] = []
        agent_critic_losses: list[float] = []

        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with torch.no_grad():
            next_actions_list: list[torch.Tensor] = [self.target_actors[i](next_obs_tensor[:, i, :]) for i in range(self.num_agents)]
            clipped_noise: list[torch.Tensor] = [torch.clamp(torch.randn_like(a) * config.TARGET_POLICY_NOISE, -config.NOISE_CLIP, config.NOISE_CLIP) for a in next_actions_list]
            next_actions_list = [torch.clamp(next_actions_list[i] + clipped_noise[i], -1.0, 1.0) for i in range(self.num_agents)]
            next_actions_tensor: torch.Tensor = torch.cat(next_actions_list, dim=1)

        # 循环处理：遍历 agent_idx 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for agent_idx in range(self.num_agents):
            # Update Critic
            with torch.no_grad():
                target_q1: torch.Tensor = self.target_critics_1[agent_idx](next_obs_flat, next_actions_tensor)
                target_q2: torch.Tensor = self.target_critics_2[agent_idx](next_obs_flat, next_actions_tensor)
                target_q_min: torch.Tensor = torch.min(target_q1, target_q2)

                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_done: torch.Tensor = dones_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_min * (1 - agent_done)

            self.critic_1_optimizers[agent_idx].zero_grad(set_to_none=True)
            current_q1: torch.Tensor = self.critics_1[agent_idx](obs_flat, actions_flat)
            critic_1_loss: torch.Tensor = F.mse_loss(current_q1, y)
            critic_1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_1[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_1_optimizers[agent_idx].step()

            self.critic_2_optimizers[agent_idx].zero_grad(set_to_none=True)
            current_q2: torch.Tensor = self.critics_2[agent_idx](obs_flat, actions_flat)
            critic_2_loss: torch.Tensor = F.mse_loss(current_q2, y)
            critic_2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_2[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_2_optimizers[agent_idx].step()

            avg_critic_loss = (float(critic_1_loss.item()) + float(critic_2_loss.item())) / 2.0
            agent_critic_losses.append(avg_critic_loss)

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if self.update_counter % config.POLICY_UPDATE_FREQ == 0:
            # 循环处理：遍历 agent_idx 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for agent_idx in range(self.num_agents):
                # Update Actor
                pred_actions_tensor: torch.Tensor = actions_tensor.detach().clone()
                pred_actions_tensor[:, agent_idx, :] = self.actors[agent_idx](obs_tensor[:, agent_idx, :])
                pred_actions_flat: torch.Tensor = pred_actions_tensor.reshape(batch_size, -1)

                self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
                actor_loss: torch.Tensor = -self.critics_1[agent_idx](obs_flat, pred_actions_flat).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
                self.actor_optimizers[agent_idx].step()
                agent_losses.append(float(actor_loss.item()))

                soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_1[agent_idx], self.critics_1[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_2[agent_idx], self.critics_2[agent_idx], config.UPDATE_FACTOR)

            # 循环处理：遍历 n 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for n in self.noise:
                n.decay()

        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {
            "actor": float(np.mean(agent_losses)) if agent_losses else 0.0,
            "critic": float(np.mean(agent_critic_losses)),
        }

    # 函数 _init_target_networks：关键函数，承载本模块的一段可复用实验逻辑。
    def _init_target_networks(self) -> None:
        # 循环处理：遍历 (actor, target_actor) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        # 循环处理：遍历 (critic1, target_critic1) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for critic1, target_critic1 in zip(self.critics_1, self.target_critics_1):
            target_critic1.load_state_dict(critic1.state_dict())
        # 循环处理：遍历 (critic2, target_critic2) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for critic2, target_critic2 in zip(self.critics_2, self.target_critics_2):
            target_critic2.load_state_dict(critic2.state_dict())

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
                    "critic_1": get_state_dict(self.critics_1[i]),
                    "critic_2": get_state_dict(self.critics_2[i]),
                    "target_actor": self.target_actors[i].state_dict(),
                    "target_critic_1": self.target_critics_1[i].state_dict(),
                    "target_critic_2": self.target_critics_2[i].state_dict(),
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic_1_optimizer": self.critic_1_optimizers[i].state_dict(),
                    "critic_2_optimizer": self.critic_2_optimizers[i].state_dict(),
                    "noise_scale": self.noise[i].scale,
                },
                os.path.join(directory, f"agent_{i}.pth"),
            )
        update_counter_path: str = os.path.join(directory, "update_counter.txt")
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with open(update_counter_path, "w") as f:
            f.write(str(self.update_counter))

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
            load_safe(self.critics_1[i], checkpoint["critic_1"])
            load_safe(self.critics_2[i], checkpoint["critic_2"])
            self.target_actors[i].load_state_dict(checkpoint["target_actor"])
            self.target_critics_1[i].load_state_dict(checkpoint["target_critic_1"])
            self.target_critics_2[i].load_state_dict(checkpoint["target_critic_2"])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_1_optimizers[i].load_state_dict(checkpoint["critic_1_optimizer"])
            self.critic_2_optimizers[i].load_state_dict(checkpoint["critic_2_optimizer"])
            self.noise[i].scale = checkpoint["noise_scale"]

        update_counter_path: str = os.path.join(directory, "update_counter.txt")
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if os.path.exists(update_counter_path):
            # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
            with open(update_counter_path, "r") as f:
                self.update_counter = int(f.read())
        else:
            self.update_counter = 0
            print(f"⚠️ Update counter file not found: {update_counter_path}. Setting update_counter to 0.")

