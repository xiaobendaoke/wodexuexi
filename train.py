"""
中文注释说明：train.py

文件作用：
    封装强化学习模型的训练流程，区分离策略、在策略和基线模型，并记录训练日志与模型检查点。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - _get_episode_runtime_audit(): 训练或测试的回合编号。
    - train_on_policy(): 执行模型训练流程。
    - train_off_policy(): 执行模型训练流程。
    - train_baselines(): 执行模型训练流程。

主要依赖：
    marl_models, environment, utils, config, torch, numpy, time, optuna

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

from marl_models.base_model import MARLModel
from marl_models.buffer_and_helpers import ReplayBuffer, RolloutBuffer, AttentionRolloutBuffer
from marl_models.utils import save_models
from environment.env import Env
from utils.logger import Logger, Log

# from utils.plot_snapshots import plot_snapshot  # snapshot plotting, comment if not needed

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import torch
import numpy as np
import time
import optuna


# 函数 _get_episode_runtime_audit：训练或测试的回合编号，主要参数：env。
def _get_episode_runtime_audit(env: Env) -> dict[str, object]:
    """Read the latest episode-level offloading audit snapshot from the environment."""

    audit: dict[str, object] = env.last_runtime_audit or {}
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "service_learned_decision_count": float(audit.get("episode_service_learned_decision_count", 0.0)),
        "service_heuristic_decision_count": float(audit.get("episode_service_heuristic_decision_count", 0.0)),
        "service_fallback_count": float(audit.get("episode_service_fallback_count", 0.0)),
        "service_predict_exception_fallback_count": float(
            audit.get("episode_service_predict_exception_fallback_count", 0.0)
        ),
        "service_offload_policy_requested": str(audit.get("service_offload_policy_requested", "heuristic")),
        "service_offload_policy_loaded": bool(audit.get("service_offload_policy_loaded", False)),
        "service_offload_policy_checkpoint_path": audit.get("service_offload_policy_checkpoint_path"),
        "service_offload_policy_feature_family": audit.get("service_offload_policy_feature_family"),
    }


# 函数 train_on_policy：执行模型训练流程，主要参数：env, model, logger, num_episodes, trial。
def train_on_policy(env: Env, model: MARLModel, logger: Logger, num_episodes: int, trial: optuna.Trial | None = None) -> float:
    start_time: float = time.time()
    BufferClass: type[RolloutBuffer] = AttentionRolloutBuffer if "attention" in model.model_name.lower() else RolloutBuffer
    buffer: RolloutBuffer = BufferClass(num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, buffer_size=config.PPO_ROLLOUT_LENGTH, device=model.device)
    max_time_steps: int = num_episodes * config.STEPS_PER_EPISODE
    num_updates: int = max_time_steps // config.PPO_ROLLOUT_LENGTH
    assert num_updates > 0, "num_updates is 0, please modify settings."
    save_freq: int = max(num_updates // 10, 100)

    recent_rewards: list[float] = []  # Tracking metrics for tuning

    recent_losses: dict = {"actor": None, "critic": None, "entropy": None}  # For logging most recent losses with episodes

    episode_log: Log = Log()
    episode: int = 1
    episode_step: int = 0
    episode_reward: float = 0.0
    episode_latency: float = 0.0
    episode_energy: float = 0.0
    episode_fairness: float = 0.0
    episode_offline_rate: float = 0.0
    episode_deadline_satisfaction_sum: float = 0.0
    episode_offload_local_sum: float = 0.0
    episode_offload_cooperative_sum: float = 0.0
    episode_offload_mbs_sum: float = 0.0
    episode_mbs_load_sum: float = 0.0

    obs: list[np.ndarray] = env.reset()
    obs_arr: np.ndarray = np.asarray(obs, dtype=np.float32)
    state: np.ndarray = np.concatenate(obs, axis=0, dtype=np.float32)
    last_obs: list[np.ndarray] = obs

    # reset_trajectories(env)  # tracking code, comment if not needed
    # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

    # 循环处理：遍历 update 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for update in range(1, num_updates + 1):
        # 循环处理：遍历 _ 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for _ in range(1, config.PPO_ROLLOUT_LENGTH + 1):
            # if episode_step > 0 and episode_step % config.IMG_FREQ == 0:
            # plot_snapshot(env, episode, episode_step, logger.log_dir, logger.timestamp)

            raw_actions, log_probs, values = model.get_action_and_value(obs_arr, state)
            actions: np.ndarray = np.clip(raw_actions, -1.0, 1.0)

            next_obs, rewards, metrics = env.step(actions)
            next_state: np.ndarray = np.concatenate(next_obs, axis=0, dtype=np.float32)
            # update_trajectories(env)  # tracking code, comment if not needed

            episode_step += 1
            done: bool = episode_step >= config.STEPS_PER_EPISODE
            buffer.add(state, obs_arr, raw_actions, log_probs, rewards, done, values)
            obs = next_obs
            obs_arr = np.asarray(obs, dtype=np.float32)
            state = next_state
            last_obs = obs

            total_latency = metrics["latency"]
            total_energy = metrics["energy"]
            jfi = metrics["fairness"]
            offline_rate = metrics["offline_rate"]
            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_offline_rate = offline_rate
            episode_deadline_satisfaction_sum += metrics["deadline_satisfaction_rate"]
            episode_offload_local_sum += metrics["offloading_ratio_local"]
            episode_offload_cooperative_sum += metrics["offloading_ratio_cooperative"]
            episode_offload_mbs_sum += metrics["offloading_ratio_mbs"]
            episode_mbs_load_sum += metrics["mbs_load_ratio"]

            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if done:
                # plot_snapshot(env, episode, episode_step, logger.log_dir, logger.timestamp)  # Final snapshot of episode
                recent_rewards.append(episode_reward)
                # Request-level metrics are currently logged as means of per-step ratios.
                episode_length: float = max(float(episode_step), 1.0)
                runtime_audit: dict[str, object] = _get_episode_runtime_audit(env)
                episode_log.append(
                    episode_reward,
                    episode_latency,
                    episode_energy,
                    episode_fairness,
                    episode_offline_rate,
                    deadline_satisfaction_rate=episode_deadline_satisfaction_sum / episode_length,
                    offloading_ratio_local=episode_offload_local_sum / episode_length,
                    offloading_ratio_cooperative=episode_offload_cooperative_sum / episode_length,
                    offloading_ratio_mbs=episode_offload_mbs_sum / episode_length,
                    mbs_load_ratio=episode_mbs_load_sum / episode_length,
                    service_learned_decision_count=float(runtime_audit["service_learned_decision_count"]),
                    service_heuristic_decision_count=float(runtime_audit["service_heuristic_decision_count"]),
                    service_fallback_count=float(runtime_audit["service_fallback_count"]),
                    service_predict_exception_fallback_count=float(
                        runtime_audit["service_predict_exception_fallback_count"]
                    ),
                    service_offload_policy_requested=str(runtime_audit["service_offload_policy_requested"]),
                    service_offload_policy_loaded=bool(runtime_audit["service_offload_policy_loaded"]),
                    service_offload_policy_checkpoint_path=runtime_audit["service_offload_policy_checkpoint_path"],
                    service_offload_policy_feature_family=runtime_audit["service_offload_policy_feature_family"],
                )

                # Optuna Pruning Check
                if trial:
                    current_avg_reward: float = float(np.mean(recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards))
                    trial.report(current_avg_reward, episode)
                    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                    if trial.should_prune():
                        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
                        raise optuna.TrialPruned()

                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if episode % config.LOG_FREQ == 0:
                    elapsed_time: float = time.time() - start_time
                    logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, losses=recent_losses)

                obs = env.reset()
                obs_arr = np.asarray(obs, dtype=np.float32)
                state = np.concatenate(obs, axis=0, dtype=np.float32)

                episode += 1
                episode_step = 0
                episode_reward, episode_latency, episode_energy, episode_fairness, episode_offline_rate = 0.0, 0.0, 0.0, 0.0, 0.0
                episode_deadline_satisfaction_sum = 0.0
                episode_offload_local_sum = 0.0
                episode_offload_cooperative_sum = 0.0
                episode_offload_mbs_sum = 0.0
                episode_mbs_load_sum = 0.0
                # reset_trajectories(env)  # tracking code, comment if not needed
                # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with torch.no_grad():
            last_obs_arr: np.ndarray = np.asarray(last_obs, dtype=np.float32)
            last_state: np.ndarray = np.concatenate(last_obs, axis=0, dtype=np.float32)
            _, _, last_values = model.get_action_and_value(last_obs_arr, last_state)

        buffer.compute_returns_and_advantages(last_values, config.DISCOUNT_FACTOR, config.PPO_GAE_LAMBDA)

        temp_losses: dict = {"actor": [], "critic": [], "entropy": []}  # Only for this update

        # 循环处理：遍历 _ 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for _ in range(config.PPO_EPOCHS):
            # 循环处理：遍历 batch 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for batch in buffer.get_batches(config.PPO_BATCH_SIZE):
                loss_dict = model.update(batch)
                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if loss_dict:
                    temp_losses["actor"].append(loss_dict.get("actor"))
                    temp_losses["critic"].append(loss_dict.get("critic"))
                    temp_losses["entropy"].append(loss_dict.get("entropy"))

        buffer.clear()

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if temp_losses["actor"]:
            recent_losses = {
                "actor": float(np.mean([x for x in temp_losses["actor"] if x is not None])),
                "critic": float(np.mean([x for x in temp_losses["critic"] if x is not None])),
                "entropy": float(np.mean([x for x in temp_losses["entropy"] if x is not None])),
            }
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if update % save_freq == 0 and update < num_updates:
            save_models(model, update, "update", logger.timestamp)

    save_models(model, -1, "update", logger.timestamp, final=True)

    # Return average reward of last 10% of training for optimization score
    return float(np.mean(recent_rewards[-max(1, int(num_episodes * 0.1)) :]))


# 函数 train_off_policy：执行模型训练流程，主要参数：env, model, logger, num_episodes, total_step_count, trial。
def train_off_policy(env: Env, model: MARLModel, logger: Logger, num_episodes: int, total_step_count: int, trial: optuna.Trial | None = None) -> float:
    start_time: float = time.time()
    buffer: ReplayBuffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
    save_freq: int = max(num_episodes // 10, 100)
    episode_log: Log = Log()

    accumulated_losses: dict = {"actor": [], "critic": []}
    has_alpha: bool = "sac" in model.model_name.lower()  # Only track alpha loss for SAC-based algorithms
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if has_alpha:
        accumulated_losses["alpha"] = []
    recent_rewards: list[float] = []  # Tracking metrics for tuning

    # 循环处理：遍历 episode 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        model.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_offline_rate: float = 0.0
        episode_deadline_satisfaction_sum: float = 0.0
        episode_offload_local_sum: float = 0.0
        episode_offload_cooperative_sum: float = 0.0
        episode_offload_mbs_sum: float = 0.0
        episode_mbs_load_sum: float = 0.0
        # reset_trajectories(env)  # tracking code, comment if not needed
        # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

        # 循环处理：遍历 step 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for step in range(1, config.STEPS_PER_EPISODE + 1):
            # if step % config.IMG_FREQ == 0:
            # plot_snapshot(env, episode, step, logger.log_dir, logger.timestamp)

            total_step_count += 1
            obs_arr: np.ndarray = np.array(obs, dtype=np.float32)
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if total_step_count <= config.INITIAL_RANDOM_STEPS:
                actions: np.ndarray = np.array([np.random.uniform(-1, 1, config.ACTION_DIM) for _ in range(config.NUM_UAVS)])
            else:
                actions = model.select_actions(obs_arr, exploration=True)

            next_obs, rewards, metrics = env.step(actions)
            next_obs_arr: np.ndarray = np.array(next_obs, dtype=np.float32)
            # update_trajectories(env)  # tracking code, comment if not needed
            done: bool = step >= config.STEPS_PER_EPISODE
            buffer.add(obs_arr, actions, rewards, next_obs_arr, done)

            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if (total_step_count > config.INITIAL_RANDOM_STEPS) and (step % config.LEARN_FREQ == 0) and (len(buffer) > config.REPLAY_BATCH_SIZE):
                batch = buffer.sample(config.REPLAY_BATCH_SIZE)
                loss_dict = model.update(batch)
                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if loss_dict:
                    accumulated_losses["actor"].append(loss_dict.get("actor"))
                    accumulated_losses["critic"].append(loss_dict.get("critic"))
                    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                    if has_alpha and "alpha" in loss_dict:
                        accumulated_losses["alpha"].append(loss_dict.get("alpha"))

            obs = next_obs

            total_latency = metrics["latency"]
            total_energy = metrics["energy"]
            jfi = metrics["fairness"]
            offline_rate = metrics["offline_rate"]
            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_offline_rate = offline_rate
            episode_deadline_satisfaction_sum += metrics["deadline_satisfaction_rate"]
            episode_offload_local_sum += metrics["offloading_ratio_local"]
            episode_offload_cooperative_sum += metrics["offloading_ratio_cooperative"]
            episode_offload_mbs_sum += metrics["offloading_ratio_mbs"]
            episode_mbs_load_sum += metrics["mbs_load_ratio"]
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if done:
                break

        # Request-level metrics are currently logged as means of per-step ratios.
        episode_length = max(float(config.STEPS_PER_EPISODE), 1.0)
        runtime_audit = _get_episode_runtime_audit(env)
        episode_log.append(
            episode_reward,
            episode_latency,
            episode_energy,
            episode_fairness,
            episode_offline_rate,
            deadline_satisfaction_rate=episode_deadline_satisfaction_sum / episode_length,
            offloading_ratio_local=episode_offload_local_sum / episode_length,
            offloading_ratio_cooperative=episode_offload_cooperative_sum / episode_length,
            offloading_ratio_mbs=episode_offload_mbs_sum / episode_length,
            mbs_load_ratio=episode_mbs_load_sum / episode_length,
            service_learned_decision_count=float(runtime_audit["service_learned_decision_count"]),
            service_heuristic_decision_count=float(runtime_audit["service_heuristic_decision_count"]),
            service_fallback_count=float(runtime_audit["service_fallback_count"]),
            service_predict_exception_fallback_count=float(runtime_audit["service_predict_exception_fallback_count"]),
            service_offload_policy_requested=str(runtime_audit["service_offload_policy_requested"]),
            service_offload_policy_loaded=bool(runtime_audit["service_offload_policy_loaded"]),
            service_offload_policy_checkpoint_path=runtime_audit["service_offload_policy_checkpoint_path"],
            service_offload_policy_feature_family=runtime_audit["service_offload_policy_feature_family"],
        )
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            # Prepare averaged losses for logging
            avg_losses: dict | None = None
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if accumulated_losses["actor"]:
                avg_losses = {
                    "actor": float(np.mean([x for x in accumulated_losses["actor"] if x is not None])),
                    "critic": float(np.mean([x for x in accumulated_losses["critic"] if x is not None])),
                }
                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if has_alpha and accumulated_losses["alpha"]:
                    avg_losses["alpha"] = float(np.mean([x for x in accumulated_losses["alpha"] if x is not None]))
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, losses=avg_losses)
            # Reset accumulated losses for next logging interval
            accumulated_losses = {"actor": [], "critic": []}
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if has_alpha:
                accumulated_losses["alpha"] = []

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if episode % save_freq == 0 and episode < num_episodes:
            save_models(model, episode, "episode", logger.timestamp, total_steps=total_step_count)

        recent_rewards.append(episode_reward)
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if trial:
            # Report average of last 10 episodes
            current_avg_reward: float = float(np.mean(recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards))
            trial.report(current_avg_reward, episode)
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if trial.should_prune():
                # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
                raise optuna.TrialPruned()

    save_models(model, -1, "episode", logger.timestamp, final=True, total_steps=total_step_count)

    # Return average reward of last 10% of training for optimization score
    return float(np.mean(recent_rewards[-max(1, int(num_episodes * 0.1)) :]))


# 函数 train_baselines：执行模型训练流程，主要参数：env, model, logger, num_episodes。
def train_baselines(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> float:
    start_time: float = time.time()
    episode_log: Log = Log()

    # 循环处理：遍历 episode 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for episode in range(1, num_episodes + 1):
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if hasattr(model, "static_positions") and model.static_positions is not None:
            obs = env.reset(initial_positions=model.static_positions)
        else:
            obs = env.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_offline_rate: float = 0.0
        episode_deadline_satisfaction_sum: float = 0.0
        episode_offload_local_sum: float = 0.0
        episode_offload_cooperative_sum: float = 0.0
        episode_offload_mbs_sum: float = 0.0
        episode_mbs_load_sum: float = 0.0
        # reset_trajectories(env)  # tracking code, comment if not needed
        # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

        # 循环处理：遍历 step 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for step in range(1, config.STEPS_PER_EPISODE + 1):
            # if step % config.IMG_FREQ == 0:
            # plot_snapshot(env, episode, step, logger.log_dir, logger.timestamp)

            obs_arr: np.ndarray = np.array(obs, dtype=np.float32)
            actions: np.ndarray = model.select_actions(obs_arr, exploration=False)
            next_obs, rewards, metrics = env.step(actions)
            # update_trajectories(env)  # tracking code, comment if not needed
            done: bool = step >= config.STEPS_PER_EPISODE
            obs = next_obs

            total_latency = metrics["latency"]
            total_energy = metrics["energy"]
            jfi = metrics["fairness"]
            offline_rate = metrics["offline_rate"]
            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_offline_rate = offline_rate
            episode_deadline_satisfaction_sum += metrics["deadline_satisfaction_rate"]
            episode_offload_local_sum += metrics["offloading_ratio_local"]
            episode_offload_cooperative_sum += metrics["offloading_ratio_cooperative"]
            episode_offload_mbs_sum += metrics["offloading_ratio_mbs"]
            episode_mbs_load_sum += metrics["mbs_load_ratio"]
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if done:
                break

        # Request-level metrics are currently logged as means of per-step ratios.
        episode_length = max(float(config.STEPS_PER_EPISODE), 1.0)
        runtime_audit = _get_episode_runtime_audit(env)
        episode_log.append(
            episode_reward,
            episode_latency,
            episode_energy,
            episode_fairness,
            episode_offline_rate,
            deadline_satisfaction_rate=episode_deadline_satisfaction_sum / episode_length,
            offloading_ratio_local=episode_offload_local_sum / episode_length,
            offloading_ratio_cooperative=episode_offload_cooperative_sum / episode_length,
            offloading_ratio_mbs=episode_offload_mbs_sum / episode_length,
            mbs_load_ratio=episode_mbs_load_sum / episode_length,
            service_learned_decision_count=float(runtime_audit["service_learned_decision_count"]),
            service_heuristic_decision_count=float(runtime_audit["service_heuristic_decision_count"]),
            service_fallback_count=float(runtime_audit["service_fallback_count"]),
            service_predict_exception_fallback_count=float(runtime_audit["service_predict_exception_fallback_count"]),
            service_offload_policy_requested=str(runtime_audit["service_offload_policy_requested"]),
            service_offload_policy_loaded=bool(runtime_audit["service_offload_policy_loaded"]),
            service_offload_policy_checkpoint_path=runtime_audit["service_offload_policy_checkpoint_path"],
            service_offload_policy_feature_family=runtime_audit["service_offload_policy_feature_family"],
        )
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, losses=None)

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return 0.0  # Baseline training does not need tuning
