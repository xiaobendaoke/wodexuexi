"""
中文注释说明：test.py

文件作用：
    封装训练后模型的测试评估流程，按回合运行环境并记录奖励、时延、能耗和任务处理效果。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - _get_episode_runtime_audit(): 训练或测试的回合编号。
    - test_model(): 当前训练或测试的多智能体模型实例。

主要依赖：
    marl_models, environment, utils, config, numpy, time

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import MARLModel
from environment.env import Env
from utils.logger import Logger, Log

# from utils.plot_snapshots import plot_snapshot

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import numpy as np
import time


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


# 函数 test_model：当前训练或测试的多智能体模型实例，主要参数：env, model, logger, num_episodes。
def test_model(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    episode_log: Log = Log()

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
            # if step % config.TEST_IMG_FREQ == 0:
            # plot_snapshot(env, episode, step, logger.log_dir, logger.timestamp)

            obs_arr: np.ndarray = np.asarray(obs, dtype=np.float32)
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
        if episode % config.TEST_LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.TEST_LOG_FREQ, elapsed_time)
