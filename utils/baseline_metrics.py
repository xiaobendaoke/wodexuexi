"""
统一 baseline 指标收集模块。

提供 run_single_episode()、run_single_episode_joint() 和聚合函数，
供 run_baseline_comparison_experiment.py 调用。
所有 baseline 共用同一套指标采集和 JSON 输出格式。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from environment.env import Env


# 方案 5.4 定义的统一 JSON 字段
SUMMARY_METRIC_NAMES: list[str] = [
    "reward",
    "latency",
    "energy",
    "energy_efficiency",
    "fairness",
    "offline_rate",
    "deadline_satisfaction_rate",
    "offloading_ratio_local",
    "offloading_ratio_cooperative",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
]

# 新增指标字段（v2 版本，不删除旧字段）
SUMMARY_METRIC_NAMES_V2: list[str] = [
    "offline_rate_final",
    "offline_rate_step_mean",
    "fairness_final",
    "fairness_step_mean",
    "dsr_step_mean",
    "dsr_request_weighted",
    "energy_efficiency_episode",
    "energy_efficiency_global",
    "service_requests_generated",
    "service_requests_processed",
    "deadline_satisfied_service_requests",
    "service_offloads_local",
    "service_offloads_cooperative",
    "service_offloads_mbs",
    "offloading_ratio_local_processed",
    "offloading_ratio_cooperative_processed",
    "offloading_ratio_mbs_processed",
    "mbs_load_ratio_generated",
    "processed_request_ratio",
    "deadline_satisfied_per_processed",
]


def set_global_seed(seed: int | None) -> None:
    """设置全局随机种子，保证实验可复现。"""
    if seed is None:
        return
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _collect_episode_metrics(env: Env, ep_reward: float, ep_latency: float, ep_energy: float,
                             ep_fairness: float, ep_offline_rate: float, ep_deadline: float,
                             ep_local: float, ep_coop: float, ep_mbs: float, ep_mbs_load: float,
                             ep_service_generated: float, ep_service_processed: float,
                             ep_deadline_satisfied: float,
                             ep_fairness_values: list[float] | None = None,
                             ep_offline_rate_values: list[float] | None = None,
                             ep_dsr_values: list[float] | None = None,
                             ep_local_offloads: float = 0.0,
                             ep_coop_offloads: float = 0.0,
                             ep_mbs_offloads: float = 0.0) -> dict[str, float]:
    """从 episode 累计量生成标准化指标 dict。

    新增参数（v2）：
        ep_fairness_values: 每步的 fairness 值列表，用于计算 step_mean
        ep_offline_rate_values: 每步的 offline_rate 值列表，用于计算 step_mean
        ep_dsr_values: 每步的 deadline_satisfaction_rate 值列表，用于计算 step_mean
        ep_local_offloads: episode 内本地卸载总数
        ep_coop_offloads: episode 内协作卸载总数
        ep_mbs_offloads: episode 内 MBS 卸载总数
    """
    episode_length = max(float(config.STEPS_PER_EPISODE), 1.0)
    runtime_audit = dict(env.last_runtime_audit)
    energy_efficiency = ep_deadline_satisfied / max(float(ep_energy), float(config.EPSILON))

    # 计算 step_mean 指标（v2 新增）
    fairness_step_mean = float(np.mean(ep_fairness_values)) if ep_fairness_values else float(ep_fairness)
    offline_rate_step_mean = float(np.mean(ep_offline_rate_values)) if ep_offline_rate_values else float(ep_offline_rate)
    dsr_step_mean = float(np.mean(ep_dsr_values)) if ep_dsr_values else float(ep_deadline / episode_length)

    # 计算 request-weighted DSR（v2 新增）
    dsr_request_weighted = float(ep_deadline_satisfied) / max(float(ep_service_generated), float(config.EPSILON))

    # 计算卸载比例（以 processed 为分母）（v2 新增）
    offloading_ratio_local_processed = float(ep_local_offloads) / max(float(ep_service_processed), float(config.EPSILON))
    offloading_ratio_cooperative_processed = float(ep_coop_offloads) / max(float(ep_service_processed), float(config.EPSILON))
    offloading_ratio_mbs_processed = float(ep_mbs_offloads) / max(float(ep_service_processed), float(config.EPSILON))

    # 计算 MBS load ratio（以 generated 为分母）（v2 新增）
    mbs_load_ratio_generated = float(ep_mbs_offloads) / max(float(ep_service_generated), float(config.EPSILON))

    # 计算 processed request ratio（v2 新增）
    processed_request_ratio = float(ep_service_processed) / max(float(ep_service_generated), float(config.EPSILON))

    # 计算 deadline_satisfied_per_processed（v2 新增）
    deadline_satisfied_per_processed = float(ep_deadline_satisfied) / max(float(ep_service_processed), float(config.EPSILON))

    return {
        # 旧字段（保留兼容性）
        "reward": float(ep_reward),
        "latency": float(ep_latency),
        "energy": float(ep_energy),
        "energy_efficiency": float(energy_efficiency),
        "fairness": float(ep_fairness),
        "offline_rate": float(ep_offline_rate),
        "deadline_satisfaction_rate": float(ep_deadline / episode_length),
        "offloading_ratio_local": float(ep_local / episode_length),
        "offloading_ratio_cooperative": float(ep_coop / episode_length),
        "offloading_ratio_mbs": float(ep_mbs / episode_length),
        "mbs_load_ratio": float(ep_mbs_load / episode_length),
        "service_learned_decision_count": float(runtime_audit.get("episode_service_learned_decision_count", 0.0)),
        "service_heuristic_decision_count": float(runtime_audit.get("episode_service_heuristic_decision_count", 0.0)),
        "service_fallback_count": float(runtime_audit.get("episode_service_fallback_count", 0.0)),
        "service_predict_exception_fallback_count": float(runtime_audit.get("episode_service_predict_exception_fallback_count", 0.0)),
        "service_requests_generated": float(ep_service_generated),
        "service_requests_processed": float(ep_service_processed),
        "deadline_satisfied_service_requests": float(ep_deadline_satisfied),
        # 新增字段（v2）
        "offline_rate_final": float(ep_offline_rate),
        "offline_rate_step_mean": float(offline_rate_step_mean),
        "fairness_final": float(ep_fairness),
        "fairness_step_mean": float(fairness_step_mean),
        "dsr_step_mean": float(dsr_step_mean),
        "dsr_request_weighted": float(dsr_request_weighted),
        "energy_efficiency_episode": float(energy_efficiency),
        "energy_efficiency_global": float(energy_efficiency),  # 对于单 episode，两者相同
        "service_offloads_local": float(ep_local_offloads),
        "service_offloads_cooperative": float(ep_coop_offloads),
        "service_offloads_mbs": float(ep_mbs_offloads),
        "offloading_ratio_local_processed": float(offloading_ratio_local_processed),
        "offloading_ratio_cooperative_processed": float(offloading_ratio_cooperative_processed),
        "offloading_ratio_mbs_processed": float(offloading_ratio_mbs_processed),
        "mbs_load_ratio_generated": float(mbs_load_ratio_generated),
        "processed_request_ratio": float(processed_request_ratio),
        "deadline_satisfied_per_processed": float(deadline_satisfied_per_processed),
    }


def run_single_episode(
    env: Env,
    trajectory_model,
    offload_model=None,
    exploration: bool = False,
    record_trajectory: bool = False,
    policy_type: str = "learned",
) -> tuple[dict[str, float], list[dict] | None]:
    """运行单个 episode，返回 (episode_metrics, trajectory_data)。

    支持两种模式：
    - 分层模式（proposed/vanilla_mappo/ippo）：trajectory_model + offload_model
    - 非学习模式（random/uniform）：trajectory_model 自带 get_offload_action()
    """
    obs = env.reset()
    trajectory_obs = np.asarray(obs, dtype=np.float32)

    ep_reward = 0.0
    ep_latency = 0.0
    ep_energy = 0.0
    ep_fairness = 0.0
    ep_offline_rate = 0.0
    ep_deadline = 0.0
    ep_local = 0.0
    ep_coop = 0.0
    ep_mbs = 0.0
    ep_mbs_load = 0.0
    ep_service_generated = 0.0
    ep_service_processed = 0.0
    ep_deadline_satisfied = 0.0
    ep_local_offloads = 0.0
    ep_coop_offloads = 0.0
    ep_mbs_offloads = 0.0

    # v2 新增：记录每步的 fairness、offline_rate、DSR 值
    ep_fairness_values: list[float] = []
    ep_offline_rate_values: list[float] = []
    ep_dsr_values: list[float] = []

    trajectory_frames: list[dict] = []

    for step_idx in range(config.STEPS_PER_EPISODE):
        # 获取卸载观测和 mask
        offload_obs, offload_masks = env.get_offloading_obs_and_masks()

        # 轨迹动作
        traj_actions = trajectory_model.select_actions(trajectory_obs, exploration=exploration)

        # 卸载动作
        offload_actions = None
        if offload_model is not None:
            # 学习卸载模型
            offload_actions, _, _ = offload_model.get_action_and_value(
                offload_obs, masks=offload_masks, exploration=exploration,
            )
        elif hasattr(trajectory_model, "get_offload_action"):
            # 非学习模型自带卸载逻辑（Random/Uniform）
            offload_actions = trajectory_model.get_offload_action(offload_obs, offload_masks)

        # 环境步进
        next_obs, rewards, metrics = env.step(traj_actions, offloading_actions=offload_actions)
        obs = next_obs
        trajectory_obs = np.asarray(next_obs, dtype=np.float32)

        # 记录轨迹帧（在 env.step 之后，确保 offloading ratio 是当前步的）
        if record_trajectory:
            uav_positions = []
            for uav in env.uavs:
                uav_positions.append([float(uav.pos[0]), float(uav.pos[1])])
            trajectory_frames.append({
                "step": step_idx,
                "uav_positions": uav_positions,
                "offloading_ratio_local": float(metrics.get("offloading_ratio_local", 0.0)),
                "offloading_ratio_cooperative": float(metrics.get("offloading_ratio_cooperative", 0.0)),
                "offloading_ratio_mbs": float(metrics.get("offloading_ratio_mbs", 0.0)),
                "mbs_load_ratio": float(metrics.get("mbs_load_ratio", 0.0)),
                "deadline_satisfaction_rate": float(metrics.get("deadline_satisfaction_rate", 0.0)),
            })

        ep_reward += float(np.sum(rewards))
        ep_latency += float(metrics["latency"])
        ep_energy += float(metrics["energy"])
        ep_fairness = float(metrics["fairness"])
        ep_offline_rate = float(metrics["offline_rate"])
        ep_deadline += float(metrics["deadline_satisfaction_rate"])
        ep_local += float(metrics["offloading_ratio_local"])
        ep_coop += float(metrics["offloading_ratio_cooperative"])
        ep_mbs += float(metrics["offloading_ratio_mbs"])
        ep_mbs_load += float(metrics["mbs_load_ratio"])
        service_generated = float(metrics.get("service_requests_generated", 0.0))
        ep_service_generated += service_generated
        ep_service_processed += float(metrics.get("service_requests_processed", 0.0))
        ep_deadline_satisfied += float(metrics.get("deadline_satisfaction_rate", 0.0)) * service_generated

        # v2 新增：记录卸载计数
        ep_local_offloads += float(metrics.get("service_offloads_local", 0.0))
        ep_coop_offloads += float(metrics.get("service_offloads_cooperative", 0.0))
        ep_mbs_offloads += float(metrics.get("service_offloads_mbs", 0.0))

        # v2 新增：记录每步的 fairness、offline_rate、DSR 值
        ep_fairness_values.append(float(metrics["fairness"]))
        ep_offline_rate_values.append(float(metrics["offline_rate"]))
        ep_dsr_values.append(float(metrics["deadline_satisfaction_rate"]))

    episode_metrics = _collect_episode_metrics(
        env, ep_reward, ep_latency, ep_energy, ep_fairness, ep_offline_rate,
        ep_deadline, ep_local, ep_coop, ep_mbs, ep_mbs_load,
        ep_service_generated, ep_service_processed, ep_deadline_satisfied,
        ep_fairness_values=ep_fairness_values,
        ep_offline_rate_values=ep_offline_rate_values,
        ep_dsr_values=ep_dsr_values,
        ep_local_offloads=ep_local_offloads,
        ep_coop_offloads=ep_coop_offloads,
        ep_mbs_offloads=ep_mbs_offloads,
    )
    episode_metrics["policy_type"] = policy_type
    return episode_metrics, trajectory_frames if record_trajectory else None


def run_single_episode_joint(
    env: Env,
    model,
    exploration: bool = False,
    record_trajectory: bool = False,
    policy_type: str = "learned",
) -> tuple[dict[str, float], list[dict] | None]:
    """Joint MAPPO 的单 episode 评估。"""
    obs = env.reset()
    trajectory_obs = np.asarray(obs, dtype=np.float32)

    ep_reward = 0.0
    ep_latency = 0.0
    ep_energy = 0.0
    ep_fairness = 0.0
    ep_offline_rate = 0.0
    ep_deadline = 0.0
    ep_local = 0.0
    ep_coop = 0.0
    ep_mbs = 0.0
    ep_mbs_load = 0.0
    ep_service_generated = 0.0
    ep_service_processed = 0.0
    ep_deadline_satisfied = 0.0
    ep_local_offloads = 0.0
    ep_coop_offloads = 0.0
    ep_mbs_offloads = 0.0

    # v2 新增：记录每步的 fairness、offline_rate、DSR 值
    ep_fairness_values: list[float] = []
    ep_offline_rate_values: list[float] = []
    ep_dsr_values: list[float] = []

    trajectory_frames: list[dict] = []

    for step_idx in range(config.STEPS_PER_EPISODE):
        offload_obs, offload_masks = env.get_offloading_obs_and_masks()
        joint_obs = np.concatenate([trajectory_obs, offload_obs], axis=-1).astype(np.float32)
        traj_actions, offload_actions, _, _ = model.get_action_and_value(
            joint_obs, masks=offload_masks, exploration=exploration,
        )
        traj_actions = np.clip(traj_actions, -1.0, 1.0)

        next_obs, rewards, metrics = env.step(traj_actions, offloading_actions=offload_actions)

        # 记录轨迹帧（在 env.step 之后，确保 offloading ratio 是当前步的）
        if record_trajectory:
            uav_positions = []
            for uav in env.uavs:
                uav_positions.append([float(uav.pos[0]), float(uav.pos[1])])
            trajectory_frames.append({
                "step": step_idx,
                "uav_positions": uav_positions,
                "offloading_ratio_local": float(metrics.get("offloading_ratio_local", 0.0)),
                "offloading_ratio_cooperative": float(metrics.get("offloading_ratio_cooperative", 0.0)),
                "offloading_ratio_mbs": float(metrics.get("offloading_ratio_mbs", 0.0)),
                "mbs_load_ratio": float(metrics.get("mbs_load_ratio", 0.0)),
                "deadline_satisfaction_rate": float(metrics.get("deadline_satisfaction_rate", 0.0)),
            })
        trajectory_obs = np.asarray(next_obs, dtype=np.float32)

        ep_reward += float(np.sum(rewards))
        ep_latency += float(metrics["latency"])
        ep_energy += float(metrics["energy"])
        ep_fairness = float(metrics["fairness"])
        ep_offline_rate = float(metrics["offline_rate"])
        ep_deadline += float(metrics["deadline_satisfaction_rate"])
        ep_local += float(metrics["offloading_ratio_local"])
        ep_coop += float(metrics["offloading_ratio_cooperative"])
        ep_mbs += float(metrics["offloading_ratio_mbs"])
        ep_mbs_load += float(metrics["mbs_load_ratio"])
        service_generated = float(metrics.get("service_requests_generated", 0.0))
        ep_service_generated += service_generated
        ep_service_processed += float(metrics.get("service_requests_processed", 0.0))
        ep_deadline_satisfied += float(metrics.get("deadline_satisfaction_rate", 0.0)) * service_generated

        # v2 新增：记录卸载计数
        ep_local_offloads += float(metrics.get("service_offloads_local", 0.0))
        ep_coop_offloads += float(metrics.get("service_offloads_cooperative", 0.0))
        ep_mbs_offloads += float(metrics.get("service_offloads_mbs", 0.0))

        # v2 新增：记录每步的 fairness、offline_rate、DSR 值
        ep_fairness_values.append(float(metrics["fairness"]))
        ep_offline_rate_values.append(float(metrics["offline_rate"]))
        ep_dsr_values.append(float(metrics["deadline_satisfaction_rate"]))

    episode_metrics = _collect_episode_metrics(
        env, ep_reward, ep_latency, ep_energy, ep_fairness, ep_offline_rate,
        ep_deadline, ep_local, ep_coop, ep_mbs, ep_mbs_load,
        ep_service_generated, ep_service_processed, ep_deadline_satisfied,
        ep_fairness_values=ep_fairness_values,
        ep_offline_rate_values=ep_offline_rate_values,
        ep_dsr_values=ep_dsr_values,
        ep_local_offloads=ep_local_offloads,
        ep_coop_offloads=ep_coop_offloads,
        ep_mbs_offloads=ep_mbs_offloads,
    )
    episode_metrics["policy_type"] = policy_type
    return episode_metrics, trajectory_frames if record_trajectory else None


def aggregate_metric_dicts(entries: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """将多 episode 指标聚合为 mean/std。

    v2 版本：同时处理旧字段和新增字段。
    """
    # 合并所有指标名称
    all_metrics = SUMMARY_METRIC_NAMES + SUMMARY_METRIC_NAMES_V2
    return {
        metric: {
            "mean": float(np.mean([entry[metric] for entry in entries])) if entries else 0.0,
            "std": float(np.std([entry[metric] for entry in entries])) if entries else 0.0,
        }
        for metric in all_metrics
    }


def save_baseline_result(
    method: str,
    seed: int,
    workload_seed: int,
    episode_metrics_list: list[dict[str, float]],
    output_dir: str = "results/baseline_comparison",
    training_curve: list[dict] | None = None,
    trajectory_data: list[dict] | None = None,
) -> str:
    """保存 baseline 结果到统一目录结构。

    输出文件：
        <output_dir>/<method>/metrics_seed_<S>_workload_<W>.json
        <output_dir>/<method>/training_curve_seed_<S>.json  (可选)
        <output_dir>/<method>/trajectory_seed_<S>_workload_<W>.json  (可选)
        <output_dir>/<method>/config_seed_<S>.json
    """
    method_dir = Path(output_dir) / method
    method_dir.mkdir(parents=True, exist_ok=True)

    # 聚合指标
    aggregate = aggregate_metric_dicts(episode_metrics_list)

    # 保存 metrics
    metrics_path = method_dir / f"metrics_seed_{seed}_workload_{workload_seed}.json"
    metrics_data = {
        "method": method,
        "seed": seed,
        "workload_seed": workload_seed,
        "num_episodes": len(episode_metrics_list),
        "aggregate": aggregate,
        "per_episode": episode_metrics_list,
    }
    metrics_path.write_text(json.dumps(metrics_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 保存 training curve（如果提供）
    if training_curve is not None:
        curve_path = method_dir / f"training_curve_seed_{seed}.json"
        curve_path.write_text(json.dumps(training_curve, indent=2, ensure_ascii=False), encoding="utf-8")

    # 保存 trajectory（如果提供）
    if trajectory_data is not None:
        traj_path = method_dir / f"trajectory_seed_{seed}_workload_{workload_seed}.json"
        traj_path.write_text(json.dumps(trajectory_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 保存 config
    config_path = method_dir / f"config_seed_{seed}.json"
    config_dict = {
        key: getattr(config, key)
        for key in dir(config)
        if key.isupper() and not key.startswith("__") and not callable(getattr(config, key))
    }

    def _numpy_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    config_path.write_text(json.dumps(config_dict, indent=2, default=_numpy_encoder), encoding="utf-8")

    print(f"Baseline results saved to {method_dir}")
    return str(metrics_path)
