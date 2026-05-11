"""
中文注释说明：run_joint_trajectory_offload_experiment.py

文件作用：
    运行轨迹控制与任务卸载联合实验，用于评估无人机移动决策和卸载策略耦合后的整体性能。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - SUMMARY_METRIC_NAMES: 实验汇总信息，最终写入报告或 manifest 文件。
    - configure_fp32_precision(): 关键函数，承载本模块的一段可复用实验逻辑。
    - configure_compile_backend(): 关键函数，承载本模块的一段可复用实验逻辑。
    - snapshot_config(): 全局配置模块，保存环境参数和训练超参数。
    - restore_config(): 全局配置模块，保存环境参数和训练超参数。
    - _get_episode_runtime_audit(): 训练或测试的回合编号。
    - resolve_latest_training_artifacts(): 关键函数，承载本模块的一段可复用实验逻辑。
    - resolve_offload_checkpoints(): 关键函数，承载本模块的一段可复用实验逻辑。
    - aggregate_metric_dicts(): 关键函数，承载本模块的一段可复用实验逻辑。
    - set_service_offload_policy(): 关键函数，承载本模块的一段可复用实验逻辑。
    - run_single_episode(): 训练或测试的回合编号。
    - evaluate_joint_policy(): 关键函数，承载本模块的一段可复用实验逻辑。
    - build_delta_vs_reference(): 关键函数，承载本模块的一段可复用实验逻辑。
    - parse_args(): 解析命令行参数，并为实验脚本提供可覆盖的默认配置。
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    argparse, copy, json, os, time, warnings, datetime, pathlib, numpy, torch

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from analyze_experiment_statistics import build_statistics, extract_policy_units
from environment.env import Env
from environment.uavs import UAV
from marl_models.base_model import MARLModel
from marl_models.utils import get_model
from paths import results_path
from utils.comparative_plots import compare_algorithms
from utils.logger import Log, Logger, load_configs
from utils.plot_logs import generate_plots


# 关键变量 SUMMARY_METRIC_NAMES：实验汇总信息，最终写入报告或 manifest 文件。
SUMMARY_METRIC_NAMES: tuple[str, ...] = (
    "reward",
    "latency",
    "energy",
    "fairness",
    "offline_rate",
    "deadline_satisfaction_rate",
    "offloading_ratio_local",
    "offloading_ratio_cooperative",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
    "service_learned_decision_count",
    "service_heuristic_decision_count",
    "service_fallback_count",
    "service_predict_exception_fallback_count",
)

POLICY_ALIASES: dict[str, str] = {
    "heuristic": "heuristic",
    "surrogate": "surrogate",
    "oracle_guided": "surrogate",
    "oracle-guided": "surrogate",
    "rich_reduced": "rich_reduced",
    "lower_mappo": "lower_mappo",
    "lower-mappo": "lower_mappo",
}


def _parse_key_value_entries(entries: list[str] | None, option_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must use NAME=VALUE format, got: {entry}")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"{option_name} entries must use non-empty NAME=VALUE pairs, got: {entry}")
        mapping[key] = value
    return mapping


def _normalize_offload_policy(policy_label: str) -> str:
    normalized = POLICY_ALIASES.get(policy_label)
    if normalized is None:
        raise ValueError(f"Unsupported joint-eval policy label: {policy_label}")
    return normalized


# 函数 configure_fp32_precision：关键函数，承载本模块的一段可复用实验逻辑。
def configure_fp32_precision() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Please use the new API settings to control TF32 behavior.*",
        category=UserWarning,
    )
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# 函数 configure_compile_backend：关键函数，承载本模块的一段可复用实验逻辑。
def configure_compile_backend() -> None:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if os.name == "nt":
        # 函数 _no_compile：关键函数，承载本模块的一段可复用实验逻辑，主要参数：module。
        def _no_compile(module, *args, **kwargs):
            # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
            return module

        torch.compile = _no_compile  # type: ignore[attr-defined]


# 函数 snapshot_config：全局配置模块，保存环境参数和训练超参数。
def snapshot_config() -> dict[str, object]:
    snapshot: dict[str, object] = {}
    # 循环处理：遍历 key 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for key in dir(config):
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if key.isupper() and not key.startswith("__"):
            value = getattr(config, key)
            snapshot[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return snapshot


# 函数 restore_config：全局配置模块，保存环境参数和训练超参数，主要参数：snapshot。
def restore_config(snapshot: dict[str, object]) -> None:
    # 循环处理：遍历 (key, value) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for key, value in snapshot.items():
        setattr(config, key, value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))


# 函数 _get_episode_runtime_audit：训练或测试的回合编号，主要参数：env。
def _get_episode_runtime_audit(env: Env) -> dict[str, object]:
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


# 函数 resolve_latest_training_artifacts：关键函数，承载本模块的一段可复用实验逻辑，主要参数：trajectory_run_root, model_name。
def resolve_latest_training_artifacts(trajectory_run_root: str | Path, model_name: str) -> tuple[Path | None, Path | None]:
    run_root = Path(trajectory_run_root)
    config_candidates: list[Path] = []
    primary_config_dir = run_root / "train_logs" / model_name
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if primary_config_dir.exists():
        config_candidates.extend(primary_config_dir.glob("config_*.json"))

    fallback_config_dir = Path("train_logs") / model_name
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if fallback_config_dir.exists():
        config_candidates.extend(fallback_config_dir.glob("config_*.json"))

    config_candidates = sorted(
        {path.resolve(): path for path in config_candidates}.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    model_candidates: list[Path] = []
    primary_model_root = run_root / "saved_models"
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if primary_model_root.exists():
        model_candidates.extend([path for path in primary_model_root.glob(f"{model_name}_*") if (path / "final").exists()])

    fallback_model_root = Path("saved_models")
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if fallback_model_root.exists():
        model_candidates.extend([path for path in fallback_model_root.glob(f"{model_name}_*") if (path / "final").exists()])

    model_candidates = sorted(
        {path.resolve(): path for path in model_candidates}.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not model_candidates:
        if model_name.endswith("_greedy") or model_name in {"random", "static", "nearest_greedy", "uncoordinated_greedy"}:
            resolved_config = config_candidates[0] if config_candidates else None
            return resolved_config, None
        raise FileNotFoundError(
            f"No saved model run with a 'final' directory found under either '{primary_model_root}' "
            f"or '{fallback_model_root}' for model '{model_name}'."
        )

    resolved_config = config_candidates[0] if config_candidates else None
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return resolved_config, model_candidates[0] / "final"


# 函数 resolve_offload_checkpoints：关键函数，承载本模块的一段可复用实验逻辑，主要参数：offload_experiment_root。
def resolve_offload_checkpoints(offload_experiment_root: str | Path) -> tuple[Path, Path]:
    root = Path(offload_experiment_root)
    surrogate_checkpoint = root / "checkpoints" / "offload_policy_surrogate_runtime.pt"
    rich_checkpoint = root / "checkpoints" / "offload_policy_rich_runtime.pt"
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not surrogate_checkpoint.exists():
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise FileNotFoundError(f"Surrogate checkpoint not found: {surrogate_checkpoint}")
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not rich_checkpoint.exists():
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise FileNotFoundError(f"Rich reduced checkpoint not found: {rich_checkpoint}")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return surrogate_checkpoint, rich_checkpoint


# 函数 aggregate_metric_dicts：关键函数，承载本模块的一段可复用实验逻辑，主要参数：metric_dicts。
def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not metric_dicts:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {metric_name: {"mean": 0.0, "std": 0.0} for metric_name in SUMMARY_METRIC_NAMES}
    arrays = {
        metric_name: np.asarray([entry[metric_name] for entry in metric_dicts], dtype=np.float64)
        for metric_name in SUMMARY_METRIC_NAMES
    }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        metric_name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for metric_name, values in arrays.items()
    }


# 函数 set_service_offload_policy：关键函数，承载本模块的一段可复用实验逻辑，主要参数：policy_label, checkpoint_path。
def set_service_offload_policy(policy_label: str, checkpoint_path: str | None) -> None:
    normalized_policy = _normalize_offload_policy(policy_label)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if normalized_policy == "heuristic":
        config.SERVICE_OFFLOAD_POLICY = "heuristic"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif normalized_policy in {"surrogate", "rich_reduced"}:
        config.SERVICE_OFFLOAD_POLICY = "learned"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = checkpoint_path
    elif normalized_policy == "lower_mappo":
        config.SERVICE_OFFLOAD_POLICY = "heuristic"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None
    else:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported joint-eval policy label: {policy_label}")
    UAV._policy_cache.clear()


def capture_spatial_trace_step(env: Env, step: int, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "step": int(step),
        "uav_positions": [[float(v) for v in uav.pos[:2]] for uav in env.uavs],
        "ue_positions": [[float(v) for v in ue.pos[:2]] for ue in env.ues],
        "metrics": {
            "deadline_satisfaction_rate": float(metrics.get("deadline_satisfaction_rate", 0.0)),
            "offloading_ratio_local": float(metrics.get("offloading_ratio_local", 0.0)),
            "offloading_ratio_cooperative": float(metrics.get("offloading_ratio_cooperative", 0.0)),
            "offloading_ratio_mbs": float(metrics.get("offloading_ratio_mbs", 0.0)),
            "mbs_load_ratio": float(metrics.get("mbs_load_ratio", 0.0)),
            "service_requests_generated": float(metrics.get("service_requests_generated", 0.0)),
            "service_requests_processed": float(metrics.get("service_requests_processed", 0.0)),
            "service_fallback_count": float(metrics.get("service_fallback_count", 0.0)),
        },
    }


# 函数 run_single_episode：训练或测试的回合编号，主要参数：env, model。
def run_single_episode(
    env: Env,
    model: MARLModel,
    *,
    offload_model: MARLModel | None = None,
    record_spatial_trace: bool = False,
    spatial_trace_interval: int = 20,
) -> tuple[dict[str, float], dict[str, object], list[dict[str, object]]]:
    obs = env.reset()
    model.reset()
    trace_steps: list[dict[str, object]] = []

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

    # 循环处理：遍历 step 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for step in range(1, config.STEPS_PER_EPISODE + 1):
        obs_arr = np.asarray(obs, dtype=np.float32)
        actions = model.select_actions(obs_arr, exploration=False)
        offloading_actions = None
        if offload_model is not None:
            offload_obs, offload_masks = env.get_offloading_obs_and_masks()
            offloading_actions, _, _ = offload_model.get_action_and_value(
                offload_obs,
                masks=offload_masks,
                exploration=False,
            )
        next_obs, rewards, metrics = env.step(actions, offloading_actions=offloading_actions)
        obs = next_obs
        if record_spatial_trace and (step == 1 or step % max(int(spatial_trace_interval), 1) == 0):
            trace_steps.append(capture_spatial_trace_step(env, step, metrics))

        episode_reward += float(np.sum(rewards))
        episode_latency += float(metrics["latency"])
        episode_energy += float(metrics["energy"])
        episode_fairness = float(metrics["fairness"])
        episode_offline_rate = float(metrics["offline_rate"])
        episode_deadline_satisfaction_sum += float(metrics["deadline_satisfaction_rate"])
        episode_offload_local_sum += float(metrics["offloading_ratio_local"])
        episode_offload_cooperative_sum += float(metrics["offloading_ratio_cooperative"])
        episode_offload_mbs_sum += float(metrics["offloading_ratio_mbs"])
        episode_mbs_load_sum += float(metrics["mbs_load_ratio"])

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if step >= config.STEPS_PER_EPISODE:
            break

    episode_length = max(float(config.STEPS_PER_EPISODE), 1.0)
    runtime_audit = _get_episode_runtime_audit(env)
    episode_metrics = {
        "reward": float(episode_reward),
        "latency": float(episode_latency),
        "energy": float(episode_energy),
        "fairness": float(episode_fairness),
        "offline_rate": float(episode_offline_rate),
        "deadline_satisfaction_rate": float(episode_deadline_satisfaction_sum / episode_length),
        "offloading_ratio_local": float(episode_offload_local_sum / episode_length),
        "offloading_ratio_cooperative": float(episode_offload_cooperative_sum / episode_length),
        "offloading_ratio_mbs": float(episode_offload_mbs_sum / episode_length),
        "mbs_load_ratio": float(episode_mbs_load_sum / episode_length),
        "service_learned_decision_count": float(runtime_audit["service_learned_decision_count"]),
        "service_heuristic_decision_count": float(runtime_audit["service_heuristic_decision_count"]),
        "service_fallback_count": float(runtime_audit["service_fallback_count"]),
        "service_predict_exception_fallback_count": float(runtime_audit["service_predict_exception_fallback_count"]),
    }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return episode_metrics, runtime_audit, trace_steps


# 函数 evaluate_joint_policy：关键函数，承载本模块的一段可复用实验逻辑。
def evaluate_joint_policy(
    *,
    combo_label: str,
    policy_label: str,
    checkpoint_path: str | None,
    trajectory_model_name: str,
    trajectory_model_dir: str | Path | None,
    trajectory_config_path: str | Path | None,
    lower_model_dir: str | Path | None,
    lower_model_name: str,
    seeds: list[int],
    episodes_per_seed: int,
    steps_per_episode: int | None,
    run_root: Path,
    record_spatial_trace: bool = False,
    spatial_trace_interval: int = 20,
) -> dict[str, object]:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if trajectory_config_path is not None:
        load_configs(str(trajectory_config_path))
    config.MODEL = trajectory_model_name
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if steps_per_episode is not None:
        config.STEPS_PER_EPISODE = int(steps_per_episode)

    set_service_offload_policy(policy_label, checkpoint_path)
    normalized_offload_policy = _normalize_offload_policy(policy_label)

    safe_combo_label = combo_label.replace("/", "__").replace(" ", "_")
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{safe_combo_label}")
    log_dir = run_root / "test_logs" / safe_combo_label / timestamp
    plot_dir = run_root / "test_plots" / safe_combo_label / timestamp
    logger = Logger(str(log_dir), timestamp)
    logger.log_configs()
    episode_log = Log()

    start_time = time.time()
    per_seed_runs: list[dict[str, object]] = []
    aggregate_units: list[dict[str, float]] = []
    spatial_trace_records: list[dict[str, object]] = []
    global_episode_idx = 0

    # 循环处理：遍历 seed 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = Env()
        model = get_model(trajectory_model_name)
        if trajectory_model_dir is not None:
            model.load(str(trajectory_model_dir))
        offload_model = None
        if normalized_offload_policy == "lower_mappo":
            if lower_model_dir is None:
                raise ValueError(f"lower_model_dir is required for combo '{combo_label}'.")
            offload_model = get_model(lower_model_name)
            offload_model.load(str(lower_model_dir))

        episode_metrics_for_seed: list[dict[str, float]] = []
        # 循环处理：遍历 episode_idx 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for episode_idx in range(episodes_per_seed):
            run_seed = int(seed + episode_idx * 1000)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

            episode_metrics, runtime_audit, trace_steps = run_single_episode(
                env,
                model,
                offload_model=offload_model,
                record_spatial_trace=record_spatial_trace,
                spatial_trace_interval=spatial_trace_interval,
            )
            episode_metrics_for_seed.append(episode_metrics)
            aggregate_units.append(episode_metrics)
            if record_spatial_trace:
                spatial_trace_records.append(
                    {
                        "combo": combo_label,
                        "trajectory_model": trajectory_model_name,
                        "offload_policy": policy_label,
                        "seed": int(seed),
                        "episode": int(episode_idx),
                        "steps": trace_steps,
                    }
                )

            episode_log.append(
                episode_metrics["reward"],
                episode_metrics["latency"],
                episode_metrics["energy"],
                episode_metrics["fairness"],
                episode_metrics["offline_rate"],
                deadline_satisfaction_rate=episode_metrics["deadline_satisfaction_rate"],
                offloading_ratio_local=episode_metrics["offloading_ratio_local"],
                offloading_ratio_cooperative=episode_metrics["offloading_ratio_cooperative"],
                offloading_ratio_mbs=episode_metrics["offloading_ratio_mbs"],
                mbs_load_ratio=episode_metrics["mbs_load_ratio"],
                service_learned_decision_count=episode_metrics["service_learned_decision_count"],
                service_heuristic_decision_count=episode_metrics["service_heuristic_decision_count"],
                service_fallback_count=episode_metrics["service_fallback_count"],
                service_predict_exception_fallback_count=episode_metrics["service_predict_exception_fallback_count"],
                service_offload_policy_requested=str(runtime_audit["service_offload_policy_requested"]),
                service_offload_policy_loaded=bool(runtime_audit["service_offload_policy_loaded"]) or offload_model is not None,
                service_offload_policy_checkpoint_path=runtime_audit["service_offload_policy_checkpoint_path"],
                service_offload_policy_feature_family=runtime_audit["service_offload_policy_feature_family"] or normalized_offload_policy,
            )
            global_episode_idx += 1
            logger.log_metrics(global_episode_idx, episode_log, 1, time.time() - start_time)

        per_seed_mean = {
            metric_name: float(np.mean([entry[metric_name] for entry in episode_metrics_for_seed]))
            for metric_name in SUMMARY_METRIC_NAMES
        }
        per_seed_runs.append(
            {
                "policy": policy_label,
                "seed": int(seed),
                "episodes": episode_metrics_for_seed,
                "per_seed_mean": per_seed_mean,
            }
        )

    generate_plots(str(logger.json_file_path), str(plot_dir), "joint_test", timestamp, smoothing_window=2)
    spatial_trace_path: str | None = None
    if record_spatial_trace:
        trace_dir = run_root / "spatial_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{safe_combo_label}_spatial_trace.json"
        trace_payload = {
            "combo": combo_label,
            "trajectory_model": trajectory_model_name,
            "offload_policy": policy_label,
            "spatial_trace_interval": int(spatial_trace_interval),
            "records": spatial_trace_records,
        }
        trace_path.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        spatial_trace_path = str(trace_path)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "log_dir": str(log_dir),
        "plot_dir": str(plot_dir),
        "log_json_path": str(logger.json_file_path),
        "config_path": str(logger.config_file_path),
        "trajectory_config_source_path": str(trajectory_config_path) if trajectory_config_path is not None else None,
        "trajectory_model": trajectory_model_name,
        "trajectory_model_dir": str(trajectory_model_dir) if trajectory_model_dir is not None else None,
        "offload_policy": policy_label,
        "lower_model_dir": str(lower_model_dir) if lower_model_dir is not None else None,
        "lower_model_name": lower_model_name if lower_model_dir is not None else None,
        "spatial_trace_path": spatial_trace_path,
        "aggregate": aggregate_metric_dicts(aggregate_units),
        "per_seed": per_seed_runs,
    }


# 函数 build_delta_vs_reference：关键函数，承载本模块的一段可复用实验逻辑。
def build_delta_vs_reference(
    *,
    results_by_policy: dict[str, dict[str, object]],
    reference_policy: str,
) -> dict[str, dict[str, dict[str, float]]]:
    reference_units = [entry["per_seed_mean"] for entry in results_by_policy[reference_policy]["per_seed"]]
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    # 循环处理：遍历 (policy_label, details) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for policy_label, details in results_by_policy.items():
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if policy_label == reference_policy:
            continue
        policy_units = [entry["per_seed_mean"] for entry in details["per_seed"]]
        deltas[policy_label] = {
            metric_name: {
                "mean_delta": float(
                    np.mean(
                        [float(policy_units[idx][metric_name] - reference_units[idx][metric_name]) for idx in range(len(policy_units))]
                    )
                ),
                "std_delta": float(
                    np.std(
                        [float(policy_units[idx][metric_name] - reference_units[idx][metric_name]) for idx in range(len(policy_units))]
                    )
                ),
            }
            for metric_name in SUMMARY_METRIC_NAMES
        }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return deltas


def build_combo_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.hmarl_main_table:
        return [
            {"label": "uncoordinated_greedy__heuristic", "trajectory_model": "uncoordinated_greedy", "offload_policy": "heuristic"},
            {"label": "attention_mappo__heuristic", "trajectory_model": "attention_mappo", "offload_policy": "heuristic"},
            {"label": "attention_mappo__oracle_guided", "trajectory_model": "attention_mappo", "offload_policy": "oracle_guided"},
            {"label": "uncoordinated_greedy__lower_mappo", "trajectory_model": "uncoordinated_greedy", "offload_policy": "lower_mappo"},
            {"label": "attention_mappo__lower_mappo", "trajectory_model": "attention_mappo", "offload_policy": "lower_mappo"},
            {"label": "full_hierarchical_marl", "trajectory_model": "attention_mappo", "offload_policy": "lower_mappo"},
        ]

    if args.four_way_ablation:
        return [
            {"label": "uncoordinated_greedy__heuristic", "trajectory_model": "uncoordinated_greedy", "offload_policy": "heuristic"},
            {"label": "attention_mappo__heuristic", "trajectory_model": "attention_mappo", "offload_policy": "heuristic"},
            {"label": "uncoordinated_greedy__oracle_guided", "trajectory_model": "uncoordinated_greedy", "offload_policy": "oracle_guided"},
            {"label": "attention_mappo__oracle_guided", "trajectory_model": "attention_mappo", "offload_policy": "oracle_guided"},
        ]

    if args.combos:
        combos: list[dict[str, str]] = []
        for entry in args.combos:
            parts = [part.strip() for part in entry.split(":")]
            if len(parts) != 3 or not all(parts):
                raise ValueError("--combos entries must use LABEL:TRAJECTORY_MODEL:OFFLOAD_POLICY format.")
            label, trajectory_model, offload_policy = parts
            combos.append({"label": label, "trajectory_model": trajectory_model, "offload_policy": offload_policy})
        return combos

    return [
        {
            "label": f"{args.trajectory_model}__{policy_label}",
            "trajectory_model": args.trajectory_model,
            "offload_policy": policy_label,
        }
        for policy_label in args.policies
    ]


def resolve_trajectory_artifacts_for_combos(args: argparse.Namespace, combo_specs: list[dict[str, str]]) -> dict[str, dict[str, str | None]]:
    run_roots = _parse_key_value_entries(args.trajectory_run_roots, "--trajectory_run_roots")
    config_paths = _parse_key_value_entries(args.trajectory_configs, "--trajectory_configs")
    model_dirs = _parse_key_value_entries(args.trajectory_model_dirs, "--trajectory_model_dirs")
    artifacts: dict[str, dict[str, str | None]] = {}

    for combo in combo_specs:
        label = combo["label"]
        model_name = combo["trajectory_model"]
        explicit_config = config_paths.get(label) or config_paths.get(model_name)
        explicit_model_dir = model_dirs.get(label) or model_dirs.get(model_name)
        run_root = run_roots.get(label) or run_roots.get(model_name) or args.trajectory_run_root
        if explicit_config is not None or explicit_model_dir is not None:
            artifacts[label] = {
                "config": str(Path(explicit_config)) if explicit_config is not None else None,
                "model_dir": str(Path(explicit_model_dir)) if explicit_model_dir is not None else None,
                "run_root": str(Path(run_root).resolve()) if run_root is not None else None,
            }
            continue

        resolved_config, resolved_model_dir = resolve_latest_training_artifacts(run_root, model_name)
        artifacts[label] = {
            "config": str(resolved_config) if resolved_config is not None else None,
            "model_dir": str(resolved_model_dir) if resolved_model_dir is not None else None,
            "run_root": str(Path(run_root).resolve()),
        }
    return artifacts


def resolve_lower_model_dirs(args: argparse.Namespace, combo_specs: list[dict[str, str]]) -> dict[str, str | None]:
    explicit_dirs = _parse_key_value_entries(args.lower_model_dirs, "--lower_model_dirs")
    lower_dirs: dict[str, str | None] = {}
    for combo in combo_specs:
        label = combo["label"]
        policy = _normalize_offload_policy(combo["offload_policy"])
        if policy != "lower_mappo":
            lower_dirs[label] = None
            continue
        candidate = explicit_dirs.get(label) or explicit_dirs.get(combo["offload_policy"]) or args.lower_model_dir
        if candidate is None:
            raise ValueError(
                f"Combo '{label}' uses lower_mappo but no lower model dir was provided. "
                "Use --lower_model_dir or --lower_model_dirs LABEL=PATH."
            )
        path = Path(candidate)
        if not (path / "offload_mappo.pth").exists():
            raise FileNotFoundError(f"Lower MAPPO checkpoint not found: {path / 'offload_mappo.pth'}")
        lower_dirs[label] = str(path)
    return lower_dirs


def resolve_lower_model_names(args: argparse.Namespace, combo_specs: list[dict[str, str]]) -> dict[str, str]:
    explicit_names = _parse_key_value_entries(args.lower_model_names, "--lower_model_names")
    lower_names: dict[str, str] = {}
    for combo in combo_specs:
        label = combo["label"]
        default_name = str(args.lower_model_name)
        lower_names[label] = explicit_names.get(label) or explicit_names.get(combo["offload_policy"]) or default_name
    return lower_names


# 函数 parse_args：解析命令行参数，并为实验脚本提供可覆盖的默认配置。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run joint trajectory-control + offloading-policy comparison experiments.")
    parser.add_argument("--name", type=str, default=time.strftime("joint_exp_%Y%m%d_%H%M%S"), help="Joint experiment name.")
    parser.add_argument("--trajectory_run_root", type=str, required=True, help="Run root containing the trained trajectory model artifacts.")
    parser.add_argument("--trajectory_model", type=str, default="attention_mappo", help="Trajectory model name to load.")
    parser.add_argument("--trajectory_config", type=str, default=None, help="Optional explicit config_*.json path.")
    parser.add_argument("--trajectory_model_dir", type=str, default=None, help="Optional explicit saved model final directory.")
    parser.add_argument(
        "--trajectory_run_roots",
        nargs="*",
        default=None,
        help="Optional NAME=PATH overrides for multiple trajectory models in combo mode.",
    )
    parser.add_argument(
        "--trajectory_configs",
        nargs="*",
        default=None,
        help="Optional NAME=PATH config overrides for multiple trajectory models in combo mode.",
    )
    parser.add_argument(
        "--trajectory_model_dirs",
        nargs="*",
        default=None,
        help="Optional NAME=PATH saved-model overrides for multiple trajectory models in combo mode.",
    )
    parser.add_argument("--offload_experiment_root", type=str, default=None, help="Offload experiment root containing classifier checkpoints/.")
    parser.add_argument("--surrogate_checkpoint", type=str, default=None, help="Optional explicit surrogate checkpoint path.")
    parser.add_argument("--rich_checkpoint", type=str, default=None, help="Optional explicit rich reduced checkpoint path.")
    parser.add_argument("--lower_model_dir", type=str, default=None, help="Saved lower-MAPPO final directory for lower_mappo policies.")
    parser.add_argument("--lower_model_name", type=str, default="constrained_attention_offload_mappo", help="Model name used to load lower-MAPPO checkpoints.")
    parser.add_argument(
        "--lower_model_dirs",
        nargs="*",
        default=None,
        help="Optional LABEL=PATH lower-MAPPO final-directory overrides for combo mode.",
    )
    parser.add_argument(
        "--lower_model_names",
        nargs="*",
        default=None,
        help="Optional LABEL=MODEL lower-MAPPO model-name overrides for combo mode.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["heuristic", "surrogate", "rich_reduced"],
        choices=["heuristic", "surrogate", "oracle_guided", "oracle-guided", "rich_reduced", "lower_mappo", "lower-mappo"],
        help="Offloading policies to compare.",
    )
    parser.add_argument(
        "--combos",
        nargs="+",
        default=None,
        help="Explicit LABEL:TRAJECTORY_MODEL:OFFLOAD_POLICY combinations for orthogonal joint evaluation.",
    )
    parser.add_argument(
        "--four_way_ablation",
        action="store_true",
        help="Evaluate uncoordinated/heuristic, attention_mappo/heuristic, uncoordinated/oracle_guided, and attention_mappo/oracle_guided.",
    )
    parser.add_argument(
        "--hmarl_main_table",
        action="store_true",
        help="Evaluate the six-combo thesis table, including lower-MAPPO and full hierarchical MARL.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Evaluation seeds.")
    parser.add_argument("--episodes_per_seed", type=int, default=8, help="Episodes per seed and policy.")
    parser.add_argument("--steps_per_episode", type=int, default=None, help="Optional override for STEPS_PER_EPISODE.")
    parser.add_argument("--comparison_smoothing", type=int, default=3, help="Smoothing window for cross-policy comparison plots.")
    parser.add_argument("--record_spatial_trace", action="store_true", help="Record sampled UAV/UE positions and per-step offload ratios.")
    parser.add_argument("--spatial_trace_interval", type=int, default=20, help="Step interval used when recording spatial trace.")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return parser.parse_args()


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main() -> None:
    args = parse_args()
    configure_fp32_precision()
    configure_compile_backend()

    base_snapshot = snapshot_config()
    run_root = results_path("joint_experiments", args.name)
    run_root.mkdir(parents=True, exist_ok=True)
    combo_specs = build_combo_specs(args)

    if args.trajectory_config is not None:
        args.trajectory_configs = list(args.trajectory_configs or []) + [f"{args.trajectory_model}={args.trajectory_config}"]
    if args.trajectory_model_dir is not None:
        args.trajectory_model_dirs = list(args.trajectory_model_dirs or []) + [f"{args.trajectory_model}={args.trajectory_model_dir}"]
    trajectory_artifacts = resolve_trajectory_artifacts_for_combos(args, combo_specs)
    lower_model_dirs = resolve_lower_model_dirs(args, combo_specs)
    lower_model_names = resolve_lower_model_names(args, combo_specs)
    for combo_label, artifact in trajectory_artifacts.items():
        if artifact["config"] is None:
            print(f"[joint] warning: no config_*.json was found for combo '{combo_label}'; using current config.py values.")

    needs_classifier_checkpoints = any(_normalize_offload_policy(combo["offload_policy"]) in {"surrogate", "rich_reduced"} for combo in combo_specs)
    if needs_classifier_checkpoints and args.offload_experiment_root is None and (args.surrogate_checkpoint is None or args.rich_checkpoint is None):
        raise ValueError("--offload_experiment_root is required when evaluating surrogate/oracle_guided/rich_reduced policies without explicit checkpoints.")
    if needs_classifier_checkpoints and (args.surrogate_checkpoint is None or args.rich_checkpoint is None):
        inferred_surrogate_checkpoint, inferred_rich_checkpoint = resolve_offload_checkpoints(args.offload_experiment_root)
        surrogate_checkpoint = Path(args.surrogate_checkpoint) if args.surrogate_checkpoint is not None else inferred_surrogate_checkpoint
        rich_checkpoint = Path(args.rich_checkpoint) if args.rich_checkpoint is not None else inferred_rich_checkpoint
    else:
        surrogate_checkpoint = Path(args.surrogate_checkpoint) if args.surrogate_checkpoint is not None else None
        rich_checkpoint = Path(args.rich_checkpoint) if args.rich_checkpoint is not None else None

    policy_to_checkpoint = {
        "heuristic": None,
        "surrogate": str(surrogate_checkpoint) if surrogate_checkpoint is not None else None,
        "oracle_guided": str(surrogate_checkpoint) if surrogate_checkpoint is not None else None,
        "oracle-guided": str(surrogate_checkpoint) if surrogate_checkpoint is not None else None,
        "rich_reduced": str(rich_checkpoint) if rich_checkpoint is not None else None,
        "lower_mappo": None,
    }

    results_by_combo: dict[str, dict[str, object]] = {}
    # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
    try:
        for combo in combo_specs:
            combo_label = combo["label"]
            trajectory_model = combo["trajectory_model"]
            policy_label = combo["offload_policy"]
            checkpoint_key = _normalize_offload_policy(policy_label)
            artifact = trajectory_artifacts[combo_label]
            print(f"[joint] evaluating combo={combo_label} trajectory={trajectory_model} offload={policy_label}")
            restore_config(base_snapshot)
            results_by_combo[combo_label] = evaluate_joint_policy(
                combo_label=combo_label,
                policy_label=policy_label,
                checkpoint_path=policy_to_checkpoint[checkpoint_key],
                trajectory_model_name=trajectory_model,
                trajectory_model_dir=artifact["model_dir"],
                trajectory_config_path=artifact["config"],
                lower_model_dir=lower_model_dirs[combo_label],
                lower_model_name=lower_model_names[combo_label],
                seeds=[int(seed) for seed in args.seeds],
                episodes_per_seed=args.episodes_per_seed,
                steps_per_episode=args.steps_per_episode,
                run_root=run_root,
                record_spatial_trace=bool(args.record_spatial_trace),
                spatial_trace_interval=int(args.spatial_trace_interval),
            )

        combo_labels = [combo["label"] for combo in combo_specs]
        log_dirs = [results_by_combo[combo_label]["log_dir"] for combo_label in combo_labels]
        comparison_dir = run_root / "comparisons"
        compare_algorithms(log_dirs, combo_labels, str(comparison_dir), smoothing_window=args.comparison_smoothing)

        summary = {
            "metadata": {
                "experiment_name": args.name,
                "trajectory_run_root": str(Path(args.trajectory_run_root).resolve()),
                "trajectory_model": args.trajectory_model,
                "trajectory_artifacts": trajectory_artifacts,
                "lower_model_dirs": lower_model_dirs,
                "lower_model_names": lower_model_names,
                "offload_experiment_root": str(Path(args.offload_experiment_root).resolve()) if args.offload_experiment_root is not None else None,
                "surrogate_checkpoint": str(surrogate_checkpoint.resolve()) if surrogate_checkpoint is not None else None,
                "rich_checkpoint": str(rich_checkpoint.resolve()) if rich_checkpoint is not None else None,
                "policies": args.policies,
                "combos": combo_specs,
                "seeds": [int(seed) for seed in args.seeds],
                "episodes_per_seed": int(args.episodes_per_seed),
                "steps_per_episode": int(args.steps_per_episode) if args.steps_per_episode is not None else None,
                "comparison_dir": str(comparison_dir),
                "record_spatial_trace": bool(args.record_spatial_trace),
                "spatial_trace_interval": int(args.spatial_trace_interval),
            },
            "per_policy": results_by_combo,
            "per_combo": results_by_combo,
            "delta_vs_reference": build_delta_vs_reference(results_by_policy=results_by_combo, reference_policy=combo_labels[0])
            if combo_labels
            else {},
            "delta_vs_heuristic": build_delta_vs_reference(results_by_policy=results_by_combo, reference_policy="heuristic")
            if "heuristic" in results_by_combo
            else {},
        }
        try:
            summary["statistics"] = build_statistics(
                extract_policy_units(summary),
                reference=combo_labels[0],
                confidence=0.95,
                bootstrap_samples=3000,
            )
        except Exception as exc:
            summary["statistics_error"] = str(exc)
        summary_path = run_root / "joint_experiment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        restore_config(base_snapshot)
        UAV._policy_cache.clear()


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
