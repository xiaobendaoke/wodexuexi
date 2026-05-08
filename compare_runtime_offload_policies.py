"""
中文注释说明：compare_runtime_offload_policies.py

文件作用：
    比较不同运行时任务卸载策略在多回合仿真中的性能差异。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - METRIC_NAMES: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - RuntimeScenario: 核心类，封装本模块中的主要状态和行为。
    - snapshot_config(): 全局配置模块，保存环境参数和训练超参数。
    - restore_config(): 全局配置模块，保存环境参数和训练超参数。
    - _scaled_int_array(): 关键函数，承载本模块的一段可复用实验逻辑。
    - build_runtime_scenarios(): 关键函数，承载本模块的一段可复用实验逻辑。
    - apply_runtime_scenario(): 关键函数，承载本模块的一段可复用实验逻辑。
    - set_runtime_policy(): 关键函数，承载本模块的一段可复用实验逻辑。
    - aggregate_metric_dicts(): 关键函数，承载本模块的一段可复用实验逻辑。
    - run_policy_for_seed(): 关键函数，承载本模块的一段可复用实验逻辑。
    - compare_runtime_offload_policies(): 关键函数，承载本模块的一段可复用实验逻辑。
    - parse_args(): 解析命令行参数，并为实验脚本提供可覆盖的默认配置。
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    argparse, copy, json, dataclasses, pathlib, numpy, config, paths, environment, marl_models

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from paths import results_path
from environment.env import Env
from environment.uavs import UAV
from marl_models.static_baseline.static_model import StaticModel


# 关键变量 METRIC_NAMES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
METRIC_NAMES: tuple[str, ...] = (
    "latency",
    "energy",
    "deadline_satisfaction_rate",
    "offloading_ratio_local",
    "offloading_ratio_cooperative",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
)


# 类 RuntimeScenario：核心类，封装本模块中的主要状态和行为。
@dataclass(frozen=True)
class RuntimeScenario:
    name: str
    description: str


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


# 函数 _scaled_int_array：关键函数，承载本模块的一段可复用实验逻辑，主要参数：base_values, scale, minimum。
def _scaled_int_array(base_values: np.ndarray, scale: float, minimum: int) -> np.ndarray:
    scaled = np.maximum(np.round(base_values.astype(np.float64) * scale).astype(np.int64), minimum)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return scaled.astype(np.int64)


# 函数 build_runtime_scenarios：关键函数，承载本模块的一段可复用实验逻辑。
def build_runtime_scenarios() -> list[RuntimeScenario]:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return [
        RuntimeScenario(
            name="default_uniform",
            description="Repository default configuration with uniformly distributed UEs.",
        ),
        RuntimeScenario(
            name="hotspot_deadline_stress",
            description="Hotspot-heavy traffic with tighter deadlines, heavier compute, and weaker backhaul.",
        ),
        RuntimeScenario(
            name="local_cache_friendly",
            description="Backhaul-constrained regime where warm UAV caches and stronger local compute can beat MBS offloading.",
        ),
        RuntimeScenario(
            name="cooperative_friendly",
            description="Backhaul-limited cooperative regime with heterogeneous UAV compute and denser neighbor availability.",
        ),
        RuntimeScenario(
            name="mbs_heavy_jobs",
            description="Large service jobs and strong backhaul that make MBS routing more attractive.",
        ),
    ]


# 函数 apply_runtime_scenario：关键函数，承载本模块的一段可复用实验逻辑，主要参数：scenario, base_snapshot。
def apply_runtime_scenario(scenario: RuntimeScenario, base_snapshot: dict[str, object]) -> None:
    restore_config(base_snapshot)

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if scenario.name == "default_uniform":
        config.USE_HOTSPOTS = False
        config.BANDWIDTH_BACKHAUL = 900_000
        config.UAV_SENSING_RANGE = 420.0
        config.UAV_STORAGE_CAPACITY = np.full(config.NUM_UAVS, 140 * 10**6, dtype=np.int64)
        config.UAV_COMPUTING_CAPACITY = np.full(config.NUM_UAVS, 55 * 10**9, dtype=np.int64)
        config.SERVICE_DEADLINE_MIN = 0.65 * config.TIME_SLOT_DURATION
        config.SERVICE_DEADLINE_MAX = 2.00 * config.TIME_SLOT_DURATION
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif scenario.name == "hotspot_deadline_stress":
        config.USE_HOTSPOTS = True
        config.HOTSPOT_UE_PROB = 0.9
        config.SERVICE_DEADLINE_MIN = 0.50 * config.TIME_SLOT_DURATION
        config.SERVICE_DEADLINE_MAX = 1.55 * config.TIME_SLOT_DURATION
        config.BANDWIDTH_BACKHAUL = 350_000
        config.UAV_SENSING_RANGE = 460.0
        config.UAV_STORAGE_CAPACITY = np.full(config.NUM_UAVS, 160 * 10**6, dtype=np.int64)
        config.UAV_COMPUTING_CAPACITY = np.full(config.NUM_UAVS, 70 * 10**9, dtype=np.int64)
        config.CPU_CYCLES_PER_BYTE = _scaled_int_array(np.asarray(base_snapshot["CPU_CYCLES_PER_BYTE"]), scale=1.15, minimum=200)
        config.FILE_SIZES = _scaled_int_array(np.asarray(base_snapshot["FILE_SIZES"]), scale=1.10, minimum=1)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif scenario.name == "local_cache_friendly":
        config.USE_HOTSPOTS = True
        config.HOTSPOT_UE_PROB = 0.85
        config.SERVICE_DEADLINE_MIN = 0.75 * config.TIME_SLOT_DURATION
        config.SERVICE_DEADLINE_MAX = 2.35 * config.TIME_SLOT_DURATION
        config.BANDWIDTH_BACKHAUL = 120_000
        config.UAV_SENSING_RANGE = 420.0
        config.UAV_STORAGE_CAPACITY = np.full(config.NUM_UAVS, 220 * 10**6, dtype=np.int64)
        config.UAV_COMPUTING_CAPACITY = np.full(config.NUM_UAVS, 95 * 10**9, dtype=np.int64)
        config.CPU_CYCLES_PER_BYTE = _scaled_int_array(np.asarray(base_snapshot["CPU_CYCLES_PER_BYTE"]), scale=0.80, minimum=200)
        config.FILE_SIZES = _scaled_int_array(np.asarray(base_snapshot["FILE_SIZES"]), scale=0.90, minimum=1)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif scenario.name == "cooperative_friendly":
        config.USE_HOTSPOTS = True
        config.HOTSPOT_UE_PROB = 0.85
        config.SERVICE_DEADLINE_MIN = 0.70 * config.TIME_SLOT_DURATION
        config.SERVICE_DEADLINE_MAX = 2.20 * config.TIME_SLOT_DURATION
        config.BANDWIDTH_BACKHAUL = 150_000
        config.BANDWIDTH_INTER = 60 * 10**6
        config.UAV_SENSING_RANGE = 520.0
        config.UAV_STORAGE_CAPACITY = np.full(config.NUM_UAVS, 180 * 10**6, dtype=np.int64)
        config.MBS_POS = np.array([900.0, 900.0, 30.0], dtype=np.float32)
        base_compute = np.asarray(base_snapshot["UAV_COMPUTING_CAPACITY"], dtype=np.int64)
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if base_compute.size >= 5:
            cooperative_compute = np.array([30, 180, 160, 120, 100], dtype=np.int64)[: base_compute.size] * 10**9
            config.UAV_COMPUTING_CAPACITY = cooperative_compute.astype(np.int64)
        else:
            boosted = base_compute.copy()
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if boosted.size > 1:
                boosted[1:] = _scaled_int_array(boosted[1:], scale=2.6, minimum=80 * 10**9)
            boosted[0] = int(max(boosted[0] * 0.7, 25 * 10**9))
            config.UAV_COMPUTING_CAPACITY = boosted.astype(np.int64)
        config.CPU_CYCLES_PER_BYTE = _scaled_int_array(np.asarray(base_snapshot["CPU_CYCLES_PER_BYTE"]), scale=0.95, minimum=200)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif scenario.name == "mbs_heavy_jobs":
        config.USE_HOTSPOTS = False
        config.SERVICE_DEADLINE_MIN = 0.45 * config.TIME_SLOT_DURATION
        config.SERVICE_DEADLINE_MAX = 1.25 * config.TIME_SLOT_DURATION
        config.BANDWIDTH_BACKHAUL = 18 * 10**6
        config.MIN_INPUT_SIZE = int(1.5 * 10**6)
        config.MAX_INPUT_SIZE = int(6.5 * 10**6)
        config.CPU_CYCLES_PER_BYTE = _scaled_int_array(np.asarray(base_snapshot["CPU_CYCLES_PER_BYTE"]), scale=1.40, minimum=200)
        config.FILE_SIZES = _scaled_int_array(np.asarray(base_snapshot["FILE_SIZES"]), scale=1.20, minimum=1)
    else:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported runtime scenario: {scenario.name}")

    config.AVG_FILE_SIZE = float(np.mean(config.FILE_SIZES))


# 函数 set_runtime_policy：关键函数，承载本模块的一段可复用实验逻辑，主要参数：policy_mode, checkpoint_path。
def set_runtime_policy(policy_mode: str, checkpoint_path: str | None) -> None:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if policy_mode == "heuristic":
        config.SERVICE_OFFLOAD_POLICY = "heuristic"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif policy_mode == "learned":
        config.SERVICE_OFFLOAD_POLICY = "learned"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = checkpoint_path
    elif policy_mode == "cql":
        config.SERVICE_OFFLOAD_POLICY = "cql"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = checkpoint_path
    elif policy_mode == "radcc":
        config.SERVICE_OFFLOAD_POLICY = "radcc"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = checkpoint_path
    elif policy_mode == "sc_ogo":
        config.SERVICE_OFFLOAD_POLICY = "sc_ogo"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = checkpoint_path
    else:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported runtime policy mode: {policy_mode}")
    UAV._policy_cache.clear()


# 函数 aggregate_metric_dicts：关键函数，承载本模块的一段可复用实验逻辑，主要参数：metric_dicts。
def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not metric_dicts:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {metric_name: {"mean": 0.0, "std": 0.0} for metric_name in METRIC_NAMES}
    arrays = {metric_name: np.asarray([entry[metric_name] for entry in metric_dicts], dtype=np.float64) for metric_name in METRIC_NAMES}
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        metric_name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for metric_name, values in arrays.items()
    }


# 函数 run_policy_for_seed：关键函数，承载本模块的一段可复用实验逻辑。
def run_policy_for_seed(
    *,
    scenario: RuntimeScenario,
    policy_label: str,
    policy_mode: str,
    checkpoint_path: str | None,
    base_snapshot: dict[str, object],
    seed: int,
    episodes_per_seed: int,
    steps_per_episode: int,
) -> dict[str, object]:
    seed_episode_metrics: list[dict[str, float]] = []

    # 循环处理：遍历 episode_idx 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for episode_idx in range(episodes_per_seed):
        apply_runtime_scenario(scenario, base_snapshot)
        set_runtime_policy(policy_mode, checkpoint_path)

        run_seed = int(seed + episode_idx * 1000)
        np.random.seed(run_seed)
        static_model = StaticModel("static", config.NUM_UAVS, config.OBS_DIM_SINGLE, config.ACTION_DIM, "cpu")
        env = Env()
        env.reset(initial_positions=static_model.static_positions)

        episode_totals: dict[str, float] = {metric_name: 0.0 for metric_name in METRIC_NAMES}
        # 循环处理：遍历 _ 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for _ in range(steps_per_episode):
            actions = np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
            _, _, metrics = env.step(actions)
            # 循环处理：遍历 metric_name 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for metric_name in METRIC_NAMES:
                episode_totals[metric_name] += float(metrics[metric_name])

        seed_episode_metrics.append(
            {metric_name: total_value / float(steps_per_episode) for metric_name, total_value in episode_totals.items()}
        )

    per_seed_mean = {
        metric_name: float(np.mean([episode_metrics[metric_name] for episode_metrics in seed_episode_metrics]))
        for metric_name in METRIC_NAMES
    }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "policy": policy_label,
        "seed": seed,
        "scenario": scenario.name,
        "episode_metrics": seed_episode_metrics,
        "per_seed_mean": per_seed_mean,
    }


# 函数 compare_runtime_offload_policies：关键函数，承载本模块的一段可复用实验逻辑。
def compare_runtime_offload_policies(
    *,
    surrogate_checkpoint: str | Path,
    rich_reduced_checkpoint: str | Path,
    cql_checkpoint: str | Path | None = None,
    radcc_checkpoint: str | Path | None = None,
    sc_ogo_checkpoint: str | Path | None = None,
    seeds: list[int],
    episodes_per_seed: int,
    steps_per_episode: int,
    output_path: str | Path,
) -> dict[str, object]:
    base_snapshot = snapshot_config()
    scenarios = build_runtime_scenarios()
    policy_specs: dict[str, tuple[str, str | None]] = {
        "heuristic_offloading": ("heuristic", None),
        "surrogate_baseline": ("learned", str(surrogate_checkpoint)),
        "rich_reduced_runtime_policy": ("learned", str(rich_reduced_checkpoint)),
    }
    if cql_checkpoint is not None:
        policy_specs["cql_dqn_offloading"] = ("cql", str(cql_checkpoint))
    if radcc_checkpoint is not None:
        policy_specs["radcc_offloading"] = ("radcc", str(radcc_checkpoint))
    if sc_ogo_checkpoint is not None:
        policy_specs["sc_ogo_offloading"] = ("sc_ogo", str(sc_ogo_checkpoint))

    scenario_results: dict[str, dict[str, object]] = {}
    overall_seed_summaries: dict[str, list[dict[str, float]]] = {policy_name: [] for policy_name in policy_specs}

    # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
    try:
        # 循环处理：遍历 scenario 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for scenario in scenarios:
            scenario_policy_results: dict[str, object] = {}
            # 循环处理：遍历 (policy_name, checkpoint_path) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for policy_name, (policy_mode, checkpoint_path) in policy_specs.items():
                per_seed_runs: list[dict[str, object]] = []
                per_seed_means: list[dict[str, float]] = []
                # 循环处理：遍历 seed 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
                for seed in seeds:
                    run_result = run_policy_for_seed(
                        scenario=scenario,
                        policy_label=policy_name,
                        policy_mode=policy_mode,
                        checkpoint_path=checkpoint_path,
                        base_snapshot=base_snapshot,
                        seed=seed,
                        episodes_per_seed=episodes_per_seed,
                        steps_per_episode=steps_per_episode,
                    )
                    per_seed_runs.append(run_result)
                    per_seed_means.append(run_result["per_seed_mean"])
                    overall_seed_summaries[policy_name].append(run_result["per_seed_mean"])

                scenario_policy_results[policy_name] = {
                    "description": scenario.description,
                    "per_seed": per_seed_runs,
                    "aggregate": aggregate_metric_dicts(per_seed_means),
                }
            scenario_results[scenario.name] = scenario_policy_results
    finally:
        restore_config(base_snapshot)
        UAV._policy_cache.clear()

    overall_results = {
        policy_name: aggregate_metric_dicts(per_seed_means) for policy_name, per_seed_means in overall_seed_summaries.items()
    }

    delta_vs_heuristic: dict[str, dict[str, object]] = {}
    heuristic_seed_units = overall_seed_summaries["heuristic_offloading"]
    comparison_policy_names = [policy_name for policy_name in policy_specs if policy_name != "heuristic_offloading"]
    # 循环处理：遍历 policy_name 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for policy_name in comparison_policy_names:
        policy_seed_units = overall_seed_summaries[policy_name]
        delta_vs_heuristic[policy_name] = {
            metric_name: {
                "mean_delta": float(
                    np.mean(
                        [
                            float(policy_seed_units[idx][metric_name] - heuristic_seed_units[idx][metric_name])
                            for idx in range(len(policy_seed_units))
                        ]
                    )
                ),
                "std_delta": float(
                    np.std(
                        [
                            float(policy_seed_units[idx][metric_name] - heuristic_seed_units[idx][metric_name])
                            for idx in range(len(policy_seed_units))
                        ]
                    )
                ),
            }
            for metric_name in METRIC_NAMES
        }

    scenario_wins: dict[str, dict[str, int]] = {}
    # 循环处理：遍历 policy_name 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for policy_name in comparison_policy_names:
        latency_wins = 0
        energy_wins = 0
        deadline_wins = 0
        # 循环处理：遍历 scenario 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for scenario in scenarios:
            scenario_name = scenario.name
            heuristic_metrics = scenario_results[scenario_name]["heuristic_offloading"]["aggregate"]
            policy_metrics = scenario_results[scenario_name][policy_name]["aggregate"]
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if policy_metrics["latency"]["mean"] < heuristic_metrics["latency"]["mean"]:
                latency_wins += 1
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if policy_metrics["energy"]["mean"] < heuristic_metrics["energy"]["mean"]:
                energy_wins += 1
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if policy_metrics["deadline_satisfaction_rate"]["mean"] > heuristic_metrics["deadline_satisfaction_rate"]["mean"]:
                deadline_wins += 1
        scenario_wins[policy_name] = {
            "latency_wins": latency_wins,
            "energy_wins": energy_wins,
            "deadline_wins": deadline_wins,
            "num_scenarios": len(scenarios),
        }

    report: dict[str, object] = {
        "metadata": {
            "surrogate_checkpoint": str(surrogate_checkpoint),
            "rich_reduced_checkpoint": str(rich_reduced_checkpoint),
            "cql_checkpoint": str(cql_checkpoint) if cql_checkpoint is not None else None,
            "radcc_checkpoint": str(radcc_checkpoint) if radcc_checkpoint is not None else None,
            "sc_ogo_checkpoint": str(sc_ogo_checkpoint) if sc_ogo_checkpoint is not None else None,
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "steps_per_episode": steps_per_episode,
            "trajectory_policy": "static baseline with zero motion actions",
            "metric_semantics": "Per-episode means of per-step metrics, then aggregated across seeds.",
        },
        "scenarios": [{ "name": scenario.name, "description": scenario.description } for scenario in scenarios],
        "per_scenario": scenario_results,
        "overall": overall_results,
        "delta_vs_heuristic": delta_vs_heuristic,
        "scenario_win_counts_vs_heuristic": scenario_wins,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return report


# 函数 parse_args：解析命令行参数，并为实验脚本提供可覆盖的默认配置。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heuristic, learned classifier, and CQL-DQN runtime offloading policies.")
    parser.add_argument("--surrogate_checkpoint", type=str, required=True, help="Full-feature surrogate checkpoint.")
    parser.add_argument("--rich_checkpoint", type=str, required=True, help="Rich reduced runtime checkpoint.")
    parser.add_argument("--cql_checkpoint", type=str, default=None, help="Optional constrained CQL-DQN checkpoint.")
    parser.add_argument("--radcc_checkpoint", type=str, default=None, help="Optional RADCC-Offload checkpoint.")
    parser.add_argument("--sc_ogo_checkpoint", type=str, default=None, help="Optional SC-OGO surrogate checkpoint.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(results_path("reports", "runtime_offload_policy_comparison.json")),
        help="Output JSON report.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Seeds used for the multi-seed comparison.")
    parser.add_argument("--episodes_per_seed", type=int, default=4, help="Episodes per seed and scenario.")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="Steps per episode.")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return parser.parse_args()


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main() -> None:
    args = parse_args()
    report = compare_runtime_offload_policies(
        surrogate_checkpoint=args.surrogate_checkpoint,
        rich_reduced_checkpoint=args.rich_checkpoint,
        cql_checkpoint=args.cql_checkpoint,
        radcc_checkpoint=args.radcc_checkpoint,
        sc_ogo_checkpoint=args.sc_ogo_checkpoint,
        seeds=[int(seed) for seed in args.seeds],
        episodes_per_seed=args.episodes_per_seed,
        steps_per_episode=args.steps_per_episode,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
