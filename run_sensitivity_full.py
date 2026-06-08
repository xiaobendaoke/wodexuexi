#!/usr/bin/env python3
"""
完整敏感性实验：使用已训练的兼容模型运行所有三个实验。

模型兼容性已验证：
- Proposed: paper_revised_full_20260518 (seed 42, 84, 126)
- Vanilla MAPPO: 20260603/20260604 系列（8个版本，需映射到seed）
- Heuristic: 不需要训练
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from environment.env import Env
from marl_models.utils import get_model
from utils.baseline_metrics import (
    set_global_seed,
    run_single_episode,
    aggregate_metric_dicts,
    SUMMARY_METRIC_NAMES,
)

# ─── 已验证的兼容模型路径 ──────────────────────────────────────────────────────

PROPOSED_MODELS = {
    42: {
        "traj": "saved_models/attention_mappo_cpu_full_paper_revised_full_20260518_full_hmarl_seed42_200ep/final",
        "offload": "saved_models/offload_mappo_cpu_full_paper_revised_full_20260518_full_hmarl_seed42_200ep/final",
    },
    84: {
        "traj": "saved_models/attention_mappo_cpu_full_paper_revised_full_20260518_full_hmarl_seed84_200ep/final",
        "offload": "saved_models/offload_mappo_cpu_full_paper_revised_full_20260518_full_hmarl_seed84_200ep/final",
    },
    126: {
        "traj": "saved_models/attention_mappo_cpu_full_paper_revised_full_20260518_full_hmarl_seed126_200ep/final",
        "offload": "saved_models/offload_mappo_cpu_full_paper_revised_full_20260518_full_hmarl_seed126_200ep/final",
    },
}

VANILLA_MAPPO_MODELS = {
    42: {
        "traj": "saved_models/vanilla_mappo_20260603_174250_vanilla_mappo/final",
        "offload": "saved_models/offload_mappo_20260603_174250_vanilla_mappo/final",
    },
    84: {
        "traj": "saved_models/vanilla_mappo_20260603_180500_vanilla_mappo/final",
        "offload": "saved_models/offload_mappo_20260603_180500_vanilla_mappo/final",
    },
    126: {
        "traj": "saved_models/vanilla_mappo_20260603_181404_vanilla_mappo/final",
        "offload": "saved_models/offload_mappo_20260603_181404_vanilla_mappo/final",
    },
}


def load_model_pair(algo_name: str, seed: int):
    """加载模型对 (trajectory_model, offload_model)。"""
    if algo_name == "proposed":
        paths = PROPOSED_MODELS.get(seed)
        if not paths:
            raise ValueError(f"No proposed model for seed {seed}")
        traj_model = get_model("attention_mappo")
        traj_model.load(paths["traj"])
        offload_model = get_model("constrained_attention_offload_mappo")
        offload_model.load(paths["offload"])
        return traj_model, offload_model

    elif algo_name == "vanilla_mappo":
        paths = VANILLA_MAPPO_MODELS.get(seed)
        if not paths:
            raise ValueError(f"No vanilla_mappo model for seed {seed}")
        traj_model = get_model("vanilla_mappo")
        traj_model.load(paths["traj"])
        offload_model = get_model("constrained_attention_offload_mappo")
        offload_model.load(paths["offload"])
        return traj_model, offload_model

    elif algo_name == "heuristic":
        traj_model = get_model("uncoordinated_greedy")
        return traj_model, None

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_single_eval_episode(env, traj_model, offload_model, algo_name, run_seed):
    """运行单个评估episode。"""
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    if algo_name == "heuristic":
        episode_metrics, _ = run_single_episode(
            env, traj_model, offload_model=None,
            exploration=False, record_trajectory=False,
            policy_type="non_learning",
        )
    else:
        episode_metrics, _ = run_single_episode(
            env, traj_model, offload_model=offload_model,
            exploration=False, record_trajectory=False,
            policy_type="learned",
        )

    return episode_metrics


# ─── 实验A：收敛曲线 ──────────────────────────────────────────────────────────

def run_convergence_experiment(
    training_seeds: list[int] = [42, 84, 126],
    workload_seed: int = 42,
    eval_episodes: int = 6,
    output_dir: str = "results/sensitivity/effective_efficiency_convergence",
) -> dict:
    """运行收敛曲线实验。

    使用已训练模型在固定workload下评估不同checkpoint的EEE。
    由于没有中间checkpoint，只评估最终模型（200 episodes）。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    algorithms = ["proposed", "vanilla_mappo"]
    all_results = {}

    for algo_name in algorithms:
        print(f"\n{'='*60}")
        print(f"Evaluating {algo_name}")
        print(f"{'='*60}")

        algo_results = []

        for seed in training_seeds:
            print(f"\n--- Seed {seed} ---")
            set_global_seed(seed)

            try:
                traj_model, offload_model = load_model_pair(algo_name, seed)
            except Exception as e:
                print(f"  Failed to load model: {e}")
                continue

            env = Env()
            episode_metrics_list = []

            for ep_idx in range(eval_episodes):
                run_seed = int(workload_seed + ep_idx * 1000)
                metrics = run_single_eval_episode(env, traj_model, offload_model, algo_name, run_seed)
                episode_metrics_list.append(metrics)

            # 计算统计量
            unit_mean = {}
            for metric in SUMMARY_METRIC_NAMES:
                values = [m.get(metric, 0.0) for m in episode_metrics_list if metric in m]
                if values:
                    unit_mean[metric] = float(np.mean(values))

            algo_results.append({
                "seed": seed,
                "episode": 200,
                "unit_mean": unit_mean,
                "episode_metrics": episode_metrics_list,
            })

            print(f"  EEE: {unit_mean.get('energy_efficiency', 0):.6f}")
            print(f"  Reward: {unit_mean.get('reward', 0):.2f}")
            print(f"  DSR: {unit_mean.get('deadline_satisfaction_rate', 0):.4f}")

        all_results[algo_name] = algo_results

    # 保存结果
    raw_path = output_path / "raw"
    raw_path.mkdir(exist_ok=True)
    with open(raw_path / "convergence_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # 生成汇总
    summary = generate_convergence_summary(all_results)
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def generate_convergence_summary(all_results: dict) -> dict:
    """生成收敛曲线汇总统计。"""
    summary = {
        "experiment": "effective_efficiency_convergence",
        "algorithms": {},
    }

    for algo_name, seed_results in all_results.items():
        if not seed_results:
            continue

        # 收集所有seed的unit_mean
        unit_means = [sr["unit_mean"] for sr in seed_results]

        # 计算统计量
        stats = {}
        for metric in SUMMARY_METRIC_NAMES:
            values = [um.get(metric, 0.0) for um in unit_means if metric in um]
            if values:
                stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "n": len(values),
                }

        summary["algorithms"][algo_name] = {
            "num_seeds": len(seed_results),
            "seeds": [sr["seed"] for sr in seed_results],
            "episode": 200,
            "stats": stats,
        }

    return summary


# ─── 实验B：UE数量敏感性 ──────────────────────────────────────────────────────

def run_ue_count_experiment(
    ue_counts: list[int] = [60, 80, 100, 120, 140],
    training_seeds: list[int] = [42, 84, 126],
    workload_seeds: list[int] = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420],
    eval_episodes: int = 6,
    output_dir: str = "results/sensitivity/ue_count",
) -> dict:
    """运行UE数量敏感性实验。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    algorithms = ["heuristic", "vanilla_mappo", "proposed"]
    all_results = {}

    for ue_count in ue_counts:
        print(f"\n{'='*60}")
        print(f"Testing UE count = {ue_count}")
        print(f"{'='*60}")

        ue_results = {}

        for algo_name in algorithms:
            print(f"\n--- Algorithm: {algo_name} ---")
            algo_metrics = []

            for train_seed in training_seeds:
                print(f"  Training seed: {train_seed}")

                # 临时修改config
                original_num_ues = config.NUM_UES
                config.NUM_UES = ue_count

                try:
                    traj_model, offload_model = load_model_pair(algo_name, train_seed)
                except Exception as e:
                    print(f"    Failed to load model: {e}")
                    config.NUM_UES = original_num_ues
                    continue

                env = Env()

                for workload_seed in workload_seeds:
                    for ep_idx in range(eval_episodes):
                        run_seed = int(workload_seed + ep_idx * 1000)
                        metrics = run_single_eval_episode(env, traj_model, offload_model, algo_name, run_seed)
                        metrics["ue_count"] = ue_count
                        metrics["train_seed"] = train_seed
                        metrics["workload_seed"] = workload_seed
                        algo_metrics.append(metrics)

                # 恢复config
                config.NUM_UES = original_num_ues

            ue_results[algo_name] = algo_metrics

        all_results[ue_count] = ue_results

    # 保存原始结果
    raw_path = output_path / "raw"
    raw_path.mkdir(exist_ok=True)
    with open(raw_path / "ue_count_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # 生成汇总
    summary = generate_sensitivity_summary(all_results, "ue_count")
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成统计报告
    stats_md = generate_statistics_markdown(summary, "ue_count")
    with open(output_path / "statistics.md", "w") as f:
        f.write(stats_md)

    return summary


# ─── 实验C：UAV CPU敏感性 ──────────────────────────────────────────────────────

def run_uav_cpu_scale_experiment(
    scales: list[float] = [0.6, 0.8, 1.0, 1.2, 1.4],
    training_seeds: list[int] = [42, 84, 126],
    workload_seeds: list[int] = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420],
    eval_episodes: int = 6,
    output_dir: str = "results/sensitivity/uav_cpu_scale",
) -> dict:
    """运行UAV CPU敏感性实验。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    algorithms = ["heuristic", "vanilla_mappo", "proposed"]
    all_results = {}

    for scale in scales:
        print(f"\n{'='*60}")
        print(f"Testing UAV CPU scale = {scale}x")
        print(f"{'='*60}")

        scale_results = {}

        for algo_name in algorithms:
            print(f"\n--- Algorithm: {algo_name} ---")
            algo_metrics = []

            for train_seed in training_seeds:
                print(f"  Training seed: {train_seed}")

                # 临时修改config中的UAV计算能力
                original_computing_capacity = config.UAV_COMPUTING_CAPACITY.copy()
                base_range = np.arange(40 * 10**9, 90 * 10**9, 5 * 10**9)
                scaled_range = base_range * scale
                config.UAV_COMPUTING_CAPACITY = np.random.choice(
                    scaled_range, size=config.NUM_UAVS
                ).astype(np.int64)

                try:
                    traj_model, offload_model = load_model_pair(algo_name, train_seed)
                except Exception as e:
                    print(f"    Failed to load model: {e}")
                    config.UAV_COMPUTING_CAPACITY = original_computing_capacity
                    continue

                env = Env()

                for workload_seed in workload_seeds:
                    for ep_idx in range(eval_episodes):
                        run_seed = int(workload_seed + ep_idx * 1000)
                        metrics = run_single_eval_episode(env, traj_model, offload_model, algo_name, run_seed)
                        metrics["cpu_scale"] = scale
                        metrics["train_seed"] = train_seed
                        metrics["workload_seed"] = workload_seed
                        algo_metrics.append(metrics)

                # 恢复config
                config.UAV_COMPUTING_CAPACITY = original_computing_capacity

            scale_results[algo_name] = algo_metrics

        all_results[scale] = scale_results

    # 保存原始结果
    raw_path = output_path / "raw"
    raw_path.mkdir(exist_ok=True)
    with open(raw_path / "uav_cpu_scale_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # 生成汇总
    summary = generate_sensitivity_summary(all_results, "uav_cpu_scale")
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成统计报告
    stats_md = generate_statistics_markdown(summary, "uav_cpu_scale")
    with open(output_path / "statistics.md", "w") as f:
        f.write(stats_md)

    return summary


# ─── 通用汇总函数 ──────────────────────────────────────────────────────────────

def generate_sensitivity_summary(all_results: dict, experiment_name: str) -> dict:
    """生成敏感性实验汇总统计。"""
    summary = {
        "experiment": experiment_name,
        "variables": {},
    }

    for variable_value, algo_results in all_results.items():
        var_summary = {}

        for algo_name, metrics_list in algo_results.items():
            if not metrics_list:
                continue

            # 按(train_seed, workload_seed)分组，计算episode mean
            grouped = {}
            for m in metrics_list:
                key = (m.get("train_seed", 0), m.get("workload_seed", 0))
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(m)

            # 计算每个组的episode mean
            unit_means = []
            for key, group in grouped.items():
                unit_mean = {}
                for metric in SUMMARY_METRIC_NAMES:
                    values = [m.get(metric, 0.0) for m in group if metric in m]
                    if values:
                        unit_mean[metric] = float(np.mean(values))
                unit_means.append(unit_mean)

            # 计算统计量
            stats = {}
            for metric in SUMMARY_METRIC_NAMES:
                values = [um.get(metric, 0.0) for um in unit_means if metric in um]
                if values:
                    stats[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "ci_low": float(np.percentile(values, 2.5)),
                        "ci_high": float(np.percentile(values, 97.5)),
                        "n": len(values),
                    }

            var_summary[algo_name] = {
                "num_samples": len(unit_means),
                "stats": stats,
            }

        summary["variables"][str(variable_value)] = var_summary

    # 计算相对提升
    summary["improvements"] = calculate_improvements(summary)

    return summary


def calculate_improvements(summary: dict) -> dict:
    """计算Proposed相对于baseline的提升百分比。"""
    improvements = {}

    for var_value, var_data in summary["variables"].items():
        if "proposed" not in var_data:
            continue

        proposed_stats = var_data["proposed"]["stats"]
        improvements[var_value] = {}

        for algo_name in ["heuristic", "vanilla_mappo"]:
            if algo_name not in var_data:
                continue

            baseline_stats = var_data[algo_name]["stats"]
            algo_improvements = {}

            for metric in ["energy_efficiency", "reward", "energy", "deadline_satisfaction_rate"]:
                if metric in proposed_stats and metric in baseline_stats:
                    proposed_mean = proposed_stats[metric]["mean"]
                    baseline_mean = baseline_stats[metric]["mean"]

                    if abs(baseline_mean) > 1e-6:
                        if metric in ["energy", "latency", "offline_rate"]:
                            improvement = (baseline_mean - proposed_mean) / baseline_mean * 100
                        else:
                            improvement = (proposed_mean - baseline_mean) / abs(baseline_mean) * 100

                        algo_improvements[metric] = {
                            "proposed_mean": proposed_mean,
                            "baseline_mean": baseline_mean,
                            "improvement_pct": float(improvement),
                        }

            improvements[var_value][algo_name] = algo_improvements

    return improvements


def generate_statistics_markdown(summary: dict, experiment_name: str) -> str:
    """生成统计报告Markdown。"""
    lines = [
        f"# {experiment_name.replace('_', ' ').title()} 敏感性实验统计报告",
        "",
        f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # 概览
    lines.extend([
        "## 实验概览",
        "",
        "| 变量值 | 算法 | 样本量 | Energy Efficiency (mean±std) | Reward (mean±std) | DSR (mean±std) |",
        "|--------|------|--------|------------------------------|-------------------|----------------|",
    ])

    for var_value, var_data in sorted(summary["variables"].items()):
        for algo_name, algo_data in var_data.items():
            n = algo_data["num_samples"]
            ee = algo_data["stats"].get("energy_efficiency", {})
            rw = algo_data["stats"].get("reward", {})
            dsr = algo_data["stats"].get("deadline_satisfaction_rate", {})

            ee_str = f"{ee.get('mean', 0):.6f}±{ee.get('std', 0):.6f}" if ee else "N/A"
            rw_str = f"{rw.get('mean', 0):.2f}±{rw.get('std', 0):.2f}" if rw else "N/A"
            dsr_str = f"{dsr.get('mean', 0):.4f}±{dsr.get('std', 0):.4f}" if dsr else "N/A"

            lines.append(f"| {var_value} | {algo_name} | {n} | {ee_str} | {rw_str} | {dsr_str} |")

    lines.append("")

    return "\n".join(lines)


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="完整敏感性实验")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["convergence", "ue_count", "uav_cpu_scale", "all"],
                        help="实验类型")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126],
                        help="训练种子")
    parser.add_argument("--workload_seeds", type=int, nargs="+",
                        default=[42, 84, 126, 168, 210, 252, 294, 336, 378, 420],
                        help="工作负载种子")
    parser.add_argument("--eval_episodes", type=int, default=6,
                        help="每个配置的评估episode数")
    parser.add_argument("--ue_counts", type=int, nargs="+", default=[60, 80, 100, 120, 140],
                        help="UE数量列表")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4],
                        help="CPU缩放因子列表")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Running sensitivity experiment: {args.experiment}")
    print(f"Training seeds: {args.seeds}")
    print(f"Workload seeds: {args.workload_seeds}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"{'='*60}")

    if args.experiment in ["convergence", "all"]:
        print("\n>>> Running convergence experiment...")
        run_convergence_experiment(
            training_seeds=args.seeds,
            workload_seed=args.workload_seeds[0],
            eval_episodes=args.eval_episodes,
        )

    if args.experiment in ["ue_count", "all"]:
        print("\n>>> Running UE count experiment...")
        run_ue_count_experiment(
            ue_counts=args.ue_counts,
            training_seeds=args.seeds,
            workload_seeds=args.workload_seeds,
            eval_episodes=args.eval_episodes,
        )

    if args.experiment in ["uav_cpu_scale", "all"]:
        print("\n>>> Running UAV CPU scale experiment...")
        run_uav_cpu_scale_experiment(
            scales=args.scales,
            training_seeds=args.seeds,
            workload_seeds=args.workload_seeds,
            eval_episodes=args.eval_episodes,
        )

    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
