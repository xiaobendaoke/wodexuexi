#!/usr/bin/env python3
"""
敏感性实验统一入口。

支持三个实验：
- convergence: 系统有效能效随训练回合变化
- ue_count: 不同地面节点数量对系统有效能效的影响
- uav_cpu_scale: 不同UAV CPU处理速率对系统有效能效的影响

用法：
    python run_sensitivity_experiments.py --experiment convergence --seeds 42 84 126
    python run_sensitivity_experiments.py --experiment ue_count --ue_counts 60 80 100 120 140
    python run_sensitivity_experiments.py --experiment uav_cpu_scale --scales 0.6 0.8 1.0 1.2 1.4
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
from marl_models.utils import get_model, save_models
from utils.baseline_metrics import (
    set_global_seed,
    run_single_episode,
    run_single_episode_joint,
    aggregate_metric_dicts,
    save_baseline_result,
    SUMMARY_METRIC_NAMES,
)
from utils.logger import Log, Logger


# ─── 实验A：收敛曲线 ──────────────────────────────────────────────────────────

def run_convergence_experiment(
    num_episodes: int = 200,
    eval_interval: int = 20,
    training_seeds: list[int] = [42, 84, 126],
    workload_seed: int = 42,
    eval_episodes: int = 6,
    output_dir: str = "results/sensitivity/effective_efficiency_convergence",
) -> dict:
    """运行收敛曲线实验，记录不同训练回合下的有效能效。"""
    from run_baseline_comparison_experiment import train_hierarchical_baseline, train_ippo_baseline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    algorithms = {
        "proposed": {"model": "attention_mappo", "train_func": "hierarchical"},
        "vanilla_mappo": {"model": "vanilla_mappo", "train_func": "hierarchical"},
        "ippo": {"model": "ippo_baseline", "train_func": "ippo"},
    }

    all_results = {}

    for algo_name, algo_config in algorithms.items():
        print(f"\n{'='*60}")
        print(f"Training {algo_name} for convergence analysis")
        print(f"{'='*60}")

        algo_results = []

        for seed in training_seeds:
            print(f"\n--- Seed {seed} ---")
            set_global_seed(seed)
            timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{algo_name}_seed{seed}")

            # 训练模型
            if algo_config["train_func"] == "hierarchical":
                from run_hierarchical_mappo_experiment import train_hierarchical_mappo
                summary = train_hierarchical_mappo(
                    num_episodes=num_episodes,
                    timestamp=timestamp,
                    mode="full_hierarchical",
                    seed=seed,
                    trajectory_model_name=algo_config["model"],
                    lower_ablation="full",
                )
                traj_dir = summary.get("trajectory_model_dir")
                offload_dir = summary.get("offload_model_dir")
                log_json = summary.get("log_json")
            else:  # IPPO
                trajectory_model, offload_model, training_curve = train_ippo_baseline(
                    num_episodes, seed, timestamp,
                )
                traj_dir = f"saved_models/{algo_config['model']}_{timestamp}/final"
                offload_dir = f"saved_models/offload_mappo_{timestamp}/final"
                log_json = None

            # 从训练日志提取收敛曲线
            if log_json and Path(log_json).exists():
                with open(log_json, "r") as f:
                    log_data = json.load(f)
                # 提取每个eval_interval的指标
                convergence_points = []
                for entry in log_data:
                    ep = entry.get("episode", 0)
                    if ep % eval_interval == 0 and ep > 0:
                        convergence_points.append({
                            "episode": ep,
                            "energy_efficiency": entry.get("energy_efficiency", 0.0),
                            "reward": entry.get("reward", 0.0),
                            "energy": entry.get("energy", 0.0),
                            "deadline_satisfaction_rate": entry.get("deadline_satisfaction_rate", 0.0),
                            "offline_rate": entry.get("offline_rate", 0.0),
                            "mbs_load_ratio": entry.get("mbs_load_ratio", 0.0),
                        })
                algo_results.append({
                    "seed": seed,
                    "convergence_points": convergence_points,
                })
            else:
                # 如果没有日志，手动评估
                print(f"Warning: No log found for {algo_name} seed {seed}, skipping convergence extraction")

        all_results[algo_name] = algo_results

    # 保存原始结果
    raw_path = output_path / "raw"
    raw_path.mkdir(exist_ok=True)
    with open(raw_path / "convergence_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 生成汇总
    summary = generate_convergence_summary(all_results, eval_interval)
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def generate_convergence_summary(all_results: dict, eval_interval: int) -> dict:
    """生成收敛曲线汇总统计。"""
    summary = {
        "experiment": "effective_efficiency_convergence",
        "eval_interval": eval_interval,
        "algorithms": {},
    }

    for algo_name, seed_results in all_results.items():
        if not seed_results:
            continue

        # 收集所有seed在每个episode的指标
        episode_data = {}
        for seed_result in seed_results:
            for point in seed_result["convergence_points"]:
                ep = point["episode"]
                if ep not in episode_data:
                    episode_data[ep] = []
                episode_data[ep].append(point)

        # 计算每个episode的统计量
        episode_stats = []
        for ep in sorted(episode_data.keys()):
            points = episode_data[ep]
            if len(points) < 2:
                continue

            energy_eff_values = [p["energy_efficiency"] for p in points]
            reward_values = [p["reward"] for p in points]
            energy_values = [p["energy"] for p in points]
            dsr_values = [p["deadline_satisfaction_rate"] for p in points]

            episode_stats.append({
                "episode": ep,
                "energy_efficiency": {
                    "mean": float(np.mean(energy_eff_values)),
                    "std": float(np.std(energy_eff_values)),
                    "ci_low": float(np.percentile(energy_eff_values, 2.5)),
                    "ci_high": float(np.percentile(energy_eff_values, 97.5)),
                },
                "reward": {
                    "mean": float(np.mean(reward_values)),
                    "std": float(np.std(reward_values)),
                },
                "energy": {
                    "mean": float(np.mean(energy_values)),
                    "std": float(np.std(energy_values)),
                },
                "deadline_satisfaction_rate": {
                    "mean": float(np.mean(dsr_values)),
                    "std": float(np.std(dsr_values)),
                },
            })

        summary["algorithms"][algo_name] = {
            "num_seeds": len(seed_results),
            "episode_stats": episode_stats,
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

    algorithms = {
        "proposed": {"model": "attention_mappo", "offload": "constrained_attention_offload_mappo"},
        "vanilla_mappo": {"model": "vanilla_mappo", "offload": "constrained_attention_offload_mappo"},
        "heuristic": {"model": "uncoordinated_greedy_baseline", "offload": None},
    }

    # 已训练模型路径映射
    trained_models = {
        "proposed": {
            42: {
                "traj": "saved_models/attention_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed42_200ep/final",
                "offload": "saved_models/offload_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed42_200ep/final",
            },
            84: {
                "traj": "saved_models/attention_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed84_200ep/final",
                "offload": "saved_models/offload_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed84_200ep/final",
            },
            126: {
                "traj": "saved_models/attention_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed126_200ep/final",
                "offload": "saved_models/offload_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed126_200ep/final",
            },
        },
    }

    all_results = {}

    for ue_count in ue_counts:
        print(f"\n{'='*60}")
        print(f"Testing UE count = {ue_count}")
        print(f"{'='*60}")

        ue_results = {}

        for algo_name, algo_config in algorithms.items():
            print(f"\n--- Algorithm: {algo_name} ---")
            algo_metrics = []

            for train_seed in training_seeds:
                for workload_seed in workload_seeds:
                    set_global_seed(workload_seed)

                    # 临时修改config
                    original_num_ues = config.NUM_UES
                    config.NUM_UES = ue_count

                    # 创建环境和模型
                    env = Env()
                    trajectory_model = get_model(algo_config["model"])

                    # 尝试加载已训练模型
                    offload_model = None
                    if algo_name != "heuristic":
                        # 使用预定义的模型路径
                        if algo_name in trained_models and train_seed in trained_models[algo_name]:
                            traj_path = trained_models[algo_name][train_seed]["traj"]
                            offload_path = trained_models[algo_name][train_seed]["offload"]
                            if Path(traj_path).exists():
                                trajectory_model.load(traj_path)
                                print(f"  Loaded trajectory model: {traj_path}")
                            if algo_config["offload"] and Path(offload_path).exists():
                                offload_model = get_model(algo_config["offload"])
                                offload_model.load(offload_path)
                                print(f"  Loaded offload model: {offload_path}")
                        else:
                            # 回退到搜索模式
                            model_dir = Path("saved_models")
                            possible_dirs = list(model_dir.glob(f"{algo_config['model']}*seed{train_seed}*"))
                            if possible_dirs:
                                latest_dir = max(possible_dirs, key=lambda p: p.stat().st_mtime)
                                final_dir = latest_dir / "final"
                                if final_dir.exists():
                                    trajectory_model.load(str(final_dir))

                            if algo_config["offload"]:
                                offload_model = get_model(algo_config["offload"])
                                offload_possible = list(model_dir.glob(f"offload_mappo*seed{train_seed}*"))
                                if offload_possible:
                                    offload_latest = max(offload_possible, key=lambda p: p.stat().st_mtime)
                                    offload_final = offload_latest / "final"
                                    if offload_final.exists():
                                        offload_model.load(str(offload_final))

                    # 运行评估
                    for ep_idx in range(eval_episodes):
                        run_seed = int(workload_seed + ep_idx * 1000)
                        np.random.seed(run_seed)
                        torch.manual_seed(run_seed)

                        if algo_name == "heuristic":
                            episode_metrics, _ = run_single_episode(
                                env, trajectory_model, offload_model=None,
                                exploration=False, record_trajectory=False,
                                policy_type="non_learning",
                            )
                        else:
                            episode_metrics, _ = run_single_episode(
                                env, trajectory_model, offload_model=offload_model,
                                exploration=False, record_trajectory=False,
                                policy_type="learned",
                            )

                        episode_metrics["ue_count"] = ue_count
                        episode_metrics["train_seed"] = train_seed
                        episode_metrics["workload_seed"] = workload_seed
                        algo_metrics.append(episode_metrics)

                    # 恢复config
                    config.NUM_UES = original_num_ues

            ue_results[algo_name] = algo_metrics

        all_results[ue_count] = ue_results

    # 保存原始结果
    raw_path = output_path / "raw"
    raw_path.mkdir(exist_ok=True)
    with open(raw_path / "ue_count_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

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

    algorithms = {
        "proposed": {"model": "attention_mappo", "offload": "constrained_attention_offload_mappo"},
        "vanilla_mappo": {"model": "vanilla_mappo", "offload": "constrained_attention_offload_mappo"},
        "heuristic": {"model": "uncoordinated_greedy_baseline", "offload": None},
    }

    # 已训练模型路径映射
    trained_models = {
        "proposed": {
            42: {
                "traj": "saved_models/attention_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed42_200ep/final",
                "offload": "saved_models/offload_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed42_200ep/final",
            },
            84: {
                "traj": "saved_models/attention_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed84_200ep/final",
                "offload": "saved_models/offload_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed84_200ep/final",
            },
            126: {
                "traj": "saved_models/attention_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed126_200ep/final",
                "offload": "saved_models/offload_mappo_cpu_full_formula_fixed_full_20260511_full_hmarl_seed126_200ep/final",
            },
        },
    }

    all_results = {}

    for scale in scales:
        print(f"\n{'='*60}")
        print(f"Testing UAV CPU scale = {scale}x")
        print(f"{'='*60}")

        scale_results = {}

        for algo_name, algo_config in algorithms.items():
            print(f"\n--- Algorithm: {algo_name} ---")
            algo_metrics = []

            for train_seed in training_seeds:
                for workload_seed in workload_seeds:
                    set_global_seed(workload_seed)

                    # 临时修改config中的UAV计算能力
                    original_computing_capacity = config.UAV_COMPUTING_CAPACITY.copy()
                    base_range = np.arange(40 * 10**9, 90 * 10**9, 5 * 10**9)
                    scaled_range = base_range * scale
                    config.UAV_COMPUTING_CAPACITY = np.random.choice(
                        scaled_range, size=config.NUM_UAVS
                    ).astype(np.int64)

                    # 创建环境和模型
                    env = Env()
                    trajectory_model = get_model(algo_config["model"])

                    # 尝试加载已训练模型
                    offload_model = None
                    if algo_name != "heuristic":
                        # 使用预定义的模型路径
                        if algo_name in trained_models and train_seed in trained_models[algo_name]:
                            traj_path = trained_models[algo_name][train_seed]["traj"]
                            offload_path = trained_models[algo_name][train_seed]["offload"]
                            if Path(traj_path).exists():
                                trajectory_model.load(traj_path)
                                print(f"  Loaded trajectory model: {traj_path}")
                            if algo_config["offload"] and Path(offload_path).exists():
                                offload_model = get_model(algo_config["offload"])
                                offload_model.load(offload_path)
                                print(f"  Loaded offload model: {offload_path}")
                        else:
                            # 回退到搜索模式
                            model_dir = Path("saved_models")
                            possible_dirs = list(model_dir.glob(f"{algo_config['model']}*seed{train_seed}*"))
                            if possible_dirs:
                                latest_dir = max(possible_dirs, key=lambda p: p.stat().st_mtime)
                                final_dir = latest_dir / "final"
                                if final_dir.exists():
                                    trajectory_model.load(str(final_dir))

                            if algo_config["offload"]:
                                offload_model = get_model(algo_config["offload"])
                                offload_possible = list(model_dir.glob(f"offload_mappo*seed{train_seed}*"))
                                if offload_possible:
                                    offload_latest = max(offload_possible, key=lambda p: p.stat().st_mtime)
                                    offload_final = offload_latest / "final"
                                    if offload_final.exists():
                                        offload_model.load(str(offload_final))

                    # 运行评估
                    for ep_idx in range(eval_episodes):
                        run_seed = int(workload_seed + ep_idx * 1000)
                        np.random.seed(run_seed)
                        torch.manual_seed(run_seed)

                        if algo_name == "heuristic":
                            episode_metrics, _ = run_single_episode(
                                env, trajectory_model, offload_model=None,
                                exploration=False, record_trajectory=False,
                                policy_type="non_learning",
                            )
                        else:
                            episode_metrics, _ = run_single_episode(
                                env, trajectory_model, offload_model=offload_model,
                                exploration=False, record_trajectory=False,
                                policy_type="learned",
                            )

                        episode_metrics["cpu_scale"] = scale
                        episode_metrics["train_seed"] = train_seed
                        episode_metrics["workload_seed"] = workload_seed
                        algo_metrics.append(episode_metrics)

                    # 恢复config
                    config.UAV_COMPUTING_CAPACITY = original_computing_capacity

            scale_results[algo_name] = algo_metrics

        all_results[scale] = scale_results

    # 保存原始结果
    raw_path = output_path / "raw"
    raw_path.mkdir(exist_ok=True)
    with open(raw_path / "uav_cpu_scale_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

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
                            # 越低越好
                            improvement = (baseline_mean - proposed_mean) / baseline_mean * 100
                        else:
                            # 越高越好
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
        "| 变量值 | 算法 | 样本量 | Energy Efficiency (mean±std) | Reward (mean±std) |",
        "|--------|------|--------|------------------------------|-------------------|",
    ])

    for var_value, var_data in sorted(summary["variables"].items()):
        for algo_name, algo_data in var_data.items():
            n = algo_data["num_samples"]
            ee = algo_data["stats"].get("energy_efficiency", {})
            rw = algo_data["stats"].get("reward", {})

            ee_str = f"{ee.get('mean', 0):.6f}±{ee.get('std', 0):.6f}" if ee else "N/A"
            rw_str = f"{rw.get('mean', 0):.2f}±{rw.get('std', 0):.2f}" if rw else "N/A"

            lines.append(f"| {var_value} | {algo_name} | {n} | {ee_str} | {rw_str} |")

    lines.append("")

    # 详细统计
    lines.extend([
        "## 详细统计",
        "",
    ])

    for var_value, var_data in sorted(summary["variables"].items()):
        lines.append(f"### 变量值 = {var_value}")
        lines.append("")

        for algo_name, algo_data in var_data.items():
            lines.append(f"**{algo_name}** (N={algo_data['num_samples']}):")
            lines.append("")
            lines.append("| 指标 | Mean | Std | 95% CI |")
            lines.append("|------|------|-----|--------|")

            for metric, stats in algo_data["stats"].items():
                ci_str = f"[{stats.get('ci_low', 0):.4f}, {stats.get('ci_high', 0):.4f}]"
                lines.append(
                    f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {ci_str} |"
                )

            lines.append("")

    # 相对提升
    if "improvements" in summary:
        lines.extend([
            "## Proposed 相对 Baseline 的提升",
            "",
        ])

        for var_value, var_improvements in sorted(summary["improvements"].items()):
            lines.append(f"### 变量值 = {var_value}")
            lines.append("")

            for algo_name, metrics in var_improvements.items():
                lines.append(f"**vs {algo_name}:**")
                lines.append("")
                lines.append("| 指标 | Proposed | Baseline | 提升% |")
                lines.append("|------|----------|----------|-------|")

                for metric, data in metrics.items():
                    lines.append(
                        f"| {metric} | {data['proposed_mean']:.4f} | {data['baseline_mean']:.4f} | "
                        f"{data['improvement_pct']:.2f}% |"
                    )

                lines.append("")

    # 结论
    lines.extend([
        "## 结论",
        "",
        "根据统计结果分析：",
        "",
    ])

    # 提取关键结论
    for var_value, var_improvements in summary.get("improvements", {}).items():
        for algo_name, metrics in var_improvements.items():
            if "energy_efficiency" in metrics:
                improvement = metrics["energy_efficiency"]["improvement_pct"]
                if improvement > 0:
                    lines.append(
                        f"- 在{var_value}条件下，Proposed相对{algo_name}的能效提升{improvement:.1f}%"
                    )
                else:
                    lines.append(
                        f"- 在{var_value}条件下，Proposed相对{algo_name}的能效下降{abs(improvement):.1f}%"
                    )

    lines.append("")

    return "\n".join(lines)


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="敏感性实验统一入口")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["convergence", "ue_count", "uav_cpu_scale"],
                        help="实验类型")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126],
                        help="训练种子")
    parser.add_argument("--workload_seeds", type=int, nargs="+",
                        default=[42, 84, 126, 168, 210, 252, 294, 336, 378, 420],
                        help="工作负载种子")
    parser.add_argument("--eval_episodes", type=int, default=6,
                        help="每个配置的评估episode数")

    # 实验A特有参数
    parser.add_argument("--num_episodes", type=int, default=200,
                        help="训练episode数（收敛实验）")
    parser.add_argument("--eval_interval", type=int, default=20,
                        help="评估间隔（收敛实验）")

    # 实验B特有参数
    parser.add_argument("--ue_counts", type=int, nargs="+", default=[60, 80, 100, 120, 140],
                        help="UE数量列表（UE数量实验）")

    # 实验C特有参数
    parser.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4],
                        help="CPU缩放因子列表（CPU实验）")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Running sensitivity experiment: {args.experiment}")
    print(f"{'='*60}")

    if args.experiment == "convergence":
        summary = run_convergence_experiment(
            num_episodes=args.num_episodes,
            eval_interval=args.eval_interval,
            training_seeds=args.seeds,
            workload_seed=args.workload_seeds[0] if args.workload_seeds else 42,
            eval_episodes=args.eval_episodes,
        )
    elif args.experiment == "ue_count":
        summary = run_ue_count_experiment(
            ue_counts=args.ue_counts,
            training_seeds=args.seeds,
            workload_seeds=args.workload_seeds,
            eval_episodes=args.eval_episodes,
        )
    elif args.experiment == "uav_cpu_scale":
        summary = run_uav_cpu_scale_experiment(
            scales=args.scales,
            training_seeds=args.seeds,
            workload_seeds=args.workload_seeds,
            eval_episodes=args.eval_episodes,
        )

    print(f"\n{'='*60}")
    print(f"Experiment {args.experiment} completed!")
    print(f"Results saved to results/sensitivity/{args.experiment.replace('convergence', 'effective_efficiency_convergence')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
