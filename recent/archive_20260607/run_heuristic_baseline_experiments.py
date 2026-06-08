#!/usr/bin/env python3
"""
运行启发式baseline的敏感性实验（不需要训练）。

用法：
    python run_heuristic_baseline_experiments.py --experiment ue_count
    python run_heuristic_baseline_experiments.py --experiment uav_cpu_scale
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


def run_ue_count_experiment(
    ue_counts: list[int] = [60, 80, 100, 120, 140],
    workload_seeds: list[int] = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420],
    eval_episodes: int = 6,
    output_dir: str = "results/sensitivity/ue_count",
) -> dict:
    """运行UE数量敏感性实验（仅启发式baseline）。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ue_count in ue_counts:
        print(f"\n{'='*60}")
        print(f"Testing UE count = {ue_count}")
        print(f"{'='*60}")

        algo_metrics = []

        for workload_seed in workload_seeds:
            set_global_seed(workload_seed)

            # 临时修改config
            original_num_ues = config.NUM_UES
            config.NUM_UES = ue_count

            # 创建环境和模型
            env = Env()
            model = get_model("uncoordinated_greedy")

            # 运行评估
            for ep_idx in range(eval_episodes):
                run_seed = int(workload_seed + ep_idx * 1000)
                np.random.seed(run_seed)
                torch.manual_seed(run_seed)

                episode_metrics, _ = run_single_episode(
                    env, model, offload_model=None,
                    exploration=False, record_trajectory=False,
                    policy_type="non_learning",
                )

                episode_metrics["ue_count"] = ue_count
                episode_metrics["workload_seed"] = workload_seed
                algo_metrics.append(episode_metrics)

            # 恢复config
            config.NUM_UES = original_num_ues

        all_results[ue_count] = {"heuristic": algo_metrics}

    # 保存原始结果
    raw_path = output_path / "raw"
    raw_path.mkdir(exist_ok=True)
    with open(raw_path / "heuristic_ue_count_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 生成汇总
    summary = generate_sensitivity_summary(all_results, "ue_count")
    with open(output_path / "heuristic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成统计报告
    stats_md = generate_statistics_markdown(summary, "ue_count")
    with open(output_path / "heuristic_statistics.md", "w") as f:
        f.write(stats_md)

    return summary


def run_uav_cpu_scale_experiment(
    scales: list[float] = [0.6, 0.8, 1.0, 1.2, 1.4],
    workload_seeds: list[int] = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420],
    eval_episodes: int = 6,
    output_dir: str = "results/sensitivity/uav_cpu_scale",
) -> dict:
    """运行UAV CPU敏感性实验（仅启发式baseline）。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for scale in scales:
        print(f"\n{'='*60}")
        print(f"Testing UAV CPU scale = {scale}x")
        print(f"{'='*60}")

        algo_metrics = []

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
            model = get_model("uncoordinated_greedy")

            # 运行评估
            for ep_idx in range(eval_episodes):
                run_seed = int(workload_seed + ep_idx * 1000)
                np.random.seed(run_seed)
                torch.manual_seed(run_seed)

                episode_metrics, _ = run_single_episode(
                    env, model, offload_model=None,
                    exploration=False, record_trajectory=False,
                    policy_type="non_learning",
                )

                episode_metrics["cpu_scale"] = scale
                episode_metrics["workload_seed"] = workload_seed
                algo_metrics.append(episode_metrics)

            # 恢复config
            config.UAV_COMPUTING_CAPACITY = original_computing_capacity

        all_results[scale] = {"heuristic": algo_metrics}

    # 保存原始结果
    raw_path = output_path / "raw"
    raw_path.mkdir(exist_ok=True)
    with open(raw_path / "heuristic_uav_cpu_scale_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 生成汇总
    summary = generate_sensitivity_summary(all_results, "uav_cpu_scale")
    with open(output_path / "heuristic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成统计报告
    stats_md = generate_statistics_markdown(summary, "uav_cpu_scale")
    with open(output_path / "heuristic_statistics.md", "w") as f:
        f.write(stats_md)

    return summary


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

            # 按workload_seed分组，计算episode mean
            grouped = {}
            for m in metrics_list:
                key = m.get("workload_seed", 0)
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

    return summary


def generate_statistics_markdown(summary: dict, experiment_name: str) -> str:
    """生成统计报告Markdown。"""
    lines = [
        f"# {experiment_name.replace('_', ' ').title()} 敏感性实验统计报告（启发式baseline）",
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


def main():
    parser = argparse.ArgumentParser(description="启发式baseline敏感性实验")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["ue_count", "uav_cpu_scale"],
                        help="实验类型")
    parser.add_argument("--ue_counts", type=int, nargs="+", default=[60, 80, 100, 120, 140],
                        help="UE数量列表")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4],
                        help="CPU缩放因子列表")
    parser.add_argument("--workload_seeds", type=int, nargs="+",
                        default=[42, 84, 126, 168, 210, 252, 294, 336, 378, 420],
                        help="工作负载种子")
    parser.add_argument("--eval_episodes", type=int, default=6,
                        help="每个配置的评估episode数")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Running heuristic baseline experiment: {args.experiment}")
    print(f"{'='*60}")

    if args.experiment == "ue_count":
        summary = run_ue_count_experiment(
            ue_counts=args.ue_counts,
            workload_seeds=args.workload_seeds,
            eval_episodes=args.eval_episodes,
        )
    elif args.experiment == "uav_cpu_scale":
        summary = run_uav_cpu_scale_experiment(
            scales=args.scales,
            workload_seeds=args.workload_seeds,
            eval_episodes=args.eval_episodes,
        )

    print(f"\n{'='*60}")
    print(f"Experiment {args.experiment} completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
