#!/usr/bin/env python3
"""生成敏感性实验图表。

生成三张图：
- 图5-11 系统有效能效收敛曲线
- 图5-12 地面节点数量对系统有效能效的影响
- 图5-13 UAV算力对系统有效能效的影响
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

from thesis_figure_style import PALETTE, save_pub as save_pub_shared, setup_thesis_style

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRS = (ROOT / "docs" / "figures", ROOT / "latex" / "docs" / "figures")
QA_REPORT = ROOT / "docs" / "figures" / "figure_text_qa_report.json"
ALGO_COLORS = {
    "proposed": PALETTE["full"],
    "vanilla_mappo": PALETTE["upper"],
    "ippo": PALETTE["lower"],
    "heuristic": PALETTE["baseline"],
}


def setup_style() -> None:
    setup_thesis_style(font_size=7, legend_frameon=False)


def save_pub(fig: plt.Figure, name: str) -> None:
    qa = save_pub_shared(fig, name, OUTPUT_DIRS, qa_report=QA_REPORT)
    if qa["status"] != "ok":
        print(f"QA review: {name} -> {qa}")
    print(f"Saved: {name}")


def soften(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.7)


# ─── 图5-11：收敛曲线 ─────────────────────────────────────────────────────────

def fig_convergence_curve() -> None:
    """从训练日志提取的收敛数据生成系统有效能效收敛曲线。"""
    conv_path = ROOT / "results" / "sensitivity" / "effective_efficiency_convergence" / "summary.json"
    if not conv_path.exists():
        print(f"Warning: {conv_path} not found, skipping convergence figure")
        return

    with open(conv_path, "r") as f:
        data = json.load(f)

    # 同时尝试加载从训练日志提取的中间checkpoint数据
    raw_path = ROOT / "results" / "sensitivity" / "effective_efficiency_convergence" / "raw" / "convergence_results.json"
    raw_data = None
    if raw_path.exists():
        with open(raw_path, "r") as f:
            raw_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    algo_labels = {
        "proposed": "本文方法",
        "vanilla_mappo": "普通MAPPO",
        "ippo": "IPPO",
    }
    algo_colors = {
        "proposed": ALGO_COLORS["proposed"],
        "vanilla_mappo": ALGO_COLORS["vanilla_mappo"],
        "ippo": ALGO_COLORS["ippo"],
    }
    algo_markers = {
        "proposed": "o",
        "vanilla_mappo": "s",
        "ippo": "^",
    }

    # 左图：EEE 收敛（如果有中间checkpoint数据）
    ax = axes[0]
    has_curve_data = False
    if raw_data:
        for algo_name in ["proposed", "vanilla_mappo", "ippo"]:
            if algo_name not in raw_data:
                continue
            seed_results = raw_data[algo_name]
            # 收集所有seed在每个episode的EEE
            episode_data = {}
            for sr in seed_results:
                for cp in sr.get("convergence_points", []):
                    ep = cp["episode"]
                    if ep not in episode_data:
                        episode_data[ep] = []
                    episode_data[ep].append(cp["energy_efficiency"])

            if len(episode_data) < 2:
                continue

            has_curve_data = True
            episodes = sorted(episode_data.keys())
            means = [np.mean(episode_data[ep]) for ep in episodes]
            stds = [np.std(episode_data[ep]) for ep in episodes]

            ax.plot(episodes, means, label=algo_labels.get(algo_name, algo_name),
                    color=algo_colors.get(algo_name, "#999"),
                    marker=algo_markers.get(algo_name, "o"),
                    linewidth=1.5, markersize=3)
            ax.fill_between(episodes,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=algo_colors.get(algo_name, "#999"), alpha=0.15)

    if not has_curve_data:
        # 只有最终数据点，用柱状图
        algos = []
        eee_means = []
        eee_stds = []
        for algo_name in ["proposed", "vanilla_mappo"]:
            if algo_name in data.get("algorithms", {}):
                stats = data["algorithms"][algo_name].get("stats", {})
                ee = stats.get("energy_efficiency", {})
                algos.append(algo_labels.get(algo_name, algo_name))
                eee_means.append(ee.get("mean", 0))
                eee_stds.append(ee.get("std", 0))

        x = np.arange(len(algos))
        colors = [algo_colors.get(a.lower().replace(" ", "_").replace("(full hierarchical)", "").strip(), "#999")
                  for a in algos]
        # Map labels back to color keys
        color_list = []
        for a in algos:
            if "本文方法" in a:
                color_list.append(ALGO_COLORS["proposed"])
            elif "普通" in a:
                color_list.append(ALGO_COLORS["vanilla_mappo"])
            else:
                color_list.append(ALGO_COLORS["heuristic"])

        bars = ax.bar(x, eee_means, yerr=eee_stds, capsize=2,
                      color=color_list, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=15, ha="right")
        ax.set_ylabel("有效能效")

    ax.set_xlabel("训练回合" if has_curve_data else "")
    ax.set_title("训练完成后的有效能效", fontsize=8, loc="left", pad=4)
    if has_curve_data:
        ax.legend(loc="upper left", fontsize=6)
    soften(ax)

    # 右图：DSR 对比
    ax2 = axes[1]
    algos = []
    dsr_means = []
    dsr_stds = []
    for algo_name in ["proposed", "vanilla_mappo"]:
        if algo_name in data.get("algorithms", {}):
            stats = data["algorithms"][algo_name].get("stats", {})
            dsr = stats.get("deadline_satisfaction_rate", {})
            algos.append(algo_labels.get(algo_name, algo_name))
            dsr_means.append(dsr.get("mean", 0))
            dsr_stds.append(dsr.get("std", 0))

    x = np.arange(len(algos))
    color_list = [ALGO_COLORS["proposed"], ALGO_COLORS["vanilla_mappo"]]
    ax2.bar(x, dsr_means, yerr=dsr_stds, capsize=2,
            color=color_list, edgecolor="white", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, rotation=15, ha="right")
    ax2.set_ylabel("DSR")
    ax2.set_title("截止期满足率", fontsize=8, loc="left", pad=4)
    soften(ax2)

    fig.tight_layout(w_pad=1.2)
    save_pub(fig, "图5-11_系统有效能效收敛曲线")


# ─── 图5-12：UE数量敏感性 ──────────────────────────────────────────────────────

def fig_ue_count_sensitivity(data_path: Path) -> None:
    if not data_path.exists():
        print(f"Warning: {data_path} not found, skipping ue_count figure")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    algo_names = {"proposed": "本文方法", "vanilla_mappo": "普通MAPPO", "heuristic": "启发式"}
    algo_colors = {"proposed": ALGO_COLORS["proposed"], "vanilla_mappo": ALGO_COLORS["vanilla_mappo"], "heuristic": ALGO_COLORS["heuristic"]}
    algo_markers = {"proposed": "o", "vanilla_mappo": "s", "heuristic": "^"}

    ue_counts = sorted([int(v) for v in data.get("variables", {}).keys()])

    # 左图：EEE
    ax = axes[0]
    for algo_name in ["heuristic", "vanilla_mappo", "proposed"]:
        means, ci_lows, ci_highs = [], [], []
        for uc in ue_counts:
            stats = data["variables"].get(str(uc), {}).get(algo_name, {}).get("stats", {}).get("energy_efficiency", {})
            means.append(stats.get("mean", 0))
            ci_lows.append(stats.get("ci_low", 0))
            ci_highs.append(stats.get("ci_high", 0))

        ax.plot(ue_counts, means, label=algo_names[algo_name], color=algo_colors[algo_name],
                marker=algo_markers[algo_name], linewidth=1.5, markersize=5)
        ax.fill_between(ue_counts, ci_lows, ci_highs, color=algo_colors[algo_name], alpha=0.15)

    ax.set_xlabel("UE数量")
    ax.set_ylabel("有效能效")
    ax.set_title("有效能效", fontsize=8, loc="left", pad=4)
    ax.legend(loc="upper left", fontsize=6)
    ax.set_xticks(ue_counts)
    soften(ax)

    # 右图：DSR
    ax2 = axes[1]
    for algo_name in ["heuristic", "vanilla_mappo", "proposed"]:
        means, ci_lows, ci_highs = [], [], []
        for uc in ue_counts:
            stats = data["variables"].get(str(uc), {}).get(algo_name, {}).get("stats", {}).get("deadline_satisfaction_rate", {})
            means.append(stats.get("mean", 0))
            ci_lows.append(stats.get("ci_low", 0))
            ci_highs.append(stats.get("ci_high", 0))

        ax2.plot(ue_counts, means, label=algo_names[algo_name], color=algo_colors[algo_name],
                 marker=algo_markers[algo_name], linewidth=1.5, markersize=5)
        ax2.fill_between(ue_counts, ci_lows, ci_highs, color=algo_colors[algo_name], alpha=0.15)

    ax2.set_xlabel("UE数量")
    ax2.set_ylabel("DSR")
    ax2.set_title("截止期满足率", fontsize=8, loc="left", pad=4)
    ax2.legend(loc="upper right", fontsize=6)
    ax2.set_xticks(ue_counts)
    soften(ax2)

    fig.tight_layout(w_pad=1.2)
    save_pub(fig, "图5-12_地面节点数量对系统有效能效的影响")


# ─── 图5-13：UAV CPU敏感性 ──────────────────────────────────────────────────────

def fig_uav_cpu_sensitivity(data_path: Path) -> None:
    if not data_path.exists():
        print(f"Warning: {data_path} not found, skipping uav_cpu figure")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    algo_names = {"proposed": "本文方法", "vanilla_mappo": "普通MAPPO", "heuristic": "启发式"}
    algo_colors = {"proposed": ALGO_COLORS["proposed"], "vanilla_mappo": ALGO_COLORS["vanilla_mappo"], "heuristic": ALGO_COLORS["heuristic"]}
    algo_markers = {"proposed": "o", "vanilla_mappo": "s", "heuristic": "^"}

    scales = sorted([float(v) for v in data.get("variables", {}).keys()])

    # 左图：EEE
    ax = axes[0]
    for algo_name in ["heuristic", "vanilla_mappo", "proposed"]:
        means, ci_lows, ci_highs = [], [], []
        for sc in scales:
            stats = data["variables"].get(str(sc), {}).get(algo_name, {}).get("stats", {}).get("energy_efficiency", {})
            means.append(stats.get("mean", 0))
            ci_lows.append(stats.get("ci_low", 0))
            ci_highs.append(stats.get("ci_high", 0))

        ax.plot(scales, means, label=algo_names[algo_name], color=algo_colors[algo_name],
                marker=algo_markers[algo_name], linewidth=1.5, markersize=5)
        ax.fill_between(scales, ci_lows, ci_highs, color=algo_colors[algo_name], alpha=0.15)

    ax.set_xlabel("UAV算力缩放因子")
    ax.set_ylabel("有效能效")
    ax.set_title("有效能效", fontsize=8, loc="left", pad=4)
    ax.legend(loc="upper left", fontsize=6)
    ax.set_xticks(scales)
    soften(ax)

    # 右图：DSR
    ax2 = axes[1]
    for algo_name in ["heuristic", "vanilla_mappo", "proposed"]:
        means, ci_lows, ci_highs = [], [], []
        for sc in scales:
            stats = data["variables"].get(str(sc), {}).get(algo_name, {}).get("stats", {}).get("deadline_satisfaction_rate", {})
            means.append(stats.get("mean", 0))
            ci_lows.append(stats.get("ci_low", 0))
            ci_highs.append(stats.get("ci_high", 0))

        ax2.plot(scales, means, label=algo_names[algo_name], color=algo_colors[algo_name],
                 marker=algo_markers[algo_name], linewidth=1.5, markersize=5)
        ax2.fill_between(scales, ci_lows, ci_highs, color=algo_colors[algo_name], alpha=0.15)

    ax2.set_xlabel("UAV算力缩放因子")
    ax2.set_ylabel("DSR")
    ax2.set_title("截止期满足率", fontsize=8, loc="left", pad=4)
    ax2.legend(loc="upper right", fontsize=6)
    ax2.set_xticks(scales)
    soften(ax2)

    fig.tight_layout(w_pad=1.2)
    save_pub(fig, "图5-13_UAV算力对系统有效能效的影响")


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    setup_style()

    sensitivity_dir = ROOT / "results" / "sensitivity"

    fig_convergence_curve()

    ue_count_data = sensitivity_dir / "ue_count" / "summary.json"
    fig_ue_count_sensitivity(ue_count_data)

    uav_cpu_data = sensitivity_dir / "uav_cpu_scale" / "summary.json"
    fig_uav_cpu_sensitivity(uav_cpu_data)

    print("\nAll sensitivity figures generated!")


if __name__ == "__main__":
    main()
