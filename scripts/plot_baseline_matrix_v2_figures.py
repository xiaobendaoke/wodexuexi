#!/usr/bin/env python3
"""Generate thesis figures from baseline_matrix_v2 and sensitivity results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_STATS = ROOT / "results" / "baseline_matrix_v2" / "statistics.json"
UE_SUMMARY = ROOT / "results" / "sensitivity" / "ue_count" / "summary.json"
CPU_SUMMARY = ROOT / "results" / "sensitivity" / "uav_cpu_scale" / "summary.json"
OUT_DIR = ROOT / "latex" / "docs" / "figures"
DOC_OUT_DIR = ROOT / "docs" / "figures"
OUTPUT_DIRS = (DOC_OUT_DIR, OUT_DIR)
QA_REPORT = DOC_OUT_DIR / "figure_text_qa_report.json"
sys.path.insert(0, str(ROOT / "docs" / "figures" / "scripts"))
from thesis_figure_style import BASELINE_COLORS, METHOD_LABELS_CN, PALETTE, save_pub as save_pub_shared, setup_thesis_style

METHODS = ["random", "uniform", "ippo", "vanilla_mappo", "joint_mappo", "proposed"]
METHOD_LABELS = METHOD_LABELS_CN
COLORS = BASELINE_COLORS


def setup_style() -> None:
    setup_thesis_style(font_size=8, legend_frameon=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def metric_stats(stats: dict, method: str, metric: str, scale: float = 1.0) -> tuple[float, float, float]:
    item = stats["methods"][method]["metrics"][metric]
    mean = float(item["mean"]) * scale
    low_key = "ci_95_lower" if "ci_95_lower" in item else "ci_low"
    high_key = "ci_95_upper" if "ci_95_upper" in item else "ci_high"
    low = float(item[low_key]) * scale
    high = float(item[high_key]) * scale
    return mean, max(0.0, mean - low), max(0.0, high - mean)


def save_figure(fig: plt.Figure, stem: str) -> None:
    qa = save_pub_shared(fig, stem, OUTPUT_DIRS, qa_report=QA_REPORT)
    if qa["status"] != "ok":
        print(f"QA review: {stem} -> {qa}")


def plot_baseline_overview(stats: dict) -> None:
    panels = [
        ("reward", "系统奖励", "越高越好", 1.0, "up"),
        ("latency", "时延 (10^6)", "越低越好", 1e-6, "down"),
        ("energy", "能耗 (10^6 J)", "越低越好", 1e-6, "down"),
        ("fairness_final", "公平性", "越高越好", 1.0, "up"),
        ("offline_rate_final", "离线率 (%)", "越低越好", 100.0, "down"),
        ("dsr_request_weighted", "请求加权DSR", "越高越好", 1.0, "up"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    x = np.arange(len(METHODS))
    labels = [METHOD_LABELS[m] for m in METHODS]

    for ax, (metric, title, subtitle, scale, direction) in zip(axes.ravel(), panels):
        means, lows, highs = [], [], []
        for method in METHODS:
            mean, low, high = metric_stats(stats, method, metric, scale)
            means.append(mean)
            lows.append(low)
            highs.append(high)

        bars = ax.bar(
            x,
            means,
            yerr=np.vstack([lows, highs]),
            capsize=2.5,
            color=[COLORS[m] for m in METHODS],
            edgecolor="#333333",
            linewidth=0.4,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        best_idx = int(np.argmax(means) if direction == "up" else np.argmin(means))
        bars[best_idx].set_edgecolor("#111111")
        bars[best_idx].set_linewidth(1.4)
        ax.set_title(title, fontsize=9, pad=3)
        ax.text(0.02, 0.92, subtitle, transform=ax.transAxes, fontsize=6.5, color="#555555")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        if metric == "offline_rate_final":
            ax.set_ylim(0, max(means + highs) * 1.35 + 0.02)

    save_figure(fig, "图5-14_扩展baseline矩阵多指标对比")
    plt.close(fig)


def plot_reliability_offloading(stats: dict) -> None:
    fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0])
    ax_stack = fig.add_subplot(gs[:, 0])
    ax_mbs = fig.add_subplot(gs[0, 1])
    ax_quality = fig.add_subplot(gs[1, 1])

    x = np.arange(len(METHODS))
    labels = [METHOD_LABELS[m] for m in METHODS]
    local = [metric_stats(stats, m, "offloading_ratio_local_processed", 100.0)[0] for m in METHODS]
    coop = [metric_stats(stats, m, "offloading_ratio_cooperative_processed", 100.0)[0] for m in METHODS]
    mbs = [metric_stats(stats, m, "offloading_ratio_mbs_processed", 100.0)[0] for m in METHODS]
    ax_stack.barh(x, local, color="#6baed6", label="本地")
    ax_stack.barh(x, coop, left=local, color="#74c476", label="协作")
    ax_stack.barh(x, mbs, left=np.array(local) + np.array(coop), color="#fdae6b", label="MBS")
    ax_stack.set_yticks(x)
    ax_stack.set_yticklabels(labels)
    ax_stack.invert_yaxis()
    ax_stack.set_xlabel("已处理服务请求 (%)")
    ax_stack.set_title("卸载分布", fontsize=9)
    ax_stack.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=7)
    ax_stack.grid(axis="x", color="#dddddd", linewidth=0.5, alpha=0.8)
    ax_stack.set_axisbelow(True)

    mbs_load = [metric_stats(stats, m, "mbs_load_ratio_generated", 100.0)[0] for m in METHODS]
    ax_mbs.bar(x, mbs_load, color=[COLORS[m] for m in METHODS], edgecolor="#333333", linewidth=0.4)
    ax_mbs.set_xticks(x)
    ax_mbs.set_xticklabels(labels, rotation=35, ha="right")
    ax_mbs.set_ylabel("MBS负载 (%)")
    ax_mbs.set_title("生成请求口径MBS负载", fontsize=9)
    ax_mbs.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.8)
    ax_mbs.set_axisbelow(True)

    processed = [metric_stats(stats, m, "processed_request_ratio", 100.0)[0] for m in METHODS]
    satisfied = [metric_stats(stats, m, "deadline_satisfied_per_processed", 100.0)[0] for m in METHODS]
    width = 0.36
    ax_quality.bar(x - width / 2, processed, width, color="#9ecae1", label="处理/生成")
    ax_quality.bar(x + width / 2, satisfied, width, color="#31a354", label="达期/处理")
    ax_quality.set_xticks(x)
    ax_quality.set_xticklabels(labels, rotation=35, ha="right")
    ax_quality.set_ylabel("比例 (%)")
    ax_quality.set_title("请求处理质量", fontsize=9)
    ax_quality.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.8)
    ax_quality.set_axisbelow(True)
    ax_quality.legend(fontsize=7, loc="upper right")

    save_figure(fig, "图5-15_可靠性与卸载机制分析")
    plt.close(fig)


def get_sensitivity(summary: dict, variable: str, method: str, metric: str, scale: float) -> tuple[float, float]:
    item = summary["variables"][variable][method]["stats"][metric]
    return float(item["mean"]) * scale, float(item["std"]) * scale


def plot_sensitivity_grid(summary: dict, variables: list[str], methods: list[str], stem: str, xlabel: str) -> None:
    panels = [
        ("energy", "能耗 (10^6 J)", 1e-6),
        ("latency", "时延 (10^6)", 1e-6),
        ("deadline_satisfaction_rate", "DSR", 1.0),
    ]
    markers = {"heuristic": "s", "vanilla_mappo": "o", "proposed": "D"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    x = np.array([float(v) for v in variables])
    for ax, (metric, ylabel, scale) in zip(axes, panels):
        for method in methods:
            means, stds = [], []
            for variable in variables:
                mean, std = get_sensitivity(summary, variable, method, metric, scale)
                means.append(mean)
                stds.append(std)
            ax.errorbar(
                x,
                means,
                yerr=stds,
                marker=markers[method],
                markersize=4.2,
                linewidth=1.6,
                capsize=2.2,
                color=COLORS[method],
                label=METHOD_LABELS[method].replace("\n", ""),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_title(ylabel.split(" (")[0], fontsize=9)
    axes[0].legend(fontsize=7, loc="best")
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    setup_style()
    stats = load_json(BASELINE_STATS)
    plot_baseline_overview(stats)
    plot_reliability_offloading(stats)

    ue_summary = load_json(UE_SUMMARY)
    ue_vars = sorted(ue_summary["variables"].keys(), key=lambda v: float(v))
    plot_sensitivity_grid(
        ue_summary,
        ue_vars,
        ["heuristic", "vanilla_mappo", "proposed"],
        "图5-16_地面节点数量敏感性三指标对比",
        "UE数量",
    )

    cpu_summary = load_json(CPU_SUMMARY)
    cpu_vars = sorted(cpu_summary["variables"].keys(), key=lambda v: float(v))
    plot_sensitivity_grid(
        cpu_summary,
        cpu_vars,
        ["heuristic", "vanilla_mappo", "proposed"],
        "图5-17_UAV算力敏感性三指标对比",
        "UAV算力缩放",
    )


if __name__ == "__main__":
    main()
