#!/usr/bin/env python3
"""Generate Chapter 5 figures from the force-admission evaluation snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis_figure_style import PALETTE, save_pub as save_pub_shared, setup_thesis_style


ROOT = Path(__file__).resolve().parents[3]
STATS_PATH = ROOT / "results" / "final_force_admission" / "evaluation_20260630" / "variant_statistics.json"
OUTPUT_DIRS = (ROOT / "docs" / "figures", ROOT / "latex" / "docs" / "figures")
QA_REPORT = ROOT / "docs" / "figures" / "figure_text_qa_report.json"

MAIN_VARIANTS = ["main_upper_only", "main_lower_only_fixed_upper", "main_full_hierarchical"]
MAIN_LABELS = ["上层+启发式", "固定上层+下层", "完整双层"]
MAIN_COLORS = [PALETTE["upper"], PALETTE["lower"], PALETTE["full"]]

ABLATION_VARIANTS = ["ablation_full", "ablation_no_mask", "ablation_no_lagrange", "ablation_no_attention"]
ABLATION_LABELS = ["完整下层", "无Mask", "无Lagrange", "无Attention"]
ABLATION_COLORS = [PALETTE["full"], "#D8A448", "#C75E5A", "#7C70B2"]

DSR_VARIANTS = ["main_full_hierarchical", "dsr_priority_full"]
DSR_LABELS = ["完整双层", "DSR优先"]
DSR_COLORS = [PALETTE["full"], "#D8A448"]


def setup_style() -> None:
    setup_thesis_style(font_size=7, legend_frameon=False)


def load_stats() -> dict:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Missing force-admission statistics: {STATS_PATH}")
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))


def metric(stats: dict, variant: str, key: str, scale: float = 1.0) -> tuple[float, float]:
    item = stats["methods"][variant]["metrics"][key]
    return item["mean"] * scale, item.get("std", 0.0) * scale


def values(stats: dict, variants: list[str], key: str, scale: float = 1.0) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for variant in variants:
        mean, std = metric(stats, variant, key, scale)
        means.append(mean)
        stds.append(std)
    return means, stds


def save_pub(fig: plt.Figure, name: str) -> None:
    qa = save_pub_shared(fig, name, OUTPUT_DIRS, qa_report=QA_REPORT)
    if qa["status"] != "ok":
        print(f"QA review: {name} -> {qa}")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.01, 0.98, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


def soften(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.7)


def bar_panel(
    ax: plt.Axes,
    means: list[float],
    stds: list[float] | None,
    title: str,
    ylabel: str,
    label: str,
    *,
    colors: list[str],
    ticklabels: list[str],
    zero_line: bool = False,
    direction: str = "up",
) -> None:
    x = np.arange(len(means))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=2.5 if stds else 0,
        color=colors,
        edgecolor="#333333",
        linewidth=0.45,
        error_kw={"elinewidth": 0.8, "capthick": 0.8} if stds else None,
    )
    best_idx = int(np.argmax(means) if direction == "up" else np.argmin(means))
    bars[best_idx].set_edgecolor("#111111")
    bars[best_idx].set_linewidth(1.4)
    if zero_line:
        ax.axhline(0, color=PALETTE["text"], linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, rotation=28, ha="right")
    ax.set_title(title, fontsize=9, pad=3)
    ax.text(0.02, 0.92, "越高越好" if direction == "up" else "越低越好", transform=ax.transAxes, fontsize=6.5, color="#555555")
    ax.set_ylabel(ylabel)
    panel_label(ax, label)
    soften(ax)


def fig_main_comparison(stats: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65))
    panels = [
        ("reward", 1.0, "系统奖励", "回合均值", "a", "up", True),
        ("energy", 1e-6, "UAV侧能耗", r"$10^6$ J", "b", "down", False),
        ("fairness_final", 1.0, "服务公平性", "Jain指数", "c", "up", False),
        ("dsr_request_weighted", 1.0, "截止期满足率", "DSR", "d", "up", False),
        ("mbs_load_ratio_generated", 100.0, "MBS负载", "%", "e", "down", False),
        ("forced_admission_ratio", 100.0, "兜底接入比例", "%", "f", "down", False),
    ]
    for ax, (key, scale, title, ylabel, label, direction, zero_line) in zip(axes.flat, panels):
        means, stds = values(stats, MAIN_VARIANTS, key, scale)
        bar_panel(ax, means, stds, title, ylabel, label, colors=MAIN_COLORS, ticklabels=MAIN_LABELS, direction=direction, zero_line=zero_line)
    axes[0, 1].annotate("最低能耗", xy=(2, metric(stats, "main_full_hierarchical", "energy", 1e-6)[0]), xytext=(1.25, 66), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    axes[1, 0].annotate("DSR权衡", xy=(2, metric(stats, "main_full_hierarchical", "dsr_request_weighted")[0]), xytext=(1.15, 0.44), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7, color=PALETTE["accent"])
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(w_pad=1.0, h_pad=1.35)
    save_pub(fig, "图5-7_主实验多指标对比")


def fig_offloading_distribution(stats: dict) -> None:
    x = np.arange(len(MAIN_VARIANTS))
    local, _ = values(stats, MAIN_VARIANTS, "offloading_ratio_local_processed", 100.0)
    coop, _ = values(stats, MAIN_VARIANTS, "offloading_ratio_cooperative_processed", 100.0)
    mbs, _ = values(stats, MAIN_VARIANTS, "offloading_ratio_mbs_processed", 100.0)
    forced, _ = values(stats, MAIN_VARIANTS, "forced_admission_ratio", 100.0)
    natural, _ = values(stats, MAIN_VARIANTS, "natural_coverage_service_ratio", 100.0)
    local = np.array(local)
    coop = np.array(coop)
    mbs = np.array(mbs)
    natural = np.array(natural)
    forced = np.array(forced)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"width_ratios": [1.35, 1.0]})
    axes[0].bar(x, local, color="#6388C5", edgecolor="#333333", linewidth=0.4, label="本地执行")
    axes[0].bar(x, coop, bottom=local, color="#62A979", edgecolor="#333333", linewidth=0.4, label="协作执行")
    axes[0].bar(x, mbs, bottom=local + coop, color="#D8766C", edgecolor="#333333", linewidth=0.4, label="MBS")
    for i, value in enumerate(mbs):
        axes[0].text(i, min(local[i] + coop[i] + value + 1.5, 106), f"{value:.1f}", ha="center", fontsize=7, color=PALETTE["text"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MAIN_LABELS, rotation=28, ha="right")
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("卸载决策占比 (%)")
    axes[0].set_title("执行位置分布", fontsize=9, pad=3)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    panel_label(axes[0], "a")
    soften(axes[0])

    axes[1].bar(x, natural, color="#78A7C8", edgecolor="#333333", linewidth=0.4, label="自然覆盖")
    axes[1].bar(x, forced, bottom=natural, color="#E0B36A", edgecolor="#333333", linewidth=0.4, label="兜底接入")
    for i, value in enumerate(forced):
        axes[1].text(i, natural[i] + value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MAIN_LABELS, rotation=28, ha="right")
    axes[1].set_ylim(0, 106)
    axes[1].set_ylabel("服务请求占比 (%)")
    axes[1].set_title("接入来源分布", fontsize=9, pad=3)
    axes[1].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    panel_label(axes[1], "b")
    soften(axes[1])
    fig.tight_layout(w_pad=1.15)
    save_pub(fig, "图5-8_卸载分布对比")


def fig_ablation_comparison(stats: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55))
    panels = [
        ("reward", 1.0, "系统奖励", "回合均值", "a", "up", True),
        ("energy", 1e-6, "能耗", r"$10^6$ J", "b", "down", False),
        ("dsr_request_weighted", 1.0, "截止期满足率", "DSR", "c", "up", False),
        ("fairness_final", 1.0, "服务公平性", "Jain指数", "d", "up", False),
        ("mbs_load_ratio_generated", 100.0, "MBS负载", "%", "e", "down", False),
        ("offloading_ratio_cooperative_processed", 100.0, "协作执行", "%", "f", "up", False),
    ]
    for ax, (key, scale, title, ylabel, label, direction, zero_line) in zip(axes.flat, panels):
        means, stds = values(stats, ABLATION_VARIANTS, key, scale)
        bar_panel(ax, means, stds, title, ylabel, label, colors=ABLATION_COLORS, ticklabels=ABLATION_LABELS, direction=direction, zero_line=zero_line)
    axes[1, 1].annotate("MBS显著升高", xy=(2, metric(stats, "ablation_no_lagrange", "mbs_load_ratio_generated", 100.0)[0]), xytext=(1.1, 28), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    fig.tight_layout(w_pad=1.0, h_pad=1.35)
    save_pub(fig, "图5-9_下层消融实验对比")


def fig_dsr_priority_tradeoff(stats: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.65))
    panels = [
        ("reward", 1.0, "系统奖励", "回合均值", "a", "up", True),
        ("energy", 1e-6, "能耗", r"$10^6$ J", "b", "down", False),
        ("dsr_request_weighted", 1.0, "截止期满足率", "DSR", "c", "up", False),
        ("mbs_load_ratio_generated", 100.0, "MBS负载", "%", "d", "down", False),
    ]
    for ax, (key, scale, title, ylabel, label, direction, zero_line) in zip(axes.flat, panels):
        means, stds = values(stats, DSR_VARIANTS, key, scale)
        bar_panel(ax, means, stds, title, ylabel, label, colors=DSR_COLORS, ticklabels=DSR_LABELS, direction=direction, zero_line=zero_line)
    axes[2].annotate("未稳定提升", xy=(1, metric(stats, "dsr_priority_full", "dsr_request_weighted")[0]), xytext=(0.15, 0.39), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7, color=PALETTE["accent"])
    axes[3].annotate("依赖MBS", xy=(1, metric(stats, "dsr_priority_full", "mbs_load_ratio_generated", 100.0)[0]), xytext=(0.05, 18), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    fig.tight_layout(w_pad=1.05)
    save_pub(fig, "图5-10_DSR优先配置性能权衡")


def main() -> None:
    setup_style()
    stats = load_stats()
    fig_main_comparison(stats)
    fig_offloading_distribution(stats)
    fig_ablation_comparison(stats)
    fig_dsr_priority_tradeoff(stats)


if __name__ == "__main__":
    main()
