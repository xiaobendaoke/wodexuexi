#!/usr/bin/env python3
"""Generate publication-style Chapter 5 result figures from locked manuscript values."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from thesis_figure_style import PALETTE, save_pub as save_pub_shared, setup_thesis_style


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRS = (ROOT / "docs" / "figures", ROOT / "latex" / "docs" / "figures")
QA_REPORT = ROOT / "docs" / "figures" / "figure_text_qa_report.json"

METHODS = ["unco+heuristic", "att+heuristic", "unco+lower", "att+lower", "full hierarchical"]
SHORT = ["基线", "仅上层", "仅下层", "上下层", "完整"]
METHOD_COLORS = [PALETTE["baseline"], PALETTE["upper"], PALETTE["lower"], PALETTE["combined"], PALETTE["full"]]

MAIN = {
    "reward": [-5933.9, -2128.6, -5274.4, -1599.4, -1451.7],
    "reward_std": [195.6, 389.7, 141.3, 330.1, 373.7],
    "energy": [114.70, 111.81, 42.85, 53.32, 58.06],
    "energy_std": [15.4, 10.6, 13.6, 12.4, 7.0],
    "dsr": [0.2734, 0.2879, 0.1891, 0.2170, 0.2083],
    "dsr_std": [0.029, 0.032, 0.029, 0.035, 0.019],
    "fairness": [0.7745, 0.9299, 0.7737, 0.9281, 0.9363],
    "fairness_std": [0.077, 0.036, 0.080, 0.035, 0.041],
    "offline": [0.0058, 0.0010, 0.0053, 0.0011, 0.0000],
    "local": [0.471, 0.472, 0.526, 0.417, 0.464],
    "coop": [0.501, 0.500, 0.301, 0.446, 0.424],
    "mbs_load": [0.026, 0.027, 0.171, 0.137, 0.111],
}

ABLATION_METHODS = ["lower_full", "no_mask", "no_lagrange", "no_attention"]
ABLATION_SHORT = ["完整", "无Mask", "无Lagrange", "无Attention"]
ABLATION = {
    "reward": [-1599.4, -1556.3, -1585.9, -1626.7],
    "energy": [53.32, 48.26, 51.12, 54.01],
    "dsr": [0.2170, 0.1945, 0.1981, 0.2094],
    "local": [0.417, 0.410, 0.360, 0.482],
    "coop": [0.446, 0.410, 0.390, 0.344],
    "mbs_load": [0.137, 0.180, 0.250, 0.173],
}

DSR_CONFIGS = ["基线", "能耗优先", "DSR优先"]
DSR_DATA = {
    "reward": [-5933.9, -1451.7, -1147.2],
    "energy": [114.70, 58.06, 83.04],
    "dsr": [0.2734, 0.2083, 0.2435],
}


def setup_style() -> None:
    setup_thesis_style(font_size=7, legend_frameon=False)


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
    values: list[float],
    errors: list[float] | None,
    title: str,
    ylabel: str,
    label: str,
    *,
    colors: list[str] | None = None,
    ticklabels: list[str] | None = None,
    zero_line: bool = False,
) -> None:
    x = np.arange(len(values))
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=2 if errors else 0,
        color=colors or METHOD_COLORS,
        edgecolor="white",
        linewidth=0.6,
    )
    if zero_line:
        ax.axhline(0, color=PALETTE["text"], linewidth=0.7)
    ax.set_xticks(x)
    labels = ticklabels or (SHORT if len(values) == len(SHORT) else ABLATION_SHORT)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title, fontsize=8, loc="left", pad=4)
    ax.set_ylabel(ylabel)
    panel_label(ax, label)
    soften(ax)


def fig_main_comparison() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65))
    bar_panel(axes[0, 0], MAIN["reward"], MAIN["reward_std"], "系统奖励", "回合均值", "a", zero_line=True)
    bar_panel(axes[0, 1], MAIN["energy"], MAIN["energy_std"], "UAV侧能耗", r"$10^6$ J", "b")
    bar_panel(axes[0, 2], MAIN["fairness"], MAIN["fairness_std"], "服务公平性", "Jain指数", "c")
    bar_panel(axes[1, 0], MAIN["dsr"], MAIN["dsr_std"], "截止期满足率", "DSR", "d")
    bar_panel(axes[1, 1], [v * 100 for v in MAIN["offline"]], None, "离线用户比例", "%", "e")
    bar_panel(axes[1, 2], [v * 100 for v in MAIN["mbs_load"]], None, "MBS负载", "%", "f")
    axes[0, 0].annotate("+4482", xy=(4, MAIN["reward"][4]), xytext=(3.05, -3500), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    axes[0, 1].annotate("-49.4%", xy=(4, MAIN["energy"][4]), xytext=(3.0, 93), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    axes[1, 0].annotate("能耗边界", xy=(4, MAIN["dsr"][4]), xytext=(3.15, 0.25), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7, color=PALETTE["accent"])
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(w_pad=1.0, h_pad=1.35)
    save_pub(fig, "图5-7_主实验多指标对比")


def fig_offloading_distribution() -> None:
    x = np.arange(len(METHODS))
    local = np.array(MAIN["local"]) * 100
    coop = np.array(MAIN["coop"]) * 100
    mbs = np.array(MAIN["mbs_load"]) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, local, color="#6388C5", edgecolor="white", linewidth=0.6, label="本地执行")
    ax.bar(x, coop, bottom=local, color="#62A979", edgecolor="white", linewidth=0.6, label="协作执行")
    ax.bar(x, mbs, bottom=local + coop, color="#D8766C", edgecolor="white", linewidth=0.6, label="MBS")
    for i, value in enumerate(mbs):
        ax.text(i, local[i] + coop[i] + value + 1.5, f"{value:.1f}", ha="center", fontsize=7, color=PALETTE["text"])
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=20, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("卸载决策占比 (%)")
    ax.set_title("下层策略重分配服务请求", fontsize=8, loc="left", pad=4)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    soften(ax)
    save_pub(fig, "图5-8_卸载分布对比")


def fig_ablation_comparison() -> None:
    colors = [PALETTE["full"], "#D8A448", "#C75E5A", "#7C70B2"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.45))
    bar_panel(axes[0, 0], ABLATION["reward"], None, "系统奖励", "回合均值", "a", colors=colors, zero_line=True)
    bar_panel(axes[0, 1], ABLATION["energy"], None, "能耗", r"$10^6$ J", "b", colors=colors)
    bar_panel(axes[0, 2], ABLATION["dsr"], None, "截止期满足率", "DSR", "c", colors=colors)
    bar_panel(axes[1, 0], [v * 100 for v in ABLATION["mbs_load"]], None, "MBS负载", "%", "d", colors=colors)
    bar_panel(axes[1, 1], [v * 100 for v in ABLATION["coop"]], None, "协作卸载", "%", "e", colors=colors)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.88,
        "消融逻辑",
        fontsize=8,
        fontweight="bold",
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].text(
        0.02,
        0.55,
        "Mask限制低质量MBS使用。\nLagrange最明显抑制\nMBS依赖。Attention保持\n协作卸载灵活性。",
        fontsize=7,
        linespacing=1.35,
        transform=axes[1, 2].transAxes,
    )
    fig.tight_layout(w_pad=1.0, h_pad=1.35)
    save_pub(fig, "图5-9_下层消融实验对比")


def fig_dsr_priority_tradeoff() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    configs_colors = [PALETTE["baseline"], PALETTE["full"], PALETTE["green"]]
    bar_panel(axes[0], DSR_DATA["reward"], None, "系统奖励", "回合均值", "a", colors=configs_colors, ticklabels=DSR_CONFIGS, zero_line=True)
    bar_panel(axes[1], DSR_DATA["energy"], None, "能耗", r"$10^6$ J", "b", colors=configs_colors, ticklabels=DSR_CONFIGS)
    bar_panel(axes[2], DSR_DATA["dsr"], None, "截止期满足率", "DSR", "c", colors=configs_colors, ticklabels=DSR_CONFIGS)
    axes[1].annotate("-27.6%", xy=(2, DSR_DATA["energy"][2]), xytext=(1.25, 103), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    axes[2].annotate("部分恢复", xy=(2, DSR_DATA["dsr"][2]), xytext=(0.85, 0.26), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    fig.tight_layout(w_pad=1.25)
    save_pub(fig, "图5-10_DSR优先配置性能权衡")


def main() -> None:
    setup_style()
    fig_main_comparison()
    fig_offloading_distribution()
    fig_ablation_comparison()
    fig_dsr_priority_tradeoff()


if __name__ == "__main__":
    main()
