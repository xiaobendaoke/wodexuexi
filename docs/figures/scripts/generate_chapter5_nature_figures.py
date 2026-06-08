#!/usr/bin/env python3
"""Generate publication-style Chapter 5 result figures from locked manuscript values."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRS = (ROOT / "docs" / "figures", ROOT / "latex" / "docs" / "figures")

PALETTE = {
    "baseline": "#717784",
    "upper": "#6F7DBE",
    "lower": "#53A6A6",
    "combined": "#E2A64A",
    "full": "#3D6FB6",
    "accent": "#C85A54",
    "green": "#3A8D5A",
    "grid": "#D5D9E2",
    "text": "#22252A",
}

METHODS = ["unco+heuristic", "att+heuristic", "unco+lower", "att+lower", "full hierarchical"]
SHORT = ["Baseline", "Upper", "Lower", "Stacked", "Full"]
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
ABLATION_SHORT = ["Full", "No mask", "No Lagrange", "No attention"]
ABLATION = {
    "reward": [-1599.4, -1556.3, -1585.9, -1626.7],
    "energy": [53.32, 48.26, 51.12, 54.01],
    "dsr": [0.2170, 0.1945, 0.1981, 0.2094],
    "local": [0.417, 0.410, 0.360, 0.482],
    "coop": [0.446, 0.410, 0.390, 0.344],
    "mbs_load": [0.137, 0.180, 0.250, 0.173],
}

DSR_CONFIGS = ["Baseline", "Energy-priority", "DSR-priority"]
DSR_DATA = {
    "reward": [-5933.9, -1451.7, -1147.2],
    "energy": [114.70, 58.06, 83.04],
    "dsr": [0.2734, 0.2083, 0.2435],
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 600,
        }
    )


def save_pub(fig: plt.Figure, name: str) -> None:
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png", "tiff"):
            fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


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
    bar_panel(axes[0, 0], MAIN["reward"], MAIN["reward_std"], "System reward", "Episode mean", "a", zero_line=True)
    bar_panel(axes[0, 1], MAIN["energy"], MAIN["energy_std"], "UAV-side energy", r"$10^6$ J", "b")
    bar_panel(axes[0, 2], MAIN["fairness"], MAIN["fairness_std"], "Service fairness", "Jain index", "c")
    bar_panel(axes[1, 0], MAIN["dsr"], MAIN["dsr_std"], "Deadline satisfaction", "DSR", "d")
    bar_panel(axes[1, 1], [v * 100 for v in MAIN["offline"]], None, "Offline users", "%", "e")
    bar_panel(axes[1, 2], [v * 100 for v in MAIN["mbs_load"]], None, "MBS load", "%", "f")
    axes[0, 0].annotate("+4482", xy=(4, MAIN["reward"][4]), xytext=(3.05, -3500), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    axes[0, 1].annotate("-49.4%", xy=(4, MAIN["energy"][4]), xytext=(3.0, 93), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    axes[1, 0].annotate("boundary", xy=(4, MAIN["dsr"][4]), xytext=(3.15, 0.25), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7, color=PALETTE["accent"])
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(w_pad=1.0, h_pad=1.35)
    save_pub(fig, "图5-7_主实验多指标对比")


def fig_offloading_distribution() -> None:
    x = np.arange(len(METHODS))
    local = np.array(MAIN["local"]) * 100
    coop = np.array(MAIN["coop"]) * 100
    mbs = np.array(MAIN["mbs_load"]) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, local, color="#6388C5", edgecolor="white", linewidth=0.6, label="Local")
    ax.bar(x, coop, bottom=local, color="#62A979", edgecolor="white", linewidth=0.6, label="Cooperative")
    ax.bar(x, mbs, bottom=local + coop, color="#D8766C", edgecolor="white", linewidth=0.6, label="MBS")
    for i, value in enumerate(mbs):
        ax.text(i, local[i] + coop[i] + value + 1.5, f"{value:.1f}", ha="center", fontsize=7, color=PALETTE["text"])
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=20, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Decision share (%)")
    ax.set_title("Lower-layer policies reallocate service requests", fontsize=8, loc="left", pad=4)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    soften(ax)
    save_pub(fig, "图5-8_卸载分布对比")


def fig_ablation_comparison() -> None:
    colors = [PALETTE["full"], "#D8A448", "#C75E5A", "#7C70B2"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.45))
    bar_panel(axes[0, 0], ABLATION["reward"], None, "Reward", "Episode mean", "a", colors=colors, zero_line=True)
    bar_panel(axes[0, 1], ABLATION["energy"], None, "Energy", r"$10^6$ J", "b", colors=colors)
    bar_panel(axes[0, 2], ABLATION["dsr"], None, "Deadline satisfaction", "DSR", "c", colors=colors)
    bar_panel(axes[1, 0], [v * 100 for v in ABLATION["mbs_load"]], None, "MBS load", "%", "d", colors=colors)
    bar_panel(axes[1, 1], [v * 100 for v in ABLATION["coop"]], None, "Cooperative offloading", "%", "e", colors=colors)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.82,
        "Ablation logic",
        fontsize=8,
        fontweight="bold",
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].text(
        0.02,
        0.63,
        "Mask constrains low-quality MBS use.\nLagrange most strongly suppresses\nMBS dependence. Attention preserves\ncooperative offloading flexibility.",
        fontsize=7,
        linespacing=1.45,
        transform=axes[1, 2].transAxes,
    )
    fig.tight_layout(w_pad=1.0, h_pad=1.35)
    save_pub(fig, "图5-9_下层消融实验对比")


def fig_dsr_priority_tradeoff() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    configs_colors = [PALETTE["baseline"], PALETTE["full"], PALETTE["green"]]
    bar_panel(axes[0], DSR_DATA["reward"], None, "Reward", "Episode mean", "a", colors=configs_colors, ticklabels=DSR_CONFIGS, zero_line=True)
    bar_panel(axes[1], DSR_DATA["energy"], None, "Energy", r"$10^6$ J", "b", colors=configs_colors, ticklabels=DSR_CONFIGS)
    bar_panel(axes[2], DSR_DATA["dsr"], None, "Deadline satisfaction", "DSR", "c", colors=configs_colors, ticklabels=DSR_CONFIGS)
    axes[1].annotate("-27.6%", xy=(2, DSR_DATA["energy"][2]), xytext=(1.25, 103), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
    axes[2].annotate("partial recovery", xy=(2, DSR_DATA["dsr"][2]), xytext=(0.85, 0.26), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=7)
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
