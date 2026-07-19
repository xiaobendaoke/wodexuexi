#!/usr/bin/env python3
"""Generate Figure 3.3: main-configuration performance trade-off."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "docs" / "figures" / "data" / "figure_3_3_main_configuration.json"
OUTPUT_DIRS = (ROOT / "docs" / "figures", ROOT / "latex" / "docs" / "figures")
OUTPUT_NAME = "图3-3_主配置性能工作点"

COLORS = {
    "upper_heuristic": "#7884B4",
    "fixed_upper_lower": "#767676",
    "full_hierarchical": "#0F4D92",
}
MARKERS = {
    "upper_heuristic": "o",
    "fixed_upper_lower": "D",
    "full_hierarchical": "s",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "PingFang SC",
                "Hiragino Sans GB",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "Arial",
                "DejaVu Sans",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#30343B",
            "axes.labelcolor": "#30343B",
            "xtick.color": "#30343B",
            "ytick.color": "#30343B",
            "legend.frameon": False,
        }
    )


def load_data() -> tuple[list[dict], str]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return payload["configurations"], payload["unit_note"]


def metric_values(configurations: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([item[key]["mean"] for item in configurations], dtype=float)
    stds = np.array([item[key]["std"] for item in configurations], dtype=float)
    return means, stds


def style_axis(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(axis=axis, color="#D9DDE5", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom")


def plot_tradeoff(ax: plt.Axes, configurations: list[dict]) -> None:
    energy, energy_std = metric_values(configurations, "energy_mj")
    fairness, fairness_std = metric_values(configurations, "fairness")
    label_offsets = [(1.5, 0.016), (1.2, -0.034), (-16.0, 0.016)]

    for item, x, xerr, y, yerr, offset in zip(
        configurations, energy, energy_std, fairness, fairness_std, label_offsets
    ):
        color = COLORS[item["id"]]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=MARKERS[item["id"]],
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7.5,
            capsize=2.5,
            elinewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            item["label"],
            xy=(x, y),
            xytext=(x + offset[0], y + offset[1]),
            fontsize=7.2,
            color=color,
            fontweight="bold" if item["id"] == "full_hierarchical" else "normal",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
            va="center",
        )

    ax.set_xlim(38, 90)
    ax.set_ylim(0.64, 1.00)
    ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_yticks([0.65, 0.75, 0.85, 0.95])
    ax.set_xlabel("UAV侧能耗 ($10^6$ J)")
    ax.set_ylabel("服务公平性 (Jain 指数)")
    ax.set_title("能耗--公平性工作点", fontsize=9, pad=5)
    style_axis(ax)
    add_panel_label(ax, "a")


def plot_constraint_dotplot(
    ax: plt.Axes,
    configurations: list[dict],
    metric_key: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    value_format: str,
    panel_label: str,
) -> None:
    means, stds = metric_values(configurations, metric_key)
    y = np.arange(len(configurations))[::-1]
    for item, yi, mean, std in zip(configurations, y, means, stds):
        color = COLORS[item["id"]]
        ax.errorbar(
            mean,
            yi,
            xerr=std,
            fmt=MARKERS[item["id"]],
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.75,
            markersize=6.5,
            capsize=2.4,
            elinewidth=1.0,
            zorder=3,
        )
        ax.text(min(mean + std + (xlim[1] - xlim[0]) * 0.03, xlim[1] * 0.94), yi, value_format.format(mean), va="center", fontsize=6.8)

    ax.set_yticks(y)
    ax.set_yticklabels([item["label"] for item in configurations], fontsize=6.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9, pad=5)
    style_axis(ax, axis="x")
    add_panel_label(ax, panel_label)


def build_figure() -> plt.Figure:
    configurations, note = load_data()
    fig = plt.figure(figsize=(7.2, 3.15))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1.0], hspace=0.78, wspace=0.72)

    tradeoff_ax = fig.add_subplot(grid[:, :2])
    dsr_ax = fig.add_subplot(grid[0, 2])
    mbs_ax = fig.add_subplot(grid[1, 2])

    plot_tradeoff(tradeoff_ax, configurations)
    plot_constraint_dotplot(
        dsr_ax,
        configurations,
        "dsr",
        "截止期满足率",
        "DSR",
        (0.25, 0.60),
        "{:.3f}",
        "b",
    )
    plot_constraint_dotplot(
        mbs_ax,
        configurations,
        "mbs_load_pct",
        "MBS 负载",
        "占生成请求比例 (%)",
        (0.0, 10.5),
        "{:.2f}%",
        "c",
    )
    fig.text(0.50, 0.01, "误差线表示均值 ± 标准差，N=30。", ha="center", va="bottom", fontsize=6.7, color="#555B66")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.16)
    return fig


def main() -> None:
    apply_style()
    fig = build_figure()
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix, dpi in (("svg", None), ("pdf", None), ("png", 450), ("tiff", 600)):
            fig.savefig(output_dir / f"{OUTPUT_NAME}.{suffix}", bbox_inches="tight", pad_inches=0.03, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
