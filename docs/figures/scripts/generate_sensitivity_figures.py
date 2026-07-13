#!/usr/bin/env python3
"""Generate force-admission sensitivity figures for Chapter 5.

The formal sensitivity figures use the force-admission evaluation snapshot and
show DSR, effective energy efficiency, and fairness together.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis_figure_style import PALETTE, save_pub as save_pub_shared, setup_thesis_style

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRS = (ROOT / "docs" / "figures", ROOT / "latex" / "docs" / "figures")
QA_REPORT = ROOT / "docs" / "figures" / "figure_text_qa_report.json"
SENSITIVITY_DIR = ROOT / "results" / "sensitivity_force_admission"

ALGO_COLORS = {
    "proposed": PALETTE["full"],
    "vanilla_mappo": PALETTE["upper"],
    "heuristic": PALETTE["baseline"],
}
ALGO_LABELS = {"proposed": "本文方法", "vanilla_mappo": "普通MAPPO", "heuristic": "启发式"}
ALGO_MARKERS = {"proposed": "o", "vanilla_mappo": "s", "heuristic": "^"}
PLOT_ALGOS = ("heuristic", "vanilla_mappo", "proposed")
METRICS = (
    ("deadline_satisfaction_rate", "DSR ↑", "截止期满足率", True),
    ("energy_efficiency", "EEE ↑", "有效能效", False),
    ("fairness", "Fairness ↑", "服务公平性", True),
)


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


def load_summary(data_path: Path) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sensitivity summary: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def variable_key(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def metric_series(
    data: dict,
    variable_values: list[float | int],
    algo_name: str,
    metric_name: str,
) -> tuple[list[float], list[float], list[float]]:
    means, lows, highs = [], [], []
    for value in variable_values:
        stats = (
            data["variables"]
            .get(variable_key(value), {})
            .get(algo_name, {})
            .get("stats", {})
            .get(metric_name, {})
        )
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 0.0))
        means.append(mean)
        lows.append(mean - std)
        highs.append(mean + std)
    return means, lows, highs


def plot_metric_panel(
    ax: plt.Axes,
    data: dict,
    variable_values: list[float | int],
    metric_name: str,
    ylabel: str,
    title: str,
    bounded: bool,
) -> None:
    for algo_name in PLOT_ALGOS:
        means, lows, highs = metric_series(data, variable_values, algo_name, metric_name)
        if bounded:
            lows = [max(0.0, v) for v in lows]
            highs = [min(1.0, v) for v in highs]
        ax.plot(
            variable_values,
            means,
            label=ALGO_LABELS[algo_name],
            color=ALGO_COLORS[algo_name],
            marker=ALGO_MARKERS[algo_name],
            linewidth=1.45,
            markersize=4,
        )
        ax.fill_between(variable_values, lows, highs, color=ALGO_COLORS[algo_name], alpha=0.12, linewidth=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5, pad=3)
    if bounded:
        ax.set_ylim(0, 1.03)
    soften(ax)


def fig_ue_count_sensitivity(data_path: Path) -> None:
    data = load_summary(data_path)
    ue_counts = sorted(int(v) for v in data.get("variables", {}))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharex=True)
    for ax, (metric_name, ylabel, title, bounded) in zip(axes, METRICS):
        plot_metric_panel(ax, data, ue_counts, metric_name, ylabel, title, bounded)
        ax.set_xlabel("UE数量")
        ax.set_xticks(ue_counts)
    axes[0].legend(loc="lower left", fontsize=5.7, ncol=1)
    fig.tight_layout(w_pad=0.7)
    save_pub(fig, "图5-16_地面节点数量敏感性三指标对比")


def fig_uav_cpu_sensitivity(data_path: Path) -> None:
    data = load_summary(data_path)
    scales = sorted(float(v) for v in data.get("variables", {}))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharex=True)
    for ax, (metric_name, ylabel, title, bounded) in zip(axes, METRICS):
        plot_metric_panel(ax, data, scales, metric_name, ylabel, title, bounded)
        ax.set_xlabel("UAV算力缩放")
        ax.set_xticks(scales)
    axes[0].legend(loc="lower left", fontsize=5.7, ncol=1)
    fig.tight_layout(w_pad=0.7)
    save_pub(fig, "图5-17_UAV算力敏感性三指标对比")


def main() -> None:
    setup_style()
    fig_ue_count_sensitivity(SENSITIVITY_DIR / "ue_count" / "summary.json")
    fig_uav_cpu_sensitivity(SENSITIVITY_DIR / "uav_cpu_scale" / "summary.json")
    print("\nForce-admission sensitivity figures generated.")


if __name__ == "__main__":
    main()
