#!/usr/bin/env python3
"""Generate the learning-rate convergence figure for Chapter 5.

The figure uses cumulative average reward to present the convergence trend in
the traditional thesis style used by many DRL papers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis_figure_style import PALETTE, save_pub as save_pub_shared, setup_thesis_style


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "results" / "learning_rate_sensitivity_fixed" / "proposed"
OUTPUT_DIRS = (ROOT / "docs" / "figures", ROOT / "latex" / "docs" / "figures")
QA_REPORT = ROOT / "docs" / "figures" / "figure_text_qa_report.json"
FIG_NAME = "图5-11_不同学习率下训练收敛曲线"

LR_ORDER = ["1e-4", "3e-4", "5e-4", "1e-3"]
LR_LABELS = {
    "1e-4": r"学习率 $\alpha=1\times10^{-4}$",
    "3e-4": r"学习率 $\alpha=3\times10^{-4}$",
    "5e-4": r"学习率 $\alpha=5\times10^{-4}$",
    "1e-3": r"学习率 $\alpha=1\times10^{-3}$",
}
COLORS = {
    "1e-4": "#4C78A8",
    "3e-4": "#54A24B",
    "5e-4": "#C75E5A",
    "1e-3": "#717784",
}
LINESTYLES = {
    "1e-4": "-",
    "3e-4": "-.",
    "5e-4": "--",
    "1e-3": ":",
}


def setup_style() -> None:
    setup_thesis_style(font_size=8, legend_frameon=False)


def cumulative_average(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def load_curve(lr: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    path = DATA_ROOT / f"lr_{lr}" / "proposed" / "training_curve_seed_42.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = np.array([row["episode"] for row in data], dtype=float)
    rewards = np.array([row["reward"] for row in data], dtype=float)
    if len(data) != 200 or episodes.tolist() != list(range(1, 201)):
        raise ValueError(f"{path} is not a complete 200-episode curve")
    stats = {
        "last_reward": float(rewards[-1]),
        "last20_mean": float(rewards[-20:].mean()),
        "last20_std": float(rewards[-20:].std(ddof=0)),
        "cumulative_final": float(cumulative_average(rewards)[-1]),
    }
    return episodes, rewards, stats


def save_figure(fig: plt.Figure) -> None:
    qa = save_pub_shared(fig, FIG_NAME, OUTPUT_DIRS, qa_report=QA_REPORT)
    if qa["status"] != "ok":
        print(f"QA review: {FIG_NAME} -> {qa}")


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.1))

    all_stats: dict[str, dict[str, float]] = {}
    for lr in LR_ORDER:
        episodes, rewards, stats = load_curve(lr)
        all_stats[lr] = stats
        cumulative_rewards = cumulative_average(rewards) / 1000.0
        ax.plot(
            episodes,
            cumulative_rewards,
            color=COLORS[lr],
            lw=1.35 if lr == "5e-4" else 1.15,
            linestyle=LINESTYLES[lr],
            label=LR_LABELS[lr],
            alpha=0.96,
        )

    ax.set_xlim(0, 200)
    ax.set_ylim(-8.2, -1.6)
    ax.set_xticks(np.arange(0, 201, 20))
    ax.set_yticks(np.arange(-8, -1, 1))
    ax.set_xlabel("训练回合", labelpad=7)
    ax.set_ylabel(r"平均累计奖励（$\times10^3$）", labelpad=7)
    ax.grid(True, which="major", color=PALETTE["grid"], linewidth=0.55, alpha=0.78)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.5,
        width=0.7,
        top=False,
        right=False,
    )
    ax.legend(loc="lower right", fontsize=7, handlelength=2.5, borderpad=0.4)

    fig.tight_layout()
    save_figure(fig)

    for lr in LR_ORDER:
        stats = all_stats[lr]
        print(
            f"{lr}: last={stats['last_reward']:.3f}, "
            f"last20_mean={stats['last20_mean']:.3f}, "
            f"last20_std={stats['last20_std']:.3f}, "
            f"cumulative_final={stats['cumulative_final']:.3f}"
        )


if __name__ == "__main__":
    main()
