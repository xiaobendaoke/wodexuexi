from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import config
RESULT_ROOT = ROOT / "results" / "baseline_comparison"
DOC_FIG_DIR = ROOT / "docs" / "figures"
LATEX_FIG_DIR = ROOT / "latex" / "docs" / "figures"

METHODS = [
    ("random", "随机策略", "图5-6a_随机策略无人机二维轨迹"),
    ("ippo", "IPPO", "图5-6b_IPPO无人机二维轨迹"),
    ("vanilla_mappo", "Vanilla MAPPO", "图5-6c_VanillaMAPPO无人机二维轨迹"),
    ("proposed", "本文方法", "图5-6d_本文方法无人机二维轨迹"),
]

UAV_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "WenQuanYi Micro Hei",
                "Noto Sans CJK SC",
                "SimHei",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": True,
        }
    )


def load_first_episode_positions(method: str) -> np.ndarray:
    path = RESULT_ROOT / method / "trajectory_seed_42_workload_42.json"
    episodes = json.loads(path.read_text(encoding="utf-8"))
    first_episode = episodes[0]["trajectory"]
    positions = [step["uav_positions"] for step in first_episode]
    return np.asarray(positions, dtype=float)


def add_direction_arrows(ax: plt.Axes, xy: np.ndarray, color: str) -> None:
    if len(xy) < 8:
        return
    arrow_indices = np.linspace(3, len(xy) - 2, 3, dtype=int)
    for idx in arrow_indices:
        start = xy[idx - 1]
        end = xy[idx]
        delta = end - start
        if np.linalg.norm(delta) < 1e-6:
            continue
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 8,
                "alpha": 0.9,
            },
        )


def draw_method(method: str, title: str, output_stem: str) -> None:
    trajectories = load_first_episode_positions(method)
    total_path_length = float(np.linalg.norm(np.diff(trajectories, axis=0), axis=2).sum())
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=300)

    for uav_idx in range(trajectories.shape[1]):
        xy = trajectories[:, uav_idx, :]
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.35, alpha=0.95, zorder=3)
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            s=42,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=4,
        )
        ax.scatter(
            xy[-1, 0],
            xy[-1, 1],
            s=42,
            marker="s",
            facecolors=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )
        add_direction_arrows(ax, xy, color)

    ax.scatter(
        [config.MBS_POS[0]],
        [config.MBS_POS[1]],
        marker="*",
        s=110,
        c="#111111",
        edgecolors="white",
        linewidths=0.6,
        label="宏基站",
        zorder=6,
    )

    legend_handles = [
        Line2D([0], [0], color="#333333", lw=1.35, label="飞行轨迹"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", markeredgewidth=1.1, lw=0, markersize=5, label="起点"),
        Line2D([0], [0], marker="s", color="#333333", markerfacecolor="#333333", lw=0, markersize=5, label="终点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111", markeredgecolor="white", markeredgewidth=0.5, markersize=9, label="宏基站"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.92,
        borderpad=0.5,
        handlelength=1.8,
        labelspacing=0.35,
    )
    ax.text(
        0.03,
        0.04,
        f"累计飞行距离：{total_path_length / 1000:.1f} km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#b5b5b5",
            "linewidth": 0.6,
            "alpha": 0.92,
        },
        zorder=7,
    )

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("x 方向位置 / m", fontsize=9)
    ax.set_ylabel("y 方向位置 / m", fontsize=9)
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=8)

    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = DOC_FIG_DIR / f"{output_stem}.png"
    pdf_path = DOC_FIG_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    shutil.copy2(png_path, LATEX_FIG_DIR / png_path.name)
    shutil.copy2(pdf_path, LATEX_FIG_DIR / pdf_path.name)


def main() -> None:
    configure_matplotlib()
    for method, title, output_stem in METHODS:
        draw_method(method, title, output_stem)


if __name__ == "__main__":
    main()
