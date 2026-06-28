#!/usr/bin/env python3
"""Generate per-method and 2x2 comparison GIFs from saved trajectory JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from thesis_figure_style import PALETTE, setup_thesis_style

RESULT_ROOT = ROOT / "results" / "baseline_comparison"
DOC_FIG_DIR = ROOT / "docs" / "figures"
LATEX_FIG_DIR = ROOT / "latex" / "docs" / "figures"

AREA_WIDTH = 700
AREA_HEIGHT = 700
MBS_POS = np.array([350.0, 350.0])
UAV_COLORS = ["#4C78A8", "#C75E5A", "#54A24B", "#E2A64A", "#7C70B2"]

METHODS = [
    ("random", "随机策略", "图5-6a_随机策略无人机二维轨迹"),
    ("ippo", "IPPO", "图5-6b_IPPO无人机二维轨迹"),
    ("vanilla_mappo", "Vanilla MAPPO", "图5-6c_VanillaMAPPO无人机二维轨迹"),
    ("proposed", "本文方法", "图5-6d_本文方法无人机二维轨迹"),
]


def load_trajectory(method: str) -> np.ndarray:
    path = RESULT_ROOT / method / "trajectory_seed_42_workload_42.json"
    episodes = json.loads(path.read_text(encoding="utf-8"))
    first_episode = episodes[0]["trajectory"]
    positions = [step["uav_positions"] for step in first_episode]
    return np.asarray(positions, dtype=float)


def load_all_trajectories() -> dict[str, np.ndarray]:
    return {method: load_trajectory(method) for method, _, _ in METHODS}


def total_path_km(trajectory: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=2).sum() / 1000.0)


def configure_axes(ax: plt.Axes, *, title: str, compact: bool = False) -> None:
    ax.set_xlim(0, AREA_WIDTH)
    ax.set_ylim(0, AREA_HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8.5 if compact else 9, loc="left", pad=4)
    ax.set_xlabel("x方向位置 / m", fontsize=7 if compact else 8)
    ax.set_ylabel("y方向位置 / m", fontsize=7 if compact else 8)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.78)
    ax.tick_params(labelsize=6.8 if compact else 7.5, length=2.5, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#333333", lw=1.35, label="飞行轨迹"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", markeredgewidth=1.1, lw=0, markersize=5, label="起点"),
        Line2D([0], [0], marker="s", color="#333333", markerfacecolor="#333333", lw=0, markersize=5, label="终点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111", markeredgecolor="white", markeredgewidth=0.5, markersize=9, label="MBS"),
    ]


def init_panel(ax: plt.Axes, trajectory: np.ndarray, title: str, *, compact: bool = False) -> tuple[list[Line2D], list[Line2D]]:
    configure_axes(ax, title=title, compact=compact)
    lines: list[Line2D] = []
    points: list[Line2D] = []
    for uav_idx in range(trajectory.shape[1]):
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        (line,) = ax.plot([], [], color=color, lw=1.35 if compact else 1.55, alpha=0.95, zorder=3)
        (point,) = ax.plot(
            [],
            [],
            marker="o",
            markersize=5.6 if compact else 7.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            lw=0,
            zorder=6,
        )
        lines.append(line)
        points.append(point)

    starts = trajectory[0]
    ends = trajectory[-1]
    for uav_idx, color in enumerate(UAV_COLORS[: trajectory.shape[1]]):
        ax.scatter(starts[uav_idx, 0], starts[uav_idx, 1], s=30 if compact else 42, marker="o", facecolors="white", edgecolors=color, linewidths=1.1, zorder=5)
        ax.scatter(ends[uav_idx, 0], ends[uav_idx, 1], s=30 if compact else 42, marker="s", facecolors=color, edgecolors="white", linewidths=0.6, zorder=5)

    ax.scatter([MBS_POS[0]], [MBS_POS[1]], marker="*", s=85 if compact else 125, c="#111111", edgecolors="white", linewidths=0.6, zorder=7)
    ax.text(
        0.03,
        0.04,
        f"累计飞行距离：{total_path_km(trajectory):.1f} km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5 if compact else 7,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D5D9E2",
            "linewidth": 0.55,
            "alpha": 0.92,
        },
        zorder=8,
    )
    return lines, points


def update_panel(trajectory: np.ndarray, lines: list[Line2D], points: list[Line2D], step_idx: int) -> None:
    for uav_idx in range(trajectory.shape[1]):
        xy = trajectory[: step_idx + 1, uav_idx, :]
        lines[uav_idx].set_data(xy[:, 0], xy[:, 1])
        points[uav_idx].set_data([xy[-1, 0]], [xy[-1, 1]])


def save_to_latex(gif_path: Path) -> None:
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gif_path, LATEX_FIG_DIR / gif_path.name)


def build_single_gif(method: str, title: str, output_stem: str, trajectory: np.ndarray, *, frames: int, fps: int) -> Path:
    setup_thesis_style(font_size=8, legend_frameon=False)
    total_steps = trajectory.shape[0]
    frame_indices = np.linspace(0, total_steps - 1, frames, dtype=int)

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    lines, points = init_panel(ax, trajectory, f"{title} UAV 轨迹动画", compact=False)
    ax.legend(handles=legend_handles(), loc="upper right", fontsize=7, borderpad=0.25, handlelength=1.6, labelspacing=0.28)
    step_text = ax.text(0.03, 0.965, "", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=PALETTE["text"])

    def update(frame_no: int) -> list[object]:
        step_idx = int(frame_indices[frame_no])
        update_panel(trajectory, lines, points, step_idx)
        step_text.set_text(f"step {step_idx + 1}/{total_steps}")
        return [*lines, *points, step_text]

    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = DOC_FIG_DIR / f"{output_stem}.gif"
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    save_to_latex(output)
    print(f"Generated {output}")
    return output


def build_combined_gif(trajectories: dict[str, np.ndarray], *, frames: int, fps: int) -> Path:
    setup_thesis_style(font_size=8, legend_frameon=False)
    total_steps = next(iter(trajectories.values())).shape[0]
    frame_indices = np.linspace(0, total_steps - 1, frames, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.5), dpi=100)
    panel_artists: dict[str, tuple[list[Line2D], list[Line2D]]] = {}
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label, (method, title, _) in zip(axes.flat, panel_labels, METHODS):
        panel_artists[method] = init_panel(ax, trajectories[method], f"{label} {title}", compact=True)

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=1.7,
        columnspacing=1.2,
    )
    fig.suptitle("固定 seed 示例下不同算法的 UAV 二维轨迹动画", fontsize=10, x=0.5, y=0.985)
    step_text = fig.text(0.5, 0.045, "", ha="center", va="bottom", fontsize=7, color=PALETTE["text"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.085, wspace=0.18, hspace=0.24)

    def update(frame_no: int) -> list[object]:
        step_idx = int(frame_indices[frame_no])
        artists: list[object] = []
        for method, trajectory in trajectories.items():
            lines, points = panel_artists[method]
            update_panel(trajectory, lines, points, step_idx)
            artists.extend(lines)
            artists.extend(points)
        step_text.set_text(f"step {step_idx + 1}/{total_steps}")
        artists.append(step_text)
        return artists

    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = DOC_FIG_DIR / "图5-6_不同算法无人机二维轨迹动画对比.gif"
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    save_to_latex(output)
    print(f"Generated {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--skip-individual", action="store_true")
    args = parser.parse_args()

    trajectories = load_all_trajectories()
    for method, trajectory in trajectories.items():
        if trajectory.shape[:2] != (1000, 5):
            raise ValueError(f"Unexpected trajectory shape for {method}: {trajectory.shape}")

    if not args.skip_individual:
        for method, title, output_stem in METHODS:
            build_single_gif(method, title, output_stem, trajectories[method], frames=args.frames, fps=args.fps)
    build_combined_gif(trajectories, frames=args.frames, fps=args.fps)


if __name__ == "__main__":
    main()
