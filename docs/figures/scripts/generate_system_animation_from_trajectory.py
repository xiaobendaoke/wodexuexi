#!/usr/bin/env python3
"""Generate the system animation from the proposed rollout trajectory."""

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

TRAJECTORY_JSON = (
    ROOT
    / "results"
    / "baseline_comparison"
    / "proposed"
    / "trajectory_seed_42_workload_42.json"
)
OUTPUT_GIF = ROOT / "docs" / "figures" / "fig_system_animation.gif"
RANDOM_DEMO_BACKUP = ROOT / "docs" / "figures" / "fig_system_animation_random_demo.gif"

AREA_WIDTH = 700
AREA_HEIGHT = 700
MBS_POS = np.array([350.0, 350.0])
UAV_COLORS = ["#4C78A8", "#C75E5A", "#54A24B", "#E2A64A", "#7C70B2"]
EXPECTED_FIRST = np.array(
    [
        [405.9, 104.0],
        [506.2, 593.6],
        [152.0, 55.6],
        [50.0, 438.0],
        [414.6, 393.5],
    ]
)
EXPECTED_LAST = np.array(
    [
        [650.0, 50.0],
        [390.1, 320.2],
        [420.4, 53.7],
        [650.0, 383.2],
        [490.9, 628.4],
    ]
)


def load_proposed_trajectory(path: Path) -> np.ndarray:
    episodes = json.loads(path.read_text(encoding="utf-8"))
    first_episode = episodes[0]["trajectory"]
    positions = [step["uav_positions"] for step in first_episode]
    return np.asarray(positions, dtype=float)


def configure_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, AREA_WIDTH)
    ax.set_ylim(0, AREA_HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x方向位置 / m", fontsize=8)
    ax.set_ylabel("y方向位置 / m", fontsize=8)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.55, alpha=0.78)
    ax.tick_params(labelsize=7.5, length=2.5, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_animation(trajectory: np.ndarray, output_path: Path, frames: int, fps: int) -> None:
    setup_thesis_style(font_size=8, legend_frameon=False)
    total_steps = trajectory.shape[0]
    frame_indices = np.linspace(0, total_steps - 1, frames, dtype=int)
    total_path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=2).sum())

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    configure_axes(ax)

    lines: list[Line2D] = []
    current_points: list[Line2D] = []
    for uav_idx in range(trajectory.shape[1]):
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        (line,) = ax.plot([], [], color=color, lw=1.55, alpha=0.95, zorder=3)
        (point,) = ax.plot(
            [],
            [],
            marker="o",
            markersize=7.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            lw=0,
            zorder=6,
        )
        lines.append(line)
        current_points.append(point)

    starts = trajectory[0]
    ends = trajectory[-1]
    for uav_idx, color in enumerate(UAV_COLORS[: trajectory.shape[1]]):
        ax.scatter(
            starts[uav_idx, 0],
            starts[uav_idx, 1],
            s=42,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=5,
        )
        ax.scatter(
            ends[uav_idx, 0],
            ends[uav_idx, 1],
            s=42,
            marker="s",
            facecolors=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )

    ax.scatter(
        [MBS_POS[0]],
        [MBS_POS[1]],
        marker="*",
        s=125,
        c="#111111",
        edgecolors="white",
        linewidths=0.6,
        zorder=7,
    )

    legend_handles = [
        Line2D([0], [0], color="#333333", lw=1.45, label="飞行轨迹"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#333333",
            markerfacecolor="white",
            markeredgewidth=1.1,
            lw=0,
            markersize=5,
            label="起点",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="#333333",
            markerfacecolor="#333333",
            lw=0,
            markersize=5,
            label="终点",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=9,
            label="MBS",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        borderpad=0.25,
        handlelength=1.6,
        labelspacing=0.28,
    )

    distance_text = ax.text(
        0.03,
        0.04,
        f"累计飞行距离：{total_path_length / 1000:.1f} km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#D5D9E2",
            "linewidth": 0.6,
            "alpha": 0.92,
        },
        zorder=8,
    )
    title = ax.set_title("", fontsize=9, loc="left", pad=4)

    def update(frame_no: int) -> list[object]:
        step_idx = int(frame_indices[frame_no])
        for uav_idx in range(trajectory.shape[1]):
            xy = trajectory[: step_idx + 1, uav_idx, :]
            lines[uav_idx].set_data(xy[:, 0], xy[:, 1])
            current_points[uav_idx].set_data([xy[-1, 0]], [xy[-1, 1]])
        title.set_text(f"本文方法 UAV 轨迹动画  |  step {step_idx + 1}/{total_steps}")
        return [*lines, *current_points, title, distance_text]

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def validate_trajectory(trajectory: np.ndarray) -> None:
    first = np.round(trajectory[0], 1)
    last = np.round(trajectory[-1], 1)
    total_path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=2).sum())
    if not np.allclose(first, EXPECTED_FIRST, atol=0.1):
        raise ValueError(f"Unexpected first positions: {first.tolist()}")
    if not np.allclose(last, EXPECTED_LAST, atol=0.1):
        raise ValueError(f"Unexpected last positions: {last.tolist()}")
    if not np.isclose(total_path_length / 1000, 39.7, atol=0.05):
        raise ValueError(f"Unexpected total path length: {total_path_length / 1000:.3f} km")


def maybe_backup_existing(output_path: Path, backup_path: Path) -> None:
    if output_path.exists() and not backup_path.exists():
        shutil.copy2(output_path, backup_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120, help="Number of GIF frames.")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second.")
    parser.add_argument("--output", type=Path, default=OUTPUT_GIF, help="Output GIF path.")
    args = parser.parse_args()

    trajectory = load_proposed_trajectory(TRAJECTORY_JSON)
    validate_trajectory(trajectory)
    maybe_backup_existing(args.output, RANDOM_DEMO_BACKUP)
    build_animation(trajectory, args.output, frames=args.frames, fps=args.fps)
    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
