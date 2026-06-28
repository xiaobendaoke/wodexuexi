#!/usr/bin/env python3
"""Generate force-admission trajectory GIFs for the four Figure 5-6 methods."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from environment.env import Env
from marl_models.utils import get_model
from thesis_figure_style import PALETTE, setup_thesis_style
from utils.baseline_metrics import run_single_episode, set_global_seed

RESULT_ROOT = ROOT / "results" / "baseline_matrix_force_admission_20260626_resume"
DOC_FIG_DIR = ROOT / "docs" / "figures"
LATEX_FIG_DIR = ROOT / "latex" / "docs" / "figures"

AREA_WIDTH = 700
AREA_HEIGHT = 700
MBS_POS = np.array([350.0, 350.0])
UAV_COLORS = ["#4C78A8", "#C75E5A", "#54A24B", "#E2A64A", "#7C70B2"]

METHODS = [
    ("random", "随机策略", "图5-6a_随机策略无人机二维轨迹_force_admission"),
    ("ippo", "IPPO", "图5-6b_IPPO无人机二维轨迹_force_admission"),
    ("vanilla_mappo", "Vanilla MAPPO", "图5-6c_VanillaMAPPO无人机二维轨迹_force_admission"),
    ("proposed", "本文方法", "图5-6d_本文方法无人机二维轨迹_force_admission"),
]


def set_eval_seed(workload_seed: int, eval_episode_idx: int) -> None:
    config.FORCE_SERVICE_ADMISSION = True
    set_global_seed(workload_seed)
    run_seed = int(workload_seed + eval_episode_idx * 1000)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)


def load_summary(method: str, training_seed: int) -> dict[str, Any]:
    path = RESULT_ROOT / method / f"seed_{training_seed}" / "training_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def model_dir(path: str) -> str:
    return str((ROOT / path).resolve()) if not Path(path).is_absolute() else path


def load_learning_models(method: str, training_seed: int) -> tuple[Any, Any]:
    if method == "ippo":
        summary = load_summary(method, training_seed)
        timestamp = str(summary["timestamp"])
        trajectory_model = get_model("ippo_baseline")
        trajectory_model.load(str(ROOT / "saved_models" / f"ippo_baseline_{timestamp}" / "final"))
        offload_model = get_model("constrained_attention_offload_mappo")
        offload_model.load(str(ROOT / "saved_models" / f"offload_mappo_{timestamp}" / "final"))
        return trajectory_model, offload_model

    summary = load_summary(method, training_seed)
    trajectory_model = get_model(str(summary["trajectory_model_name"]))
    trajectory_model.load(model_dir(str(summary["trajectory_model_dir"])))
    offload_model = get_model("constrained_attention_offload_mappo")
    offload_model.load(model_dir(str(summary["offload_model_dir"])))
    return trajectory_model, offload_model


def run_method_episode(method: str, training_seed: int, workload_seed: int, eval_episode_idx: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    set_eval_seed(workload_seed, eval_episode_idx)
    env = Env()
    if method in {"random", "uniform"}:
        trajectory_model = get_model(f"{method}_baseline")
        metrics, trajectory = run_single_episode(
            env,
            trajectory_model,
            offload_model=None,
            exploration=False,
            record_trajectory=True,
            policy_type="non_learning",
        )
    else:
        trajectory_model, offload_model = load_learning_models(method, training_seed)
        metrics, trajectory = run_single_episode(
            env,
            trajectory_model,
            offload_model=offload_model,
            exploration=False,
            record_trajectory=True,
            policy_type="learned",
        )
    if trajectory is None:
        raise RuntimeError(f"Trajectory recording returned None for {method}.")
    return metrics, trajectory


def positions_from_trajectory(trajectory: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([frame["uav_positions"] for frame in trajectory], dtype=float)


def audit_positions(positions: np.ndarray) -> dict[str, float]:
    step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    min_pair = float("inf")
    below_200 = 0
    for step_positions in positions:
        for i in range(step_positions.shape[0]):
            for j in range(i + 1, step_positions.shape[0]):
                dist = float(np.linalg.norm(step_positions[i] - step_positions[j]))
                min_pair = min(min_pair, dist)
                if dist < 200.0:
                    below_200 += 1
    edge = (
        (positions[:, :, 0] <= 60.0)
        | (positions[:, :, 0] >= 640.0)
        | (positions[:, :, 1] <= 60.0)
        | (positions[:, :, 1] >= 640.0)
    )
    return {
        "steps": float(positions.shape[0]),
        "num_uavs": float(positions.shape[1]),
        "total_distance_km": float(step_distances.sum() / 1000.0),
        "max_step_m": float(step_distances.max()) if step_distances.size else 0.0,
        "min_pair_m": 0.0 if not np.isfinite(min_pair) else min_pair,
        "below_200m_pair_count": float(below_200),
        "edge_position_ratio": float(edge.mean()),
    }


def save_trajectory(method: str, training_seed: int, workload_seed: int, eval_episode_idx: int, metrics: dict[str, float], trajectory: list[dict[str, Any]], audit: dict[str, float]) -> Path:
    if method in {"random", "uniform"}:
        out_dir = RESULT_ROOT / method
    else:
        out_dir = RESULT_ROOT / method / f"seed_{training_seed}"
    path = out_dir / f"trajectory_workload_{workload_seed}_force_admission.json"
    payload = {
        "method": method,
        "setting": "force_service_admission",
        "training_seed": None if method in {"random", "uniform"} else int(training_seed),
        "workload_seed": int(workload_seed),
        "eval_episode_idx": int(eval_episode_idx),
        "metrics": metrics,
        "trajectory_audit": audit,
        "trajectory": trajectory,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_or_generate_trajectories(training_seed: int, workload_seed: int, eval_episode_idx: int, regenerate: bool) -> dict[str, tuple[dict[str, float], np.ndarray, dict[str, float]]]:
    out: dict[str, tuple[dict[str, float], np.ndarray, dict[str, float]]] = {}
    for method, _, _ in METHODS:
        if method in {"random", "uniform"}:
            path = RESULT_ROOT / method / f"trajectory_workload_{workload_seed}_force_admission.json"
        else:
            path = RESULT_ROOT / method / f"seed_{training_seed}" / f"trajectory_workload_{workload_seed}_force_admission.json"
        if path.exists() and not regenerate:
            payload = json.loads(path.read_text(encoding="utf-8"))
            trajectory = payload["trajectory"]
            metrics = payload["metrics"]
            positions = positions_from_trajectory(trajectory)
            audit = payload.get("trajectory_audit") or audit_positions(positions)
        else:
            metrics, trajectory = run_method_episode(method, training_seed, workload_seed, eval_episode_idx)
            positions = positions_from_trajectory(trajectory)
            validate_method_output(method, metrics, positions)
            audit = audit_positions(positions)
            path = save_trajectory(method, training_seed, workload_seed, eval_episode_idx, metrics, trajectory, audit)
            print(f"Generated trajectory JSON: {path}")
        out[method] = (metrics, positions, audit)
    return out


def validate_method_output(method: str, metrics: dict[str, float], positions: np.ndarray) -> None:
    if positions.shape != (config.STEPS_PER_EPISODE, config.NUM_UAVS, 2):
        raise ValueError(f"Unexpected trajectory shape for {method}: {positions.shape}")
    processed_ratio = float(metrics.get("processed_request_ratio", 0.0))
    if not np.isclose(processed_ratio, 1.0, atol=1e-9):
        raise ValueError(f"{method} expected processed_request_ratio=1.0, got {processed_ratio:.6f}")


def configure_axes(ax: plt.Axes, title: str, compact: bool) -> None:
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


def init_panel(ax: plt.Axes, positions: np.ndarray, audit: dict[str, float], title: str, compact: bool) -> tuple[list[Line2D], list[Line2D]]:
    configure_axes(ax, title, compact)
    lines: list[Line2D] = []
    points: list[Line2D] = []
    for uav_idx in range(positions.shape[1]):
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        (line,) = ax.plot([], [], color=color, lw=1.35 if compact else 1.55, alpha=0.95, zorder=3)
        (point,) = ax.plot([], [], marker="o", markersize=5.6 if compact else 7.2, color=color, markeredgecolor="white", markeredgewidth=0.65, lw=0, zorder=6)
        lines.append(line)
        points.append(point)
        ax.scatter(positions[0, uav_idx, 0], positions[0, uav_idx, 1], s=30 if compact else 42, marker="o", facecolors="white", edgecolors=color, linewidths=1.1, zorder=5)
        ax.scatter(positions[-1, uav_idx, 0], positions[-1, uav_idx, 1], s=30 if compact else 42, marker="s", facecolors=color, edgecolors="white", linewidths=0.6, zorder=5)
    ax.scatter([MBS_POS[0]], [MBS_POS[1]], marker="*", s=85 if compact else 125, c="#111111", edgecolors="white", linewidths=0.6, zorder=7)
    ax.text(
        0.03,
        0.04,
        f"累计飞行距离：{audit['total_distance_km']:.1f} km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5 if compact else 7,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#D5D9E2", "linewidth": 0.55, "alpha": 0.92},
        zorder=8,
    )
    return lines, points


def update_panel(positions: np.ndarray, lines: list[Line2D], points: list[Line2D], step_idx: int) -> None:
    for uav_idx in range(positions.shape[1]):
        xy = positions[: step_idx + 1, uav_idx, :]
        lines[uav_idx].set_data(xy[:, 0], xy[:, 1])
        points[uav_idx].set_data([xy[-1, 0]], [xy[-1, 1]])


def copy_to_latex(path: Path) -> None:
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, LATEX_FIG_DIR / path.name)


def build_single_gif(title: str, stem: str, positions: np.ndarray, audit: dict[str, float], frames: int, fps: int) -> Path:
    setup_thesis_style(font_size=8, legend_frameon=False)
    frame_indices = np.linspace(0, positions.shape[0] - 1, frames, dtype=int)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    lines, points = init_panel(ax, positions, audit, f"{title} UAV 轨迹动画（force-admission）", compact=False)
    ax.legend(handles=legend_handles(), loc="upper right", fontsize=7, borderpad=0.25, handlelength=1.6, labelspacing=0.28)
    step_text = ax.text(0.03, 0.965, "", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=PALETTE["text"])

    def update(frame_no: int) -> list[object]:
        step_idx = int(frame_indices[frame_no])
        update_panel(positions, lines, points, step_idx)
        step_text.set_text(f"step {step_idx + 1}/{positions.shape[0]}")
        return [*lines, *points, step_text]

    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = DOC_FIG_DIR / f"{stem}.gif"
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    copy_to_latex(output)
    print(f"Generated {output}")
    return output


def build_combined_gif(data: dict[str, tuple[dict[str, float], np.ndarray, dict[str, float]]], frames: int, fps: int) -> Path:
    setup_thesis_style(font_size=8, legend_frameon=False)
    total_steps = next(iter(data.values()))[1].shape[0]
    frame_indices = np.linspace(0, total_steps - 1, frames, dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.5), dpi=100)
    artists_by_method: dict[str, tuple[list[Line2D], list[Line2D]]] = {}
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label, (method, title, _) in zip(axes.flat, panel_labels, METHODS):
        _, positions, audit = data[method]
        artists_by_method[method] = init_panel(ax, positions, audit, f"{label} {title}", compact=True)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.015), handlelength=1.7, columnspacing=1.2)
    fig.suptitle("force-admission 设置下不同算法的 UAV 二维轨迹动画", fontsize=10, x=0.5, y=0.985)
    step_text = fig.text(0.5, 0.045, "", ha="center", va="bottom", fontsize=7, color=PALETTE["text"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.085, wspace=0.18, hspace=0.24)

    def update(frame_no: int) -> list[object]:
        step_idx = int(frame_indices[frame_no])
        artists: list[object] = []
        for method, (_, positions, _) in data.items():
            lines, points = artists_by_method[method]
            update_panel(positions, lines, points, step_idx)
            artists.extend(lines)
            artists.extend(points)
        step_text.set_text(f"step {step_idx + 1}/{total_steps}")
        artists.append(step_text)
        return artists

    output = DOC_FIG_DIR / "图5-6_不同算法无人机二维轨迹动画对比_force_admission.gif"
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    copy_to_latex(output)
    print(f"Generated {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--workload-seed", type=int, default=42)
    parser.add_argument("--eval-episode-idx", type=int, default=0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--skip-individual", action="store_true")
    args = parser.parse_args()

    data = load_or_generate_trajectories(args.training_seed, args.workload_seed, args.eval_episode_idx, args.regenerate)
    if not args.skip_individual:
        for method, title, stem in METHODS:
            _, positions, audit = data[method]
            build_single_gif(title, stem, positions, audit, args.frames, args.fps)
    build_combined_gif(data, args.frames, args.fps)
    print("Audit summary:")
    for method, _, _ in METHODS:
        metrics, _, audit = data[method]
        print(
            f"{method}: processed={metrics['processed_request_ratio']:.4f}, "
            f"distance={audit['total_distance_km']:.3f} km, "
            f"max_step={audit['max_step_m']:.3f} m, "
            f"min_pair={audit['min_pair_m']:.3f} m, "
            f"below_200m_pairs={int(audit['below_200m_pair_count'])}, "
            f"edge_ratio={audit['edge_position_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
