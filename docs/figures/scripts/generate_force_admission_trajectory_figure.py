#!/usr/bin/env python3
"""Generate a force-admission trajectory figure for the proposed method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from thesis_figure_style import PALETTE, save_pub as save_pub_shared, setup_thesis_style

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from marl_models.utils import get_model
from utils.baseline_metrics import run_single_episode, set_global_seed

RESULT_ROOT = ROOT / "results" / "baseline_matrix_force_admission_20260626_resume"
SUMMARY_PATH = RESULT_ROOT / "proposed" / "seed_42" / "training_summary.json"
TRAJECTORY_JSON = RESULT_ROOT / "proposed" / "seed_42" / "trajectory_workload_42_force_admission.json"
DOC_FIG_DIR = ROOT / "docs" / "figures"
LATEX_FIG_DIR = ROOT / "latex" / "docs" / "figures"
OUTPUT_DIRS = (DOC_FIG_DIR, LATEX_FIG_DIR)
QA_REPORT = ROOT / "docs" / "figures" / "figure_text_qa_report.json"
OUTPUT_STEM = "图3-5_本文方法无人机二维轨迹_force_admission"
UAV_COLORS = ["#4C78A8", "#C75E5A", "#54A24B", "#E2A64A", "#7C70B2"]


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_models(summary: dict[str, Any]) -> tuple[Any, Any]:
    trajectory_model_name = str(summary.get("trajectory_model_name", "attention_mappo"))
    trajectory_model = get_model(trajectory_model_name)
    trajectory_model.load(str(summary["trajectory_model_dir"]))

    offload_model = get_model("constrained_attention_offload_mappo")
    offload_model.load(str(summary["offload_model_dir"]))
    return trajectory_model, offload_model


def run_force_admission_episode(
    summary_path: Path,
    workload_seed: int,
    eval_episode_idx: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    config.FORCE_SERVICE_ADMISSION = True
    set_global_seed(workload_seed)
    run_seed = int(workload_seed + eval_episode_idx * 1000)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    summary = load_summary(summary_path)
    trajectory_model, offload_model = load_models(summary)
    env = Env()
    metrics, trajectory = run_single_episode(
        env,
        trajectory_model,
        offload_model=offload_model,
        exploration=False,
        record_trajectory=True,
        policy_type="learned",
    )
    if trajectory is None:
        raise RuntimeError("Trajectory recording returned None.")
    return metrics, trajectory


def trajectory_positions(trajectory: list[dict[str, Any]]) -> np.ndarray:
    positions = [frame["uav_positions"] for frame in trajectory]
    return np.asarray(positions, dtype=float)


def audit_trajectory(positions: np.ndarray) -> dict[str, float]:
    diffs = np.diff(positions, axis=0)
    step_distances = np.linalg.norm(diffs, axis=2)
    total_distance_m = float(step_distances.sum())
    max_step_m = float(step_distances.max()) if step_distances.size else 0.0

    min_pair_m = float("inf")
    below_min_sep_count = 0
    for step_positions in positions:
        for i in range(step_positions.shape[0]):
            for j in range(i + 1, step_positions.shape[0]):
                dist = float(np.linalg.norm(step_positions[i] - step_positions[j]))
                min_pair_m = min(min_pair_m, dist)
                if dist < 200.0:
                    below_min_sep_count += 1

    edge_mask = (
        (positions[:, :, 0] <= 60.0)
        | (positions[:, :, 0] >= 640.0)
        | (positions[:, :, 1] <= 60.0)
        | (positions[:, :, 1] >= 640.0)
    )
    edge_ratio = float(edge_mask.mean())
    return {
        "steps": float(positions.shape[0]),
        "num_uavs": float(positions.shape[1]),
        "total_distance_m": total_distance_m,
        "total_distance_km": total_distance_m / 1000.0,
        "max_step_m": max_step_m,
        "min_pair_m": 0.0 if not np.isfinite(min_pair_m) else min_pair_m,
        "below_200m_pair_count": float(below_min_sep_count),
        "edge_position_ratio": edge_ratio,
    }


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


def draw_trajectory(positions: np.ndarray, audit: dict[str, float], output_stem: str) -> dict[str, object]:
    setup_thesis_style(font_size=8, legend_frameon=False)
    fig, ax = plt.subplots(figsize=(4.95, 4.35), dpi=300)

    for uav_idx in range(positions.shape[1]):
        xy = positions[:, uav_idx, :]
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
        label="MBS",
        zorder=6,
    )

    legend_handles = [
        Line2D([0], [0], color="#333333", lw=1.35, label="飞行轨迹"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", markeredgewidth=1.1, lw=0, markersize=5, label="起点"),
        Line2D([0], [0], marker="s", color="#333333", markerfacecolor="#333333", lw=0, markersize=5, label="终点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111", markeredgecolor="white", markeredgewidth=0.5, markersize=9, label="MBS"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        borderpad=0.25,
        handlelength=1.6,
        labelspacing=0.28,
        columnspacing=0.8,
    )
    ax.text(
        0.03,
        0.04,
        f"累计飞行距离：{audit['total_distance_km']:.1f} km",
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
        zorder=7,
    )

    ax.set_title("本文方法（force-admission）", fontsize=9, loc="left", pad=4)
    ax.set_xlabel("x方向位置 / m", fontsize=8)
    ax.set_ylabel("y方向位置 / m", fontsize=8)
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.78)
    ax.tick_params(labelsize=7.5, length=2.5, width=0.7)

    return save_pub_shared(fig, output_stem, OUTPUT_DIRS, qa_report=QA_REPORT)


def save_trajectory_json(
    path: Path,
    metrics: dict[str, float],
    trajectory: list[dict[str, Any]],
    audit: dict[str, float],
    workload_seed: int,
    eval_episode_idx: int,
) -> None:
    payload = {
        "method": "proposed",
        "setting": "force_service_admission",
        "training_seed": 42,
        "workload_seed": int(workload_seed),
        "eval_episode_idx": int(eval_episode_idx),
        "metrics": metrics,
        "trajectory_audit": audit,
        "trajectory": trajectory,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_outputs(metrics: dict[str, float], positions: np.ndarray) -> None:
    if positions.shape != (config.STEPS_PER_EPISODE, config.NUM_UAVS, 2):
        raise ValueError(f"Unexpected trajectory shape: {positions.shape}")
    processed_ratio = float(metrics.get("processed_request_ratio", 0.0))
    if not np.isclose(processed_ratio, 1.0, atol=1e-9):
        raise ValueError(f"Expected processed_request_ratio=1.0, got {processed_ratio:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--trajectory-output", type=Path, default=TRAJECTORY_JSON)
    parser.add_argument("--output-stem", type=str, default=OUTPUT_STEM)
    parser.add_argument("--workload-seed", type=int, default=42)
    parser.add_argument("--eval-episode-idx", type=int, default=0)
    args = parser.parse_args()

    metrics, trajectory = run_force_admission_episode(args.summary, args.workload_seed, args.eval_episode_idx)
    positions = trajectory_positions(trajectory)
    validate_outputs(metrics, positions)
    audit = audit_trajectory(positions)
    save_trajectory_json(args.trajectory_output, metrics, trajectory, audit, args.workload_seed, args.eval_episode_idx)
    qa = draw_trajectory(positions, audit, args.output_stem)

    print(f"Trajectory JSON: {args.trajectory_output}")
    print(f"Figure stem: {args.output_stem}")
    print(
        "Audit: "
        f"steps={int(audit['steps'])}, "
        f"num_uavs={int(audit['num_uavs'])}, "
        f"processed_request_ratio={metrics['processed_request_ratio']:.4f}, "
        f"distance={audit['total_distance_km']:.3f} km, "
        f"max_step={audit['max_step_m']:.3f} m, "
        f"min_pair={audit['min_pair_m']:.3f} m, "
        f"below_200m_pairs={int(audit['below_200m_pair_count'])}, "
        f"edge_ratio={audit['edge_position_ratio']:.3f}"
    )
    if qa["status"] != "ok":
        print(f"QA review: {qa}")


if __name__ == "__main__":
    main()
