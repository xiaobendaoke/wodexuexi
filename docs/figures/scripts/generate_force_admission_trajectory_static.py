#!/usr/bin/env python3
"""Generate static Figure 5-6 trajectory panels from force-admission rollouts."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from generate_force_admission_trajectory_gifs import (  # noqa: E402
    AREA_HEIGHT,
    AREA_WIDTH,
    LATEX_FIG_DIR,
    MBS_POS,
    METHODS,
    RESULT_ROOT,
    UAV_COLORS,
    audit_positions,
    load_or_generate_trajectories,
)
from thesis_figure_style import PALETTE, setup_thesis_style  # noqa: E402

DOC_FIG_DIR = ROOT / "docs" / "figures"
FORMATS = ("pdf", "svg", "png", "tiff")


def configure_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(0, AREA_WIDTH)
    ax.set_ylim(0, AREA_HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9.0, loc="left", pad=4)
    ax.set_xlabel("x方向位置 / m")
    ax.set_ylabel("y方向位置 / m")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.55, alpha=0.78)
    ax.tick_params(labelsize=7.5, length=2.5, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_direction_arrows(ax: plt.Axes, xy: np.ndarray, color: str) -> None:
    if xy.shape[0] < 5:
        return
    indices = np.linspace(80, xy.shape[0] - 1, 4, dtype=int)
    for idx in indices:
        prev_idx = max(0, idx - 28)
        start = xy[prev_idx]
        end = xy[idx]
        if np.linalg.norm(end - start) < 1e-6:
            continue
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8, "alpha": 0.72, "shrinkA": 0, "shrinkB": 0},
            zorder=4,
        )


def plot_method(title: str, positions: np.ndarray, audit: dict[str, float]) -> plt.Figure:
    setup_thesis_style(font_size=8, legend_frameon=False)
    fig, ax = plt.subplots(figsize=(3.55, 3.55), constrained_layout=True)
    configure_axes(ax, title)
    for uav_idx in range(positions.shape[1]):
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        xy = positions[:, uav_idx, :]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.45, alpha=0.94, zorder=3)
        draw_direction_arrows(ax, xy, color)
        ax.scatter(xy[0, 0], xy[0, 1], s=42, marker="o", facecolors="white", edgecolors=color, linewidths=1.1, zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=42, marker="s", facecolors=color, edgecolors="white", linewidths=0.6, zorder=5)
    ax.scatter([MBS_POS[0]], [MBS_POS[1]], marker="*", s=125, c="#111111", edgecolors="white", linewidths=0.6, zorder=7)
    ax.text(
        0.03,
        0.04,
        f"累计飞行距离：{audit['total_distance_km']:.1f} km",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#D5D9E2", "linewidth": 0.55, "alpha": 0.92},
        zorder=8,
    )
    return fig


def save_all(fig: plt.Figure, stem: str) -> list[str]:
    saved: list[str] = []
    for out_dir in (DOC_FIG_DIR, LATEX_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in FORMATS:
            path = out_dir / f"{stem}.{ext}"
            kwargs = {"bbox_inches": "tight"}
            if ext in {"png", "tiff"}:
                kwargs.update({"dpi": 450})
            fig.savefig(path, **kwargs)
            saved.append(str(path.relative_to(ROOT)))
    return saved


def main() -> None:
    data = load_or_generate_trajectories(training_seed=42, workload_seed=42, eval_episode_idx=0, regenerate=False)
    manifest = {
        "setting": "force_service_admission",
        "result_root": str(RESULT_ROOT.relative_to(ROOT)),
        "training_seed": 42,
        "workload_seed": 42,
        "eval_episode_idx": 0,
        "figures": [],
    }
    for method, title, force_stem in METHODS:
        metrics, positions, audit = data[method]
        audit = audit or audit_positions(positions)
        original_stem = force_stem.replace("_force_admission", "")
        fig = plot_method(title, positions, audit)
        saved = save_all(fig, original_stem)
        saved.extend(save_all(fig, force_stem))
        plt.close(fig)
        manifest["figures"].append(
            {
                "method": method,
                "title": title,
                "original_stem": original_stem,
                "force_admission_stem": force_stem,
                "processed_request_ratio": float(metrics.get("processed_request_ratio", 0.0)),
                "dsr_request_weighted": float(metrics.get("dsr_request_weighted", metrics.get("deadline_satisfaction_rate", 0.0))),
                "trajectory_audit": audit,
                "saved": saved,
            }
        )
    manifest_path = DOC_FIG_DIR / "force_admission_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(manifest_path, LATEX_FIG_DIR / manifest_path.name)
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "figures": len(manifest["figures"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
