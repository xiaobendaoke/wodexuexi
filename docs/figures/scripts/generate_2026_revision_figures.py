#!/usr/bin/env python3
"""Generate the 2026-09-28 revision figures for chapters 3-5.

2026-10-02: 图3-3/图3-4 renamed to 图3-6/图3-7 (训练曲线增加奖励面板);
图4-1 moved to generate_chapter34_figures_20261002.py as the enhanced 图4-3.

Data sources are archived result files only; no experiment is rerun:
  - output/bottleneck_diagnostic_20260927/validation/bottleneck_review.json (main worktree)
  - results/canonical_all_20260920/chapter4/analysis.json (main worktree)
  - <coverage worktree>/results/coverage_observation_confirmation_20260927/training_curves.csv
  - <coverage worktree>/results/whole_system_comparison_20260928/{whole_system_paired_comparison.json,evaluation_units.csv}

Outputs PDF+PNG into docs/figures/ and latex/docs/figures/.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

OUTPUT_DIRS = [
    REPO_ROOT / "docs" / "figures",
    REPO_ROOT / "latex" / "docs" / "figures",
]

CANDIDATE = "lower_mlp_upper_uncovered_obs"
STRONG_BASELINE = "lower_no_attention"

LEARNED_METHODS = [
    (CANDIDATE, "本章候选"),
    (STRONG_BASELINE, "下层无注意力"),
    ("hierarchical", "双注意力分层"),
    ("vanilla_hierarchical", "Vanilla 分层"),
    ("ippo", "IPPO"),
    ("joint", "Joint MAPPO"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Songti SC", "SimSun", "Noto Serif CJK SC", "DejaVu Serif"],
            "font.size": 9,
            "axes.unicode_minus": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#D5D9E2",
            "grid.linewidth": 0.5,
            "figure.dpi": 150,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def fig_coverage_geometry() -> None:
    """Chap3: coverage radius vs sensing radius geometry (schematic, no data)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    uav_pos = np.array([350.0, 380.0])
    rng = np.random.default_rng(7)

    ax.add_patch(Circle(uav_pos, 460.0, fill=True, facecolor="#9EC4DF", alpha=0.25,
                        edgecolor="#3D6FB6", linestyle="--", linewidth=1.2))
    ax.add_patch(Circle(uav_pos, 100.0, fill=True, facecolor="#3D6FB6", alpha=0.35,
                        edgecolor="#3D6FB6", linewidth=1.2))

    # UEs: inside coverage (admitted), between 100-460 m (sensed, not admitted), outside.
    def scatter_ring(r_min: float, r_max: float, n: int, **kw) -> None:
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = np.sqrt(rng.uniform(r_min**2, r_max**2, n))
        pts = uav_pos + np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=22, zorder=3, **kw)

    scatter_ring(0, 95, 7, marker="o", color="#3A8D5A", label="已覆盖 UE（接纳）")
    scatter_ring(110, 450, 9, marker="s", color="#D8A448", label="感知范围内未覆盖 UE")
    scatter_ring(470, 620, 5, marker="x", color="#9AA0A6", label="感知范围外 UE")

    ax.plot(*uav_pos, marker="^", color="#C75E5A", markersize=13, zorder=4)
    ax.annotate("UAV", uav_pos + np.array([-78, -30]), fontsize=10, color="#C75E5A")
    ax.annotate("", xy=uav_pos + np.array([100, 0]), xytext=uav_pos,
                arrowprops=dict(arrowstyle="->", color="#3D6FB6", lw=1.0))
    ax.text(*(uav_pos + np.array([30, -26])), "$R_c=100$ m", fontsize=8.5, color="#1F4E79")
    ax.annotate("", xy=uav_pos + 460 * np.array([np.cos(2.4), np.sin(2.4)]), xytext=uav_pos,
                arrowprops=dict(arrowstyle="->", color="#3D6FB6", lw=1.0))
    ax.text(*(uav_pos + 300 * np.array([np.cos(2.45), np.sin(2.45)])), "$R_s=460$ m",
            fontsize=8.5, color="#1F4E79")

    ax.set_xlim(0, 700)
    ax.set_ylim(0, 700)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$y$ / m")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.5)
    save(fig, "图3-2_覆盖与感知几何示意图")


def fig_failure_decomposition() -> None:
    """Chap3: generated-request outcome decomposition from the bottleneck diagnostic."""
    review = json.loads(
        (REPO_ROOT / "output" / "bottleneck_diagnostic_20260927" / "validation" / "bottleneck_review.json").read_text()
    )
    fe = review["full_evaluation"]
    success = fe["success_fraction"]
    admitted_miss = fe["admitted_deadline_miss_fraction"]
    uncovered = fe["uncovered_fraction"]
    share = fe["fraction_of_failures_uncovered"]

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    cats = ["按时完成", "已接纳但超时", "未覆盖"]
    vals = [success, admitted_miss, uncovered]
    colors = ["#3A8D5A", "#D8A448", "#C75E5A"]
    left = 0.0
    for c, v, col in zip(cats, vals, colors):
        ax.barh([0], [v], left=left, color=col, edgecolor="white", height=0.55, label=c)
        ax.text(left + v / 2, 0, f"{v*100:.1f}%", ha="center", va="center", fontsize=9, color="white")
        left += v
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("占全部实际生成服务请求的比例")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8, frameon=False)
    ax.set_title(f"未覆盖请求占全部失败请求的 {share*100:.1f}%", fontsize=9)
    save(fig, "图3-6_主场景服务请求结果构成")


def fig_confirmation_curves(coverage_root: Path) -> None:
    """Chap3: confirmation training curves (DSR and mean reward), candidate vs strong baseline."""
    path = coverage_root / "results" / "coverage_observation_confirmation_20260927" / "training_curves.csv"
    series: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    rewards: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for row in csv.DictReader(f):
            series[row["method"]][int(row["training_seed"])].append(float(row["training_DSR"]))
            rewards[row["method"]][int(row["training_seed"])].append(float(row["mean_training_reward"]))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    styles = {CANDIDATE: ("#3D6FB6", "候选（扩展观测）"), STRONG_BASELINE: ("#717784", "下层无注意力基线")}
    for ax, src, ylabel, title in [
            (axes[0], series, "训练 DSR（每 episode）", "(a) 训练 DSR"),
            (axes[1], rewards, "平均训练奖励（每 episode）", "(b) 平均训练奖励")]:
        for method, (color, label) in styles.items():
            arr = np.array([src[method][s] for s in sorted(src[method])])
            x = np.arange(arr.shape[1])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(x, mean, color=color, lw=1.4, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
            ax.text(x[-1] + 3, mean[-1], f"{mean[-1]:.3f}", color=color, fontsize=8, va="center")
        ax.set_xlim(0, 235)
        ax.set_xlabel("训练 episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.6)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].text(0.0, -0.30, "实线：10 个训练种子的均值；阴影：种子间 ±1 SD（描述训练动态，不构成收敛性证明）",
                 transform=axes[0].transAxes, fontsize=7, color="#717784")
    save(fig, "图3-7_确认实验训练曲线")


def fig_per_seed_dsr_diff(coverage_root: Path) -> None:
    """Chap5: per-training-seed DSR difference (candidate - strong baseline)."""
    path = coverage_root / "results" / "whole_system_comparison_20260928" / "evaluation_units.csv"
    dsr: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for row in csv.DictReader(f):
            if not row["training_seed"]:
                continue  # non-learned references have no training seed
            dsr[row["method"]][int(row["training_seed"])].append(float(row["DSR"]))
    seeds = sorted(dsr[CANDIDATE])
    diffs = [
        100 * (np.mean(dsr[CANDIDATE][s]) - np.mean(dsr[STRONG_BASELINE][s])) for s in seeds
    ]

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.bar(range(len(seeds)), diffs, color="#3D6FB6", width=0.62)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.axhline(np.mean(diffs), color="#C75E5A", lw=1.0, linestyle="--")
    ax.text(-0.35, max(diffs) * 0.94,
            f"红色虚线：均值 +{np.mean(diffs):.2f} 个百分点", color="#C75E5A", fontsize=8, ha="left")
    for i, v in enumerate(diffs):
        ax.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=7.5)
    ax.set_xlabel("训练种子")
    ax.set_ylabel("DSR 差值 / 个百分点")
    ax.set_title("候选 - 下层无注意力基线（10/10 为正）", fontsize=9)
    ax.grid(True, axis="y", alpha=0.6)
    ax.margins(y=0.15)
    save(fig, "图5-2_逐训练种子DSR差值")


def fig_dsr_decomposition(coverage_root: Path) -> None:
    """Chap5: admission fraction x deadline satisfaction among admitted, learned methods."""
    path = coverage_root / "results" / "whole_system_comparison_20260928" / "whole_system_paired_comparison.json"
    data = json.loads(path.read_text())["comparisons"]
    names, adm, qual = [], [], []
    for key, label in LEARNED_METHODS:
        if key == CANDIDATE:
            m = data[STRONG_BASELINE]["metrics"]
            names.append(label)
            adm.append(m["Admission fraction"]["candidate_mean"])
            qual.append(m["Deadline satisfaction among admitted"]["candidate_mean"])
        else:
            m = data[key]["metrics"]
            names.append(label)
            adm.append(m["Admission fraction"]["comparator_mean"])
            qual.append(m["Deadline satisfaction among admitted"]["comparator_mean"])

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    b1 = ax.bar(x - w / 2, adm, w, color="#6F7DBE", label="接纳率 $A/G$")
    b2 = ax.bar(x + w / 2, qual, w, color="#E2A64A", label="接纳后按时完成率 $S/A$")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("比例")
    ax.set_ylim(0, 0.85)
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(fontsize=8)
    save(fig, "图5-3_接纳与接纳后质量分解")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-root",
        type=Path,
        default=Path.home() / ".codex" / "worktrees" / "coverage-optimization" / "wodexuexi",
        help="coverage-optimization worktree root holding the 09-27/09-28 result batches",
    )
    args = parser.parse_args()
    setup_style()
    fig_coverage_geometry()
    fig_failure_decomposition()
    fig_confirmation_curves(args.coverage_root)
    fig_per_seed_dsr_diff(args.coverage_root)
    fig_dsr_decomposition(args.coverage_root)


if __name__ == "__main__":
    main()
