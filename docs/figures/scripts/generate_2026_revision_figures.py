#!/usr/bin/env python3
"""Generate the 2026-09-28 revision figures for chapters 3-5.

2026-10-02: 图3-3/图3-4 renamed to 图3-6/图3-7 (训练曲线增加奖励面板);
图4-1 moved to generate_chapter34_figures_20261002.py as the enhanced 图4-3.

Data sources are archived result files only; no experiment is rerun:
  - output/bottleneck_diagnostic_20260927/validation/bottleneck_review.json (main worktree)
  - results/canonical_all_20260920/chapter4/analysis.json (main worktree)
  - <coverage worktree>/results/coverage_observation_confirmation_20260927/training_curves.csv
  - <coverage worktree>/results/whole_system_comparison_20260928/{whole_system_paired_comparison.json,evaluation_units.csv}

Outputs PDF/SVG/600 dpi PNG into the single formal docs/figures/ directory.
Default output is the three current chapter-3 figures. Historical chapter-5
figures require --include-historical-ch5 and their original result tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_figure_style import save_pub, setup_thesis_style
from generate_chapter34_figures_20261002 import clipped_circle, source_file

BUNDLED_QA = REPO_ROOT / "tools" / "figure_qa"
SKILL_SCRIPTS = (BUNDLED_QA if BUNDLED_QA.is_dir() else
                 Path.home() / ".codex" / "skills" / "nature-figure" / "scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUTPUT_DIRS = [
    REPO_ROOT / "docs" / "figures",
]
ALIGN_DIR = REPO_ROOT / "output" / "thesis_writing_20261004" / "validation" / "figures"
QA_REPORT = REPO_ROOT / "output" / "thesis_writing_20261004" / "validation" / "figure_text_qa.json"

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
    family = setup_thesis_style(font_size=8)
    plt.rcParams.update({"font.family": "serif", "font.serif": [family],
                         "svg.fonttype": "none", "pdf.fonttype": 42})


def save(fig: plt.Figure, name: str) -> None:
    from matplotlib.text import Text
    for item in fig.findobj(match=Text):
        if item.get_fontsize() < 8:
            item.set_fontsize(8)
    for ax in fig.axes:
        ax.grid(False)
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig,
        json_out=ALIGN_DIR / f"{name}.alignment.json",
        overlay_svg=ALIGN_DIR / f"{name}.alignment.svg", strict=True)
    save_pub(fig, name, OUTPUT_DIRS, formats=("pdf", "svg", "png"), png_dpi=600,
             qa_report=QA_REPORT)
    for exported in (OUTPUT_DIRS[0] / f"{name}.pdf", OUTPUT_DIRS[0] / f"{name}.svg",
                     OUTPUT_DIRS[0] / f"{name}.png"):
        if not exported.exists():
            raise RuntimeError(f"Required export missing: {exported}")
    subprocess.run([sys.executable, str(SKILL_SCRIPTS / "audit_figure_collisions.py"),
                    str(OUTPUT_DIRS[0] / f"{name}.pdf"),
                    "--json-out", str(ALIGN_DIR / f"{name}.collision.json")],
                   check=True, capture_output=True, text=True)
    print(f"saved {name}")


def fig_coverage_geometry() -> None:
    """Chap3: coverage radius vs sensing radius geometry (schematic, no data)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    uav_pos = np.array([350.0, 380.0])
    rng = np.random.default_rng(7)

    # Large arcs are physically truncated at the map boundary. PDF clip paths
    # otherwise leave invisible strokes in the geometry collision audit.
    clipped_circle(ax, uav_pos, 460, bounds=(0,700,0,700), color="#3D6FB6",
                   ls="--", lw=1.2)
    ax.add_patch(Circle(uav_pos, 100.0, fill=True, facecolor="#3D6FB6", alpha=0.35,
                        edgecolor="#3D6FB6", linewidth=1.2))

    # UEs: inside coverage (admitted), between 100-460 m (sensed, not admitted), outside.
    def scatter_ring(r_min: float, r_max: float, n: int, **kw) -> None:
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = np.sqrt(rng.uniform(r_min**2, r_max**2, n))
        pts = uav_pos + np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        pts = pts[(pts[:,0]>=0)&(pts[:,0]<=700)&(pts[:,1]>=0)&(pts[:,1]<=700)]
        ax.scatter(pts[:, 0], pts[:, 1], s=22, zorder=3, **kw)

    scatter_ring(0, 95, 7, marker="o", color="#3A8D5A", label="已覆盖 UE（接纳）")
    scatter_ring(110, 450, 9, marker="s", color="#D8A448", label="感知范围内未覆盖 UE")
    scatter_ring(470, 620, 5, marker="x", color="#9AA0A6", label="感知范围外 UE")

    ax.plot(*uav_pos, marker="^", color="#C75E5A", markersize=13, zorder=4)
    ax.plot([],[],marker="^",color="#C75E5A",ls="",label="UAV")

    ax.set_xlim(0, 700)
    ax.set_ylim(0, 700)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$y$ / m")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.13),ncol=2,
              fontsize=8,frameon=False)
    fig.text(.5,.015,"实线圆：100 m 接纳范围；虚线弧：460 m 感知范围（与地图边界相交）",
             ha="center",fontsize=8)
    fig.subplots_adjust(bottom=.25)
    save(fig, "图3-2_覆盖与感知几何示意图")


def fig_failure_decomposition() -> None:
    """Chap3: generated-request outcome decomposition from the bottleneck diagnostic."""
    review = json.loads(
        source_file("bottleneck_review.json", REPO_ROOT / "output" / "bottleneck_diagnostic_20260927" / "validation" / "bottleneck_review.json").read_text()
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
    path = source_file("confirmation_training.csv", coverage_root / "results" / "coverage_observation_confirmation_20260927" / "training_curves.csv")
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
        ax.set_xlim(0, arr.shape[1])
        ax.set_xlabel("训练 episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.08),
               ncol=2,fontsize=8,frameon=False)
    fig.text(.5,-.04, "实线：10 个训练种子均值；阴影：种子间 ±1 SD；训练动态不构成收敛性证明",
             ha="center",fontsize=8,color="#717784")
    fig.subplots_adjust(wspace=.38)
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
        ax.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
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
                    f"{b.get_height():.2f}", ha="center", fontsize=8)
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
    parser.add_argument("--include-historical-ch5",action="store_true",
                        help="also reproduce the superseded 09-28 chapter-5 figures")
    args = parser.parse_args()
    setup_style()
    fig_coverage_geometry()
    fig_failure_decomposition()
    fig_confirmation_curves(args.coverage_root)
    if args.include_historical_ch5:
        fig_per_seed_dsr_diff(args.coverage_root)
        fig_dsr_decomposition(args.coverage_root)


if __name__ == "__main__":
    main()
