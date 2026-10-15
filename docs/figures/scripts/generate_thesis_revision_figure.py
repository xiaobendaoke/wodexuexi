"""Plot actual request-slack sensitivity and paired method effects.

Panel a reports absolute DSR under a common delay; panel b reports paired
learning increments. Diamonds are method-specific per-call software scoring,
separated from the fixed-delay curves. All points use the complete ten blocks.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp/thesis_revision_20261014/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(ROOT))
from paths import REPO_ROOT
from thesis_figure_style import setup_thesis_style


def plot(summary, destination, validation, qa_root):
    sys.path.insert(0, str(qa_root))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    data = json.loads(summary.read_text())
    if data["status"] != "PASS" or data["episodes"] != 900:
        raise ValueError("Only the complete 900-episode audit may be plotted")
    family = setup_thesis_style(8.5)
    # Preserve the established Chinese thesis font; sans-serif is only fallback.
    plt.rcParams.update({"font.family": [family, "DejaVu Sans"],
                         "font.sans-serif": ["DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15))
    fig.subplots_adjust(left=.09, right=.99, bottom=.32, top=.82, wspace=.33)
    xs = [0, 20, 50, 100, 150]
    colors = {"learned": "#3D6FB6", "random": "#9AA0A6", "greedy": "#D8A448"}
    labels = {"learned": "学习配额", "random": "强配额随机", "greedy": "贪心＋修复"}
    for method in ("learned", "random", "greedy"):
        points = [data["scenarios"][str(x)]["groups"][method] for x in xs]
        mean = np.array([p["mean"] for p in points]) * 100
        lo = np.array([p["ci95"][0] for p in points]) * 100
        hi = np.array([p["ci95"][1] for p in points]) * 100
        axes[0].plot(xs, mean, "o-", color=colors[method], markersize=3, linewidth=1.2, label=labels[method])
        axes[0].fill_between(xs, lo, hi, color=colors[method], alpha=.12)
        p = data["scenarios"]["software"]["groups"][method]
        axes[0].errorbar(205, p["mean"] * 100, yerr=[[100 * (p["mean"] - p["ci95"][0])],
                                                  [100 * (p["ci95"][1] - p["mean"])]],
                         fmt="D", color=colors[method], markersize=4, capsize=2, linewidth=.8)
    for method, color in (("random", "#53A6A6"), ("greedy", "#3D6FB6")):
        name = "learned-" + method
        points = [data["scenarios"][str(x)]["comparisons"][name] for x in xs]
        mean = np.array([p["mean"] for p in points]) * 100
        lo = np.array([p["ci95"][0] for p in points]) * 100
        hi = np.array([p["ci95"][1] for p in points]) * 100
        axes[1].plot(xs, mean, "o-", color=color, markersize=3, linewidth=1.2,
                     label="学习减" + ("强随机" if method == "random" else "贪心"))
        axes[1].fill_between(xs, lo, hi, color=color, alpha=.12)
        p = data["scenarios"]["software"]["comparisons"][name]
        axes[1].errorbar(205, p["mean"] * 100, yerr=[[100 * (p["mean"] - p["ci95"][0])],
                                                  [100 * (p["ci95"][1] - p["mean"])]],
                         fmt="D", color=color, markersize=4, capsize=2, linewidth=.8)
    axes[0].set_ylabel("DSR / %")
    axes[1].set_ylabel("配对 DSR 差 / 百分点")
    axes[1].axhline(0, color="#777777", linewidth=.7, linestyle="--")
    for i, ax in enumerate(axes):
        ax.set_title(["a  截止期余量对附加延迟的敏感性", "b  学习增量与软件费用复核"][i], loc="left", fontsize=8.5, pad=12)
        ax.set_xlim(-8, 222)
        ax.set_xticks([0, 50, 100, 150, 205], ["0", "50", "100", "150", "软件"])
        ax.axvline(177, color="#BBBBBB", linewidth=.6, linestyle=":")
        ax.set_xlabel("统一附加延迟 / ms；右侧为逐次软件计时", fontsize=7.5)
        ax.grid(axis="y", alpha=.2, linewidth=.5)
    for ax, center, columns in ((axes[0], .29, 3), (axes[1], .77, 2)):
        handles, names = ax.get_legend_handles_labels()
        fig.legend(handles, names, loc="center", bbox_to_anchor=(center, .135), ncol=columns,
                   fontsize=7.5, frameon=False, handlelength=1.5, columnspacing=1.)
    fig.text(.5, .035, "十个训练块；阴影与误差线为逐项95% t区间；事后评分不回写环境状态", ha="center", fontsize=7.5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=validation / "delay_figure_alignment.json")
    fig.savefig(destination.with_suffix(".pdf"), facecolor="white")
    fig.savefig(destination.with_suffix(".svg"), facecolor="white")
    fig.savefig(destination.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=REPO_ROOT / "results/thesis_revision_20261014/analysis/summary.json")
    p.add_argument("--destination", type=Path, default=REPO_ROOT / "docs/figures/图5-20_调度延迟敏感性")
    p.add_argument("--validation", type=Path, default=REPO_ROOT / "output/thesis_revision_20261014/validation")
    p.add_argument("--qa-root", type=Path, default=Path.home() / ".codex/skills/nature-figure/scripts")
    args = p.parse_args()
    plot(args.summary, args.destination, args.validation, args.qa_root)
