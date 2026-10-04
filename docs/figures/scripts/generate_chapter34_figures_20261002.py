#!/usr/bin/env python3
"""Generate the 2026-10-02 long-form figures for thesis chapters 3 and 4.

Data sources are archived result files only; no experiment is rerun:
  - results/canonical_all_20260920/chapter4/{analysis.json,evaluation_units.csv} (main worktree)
  - results/bridge_20260928/full/analysis.json (main worktree)
  - <coverage worktree>/results/coverage_observation_confirmation_20260927/evaluation_units.csv

Schematic figures (3-1, 3-3, 3-4, 3-5, 4-1, 4-2) draw no experimental data;
the constructed example in 图4-1 uses formulas verified against
environment/uavs.py::_get_computing_latency_and_energy (latency = C/(F/n),
energy = K_CPU * C * (F/n)**2) and is labelled as a mechanism example.

Outputs PDF + PNG(450 dpi) into docs/figures/ and latex/docs/figures/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_figure_style import PALETTE, panel_label, save_pub, setup_thesis_style, soften_axes

SKILL_SCRIPTS = Path.home() / ".zcode" / "skills" / "nature-figure" / "scripts"
if SKILL_SCRIPTS.is_dir():
    sys.path.insert(0, str(SKILL_SCRIPTS))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except Exception:  # pragma: no cover - optional QA dependency
    require_matplotlib_panel_alignment = None

OUTPUT_DIRS = [
    REPO_ROOT / "docs" / "figures",
    REPO_ROOT / "latex" / "docs" / "figures",
]
QA_REPORT = REPO_ROOT / "output" / "chapter3_chapter4_writing_20261002" / "validation" / "figure_text_qa.json"
ALIGN_DIR = REPO_ROOT / ".tmp" / "chapter34_figures_20261002" / "alignment"

T95_DF9 = 2.2622  # t(0.975, 9), 10 paired units

CH4_METHODS = [
    ("heuristic_control", "Heuristic-Control", "#9AA0A6"),
    ("greedy_single_pass", "Greedy-Single-Pass", "#D8A448"),
    ("iterative_base", "Iterative-Base", "#53A6A6"),
    ("mc_itr", "MC-ITR", "#3D6FB6"),
]
CH4_METRIC_IDX = {"misses": 0, "latency": 1, "energy": 2, "fleet": 3, "time": 4}

CANDIDATE = "lower_mlp_upper_uncovered_obs"
STRONG_BASELINE = "lower_no_attention"

ALIGN_REPORTS: list[dict] = []


def save(fig: plt.Figure, name: str, *, multi_panel: bool = False) -> None:
    if multi_panel and require_matplotlib_panel_alignment is not None:
        ALIGN_DIR.mkdir(parents=True, exist_ok=True)
        try:
            report = require_matplotlib_panel_alignment(
                fig,
                json_out=ALIGN_DIR / f"{name}.alignment.json",
                overlay_svg=ALIGN_DIR / f"{name}.alignment.svg",
                strict=True,
            )
            ALIGN_REPORTS.append({"figure": name, "status": "pass"})
        except Exception as exc:  # noqa: BLE001 - recorded and fixed in the repair loop
            ALIGN_REPORTS.append({"figure": name, "status": "fail", "detail": str(exc)[:600]})
            print(f"[alignment] {name}: {exc}")
    save_pub(fig, name, OUTPUT_DIRS, formats=("pdf", "png"), png_dpi=450, qa_report=QA_REPORT)
    print(f"saved {name}")


# ---------------------------------------------------------------- ch3 schematics


def fig_system_overview() -> None:
    """图3-1: system scenario with natural coverage admission (schematic)."""
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    rng = np.random.default_rng(11)

    ax.add_patch(Rectangle((0, 0), 700, 700, fill=False, edgecolor=PALETTE["axis"], lw=1.0))
    hotspots = [np.array([210.0, 500.0]), np.array([520.0, 180.0])]
    for h in hotspots:
        ax.add_patch(Circle(h, 100, fill=True, facecolor=PALETTE["orange"], alpha=0.10,
                            edgecolor=PALETTE["orange"], linestyle=":", lw=0.9))
    ax.text(hotspots[0][0], hotspots[0][1] + 112, "热点区域", ha="center", fontsize=7.5,
            color=PALETTE["orange"])

    uavs = [np.array(p) for p in [(200, 480), (500, 200), (430, 470), (150, 170), (580, 420)]]
    for i, p in enumerate(uavs):
        ax.add_patch(Circle(p, 100, fill=True, facecolor=PALETTE["full"], alpha=0.13,
                            edgecolor=PALETTE["full"], lw=0.9))
        ax.plot(*p, marker="^", color=PALETTE["red"], markersize=11, zorder=5)
        ax.annotate(f"UAV{i+1}", p + np.array([8, 10]), fontsize=7.5, color=PALETTE["red"])
    # sensing range shown once to avoid clutter
    ax.add_patch(Circle(uavs[0], 460, fill=False, edgecolor=PALETTE["full"], linestyle="--", lw=1.1))
    arc_pt = uavs[0] + 460 * np.array([np.cos(5.0), np.sin(5.0)])
    ax.annotate("感知范围 $R_s=460$ m", arc_pt,
                fontsize=7.5, color=PALETTE["full"],
                xytext=(430, 60), arrowprops=dict(arrowstyle="->", lw=0.7, color=PALETTE["full"]))

    def ue_cloud(center: np.ndarray, r: float, n: int) -> np.ndarray:
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = r * np.sqrt(rng.uniform(0, 1, n))
        return center + np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)

    covered_pts, uncovered_pts = [], []
    pts = np.vstack([ue_cloud(hotspots[0], 95, 24), ue_cloud(hotspots[1], 95, 22),
                     ue_cloud(np.array([350.0, 350.0]), 320, 8)])
    for q in pts:
        if not (0 < q[0] < 700 and 0 < q[1] < 700):
            continue
        (covered_pts if any(np.linalg.norm(q - u) <= 100 for u in uavs) else uncovered_pts).append(q)
    covered_pts = np.array(covered_pts)
    uncovered_pts = np.array(uncovered_pts)
    ax.scatter(covered_pts[:, 0], covered_pts[:, 1], s=14, color=PALETTE["green"],
               marker="o", zorder=4, label="已覆盖 UE（请求被接纳）")
    ax.scatter(uncovered_pts[:, 0], uncovered_pts[:, 1], s=16, color=PALETTE["red"],
               marker="x", zorder=4, label="未覆盖 UE（请求不接纳，计入 DSR 分母）")

    mbs = np.array([350.0, 350.0])
    ax.plot(*mbs, marker="*", color=PALETTE["purple"], markersize=17, zorder=5)
    ax.annotate("MBS", mbs + np.array([12, 6]), fontsize=8, color=PALETTE["purple"])

    ax.plot([], [], marker="^", color=PALETTE["red"], linestyle="", markersize=9, label="UAV（实线圆：覆盖半径 $R_c=100$ m）")
    ax.set_xlim(-15, 715)
    ax.set_ylim(-15, 715)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$y$ / m")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.075), fontsize=7.2, frameon=False, ncol=2)
    fig.text(0.5, 0.005, "示意图：UAV/UE 位置仅为说明覆盖与接纳关系，并非仿真轨迹记录",
             ha="center", fontsize=7.2, color=PALETTE["baseline"], style="italic")
    fig.subplots_adjust(bottom=0.16)
    soften_axes(ax, axis="both")
    save(fig, "图3-1_系统与自然覆盖示意图")


def fig_slot_timeline() -> None:
    """图3-3: decision order inside one time slot (from environment/env.py step)."""
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.axis("off")
    stages = [
        ("S0", "时隙开始：\n冻结快照\n（位置/请求/负载）", PALETTE["baseline"], "环境"),
        ("S1-S2", "下层卸载决策\n（质量感知掩码\n+自适应惩罚）", PALETTE["lower"], "下层"),
        ("S3", "请求批量执行\n（按最终负载\n均分算力）", PALETTE["lower"], "下层"),
        ("S4", "上层轨迹动作执行\n（速度/边界裁剪、\n碰撞消解）", PALETTE["upper"], "上层"),
        ("S5", "能耗结算", PALETTE["baseline"], "环境"),
        ("S6", "奖励与\n指标统计", PALETTE["baseline"], "环境"),
        ("S7", "缓存提交、UE 移动、\n生成下一时隙观测；\n新请求生成与接纳", PALETTE["baseline"], "环境"),
    ]
    n = len(stages)
    w, h, gap = 0.118, 0.34, 0.022
    x0, y0 = 0.015, 0.30
    for i, (tag, text, color, actor) in enumerate(stages):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.008",
                             facecolor=color, alpha=0.16, edgecolor=color, lw=1.0,
                             transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h - 0.045, tag, transform=ax.transAxes, ha="center",
                fontsize=8, fontweight="bold", color=color)
        ax.text(x + w / 2, y0 + h / 2 - 0.075, text, transform=ax.transAxes, ha="center",
                va="center", fontsize=6.6, color=PALETTE["text"])
        ax.text(x + w / 2, y0 + h + 0.055, actor, transform=ax.transAxes, ha="center",
                fontsize=7.2, color=color)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.002, y0 + h / 2), (x + w + gap - 0.002, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                         color=PALETTE["axis"], transform=ax.transAxes))
    # loop-back arrow: S7 feeds the next slot's S0 (routed below the boxes)
    ax.add_patch(FancyArrowPatch((x0 + (n - 1) * (w + gap) + w / 2, y0 - 0.015),
                                 (x0 + w / 2, y0 - 0.015),
                                 arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                 color=PALETTE["baseline"], linestyle="--",
                                 connectionstyle="arc3,rad=-0.32", transform=ax.transAxes))
    ax.text(0.5, 0.115, "进入下一时隙", transform=ax.transAxes, ha="center", fontsize=6.8,
            color=PALETTE["baseline"])
    ax.text(0.5, 0.02, "时隙 $t$", transform=ax.transAxes, ha="center", fontsize=8.5,
            color=PALETTE["text"])
    ax.text(0.5, 0.90, "单个 1 s 时隙内的决策与执行顺序（依 environment/env.py 的 step 流程）",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=PALETTE["text"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "图3-3_单时隙决策时序图")


def fig_network_structure() -> None:
    """图3-4: Attention-MAPPO network and CTDE training structure (schematic)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=7.0, alpha=0.15, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                    facecolor=color, alpha=alpha, edgecolor=color, lw=1.0,
                                    transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=fs, color=PALETTE["text"], fontweight=weight)

    def arrow(p1, p2, color=PALETTE["axis"], style="-|>", lw=0.9, ls="-"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=9, lw=lw,
                                     color=color, linestyle=ls, transform=ax.transAxes))

    # observation groups (left)
    box(0.02, 0.62, 0.17, 0.24, "自身状态\n77 维\n（位置/能量/缓存等）", PALETTE["full"])
    box(0.02, 0.35, 0.17, 0.20, "邻居 UAV\n$4\\times 2$ 维", PALETTE["lower"])
    box(0.02, 0.08, 0.17, 0.20, "关联 UE 槽位\n$30\\times 8$ 维\n（扩展观测含感知内\n未覆盖 UE）", PALETTE["orange"])

    # encoders
    box(0.26, 0.35, 0.13, 0.51, "分组线性嵌入\n（各组映射到\n64 维向量）", PALETTE["baseline"])
    arrow((0.19, 0.74), (0.26, 0.66))
    arrow((0.19, 0.45), (0.26, 0.55))
    arrow((0.19, 0.18), (0.26, 0.42))

    # cross attention
    box(0.45, 0.42, 0.17, 0.44, "交叉注意力提取器\n（8 头、64 维）\n\n自身状态为 query\nUAV-UAV 关系\nUAV-UE 关系", PALETTE["purple"], fs=6.8)
    arrow((0.39, 0.60), (0.45, 0.62))

    # actor
    box(0.68, 0.55, 0.15, 0.22, "共享 Actor $\\pi_\\theta$\n输出二维飞行动作\n（5 架 UAV 共用\n同一网络参数）", PALETTE["green"], fs=6.8)
    arrow((0.62, 0.66), (0.68, 0.66))

    # critic
    box(0.68, 0.16, 0.15, 0.22, "集中式 Critic $V_\\phi$\n输入全局状态\n（仅训练时使用）", PALETTE["red"], fs=6.8)
    arrow((0.535, 0.42), (0.68, 0.27), color=PALETTE["red"], ls="--")

    # training loop annotation
    box(0.26, 0.02, 0.34, 0.13, "集中式训练、分布式执行（CTDE）：\nPPO 更新（clip $\\epsilon=0.2$，GAE $\\lambda=0.95$）",
        PALETTE["baseline"], fs=6.8, alpha=0.10)
    arrow((0.71, 0.16), (0.60, 0.10), color=PALETTE["red"], ls="--")
    arrow((0.43, 0.085), (0.325, 0.34), color=PALETTE["baseline"], ls="--")

    ax.text(0.02, 0.93, "上层 Attention-MAPPO 的网络与训练结构（下层为独立的无注意力 MAPPO，见第二、五章）",
            transform=ax.transAxes, fontsize=8.5, color=PALETTE["text"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "图3-4_Attention-MAPPO网络与训练结构")


def fig_observation_compare() -> None:
    """图3-5: original vs extended UE observation slot filling (schematic)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    rng = np.random.default_rng(5)
    titles = ["(a) 原观测：仅已覆盖 UE 入槽", "(b) 扩展观测：补入感知范围内未覆盖 UE"]
    for k, ax in enumerate(axes):
        uav = np.array([0.0, 0.0])
        ax.add_patch(Circle(uav, 460, fill=False, edgecolor=PALETTE["full"], linestyle="--", lw=1.0))
        ax.add_patch(Circle(uav, 100, fill=True, facecolor=PALETTE["full"], alpha=0.15,
                            edgecolor=PALETTE["full"], lw=1.0))
        ax.plot(*uav, marker="^", color=PALETTE["red"], markersize=11, zorder=5)
        n_cov, n_unc = 6, 8
        ang = rng.uniform(0, 2 * np.pi, n_cov)
        rad = np.sqrt(rng.uniform(0, 1, n_cov)) * 92
        cov = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        ang = rng.uniform(0, 2 * np.pi, n_unc)
        rad = 130 + np.sqrt(rng.uniform(0, 1, n_unc)) * 300
        unc = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        ax.scatter(cov[:, 0], cov[:, 1], s=20, color=PALETTE["green"], zorder=4, label="已覆盖 UE")
        ax.scatter(unc[:, 0], unc[:, 1], s=20, color=PALETTE["red"], marker="x", zorder=4,
                   label="感知内未覆盖 UE")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-560, 500)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[k], fontsize=8.2)
        # slot strip: 30 slots
        n_slots = 30
        filled = n_cov if k == 0 else n_cov + n_unc
        for i in range(n_slots):
            x = -470 + i * (940 / n_slots)
            color = PALETTE["green"] if i < n_cov else (
                PALETTE["red"] if (k == 1 and i < filled) else "#E4E6EB")
            ax.add_patch(Rectangle((x, -545), 940 / n_slots - 3, 52,
                                   facecolor=color, edgecolor="white", lw=0.3))
        ax.text(0, -625, f"UE 槽位（共 30 个）：填充 {filled} 个，其余补零",
                ha="center", fontsize=7.0, color=PALETTE["text"])
        if k == 0:
            ax.legend(loc="upper right", fontsize=6.8, frameon=False)
    fig.text(0.5, 0.015, "示意图：槽位填充规则对照，几何与 UE 数为示意，并非注意力权重或真实观测记录",
             ha="center", fontsize=7.2, color=PALETTE["baseline"], style="italic")
    fig.subplots_adjust(bottom=0.10, wspace=0.08)
    save(fig, "图3-5_原观测与扩展观测对照示意", multi_panel=True)


# ---------------------------------------------------------------- ch3 data figures


def _load_confirmation(cov_root: Path) -> dict[str, dict[int, dict[str, list[float]]]]:
    path = cov_root / "results" / "coverage_observation_confirmation_20260927" / "evaluation_units.csv"
    data: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    with path.open() as f:
        for row in csv.DictReader(f):
            m = data[row["method"]][int(row["training_seed"])]
            m["DSR"].append(float(row["DSR"]))
            m["adm"].append(float(row["Admission fraction"]))
            m["sat"].append(float(row["Deadline satisfaction among admitted"]))
    return data


def fig_confirmation_seed_diff(cov_root: Path) -> None:
    """图3-8: per-training-seed paired DSR difference, confirmation experiment."""
    data = _load_confirmation(cov_root)
    seeds = sorted(data[CANDIDATE])
    diffs = [100 * (np.mean(data[CANDIDATE][s]["DSR"]) - np.mean(data[STRONG_BASELINE][s]["DSR"]))
             for s in seeds]
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    ci = T95_DF9 * sd / math.sqrt(len(diffs))

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.bar(range(len(seeds)), diffs, color=PALETTE["full"], width=0.62)
    ax.axhline(0, color=PALETTE["axis"], lw=0.8)
    ax.axhline(mean, color=PALETTE["red"], lw=1.0, linestyle="--")
    for i, v in enumerate(diffs):
        ax.text(i, v + 0.13, f"{v:.1f}", ha="center", fontsize=7.2)
    ax.text(0.02, 0.95,
            f"均值 +{mean:.2f} 个百分点，配对 $t$ 95% CI [{mean-ci:.2f}, {mean+ci:.2f}]；10/10 种子为正",
            transform=ax.transAxes, fontsize=7.4, color=PALETTE["red"], va="top")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=7.0)
    ax.set_xlabel("训练种子")
    ax.set_ylabel("DSR 差值 / 个百分点")
    ax.margins(y=0.16)
    soften_axes(ax)
    save(fig, "图3-8_确认实验逐种子配对差值")


def fig_admission_quality_paired(cov_root: Path) -> None:
    """图3-9: paired admission fraction and satisfaction-among-admitted."""
    data = _load_confirmation(cov_root)
    seeds = sorted(data[CANDIDATE])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    panels = [("adm", "(a) 接纳率 $A/G$"), ("sat", "(b) 接纳后按时完成率 $S/A$")]
    for ax, (key, title) in zip(axes, panels):
        base = [np.mean(data[STRONG_BASELINE][s][key]) for s in seeds]
        cand = [np.mean(data[CANDIDATE][s][key]) for s in seeds]
        for b, c in zip(base, cand):
            ax.plot([0, 1], [b, c], color=PALETTE["baseline"], lw=0.7, alpha=0.45, zorder=2)
        mb, mc = float(np.mean(base)), float(np.mean(cand))
        ax.plot([0, 1], [mb, mc], color=PALETTE["full"], lw=1.8, marker="o", markersize=5, zorder=3)
        ax.text(0, mb + 0.012, f"{mb:.3f}", ha="center", fontsize=7.4, color=PALETTE["baseline"])
        ax.text(1, mc + 0.012, f"{mc:.3f}", ha="center", fontsize=7.4, color=PALETTE["full"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["强基线", "候选"], fontsize=8)
        ax.set_title(title, fontsize=8.5)
        vals = base + cand
        span = max(vals) - min(vals)
        ax.set_ylim(min(vals) - 0.25 * span - 0.02, max(vals) + 0.30 * span + 0.03)
        soften_axes(ax)
    axes[0].text(0.0, -0.30, "灰线：10 个训练种子各自的配对连接（每种取 5 个工作负载均值）；蓝线：种子均值",
                 transform=axes[0].transAxes, fontsize=6.8, color=PALETTE["baseline"])
    save(fig, "图3-9_接纳率与接纳后质量配对比较", multi_panel=True)


# ---------------------------------------------------------------- ch4 figures


def fig_sharing_externality() -> None:
    """图4-1: constructed mechanism example of batch compute sharing (formula-verified)."""
    L_m = 2e6          # bytes
    omega = 2000.0     # cycles / byte
    F_u = 60e9         # Hz
    kappa = 1e-27
    C = omega * L_m
    res = {}
    for n in (2, 4):
        f = F_u / n
        res[n] = (C / f, kappa * C * f**2)
    assert abs(res[2][0] - 0.13333) < 1e-3 and abs(res[4][0] - 0.26667) < 1e-3
    assert abs(res[2][1] - 3600.0) < 1 and abs(res[4][1] - 900.0) < 1

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7))
    ax = axes[0]
    for row, n in enumerate((2, 4)):
        seg_w = F_u / n / 1e9
        for i in range(n):
            ax.add_patch(Rectangle((i * seg_w, 1 - row), seg_w, 0.72,
                                   facecolor=PALETTE["full"] if row == 0 else PALETTE["orange"],
                                   alpha=0.25 + 0.12 * (i % 2), edgecolor=PALETTE["axis"], lw=0.6))
            ax.text(i * seg_w + seg_w / 2, 1 - row + 0.36, f"{seg_w:.0f} GHz", ha="center",
                    va="center", fontsize=7.2)
        ax.text(-3.5, 1 - row + 0.36, f"$n={n}$", ha="right", va="center", fontsize=8)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.15, 2.05)
    ax.set_yticks([])
    ax.set_xlabel("UAV 算力 $F_u=60$ GHz 在批内最终负载间均分")
    ax.set_title("(a) 批量算力均分", fontsize=8.5)

    for ax, (idx, title, unit, fmt) in zip(axes[1:], [
            (0, "(b) 每任务计算时延", "s", "%.3f"),
            (1, "(c) 每任务计算能耗", "J", "%.0f")]):
        vals = [res[2][idx], res[4][idx]]
        bars = ax.bar(["$n=2$", "$n=4$"], vals, width=0.55,
                      color=[PALETTE["full"], PALETTE["orange"]])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt % v, ha="center", va="bottom", fontsize=7.4)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel(unit)
        ax.margins(y=0.20)
        soften_axes(ax)
    fig.text(0.5, -0.035,
             "机制示例（构造参数，非仿真数据）：$L_m=2$ MB，$\\omega_m=2000$ cycles/byte，$\\kappa=10^{-27}$；"
             "$T^{cmp}=\\omega_m L_m/(F_u/n)$，$E^{cmp}=\\kappa\\,\\omega_m L_m\\,(F_u/n)^2$。"
             "任务迁入使批内每任务时延上升、单任务计算能耗下降。",
             ha="center", fontsize=6.9, color=PALETTE["baseline"])
    fig.subplots_adjust(bottom=0.20, wspace=0.30)
    save(fig, "图4-1_批量算力共享与迁移外部影响示意", multi_panel=True)


def fig_mc_itr_flowchart() -> None:
    """图4-2: MC-ITR algorithm flowchart."""
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                    facecolor=color, alpha=0.15, edgecolor=color, lw=1.0,
                                    transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=fs, color=PALETTE["text"])

    def diamond(cx, cy, w, h, text, color, fs=6.8):
        xs = [cx, cx + w / 2, cx, cx - w / 2]
        ys = [cy + h / 2, cy, cy - h / 2, cy]
        ax.fill(xs, ys, facecolor=color, alpha=0.15, edgecolor=color, lw=1.0,
                transform=ax.transAxes)
        ax.text(cx, cy, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=fs, color=PALETTE["text"])

    def arrow(p1, p2, color=PALETTE["axis"], cs="arc3,rad=0.0"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                     color=color, connectionstyle=cs, transform=ax.transAxes))

    cx = 0.46
    # main flow (top to bottom)
    box(cx - 0.21, 0.845, 0.42, 0.105, "输入：冻结快照（请求、位置、缓存、算力）\n共享贪心初值 $x^{(0)}$（与 Iterative-Base 相同）",
        PALETTE["baseline"])
    box(cx - 0.21, 0.665, 0.42, 0.10, "枚举当前解的全部单请求迁移邻域 $\\mathcal{N}(x)$\n（本地/MBS/协作 UAV 间的单请求迁移）",
        PALETTE["full"])
    box(cx - 0.21, 0.49, 0.42, 0.10, "对每个候选按字典序目标整体评价\n$F=(\\sum\\mathbb{1}[T>T_m],\\ \\bar{T},\\ \\bar{E})$",
        PALETTE["full"])
    diamond(cx, 0.355, 0.30, 0.13, "存在严格字典序\n改善的候选？", PALETTE["purple"])
    box(cx - 0.21, 0.155, 0.42, 0.085, "执行最佳改善迁移，$k\\leftarrow k+1$", PALETTE["green"])
    # budget checkpoint on the loop-back edge
    box(0.015, 0.40, 0.20, 0.115, "预算检查：\n轮次达到\n$R_{\\max}=10^3$？", PALETTE["orange"], fs=6.6)
    # output
    box(0.70, 0.28, 0.28, 0.15, "输出当前解\n（无改善退出：邻域局部最优；\n预算退出：不保证）", PALETTE["red"], fs=6.6)

    arrow((cx, 0.845), (cx, 0.765))
    arrow((cx, 0.665), (cx, 0.59))
    arrow((cx, 0.49), (cx, 0.42))
    # improvement branch
    arrow((cx, 0.29), (cx, 0.24), color=PALETTE["green"])
    ax.text(cx + 0.015, 0.265, "是", transform=ax.transAxes, fontsize=7, color=PALETTE["green"])
    # no-improvement branch -> output
    arrow((cx + 0.15, 0.355), (0.70, 0.355), color=PALETTE["red"])
    ax.text(cx + 0.19, 0.368, "否", transform=ax.transAxes, fontsize=7, color=PALETTE["red"])
    # loop back through the budget checkpoint
    arrow((cx - 0.21, 0.1975), (0.115, 0.1975))
    arrow((0.115, 0.1975), (0.115, 0.40))
    arrow((0.115, 0.515), (0.115, 0.715))
    arrow((0.115, 0.715), (cx - 0.21, 0.715))
    ax.text(0.022, 0.60, "未超预算：\n继续搜索", transform=ax.transAxes, fontsize=6.4,
            color=PALETTE["orange"])
    # budget exceeded -> output
    arrow((0.215, 0.4575), (0.84, 0.4575), color=PALETTE["orange"])
    arrow((0.84, 0.4575), (0.84, 0.43), color=PALETTE["orange"])
    ax.text(0.52, 0.468, "超预算退出", transform=ax.transAxes, fontsize=6.4, color=PALETTE["orange"])

    ax.text(0.5, 0.972, "MC-ITR：共享贪心初值上的最佳改善字典序局部搜索",
            transform=ax.transAxes, ha="center", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "图4-2_MC-ITR算法流程图")


def _load_ch4() -> tuple[dict, list[dict]]:
    analysis = json.loads(
        (REPO_ROOT / "results" / "canonical_all_20260920" / "chapter4" / "analysis.json").read_text())
    rows = list(csv.DictReader(
        (REPO_ROOT / "results" / "canonical_all_20260920" / "chapter4" / "evaluation_units.csv").open()))
    return analysis, rows


def fig_main_comparison() -> None:
    """图4-3: main-config quality/energy/time with per-workload-seed scatter."""
    _, rows = _load_ch4()
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["variant"] == "main":
            sums[r["method"]]["misses"].append(float(r["deadline_misses"]))
            sums[r["method"]]["energy"].append(float(r["normalized_assignment_energy"]))
            sums[r["method"]]["time"].append(float(r["solver_wall_seconds"]))

    panels = [("misses", "(a) 每快照平均超时数", "%.3f"),
              ("energy", "(b) 归一化分配能耗", "%.3f"),
              ("time", "(c) 搜索耗时 / s", "%.2f")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    rng = np.random.default_rng(3)
    for ax, (key, title, fmt) in zip(axes, panels):
        for i, (mkey, label, color) in enumerate(CH4_METHODS):
            vals = np.array(sums[mkey][key])
            mean = float(vals.mean())
            ax.bar(i, mean, width=0.62, color=color, alpha=0.85, zorder=2)
            jitter = rng.uniform(-0.16, 0.16, len(vals))
            ax.scatter(i + jitter, vals, s=8, color=PALETTE["text"], alpha=0.55, zorder=3)
            ax.text(i, mean, fmt % mean, ha="center", va="bottom", fontsize=7.2, zorder=4)
        ax.set_xticks(range(len(CH4_METHODS)))
        ax.set_xticklabels([label for _, label, _ in CH4_METHODS], fontsize=6.6, rotation=18)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.20)
        soften_axes(ax)
    axes[2].text(0.0, -0.42, "散点：10 个工作负载种子各自的每快照均值；搜索耗时不含共享贪心初值",
                 transform=axes[2].transAxes, fontsize=6.8, color=PALETTE["baseline"], ha="right")
    save(fig, "图4-3_主配置质量能耗耗时对比", multi_panel=True)


def fig_paired_diff_iterative() -> None:
    """图4-4: per-workload-seed paired difference vs Iterative-Base (deadline misses)."""
    analysis, rows = _load_ch4()
    per: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r["variant"] == "main":
            per[r["method"]][int(r["workload_seed"])] = float(r["deadline_misses"])
    ref = per["iterative_base"]

    # cross-check MC-ITR statistics against the archived comparison
    comp = next(c for c in analysis["comparisons"]
                if c["variant"] == "main" and c["reference"] == "iterative_base")
    assert abs(comp["mean_difference"][0] - (-0.1733333)) < 1e-4

    def boot_ci(diffs: np.ndarray) -> tuple[float, float]:
        """Same bootstrap as run_chapter4_suite.aggregate: 5000 resamples, rng seed 2026."""
        rng = np.random.default_rng(2026)
        draws = diffs[rng.integers(0, len(diffs), size=(5000, len(diffs)))].mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        return float(lo), float(hi)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), sharey=True)
    diff_methods = [CH4_METHODS[0], CH4_METHODS[1], CH4_METHODS[3]]  # vs iterative_base
    for ax, (mkey, label, color) in zip(axes, diff_methods):
        seeds = sorted(per[mkey])
        diffs = np.array([per[mkey][s] - ref[s] for s in seeds])
        mean = float(diffs.mean())
        lo, hi = boot_ci(diffs)
        if mkey == "mc_itr":  # archived bootstrap CI must be reproduced exactly
            assert abs(lo - comp["ci95"][0][0]) < 1e-9 and abs(hi - comp["ci95"][1][0]) < 1e-9
        ax.axhline(0, color=PALETTE["axis"], lw=0.8)
        ax.scatter(range(len(seeds)), diffs, s=16, color=color, zorder=3)
        ax.axhline(mean, color=color, lw=1.2, linestyle="--")
        ax.fill_between([-0.5, len(seeds) - 0.5], lo, hi, color=color, alpha=0.15)
        ax.set_title(label, fontsize=8.2)
        ax.set_xlabel("工作负载种子（排序）")
        ax.text(0.03, 0.94, f"均值 {mean:+.3f}，95% CI [{lo:+.3f}, {hi:+.3f}]",
                transform=ax.transAxes, fontsize=6.8, va="top", color=PALETTE["text"])
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(s) for s in seeds], fontsize=6.0, rotation=45)
        soften_axes(ax)
    axes[0].set_ylabel("超时数差值（方法 $-$ Iterative-Base）")
    axes[0].text(0.0, -0.38, "差值为 10 个工作负载种子配对；区间为 5000 次重抽样的 95% bootstrap 区间，与归档 analysis.json 一致",
                 transform=axes[0].transAxes, fontsize=6.6, color=PALETTE["baseline"])
    save(fig, "图4-4_相对Iterative-Base配对差值", multi_panel=True)


def _sensitivity(rows: list[dict], variant_map: list[tuple[str, str]], metric: str):
    series: dict[str, dict[str, tuple[float, float]]] = {}
    for mkey, label, color in CH4_METHODS:
        pts = {}
        for variant, xlabel in variant_map:
            vals = np.array([float(r[metric]) for r in rows
                             if r["variant"] == variant and r["method"] == mkey])
            mean = float(vals.mean())
            ci = T95_DF9 * float(vals.std(ddof=1)) / math.sqrt(len(vals))
            pts[xlabel] = (mean, ci)
        series[label] = pts
    return series


def fig_sensitivity(kind: str) -> None:
    """图4-5 (UE load) and 图4-6 (compute capacity) sensitivity."""
    _, rows = _load_ch4()
    if kind == "ue":
        variant_map = [("ue_60", "60"), ("ue_80", "80"), ("main", "100"),
                       ("ue_120", "120"), ("ue_140", "140")]
        xlabel = "UE 数量"
        name = "图4-5_UE负载敏感性"
    else:
        variant_map = [("compute_0.6", "0.6×"), ("compute_0.8", "0.8×"), ("main", "1.0×"),
                       ("compute_1.2", "1.2×"), ("compute_1.4", "1.4×")]
        xlabel = "UAV 算力倍率"
        name = "图4-6_算力敏感性"

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    panels = [("deadline_misses", "(a) 每快照平均超时数"), ("normalized_assignment_energy", "(b) 归一化分配能耗")]
    x = np.arange(len(variant_map))
    for ax, (metric, title) in zip(axes, panels):
        series = _sensitivity(rows, variant_map, metric)
        for mkey, label, color in CH4_METHODS:
            means = [series[label][xl][0] for _, xl in variant_map]
            cis = [series[label][xl][1] for _, xl in variant_map]
            ax.errorbar(x, means, yerr=cis, color=color, lw=1.2, marker="o", markersize=3.5,
                        capsize=2, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([xl for _, xl in variant_map], fontsize=7.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=8.5)
        soften_axes(ax)
    legend_ax = axes[0] if kind == "ue" else axes[1]
    legend_ax.legend(fontsize=6.6, loc="lower right", frameon=False)
    axes[0].text(0.0, -0.32, "误差线：10 个工作负载种子的逐项 95% $t$ 区间（非同时覆盖）；各设置独立重估",
                 transform=axes[0].transAxes, fontsize=6.8, color=PALETTE["baseline"])
    save(fig, name, multi_panel=True)


def fig_exact_small() -> None:
    """图4-7: Exact-Small snapshot composition and per-component agreement."""
    _, rows = _load_ch4()
    es = [r for r in rows if r["variant"] == "exact_small" and r["method"] == "mc_itr"]
    n_snap = sum(int(r["snapshots"]) for r in es)
    n_empty = sum(float(r["empty_snapshot_fraction"]) * int(r["snapshots"]) for r in es)
    n_req = sum(float(r["admitted_service_requests"]) * int(r["snapshots"]) for r in es)
    n_double = n_req - (n_snap - n_empty)
    n_single = (n_snap - n_empty) - n_double
    assert (int(n_empty), int(n_single), int(n_double)) == (513, 83, 4)

    means: dict[str, list[float]] = {}
    for mkey in ["heuristic_control", "greedy_single_pass", "iterative_base", "mc_itr", "exact_small"]:
        sel = [r for r in rows if r["variant"] == "exact_small" and r["method"] == mkey]
        arr = np.array([[float(r["deadline_misses"]), float(r["mean_normalized_latency"]),
                         float(r["normalized_assignment_energy"])] for r in sel])
        means[mkey] = arr.mean(axis=0).tolist()

    fig = plt.figure(figsize=(8.6, 2.8))
    gs = fig.add_gridspec(1, 4, wspace=0.34)
    ax = fig.add_subplot(gs[0])
    cats = ["空快照", "单请求", "双请求"]
    vals = [513, 83, 4]
    colors = ["#C9CDD4", PALETTE["full"], PALETTE["orange"]]
    left = 0
    for c, v, col in zip(cats, vals, colors):
        ax.barh([0], [v], left=left, color=col, height=0.5, label=f"{c}：{v}")
        if v > 30:
            ax.text(left + v / 2, 0, f"{v}\n({v/6:.1f}%)", ha="center", va="center", fontsize=7.2,
                    color="white" if col != "#C9CDD4" else PALETTE["text"])
        left += v
    ax.set_xlim(0, 600)
    ax.set_yticks([])
    ax.set_xlabel("快照数（共 600）")
    ax.set_title("(a) 快照构成", fontsize=8.5)
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
    soften_axes(ax, axis="x")

    labels = [("heuristic_control", "Heuristic"), ("greedy_single_pass", "Greedy"),
              ("iterative_base", "Iterative"), ("mc_itr", "MC-ITR"), ("exact_small", "穷举精确解")]
    comp_titles = ["(b) 超时数", "(c) 归一化时延", "(d) 归一化能耗"]
    fmts = ["%.3f", "%.4f", "%.4f"]
    for gi, (comp, title, fmt) in enumerate(zip(range(3), comp_titles, fmts)):
        ax = fig.add_subplot(gs[gi + 1])
        vals = [means[k][comp] for k, _ in labels]
        cols = ["#9AA0A6", "#D8A448", "#53A6A6", "#3D6FB6", PALETTE["purple"]]
        bars = ax.bar(range(5), vals, color=cols, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt % v, ha="center", va="bottom",
                    fontsize=6.0, rotation=0)
        ax.set_xticks(range(5))
        ax.set_xticklabels([lb for _, lb in labels], fontsize=5.8, rotation=25)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.22)
        soften_axes(ax)
    fig.text(0.5, -0.06, "三 UE 小规模设置；MC-ITR、Greedy、Iterative 与穷举精确解的逐分量均值相同（实现一致性检查，非主配置最优性证明）",
             ha="center", fontsize=6.8, color=PALETTE["baseline"])
    fig.subplots_adjust(bottom=0.30)
    save(fig, "图4-7_Exact-Small验证与快照构成", multi_panel=True)


def fig_bridge() -> None:
    """图4-8: shared-trajectory snapshot bridge (chapter 4 solver vs online lower layer)."""
    analysis = json.loads((REPO_ROOT / "results" / "bridge_20260928" / "full" / "analysis.json").read_text())
    units = analysis["units"]
    methods = [("online_lower", "在线学习下层", PALETTE["baseline"])] + CH4_METHODS

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9))
    rng = np.random.default_rng(4)
    for ax, key, title, fmt in [
            (axes[0], "deadline_misses", "(a) 每快照平均超时数", "%.2f"),
            (axes[1], "normalized_assignment_energy", "(b) 归一化分配能耗", "%.2f")]:
        for i, (mkey, label, color) in enumerate(methods):
            vals = np.array([u[mkey][key] for u in units])
            mean = float(vals.mean())
            ax.bar(i, mean, width=0.62, color=color, alpha=0.85, zorder=2)
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(i + jitter, vals, s=5, color=PALETTE["text"], alpha=0.4, zorder=3)
            ax.text(i, mean, fmt % mean, ha="center", va="bottom", fontsize=6.8, zorder=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([lb for _, lb, _ in methods], fontsize=6.0, rotation=20)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.18)
        soften_axes(ax)

    ax = axes[2]
    pair = analysis["mc_itr_minus_online_lower_deadline_misses"]
    diffs = np.array(pair["seed_mean_differences"])
    ci = pair["training_seed_t_ci95"]
    ax.axhline(0, color=PALETTE["axis"], lw=0.8)
    ax.bar(range(len(diffs)), diffs, color=PALETTE["full"], width=0.62)
    ax.axhline(float(pair["mean_difference"]), color=PALETTE["red"], lw=1.0, linestyle="--")
    for i, v in enumerate(diffs):
        ax.text(i, v - 0.13, f"{v:.2f}", ha="center", va="top", fontsize=6.2)
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([f"S{i+1}" for i in range(len(diffs))], fontsize=6.8)
    ax.set_xlabel("训练种子（排序）")
    ax.set_title("(c) MC-ITR $-$ 在线下层：逐种子差值", fontsize=8.5)
    ax.text(0.03, 0.04, f"均值 {pair['mean_difference']:.3f}，95% CI [{ci[0]:.3f}, {ci[1]:.3f}]；10/10 为负",
            transform=ax.transAxes, fontsize=6.8, color=PALETTE["red"])
    ax.margins(y=0.25)
    soften_axes(ax)
    axes[0].text(0.0, -0.40,
                 "50 个评估任务（10 训练种子 × 5 工作负载）× 8 冻结快照 = 400 快照；"
                 "快照方法在候选轨迹的冻结快照上离线求解，未回写环境",
                 transform=axes[0].transAxes, fontsize=6.6, color=PALETTE["baseline"])
    save(fig, "图4-8_候选轨迹快照桥接对比", multi_panel=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-root",
        type=Path,
        default=Path.home() / ".codex" / "worktrees" / "coverage-optimization" / "wodexuexi",
        help="coverage-optimization worktree root holding the confirmation batch",
    )
    args = parser.parse_args()
    setup_thesis_style(font_size=8)

    fig_system_overview()
    fig_slot_timeline()
    fig_network_structure()
    fig_observation_compare()
    fig_confirmation_seed_diff(args.coverage_root)
    fig_admission_quality_paired(args.coverage_root)
    fig_sharing_externality()
    fig_mc_itr_flowchart()
    fig_main_comparison()
    fig_paired_diff_iterative()
    fig_sensitivity("ue")
    fig_sensitivity("compute")
    fig_exact_small()
    fig_bridge()

    if ALIGN_REPORTS:
        out = QA_REPORT.parent / "figure_panel_alignment.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ALIGN_REPORTS, ensure_ascii=False, indent=2), encoding="utf-8")
        fails = [r for r in ALIGN_REPORTS if r["status"] != "pass"]
        print(f"alignment: {len(ALIGN_REPORTS) - len(fails)} pass, {len(fails)} fail")


if __name__ == "__main__":
    main()
