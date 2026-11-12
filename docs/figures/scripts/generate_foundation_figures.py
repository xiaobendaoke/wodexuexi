"""Vector schematics for Chapter 1 and Chapter 2 foundations (zero empirical data)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_figure_style import SEMANTIC_PALETTE, resolve_qa_scripts_dir, setup_thesis_style, soften_axes

SKILL = resolve_qa_scripts_dir(REPO_ROOT)
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = REPO_ROOT / "docs" / "figures"
QA = REPO_ROOT / "output" / "thesis_figure_upgrade_20261111" / "validation"

BLUE = SEMANTIC_PALETTE["module_blue"]
TEAL = SEMANTIC_PALETTE["module_teal"]
GREEN = SEMANTIC_PALETTE["covered_ok"]
ORANGE = SEMANTIC_PALETTE["admitted_miss"]
RED = SEMANTIC_PALETTE["uncovered_expired"]
PURPLE = SEMANTIC_PALETTE["module_purple"]
GRAY = SEMANTIC_PALETTE["neutral_slate"]
DARK = SEMANTIC_PALETTE["text_dark"]
NAVY = "#1F4E79"
SLATE_EDGE = "#2B3A4A"


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    color=BLUE,
    fontsize=8.3,
    lw=1.0,
    alpha_fill="16",
    bold=False,
    pad=0.006,
):
    ax.add_patch(
        Rectangle(
            (x - pad, y - pad),
            w + 2 * pad,
            h + 2 * pad,
            facecolor=color + alpha_fill,
            edgecolor=color,
            lw=lw,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
        fontweight="bold" if bold else "normal",
        linespacing=1.28,
    )


def arrow(ax, a, b, color=DARK, ls="-", lw=1.05, scale=10):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=scale,
            lw=lw,
            linestyle=ls,
            color=color,
        )
    )


def clipped_circle(ax, center, radius, *, bounds, **kwargs):
    """Render only circle segments strictly inside bounds so PDF paths stay unclipped."""
    theta = np.linspace(0.0, 2.0 * np.pi, 2401)
    pts = np.asarray(center, dtype=float) + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    x0, x1, y0, y1 = bounds
    visible = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    points = pts.copy()
    points[~visible] = np.nan
    ax.plot(points[:, 0], points[:, 1], **kwargs)


def _export_and_audit(fig: plt.Figure, axes: list[plt.Axes], stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        json_out=QA / f"{stem}.alignment.json",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    pdf_path = OUT / f"{stem}.pdf"
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    res_text = subprocess.run(
        [
            sys.executable,
            str(SKILL / "audit_pdf_text.py"),
            str(pdf_path),
            "--min-pt",
            "5",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (QA / f"{stem}.pdf_text.json").write_text(res_text.stdout, encoding="utf-8")
    res_col = subprocess.run(
        [
            sys.executable,
            str(SKILL / "audit_figure_collisions.py"),
            str(pdf_path),
            "--json-out",
            str(QA / f"{stem}.collision.json"),
        ],
        capture_output=True,
        text=True,
    )
    if res_col.returncode != 0:
        raise RuntimeError(f"Collision audit failed for {stem}:\n{res_col.stdout}\n{res_col.stderr}")
    print(f"exported & audited: {stem}")


def render_upgraded_fig_2_1() -> None:
    """升级图 2-1：700x700 m 空地几何、互斥覆盖/感知范围与三类服务执行路径（含缓存分支）。"""
    fig = plt.figure(figsize=(8.6, 4.65))
    gs = fig.add_gridspec(
        1, 2, left=0.06, right=0.985, bottom=0.17, top=0.90, width_ratios=[0.96, 1.24], wspace=0.16
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_flow = fig.add_subplot(gs[0, 1])

    # ---- Panel (a): 700 x 700 m^2 spatial geometry & natural coverage ----
    uavs = [
        np.array([150.0, 540.0]),
        np.array([550.0, 540.0]),
        np.array([150.0, 160.0]),
        np.array([550.0, 160.0]),
        np.array([350.0, 555.0]),
    ]
    min_pair = min(
        float(np.linalg.norm(a - b)) for i, a in enumerate(uavs) for b in uavs[i + 1 :]
    )
    assert min_pair >= 200.0, f"Pairwise UAV distance violated: {min_pair}"
    mbs = np.array([350.0, 350.0])

    # Feasible inset Q = [50, 650]^2
    ax_map.add_patch(
        Rectangle(
            (50.0, 50.0),
            600.0,
            600.0,
            fill=False,
            edgecolor=GRAY,
            linestyle=":",
            lw=0.85,
            zorder=1,
        )
    )

    # Sensing arc (460 m) for UAV 3 at (150, 160)
    clipped_circle(
        ax_map,
        uavs[2],
        460.0,
        bounds=(8.0, 692.0, 8.0, 692.0),
        color=BLUE,
        linestyle="--",
        lw=1.05,
        zorder=2,
    )

    # Mutually disjoint 100 m coverage circles
    for pos in uavs:
        ax_map.add_patch(
            Circle(
                pos,
                100.0,
                facecolor=GREEN + "18",
                edgecolor=BLUE,
                lw=1.0,
                zorder=2,
            )
        )
        ax_map.plot(pos[0], pos[1], marker="^", color=BLUE, markersize=7.2, zorder=6)

    # Schematic 100 UEs (62 covered inside disks, 38 uncovered in clear open space)
    demo_rng = np.random.default_rng(20260201)
    ue_list: list[np.ndarray] = []
    counts_per_uav = [13, 12, 13, 12, 12]
    for center, count in zip(uavs, counts_per_uav):
        while sum(1 for p in ue_list if np.linalg.norm(p - center) <= 92.0) < count:
            ang = float(demo_rng.uniform(0.0, 2.0 * np.pi))
            rad = float(np.sqrt(demo_rng.uniform(0.12, 0.76)) * 86.0)
            cand = center + np.array([rad * np.cos(ang), rad * np.sin(ang)])
            # Avoid local-link segment inside U1
            if np.linalg.norm(cand - uavs[0]) < 65.0 and cand[0] < uavs[0][0] and cand[1] < uavs[0][1]:
                continue
            if 55.0 <= cand[0] <= 645.0 and 55.0 <= cand[1] <= 645.0:
                if all(np.linalg.norm(cand - q) >= 14.0 for q in ue_list):
                    ue_list.append(cand)

    # Forbidden label corridors so text labels have 100% stroke-free clearance
    forbidden_boxes = [
        (210.0, 490.0, 340.0, 435.0),  # MBS label zone above (350,350)
        (215.0, 485.0, 52.0, 118.0),   # sensing arc label zone
        (60.0, 560.0, 645.0, 695.0),   # top strip above Q=650
    ]
    while len(ue_list) < 100:
        cand = np.array([demo_rng.uniform(60.0, 640.0), demo_rng.uniform(60.0, 635.0)])
        if any(np.linalg.norm(cand - u) <= 112.0 for u in uavs):
            continue
        # Also keep clear of sensing circle boundary r in [435, 485] around uavs[2]
        d_u3 = float(np.linalg.norm(cand - uavs[2]))
        if 435.0 <= d_u3 <= 485.0:
            continue
        # Keep clear of backhaul segment U4(550,160)->MBS(350,350)
        if 340.0 <= cand[0] <= 560.0 and 150.0 <= cand[1] <= 360.0:
            t_seg = np.clip(np.dot(cand - mbs, uavs[3] - mbs) / np.dot(uavs[3] - mbs, uavs[3] - mbs), 0.0, 1.0)
            if np.linalg.norm(cand - (mbs + t_seg * (uavs[3] - mbs))) < 26.0:
                continue
        if any(x0 <= cand[0] <= x1 and y0 <= cand[1] <= y1 for x0, x1, y0, y1 in forbidden_boxes):
            continue
        if all(np.linalg.norm(cand - q) >= 17.0 for q in ue_list):
            ue_list.append(cand)

    pts = np.asarray(ue_list)
    assert len(pts) == 100
    covered_mask = np.array([any(np.linalg.norm(p - u) <= 100.0 for u in uavs) for p in pts])

    ax_map.scatter(pts[covered_mask, 0], pts[covered_mask, 1], s=10, color=GREEN, alpha=0.88, zorder=4)
    ax_map.scatter(
        pts[~covered_mask, 0],
        pts[~covered_mask, 1],
        s=13,
        marker="x",
        linewidths=0.85,
        color=RED,
        alpha=0.85,
        zorder=4,
    )

    # Three representative links on map matching panel (b) line styles
    local_ue = uavs[0] + np.array([-52.0, -48.0])
    ax_map.plot([local_ue[0], uavs[0][0]], [local_ue[1], uavs[0][1]], color=GREEN, lw=1.45, ls="-", zorder=5)
    ax_map.plot([uavs[0][0], uavs[4][0]], [uavs[0][1], uavs[4][1]], color=TEAL, lw=1.45, ls="--", zorder=5)
    ax_map.plot([uavs[3][0], mbs[0]], [uavs[3][1], mbs[1]], color=BLUE, lw=1.45, ls="-.", zorder=5)

    # MBS marker & labels placed strictly inside stroke-free corridors
    ax_map.plot(mbs[0], mbs[1], marker="D", color=PURPLE, markersize=7.2, zorder=7)
    ax_map.text(350.0, 388.0, "MBS $(350,350,30)$ m", ha="center", va="center", fontsize=7.9, color=PURPLE)
    ax_map.text(350.0, 80.0, "感知弧 $R_{\\mathrm{sense}}=460$ m", ha="center", va="center", fontsize=7.9, color=BLUE)
    ax_map.text(350.0, 675.0, "互斥覆盖圆 $R_{\\mathrm{cov}}=100$ m", ha="center", va="center", fontsize=7.9, color=BLUE)

    ax_map.set_xlim(0.0, 700.0)
    ax_map.set_ylim(0.0, 700.0)
    ax_map.set_xticks([0, 175, 350, 525, 700])
    ax_map.set_yticks([0, 175, 350, 525, 700])
    ax_map.set_xlabel("$x$ / m", fontsize=8.4)
    ax_map.set_ylabel("$y$ / m", fontsize=8.4)
    ax_map.set_title("(a) $700\\times 700\\,\\mathrm{m}^2$ 空地几何与自然覆盖", fontsize=8.8, pad=7)
    soften_axes(ax_map, axis="none")

    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=BLUE, markersize=6.8, label="UAV（5 架）"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PURPLE, markersize=6.2, label="固定 MBS"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=5.5, label="已覆盖 UE"),
        Line2D([0], [0], marker="x", color=RED, linestyle="None", markersize=5.5, label="未覆盖 UE"),
    ]
    ax_map.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.165),
        ncol=2,
        fontsize=7.9,
        frameon=True,
        fancybox=False,
        edgecolor="#4A525D",
        columnspacing=0.9,
        handletextpad=0.35,
    )

    # ---- Panel (b): Three execution paths & cache hit/miss branches ----
    ax_flow.set_xlim(0.0, 1.0)
    ax_flow.set_ylim(0.0, 1.0)
    ax_flow.axis("off")
    ax_flow.set_title("(b) 三类执行路径与机载服务缓存分支", fontsize=8.8, pad=7)

    # Wide padded boxes so text stays comfortably inside borders
    box(
        ax_flow,
        0.01,
        0.34,
        0.24,
        0.32,
        "已覆盖终端\n$(L_m,C_m,D_m)$\n上行接入机 $u$",
        color=GREEN,
        fontsize=7.9,
    )
    arrow(ax_flow, (0.26, 0.50), (0.31, 0.50), color=GREEN, ls="-", lw=1.25)
    box(
        ax_flow,
        0.32,
        0.36,
        0.18,
        0.28,
        "接入 UAV $u$\n卸载决策\n$b_m$",
        color=BLUE,
        fontsize=7.9,
    )

    # Branch 1a: Local cache hit
    box(
        ax_flow,
        0.56,
        0.79,
        0.42,
        0.18,
        "① 本地命中（$b_m=u,\\ h_{u,s}=1$）\n免回源，机载 $F_u$ 直接计算",
        color=GREEN,
        fontsize=7.9,
    )
    # Branch 1b: Local cache miss
    box(
        ax_flow,
        0.56,
        0.54,
        0.42,
        0.20,
        "① 本地未命中（$b_m=u,\\ h_{u,s}=0$）\n回源获取镜像 $B_s$ 后由 $F_u$ 计算",
        color=ORANGE,
        fontsize=7.9,
    )
    arrow(ax_flow, (0.51, 0.57), (0.55, 0.87), color=GREEN, ls="-", lw=1.25)
    arrow(ax_flow, (0.51, 0.52), (0.55, 0.64), color=GREEN, ls="-", lw=1.25)

    # Branch 2: Neighbor UAV relay
    box(
        ax_flow,
        0.56,
        0.28,
        0.42,
        0.21,
        "② 邻机中继（$b_m=j$）\n机间链路 $(u,j)$ 转发 + $F_j$ 计算",
        color=TEAL,
        fontsize=7.9,
    )
    arrow(ax_flow, (0.51, 0.46), (0.55, 0.385), color=TEAL, ls="--", lw=1.3)

    # Branch 3: MBS backhaul
    box(
        ax_flow,
        0.56,
        0.03,
        0.42,
        0.20,
        "③ 宏基站回传（$b_m=\\mathrm{MBS}$）\n空对地回传 + 全量库 $F_{\\mathrm{MBS}}$ 计算",
        color=BLUE,
        fontsize=7.9,
    )
    arrow(ax_flow, (0.51, 0.40), (0.55, 0.13), color=BLUE, ls="-.", lw=1.3)

    path_handles = [
        Line2D([0], [0], color=GREEN, lw=1.5, ls="-", label="实线：本地执行"),
        Line2D([0], [0], color=TEAL, lw=1.5, ls="--", label="虚线：邻机中继"),
        Line2D([0], [0], color=BLUE, lw=1.5, ls="-.", label="点划线：MBS 回传"),
    ]
    ax_flow.legend(
        handles=path_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.075),
        ncol=3,
        fontsize=7.9,
        frameon=True,
        fancybox=False,
        edgecolor="#4A525D",
        columnspacing=0.8,
        handletextpad=0.35,
    )

    _export_and_audit(fig, [ax_map, ax_flow], "图2-1_MEC三层架构示意图")


def render_kinematic_collision_repair() -> None:
    """新增 2.2.2 节：内缩可行域 Q、15 m 步长圆、对称外推排斥与残余冲突原位悬停回退。"""
    fig = plt.figure(figsize=(8.5, 4.15))
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, bottom=0.14, top=0.89, width_ratios=[0.95, 1.15], wspace=0.20)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # ---- Panel (a): Feasible domain Q=[50,650]^2 & speed-ball/boundary projection ----
    ax_left.set_xlim(0.0, 700.0)
    ax_left.set_ylim(0.0, 700.0)
    ax_left.add_patch(Rectangle((0, 0), 700, 700, facecolor=RED + "10", edgecolor=DARK, lw=0.85))
    ax_left.add_patch(
        Rectangle((50, 50), 600, 600, facecolor="white", edgecolor=BLUE, lw=1.2, zorder=2)
    )
    ax_left.text(350.0, 608.0, "内缩安全可行域 $\\mathcal{Q}=[50,650]^2\\,\\mathrm{m}^2$", ha="center", va="center", fontsize=8.0, color=BLUE, zorder=5)
    ax_left.text(350.0, 23.0, "$50\\,\\mathrm{m}$ 边界安全内缩带", ha="center", va="center", fontsize=7.9, color=RED, zorder=5)

    q_cur = np.array([155.0, 250.0])
    r_vis = 92.0
    ax_left.add_patch(
        Circle(q_cur, r_vis, facecolor=TEAL + "15", edgecolor=TEAL, ls="--", lw=1.1, zorder=3)
    )
    q_raw = np.array([24.0, 340.0])
    q_clip_bound = np.array([50.0, 340.0])
    vec = q_clip_bound - q_cur
    q_proj = q_cur + vec / np.linalg.norm(vec) * r_vis

    ax_left.plot(*q_cur, marker="^", color=BLUE, markersize=7.8, zorder=6)
    ax_left.plot(*q_raw, marker="x", color=RED, markersize=7.0, mew=1.3, zorder=6)
    ax_left.plot(*q_proj, marker="^", color=GREEN, markersize=7.8, zorder=6)

    ax_left.add_patch(
        FancyArrowPatch(
            tuple(q_cur), tuple(q_raw), arrowstyle="-|>", mutation_scale=10, lw=1.1, ls="--", color=RED, zorder=5
        )
    )
    ax_left.add_patch(
        FancyArrowPatch(
            tuple(q_cur), tuple(q_proj), arrowstyle="-|>", mutation_scale=10, lw=1.25, ls="-", color=GREEN, zorder=6
        )
    )

    # All text labels strictly outside circle stroke (x >= 265 or y >= 390)
    ax_left.text(270.0, 265.0, "当前位置 $\\mathbf{q}_u(t)$", ha="left", va="center", fontsize=8.0, color=DARK)
    ax_left.text(270.0, 165.0, "步长圆 $v_{\\max}\\Delta t=15\\,\\mathrm{m}$\n（局部放大示意）", ha="left", va="center", fontsize=7.9, color=TEAL)
    ax_left.text(85.0, 415.0, "越界超步长提案 $\\tilde{\\mathbf{q}}_u$", ha="left", va="center", fontsize=8.0, color=RED)
    ax_left.text(195.0, 355.0, "边界与步长球重投影", ha="left", va="center", fontsize=8.0, color=GREEN)

    ax_left.set_xticks([0, 175, 350, 525, 700])
    ax_left.set_yticks([0, 175, 350, 525, 700])
    ax_left.set_xlabel("$x$ / m", fontsize=8.4)
    ax_left.set_ylabel("$y$ / m", fontsize=8.4)
    ax_left.set_title("(a) 内缩可行域 $\\mathcal{Q}$ 与单步最大位移圆裁剪", fontsize=8.8, pad=7)
    soften_axes(ax_left, axis="none")

    # ---- Panel (b): Symmetric repulsion when d < d_min=200m + residual hover fallback ----
    ax_right.set_xlim(0.0, 1.0)
    ax_right.set_ylim(0.0, 1.0)
    ax_right.axis("off")
    ax_right.set_title("(b) 对称外推排斥向量与残余冲突悬停回退", fontsize=8.8, pad=7)

    ax_right.add_patch(
        Rectangle(
            (0.015, 0.465), 0.97, 0.505, facecolor="#F8FAFC", edgecolor=SLATE_EDGE, lw=0.95, ls="--"
        )
    )
    ax_right.text(
        0.50, 0.895, "步骤 1：机间距 $d < d_{\\min}=200\\,\\mathrm{m}$ 时连线对称外推排斥",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color=DARK
    )

    pu = (0.38, 0.69)
    pv = (0.62, 0.69)
    fu = (0.18, 0.69)
    fv = (0.82, 0.69)
    ax_right.plot([pu[0], pv[0]], [pu[1], pv[1]], color=RED, ls=":", lw=1.2)
    ax_right.plot(*pu, marker="o", color=RED, markersize=6.2)
    ax_right.plot(*pv, marker="o", color=RED, markersize=6.2)
    ax_right.plot(*fu, marker="^", color=GREEN, markersize=7.8)
    ax_right.plot(*fv, marker="^", color=GREEN, markersize=7.8)

    arrow(ax_right, (0.355, 0.69), (0.21, 0.69), color=GREEN, ls="-", lw=1.35)
    arrow(ax_right, (0.645, 0.69), (0.79, 0.69), color=GREEN, ls="-", lw=1.35)

    ax_right.text(0.50, 0.77, "冲突间距 $d<200\\,\\mathrm{m}$", ha="center", va="center", fontsize=7.9, color=RED)
    ax_right.text(0.26, 0.59, "$-0.5(d_{\\min}-d)\\hat{\\mathbf{e}}_{uv}$", ha="center", va="center", fontsize=7.9, color=GREEN)
    ax_right.text(0.74, 0.59, "$+0.5(d_{\\min}-d)\\hat{\\mathbf{e}}_{uv}$", ha="center", va="center", fontsize=7.9, color=GREEN)
    ax_right.text(0.50, 0.515, "每轮推离后重投影至 $\\mathcal{Q}$ 与 $15\\,\\mathrm{m}$ 步长圆（最多 40 轮）", ha="center", va="center", fontsize=7.9, color=GRAY)

    ax_right.add_patch(
        Rectangle(
            (0.015, 0.025), 0.97, 0.405, facecolor="#FFF9F5", edgecolor=ORANGE, lw=0.95, ls="--"
        )
    )
    ax_right.text(
        0.50, 0.365, "步骤 2：有限轮迭代后残余冲突的原位悬停回退保护",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color=DARK
    )
    box(
        ax_right, 0.05, 0.08, 0.40, 0.21,
        "残余冲突检测\n$d < 200\\,\\mathrm{m}-10^{-3}\\,\\mathrm{m}$",
        color=RED, fontsize=7.9
    )
    arrow(ax_right, (0.465, 0.185), (0.535, 0.185), color=ORANGE, ls="-", lw=1.25)
    box(
        ax_right, 0.55, 0.08, 0.40, 0.21,
        "回退上一可行位置\n$\\mathbf{q}_u(t+1)\\leftarrow\\mathbf{q}_u(t)$",
        color=GREEN, fontsize=7.9
    )

    _export_and_audit(fig, [ax_left, ax_right], "ch2_kinematic_collision_repair")


def render_dual_protocol_comparison() -> None:
    """新增 2.3.2 节：协议 I（单步批量等份共享清零）与协议 II（跨步串联排队保留+末尾排空）上下分栏对照。"""
    fig = plt.figure(figsize=(8.6, 5.1))
    gs = fig.add_gridspec(2, 1, left=0.025, right=0.985, bottom=0.04, top=0.95, hspace=0.22)
    ax_p1 = fig.add_subplot(gs[0, 0])
    ax_p2 = fig.add_subplot(gs[1, 0])

    # ---- Top Panel: Protocol I ----
    ax_p1.set_xlim(0.0, 1.0)
    ax_p1.set_ylim(0.0, 1.0)
    ax_p1.axis("off")
    ax_p1.add_patch(
        Rectangle(
            (0.008, 0.03), 0.984, 0.92, facecolor="#F8FAFD", edgecolor=BLUE, lw=1.0, ls="--"
        )
    )
    ax_p1.text(
        0.03, 0.865,
        "(a) 协议 I：单时隙批量等份算力共享与当步清零协议（第 3、4 章）",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=BLUE
    )

    box(ax_p1, 0.03, 0.48, 0.19, 0.28, "时隙 $t$ 到达请求\n自然覆盖判定", color=BLUE, fontsize=7.9)
    box(ax_p1, 0.03, 0.09, 0.19, 0.25, "当步未覆盖请求\n立即记未接纳失败", color=RED, fontsize=7.9)
    arrow(ax_p1, (0.125, 0.465), (0.125, 0.355), color=RED, ls="-", lw=1.2)

    arrow(ax_p1, (0.23, 0.62), (0.275, 0.62), color=GREEN, ls="-", lw=1.25)
    box(
        ax_p1, 0.285, 0.42, 0.29, 0.36,
        "已接纳批次 $\\mathcal{M}_t$ 同步结算\n等份共享 $f_{mj}=F_j/N_j(\\mathbf{b})$\n计算时延 $C_m N_j(\\mathbf{b})/F_j$",
        color=TEAL, fontsize=7.9
    )
    arrow(ax_p1, (0.585, 0.66), (0.625, 0.69), color=GREEN, ls="-", lw=1.2)
    arrow(ax_p1, (0.585, 0.52), (0.625, 0.28), color=ORANGE, ls="-", lw=1.2)

    box(ax_p1, 0.635, 0.54, 0.16, 0.26, "按时完成\n$T_m(\\mathbf{b})\\leq D_m$", color=GREEN, fontsize=7.9)
    box(ax_p1, 0.635, 0.13, 0.16, 0.26, "已接纳超时\n$T_m(\\mathbf{b})>D_m$", color=ORANGE, fontsize=7.9)

    ax_p1.plot([0.818, 0.818], [0.10, 0.81], color=RED, ls="--", lw=1.25)
    arrow(ax_p1, (0.824, 0.48), (0.844, 0.48), color=GRAY, ls="-", lw=1.15)
    box(
        ax_p1, 0.85, 0.22, 0.125, 0.52,
        "时隙末清零\n队列不跨步\n进入 $t+1$",
        color=GRAY, fontsize=7.9
    )

    # ---- Bottom Panel: Protocol II ----
    ax_p2.set_xlim(0.0, 1.0)
    ax_p2.set_ylim(0.0, 1.0)
    ax_p2.axis("off")
    ax_p2.add_patch(
        Rectangle(
            (0.008, 0.03), 0.984, 0.92, facecolor="#F7FBF9", edgecolor=TEAL, lw=1.0, ls="--"
        )
    )
    ax_p2.text(
        0.03, 0.865,
        "(b) 协议 II：跨时隙持续 EDF 串联排队与末尾排空协议（第 5 章）",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=TEAL
    )

    box(
        ax_p2, 0.03, 0.45, 0.21, 0.33,
        "未覆盖驻留待接纳池\n$\\mathcal{W}_t$ 跨步等待\n入 $100\\,\\mathrm{m}$ 即接纳",
        color=BLUE, fontsize=7.9
    )
    box(
        ax_p2, 0.03, 0.09, 0.21, 0.24,
        "截止期 $d_n$ 前未覆盖\n待接纳超期淘汰",
        color=RED, fontsize=7.9
    )
    arrow(ax_p2, (0.135, 0.435), (0.135, 0.345), color=RED, ls="-", lw=1.2)

    arrow(ax_p2, (0.25, 0.615), (0.285, 0.615), color=GREEN, ls="-", lw=1.25)
    box(
        ax_p2, 0.295, 0.45, 0.23, 0.33,
        "阶段 1：无线传输队列\n共享上传 + 转发/回传\n剩余比特跨步保留",
        color=TEAL, fontsize=7.9
    )
    arrow(ax_p2, (0.535, 0.615), (0.565, 0.615), color=TEAL, ls="-", lw=1.25)
    box(
        ax_p2, 0.575, 0.45, 0.23, 0.33,
        "阶段 2：先占 EDF 计算\n按 $d_n$ 升序独占 $F_j$\n剩余周期跨步保留",
        color=GREEN, fontsize=7.9
    )
    ax_p2.text(
        0.55, 0.20,
        "跨时隙进度连续传递（$[0, 200]\\,\\mathrm{s}$），到期时刻 $t=d_n$ 未完立即过期清除",
        ha="center", va="center", fontsize=7.9, color=ORANGE
    )

    arrow(ax_p2, (0.815, 0.615), (0.835, 0.615), color=GREEN, ls="-", lw=1.25)
    box(
        ax_p2, 0.845, 0.14, 0.13, 0.65,
        "末尾排空段\n$[200,204]\\,\\mathrm{s}$\n停止新到达\n在途任务\n完整结算",
        color=PURPLE, fontsize=7.9
    )

    _export_and_audit(fig, [ax_p1, ax_p2], "ch2_dual_protocol_comparison")


def render_legacy_card(number: int) -> None:
    """Render 图2-2 and 图2-3 foundation block diagrams with crisp right-angle engineering boxes."""
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.07, top=0.94)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    if number == 2:
        name = "UAV辅助通信应用场景"
        box(ax, 0.03, 0.69, 0.28, 0.21, "灾后或临时热点\n快速形成空中接入", TEAL, fontsize=9.8)
        box(ax, 0.36, 0.69, 0.28, 0.21, "巡检与感知需求\n数据进入计算链路", TEAL, fontsize=9.8)
        box(ax, 0.69, 0.69, 0.28, 0.21, "移动用户与覆盖变化\n持续调整服务位置", TEAL, fontsize=9.8)
        box(ax, 0.18, 0.29, 0.64, 0.22, "共同决策关系\n位置影响接纳与链路，卸载影响负载与完成时间", BLUE, fontsize=9.8)
        ax.plot([0.17, 0.17, 0.83, 0.83], [0.678, 0.595, 0.595, 0.678], color=SLATE_EDGE, lw=1.05)
        ax.plot([0.50, 0.50], [0.678, 0.595], color=SLATE_EDGE, lw=1.05)
        arrow(ax, (0.50, 0.595), (0.50, 0.522), color=SLATE_EDGE, lw=1.1)
        box(ax, 0.10, 0.04, 0.80, 0.14, "本文评价：请求能否自然接纳，以及接纳后能否按时完成", GRAY, fontsize=9.8)
        arrow(ax, (0.50, 0.278), (0.50, 0.192), color=SLATE_EDGE, lw=1.1)
    else:
        name = "强化学习智能体环境交互框架"
        box(ax, 0.03, 0.34, 0.34, 0.29, "智能体／参数化策略\n根据观测选择动作", BLUE, fontsize=9.8)
        box(ax, 0.63, 0.34, 0.34, 0.29, "环境／状态转移\n执行动作并产生反馈", TEAL, fontsize=9.8)
        arrow(ax, (0.382, 0.52), (0.618, 0.52), color=SLATE_EDGE, lw=1.15)
        ax.text(0.50, 0.585, "动作 $a_t$", ha="center", fontsize=10.2, color=DARK)
        ax.plot([0.79, 0.79, 0.20], [0.328, 0.125, 0.125], color=SLATE_EDGE, lw=1.1)
        arrow(ax, (0.20, 0.125), (0.20, 0.328), color=SLATE_EDGE, lw=1.15)
        ax.text(0.50, 0.195, "下一观测 $o_{t+1}$、奖励 $r_t$、终止标记", ha="center", fontsize=9.6, color=DARK)
        box(ax, 0.24, 0.76, 0.52, 0.14, "训练阶段：采样交互轨迹 → 价值估计与策略更新", GRAY, fontsize=9.6)
        arrow(ax, (0.20, 0.642), (0.20, 0.83), color=GRAY, ls="--", lw=1.0)
        arrow(ax, (0.20, 0.83), (0.232, 0.83), color=GRAY, ls="--", lw=1.0)
    title = f"图2-{number}_{name}"
    _export_and_audit(fig, [ax], title)


def render_organization() -> None:
    """图1-1：工科框线质感章节组织架构图（深海蓝竖排章签 + 虚线研究阶段分组 + 正交折线箭头）。"""
    fig, ax = plt.subplots(figsize=(7.6, 5.65))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    name = "论文章节安排与组织结构"

    def header_bar(y: float, h: float, tab_text: str, body_text: str) -> None:
        ax.add_patch(
            Rectangle((0.060, y), 0.130, h, facecolor=NAVY, edgecolor=SLATE_EDGE, lw=1.0)
        )
        ax.text(
            0.125,
            y + h / 2,
            tab_text,
            ha="center",
            va="center",
            fontsize=9.4,
            fontweight="bold",
            color="white",
        )
        ax.add_patch(
            Rectangle((0.190, y), 0.750, h, facecolor="#EBF1F8", edgecolor=SLATE_EDGE, lw=1.0)
        )
        ax.text(
            0.565,
            y + h / 2,
            body_text,
            ha="center",
            va="center",
            fontsize=8.8,
            color=DARK,
            linespacing=1.32,
        )

    def chapter_row(y: float, h: float, chap_vertical: str, chap_title: str, bullets: str) -> None:
        yc = y + h / 2
        ax.add_patch(
            Rectangle((0.188, y), 0.052, h, facecolor=NAVY, edgecolor=SLATE_EDGE, lw=1.0)
        )
        ax.text(
            0.214,
            yc,
            chap_vertical,
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="white",
            linespacing=1.14,
        )
        ax.add_patch(
            Rectangle((0.240, y), 0.230, h, facecolor="#F2F6FA", edgecolor=SLATE_EDGE, lw=1.0)
        )
        ax.text(
            0.355,
            yc,
            chap_title,
            ha="center",
            va="center",
            fontsize=8.7,
            fontweight="bold",
            color=DARK,
            linespacing=1.32,
        )
        arrow(ax, (0.474, yc), (0.496, yc), color=SLATE_EDGE, lw=1.05, scale=9)
        ax.add_patch(
            Rectangle((0.500, y), 0.460, h, facecolor="#FAFBFD", edgecolor=SLATE_EDGE, lw=1.0)
        )
        ax.text(
            0.514,
            yc,
            bullets,
            ha="left",
            va="center",
            fontsize=8.3,
            color=DARK,
            linespacing=1.30,
        )

    # 1. 第一章：绪论
    header_bar(
        0.896,
        0.086,
        "第一章",
        "绪  论\n课题背景与研究意义  |  国内外研究现状综述  |  主要研究内容与章节组织",
    )
    arrow(ax, (0.50, 0.892), (0.50, 0.858), color=SLATE_EDGE, lw=1.15)

    # 2. 阶段一：系统建模基础研究（虚线外框）
    ax.add_patch(
        Rectangle((0.025, 0.680), 0.950, 0.174, facecolor="none", edgecolor=NAVY, lw=1.0, ls="--")
    )
    ax.add_patch(
        Rectangle((0.040, 0.698), 0.116, 0.138, facecolor="#E8EEF6", edgecolor=SLATE_EDGE, lw=1.0)
    )
    ax.text(
        0.098,
        0.767,
        "系统建模\n基础研究",
        ha="center",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color=NAVY,
        linespacing=1.32,
    )
    arrow(ax, (0.160, 0.767), (0.184, 0.767), color=SLATE_EDGE, lw=1.05, scale=9)
    chapter_row(
        0.698,
        0.138,
        "第\n二\n章",
        "无人机集群空地协同\n计算系统建模与分析",
        "• 自然覆盖几何关联、离散机动与无线信道建模\n• 协议 I（批量等份共享）与协议 II（跨时隙排队）\n• 截止期满足率双因子分解与同源配对评价准则",
    )

    arrow(ax, (0.50, 0.676), (0.50, 0.644), color=SLATE_EDGE, lw=1.15)

    # 3. 阶段二：协同规划与控制技术研究（虚线外框 + 正交总线箭头）
    ax.add_patch(
        Rectangle((0.025, 0.134), 0.950, 0.506, facecolor="none", edgecolor=NAVY, lw=1.0, ls="--")
    )
    ax.add_patch(
        Rectangle((0.040, 0.154), 0.116, 0.466, facecolor="#E8EEF6", edgecolor=SLATE_EDGE, lw=1.0)
    )
    ax.text(
        0.098,
        0.387,
        "协同规划\n与控制\n技术研究",
        ha="center",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color=NAVY,
        linespacing=1.38,
    )
    ax.plot([0.156, 0.171], [0.387, 0.387], color=SLATE_EDGE, lw=1.05)
    ax.plot([0.171, 0.171], [0.223, 0.551], color=SLATE_EDGE, lw=1.05)
    for yc in (0.551, 0.387, 0.223):
        arrow(ax, (0.171, yc), (0.184, yc), color=SLATE_EDGE, lw=1.05, scale=9)

    chapter_row(
        0.482,
        0.138,
        "第\n三\n章",
        "无人机集群需求感知\n协同轨迹规划",
        "• 原始已关联观测下未覆盖需求混叠瓶颈诊断\n• 感知半径内未覆盖终端按距离排序扩展观测\n• 上层关系自注意力轨迹与下层无注意力精简卸载",
    )
    chapter_row(
        0.318,
        0.138,
        "第\n四\n章",
        "跨机资源竞争下的无人机\n集群计算任务分配",
        "• 单时隙批量等份共享下的跨机迁移时延外部性\n• 超时请求数优先与归一化时延次之二元字典序\n• 全批次重评估多轮协同迭代（MC-ITR）与桥接",
    )
    chapter_row(
        0.154,
        0.138,
        "第\n五\n章",
        "动态排队服务中的规划蒸馏\n与反事实辅助协同控制",
        "• 释放时间工作守恒 EDF 串联排队与等变双分支\n• 两轮 DAgger 规划蒸馏与共同随机数反事实微调\n• 零前瞻快速推理与温度校准神经引导 Top-6 MPC",
    )

    arrow(ax, (0.50, 0.130), (0.50, 0.098), color=SLATE_EDGE, lw=1.15)

    # 4. 结论
    header_bar(
        0.012,
        0.082,
        "结  论",
        "全文主要研究工作总结  |  论文核心发现与方法贡献归纳  |  后续研究工作展望",
    )
    _export_and_audit(fig, [ax], f"图1-1_{name}")


if __name__ == "__main__":
    family = setup_thesis_style(8.4)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family],
            "font.sans-serif": [family],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    render_organization()
    render_upgraded_fig_2_1()
    render_kinematic_collision_repair()
    render_dual_protocol_comparison()
    for num in (2, 3):
        render_legacy_card(num)
