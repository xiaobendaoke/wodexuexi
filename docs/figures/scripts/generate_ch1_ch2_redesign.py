#!/usr/bin/env python3
"""Generate Chapter 1 and Chapter 2 redesigned figures (Figs 1-1, 2-1, 2-2, 2-3).

Canvas width is locked 1:1 to SYSU master's thesis \\textwidth = 146.6 mm (415.56 pt).
All outputs pass SVG schematic audit, panel alignment gate, PDF glyph floor (>= 5 pt),
and PyMuPDF rendered collision checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_schematic_builder import SchematicCanvas
from thesis_figure_style import (
    REDESIGN_PALETTE as P,
    TEXTWIDTH_IN,
    add_panel_tag,
    export_redesign_figure,
    setup_redesign_style,
    style_redesign_axis,
)

OUT_DIRS = [
    REPO_ROOT / "docs" / "figures",
    REPO_ROOT / "output" / "thesis_figure_redesign_20261112" / "deliverables",
]
QA_DIR = REPO_ROOT / "output" / "thesis_figure_redesign_20261112" / "validation"


def build_fig1_01_roadmap() -> dict[str, object]:
    """图 1-1：论文章节安排与总体技术路线组织结构图（自顶向下五层章节架构）。"""
    c = SchematicCanvas(height_mm=123.0)

    chapter_blue = "#2B5B9E"
    card_bg = "#E9EFF6"
    card_stroke = "#2B5B9E"
    divider_col = "#AEC0D6"
    group_stroke = "#3B6FB6"
    arrow_col = P["ink"]
    cx = c.w / 2.0  # 207.78 pt

    def draw_chapter_card(
        x: float,
        y: float,
        w: float,
        h: float,
        chap_chars: tuple[str, ...] | None,
        title: str,
        lines: list[str],
        *,
        strip_w: float = 22.0,
        title_size: float = 8.7,
        sub_size: float = 7.4,
    ) -> None:
        c.ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=card_bg,
                edgecolor=card_stroke,
                linewidth=0.95,
                joinstyle="miter",
                zorder=2.2,
            )
        )
        if chap_chars:
            c.ax.add_patch(
                Rectangle(
                    (x, y),
                    strip_w,
                    h,
                    facecolor=chapter_blue,
                    edgecolor=card_stroke,
                    linewidth=0.95,
                    joinstyle="miter",
                    zorder=2.5,
                )
            )
            mid_y = y + h / 2.0
            step_y = 10.4 if h >= 42.0 else 9.4
            offsets = [(i - (len(chap_chars) - 1) / 2.0) * step_y for i in range(len(chap_chars))]
            for ch, dy in zip(chap_chars, offsets):
                c.text(
                    x + strip_w / 2.0,
                    mid_y + dy,
                    ch,
                    fontsize=8.0,
                    color="#FFFFFF",
                    bold=True,
                    zorder=3.0,
                )
            rx = x + strip_w
            rw = w - strip_w
        else:
            rx = x
            rw = w

        rcx = rx + rw / 2.0
        title_h = 20.5
        c.text(rcx, y + title_h * 0.52, title, fontsize=title_size, color=P["ink"], bold=True, zorder=3.0)
        c.line((rx + 7.5, y + title_h), (x + w - 7.5, y + title_h), color=divider_col, lw=0.70, zorder=2.6)

        body_top = y + title_h
        body_h = h - title_h
        if len(lines) == 1:
            c.text(rcx, body_top + body_h * 0.52, lines[0], fontsize=sub_size, color=P["ink"], zorder=3.0)
        elif len(lines) == 2:
            c.text(rcx, body_top + body_h * 0.31, lines[0], fontsize=sub_size, color=P["ink"], zorder=3.0)
            c.text(rcx, body_top + body_h * 0.72, lines[1], fontsize=sub_size, color=P["ink"], zorder=3.0)

    def draw_group_frame(x: float, y: float, w: float, h: float, label: str) -> None:
        c.ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="none",
                edgecolor=group_stroke,
                linewidth=0.95,
                linestyle=(0, (5.0, 3.2)),
                joinstyle="miter",
                zorder=1.2,
            )
        )
        c.text(
            x + 8.5,
            y + 5.8,
            label,
            fontsize=8.4,
            color=chapter_blue,
            bold=True,
            ha="left",
            va="top",
            zorder=2.0,
        )

    # 1. Chapter 1: Introduction
    w1, h1 = 240.0, 39.0
    x1, y1 = cx - w1 / 2.0, 6.0
    draw_chapter_card(
        x1,
        y1,
        w1,
        h1,
        ("第", "一", "章"),
        "绪　论",
        ["课题背景与研究意义  ·  国内外研究现状  ·  主要研究内容"],
        title_size=8.9,
        sub_size=7.4,
    )

    # Group 1: System Modeling & Evaluation Foundation (Chapter 2)
    draw_group_frame(10.0, 56.0, c.w - 20.0, 68.0, "系统建模与评价基础")
    w2, h2 = 318.0, 41.0
    x2, y2 = cx - w2 / 2.0, 76.5
    draw_chapter_card(
        x2,
        y2,
        w2,
        h2,
        ("第", "二", "章"),
        "无人机集群空地协同计算系统建模与分析",
        ["自然覆盖与安全几何  |  多链路与三级执行路径  |  双协议负载与归因评价"],
        title_size=8.7,
        sub_size=7.4,
    )
    c.arrow((cx, y1 + h1 + 0.5), (cx, y2 - 0.5), color=arrow_col, lw=1.05)

    # Group 2: Cooperative Planning, Allocation & Dynamic Queuing Control (Chapters 3, 4, 5)
    draw_group_frame(10.0, 135.0, c.w - 20.0, 158.0, "协同规划与控制方法研究")

    w34, h34 = 181.0, 54.5
    x3, y34 = 16.0, 166.0
    x4 = c.w - 16.0 - w34
    cx3 = x3 + w34 / 2.0
    cx4 = x4 + w34 / 2.0

    draw_chapter_card(
        x3,
        y34,
        w34,
        h34,
        ("第", "三", "章"),
        "无人机集群需求感知协同轨迹规划",
        ["预接纳局部观测梯度盲区分析", "感知域未覆盖需求扩展与协同训练"],
        strip_w=20.5,
        title_size=8.2,
        sub_size=7.3,
    )
    draw_chapter_card(
        x4,
        y34,
        w34,
        h34,
        ("第", "四", "章"),
        "跨机资源竞争下的集群计算任务分配",
        ["同批算力共享跨机迁移耦合分析", "严格二元字典序与全批次协同迭代"],
        strip_w=20.5,
        title_size=8.2,
        sub_size=7.3,
    )

    # Split connector: Ch 2 -> Ch 3 & Ch 4 (clean miter corners via ortho_arrow + stem)
    y_split = 154.0
    c.line((cx, y2 + h2 + 0.5), (cx, y_split), color=arrow_col, lw=1.05, zorder=3.0)
    c.ortho_arrow([(cx, y_split), (cx3, y_split), (cx3, y34 - 0.5)], color=arrow_col, lw=1.05)
    c.ortho_arrow([(cx, y_split), (cx4, y_split), (cx4, y34 - 0.5)], color=arrow_col, lw=1.05)

    # Horizontal handoff arrow: Ch 3 -> Ch 4
    mid_y34 = y34 + h34 / 2.0
    c.arrow((x3 + w34 + 0.6, mid_y34), (x4 - 0.6, mid_y34), color=arrow_col, lw=1.05)

    # Chapter 5: Dynamic Queuing Joint Control
    w5, h5 = 344.0, 42.0
    x5, y5 = cx - w5 / 2.0, 244.0
    draw_chapter_card(
        x5,
        y5,
        w5,
        h5,
        ("第", "五", "章"),
        "动态排队服务中的规划蒸馏与反事实辅助协同控制",
        ["两阶段串联排队与同变双分支网络  |  规划蒸馏与反事实微调  |  神经引导滚动预测控制"],
        title_size=8.6,
        sub_size=7.3,
    )

    # Merge connector: Ch 3 & Ch 4 -> Ch 5 (clean orthogonal polyline + arrow)
    y_merge = 232.0
    c.ax.plot(
        [cx3, cx3, cx4, cx4],
        [y34 + h34 + 0.5, y_merge, y_merge, y34 + h34 + 0.5],
        color=arrow_col,
        linewidth=1.05,
        solid_joinstyle="miter",
        solid_capstyle="butt",
        zorder=3.0,
    )
    c.arrow((cx, y_merge), (cx, y5 - 0.5), color=arrow_col, lw=1.05)

    # Conclusion
    w6, h6 = 240.0, 37.5
    x6, y6 = cx - w6 / 2.0, 305.0
    draw_chapter_card(
        x6,
        y6,
        w6,
        h6,
        None,
        "结　论",
        ["主要研究成果总结  ·  研究局限性剖析  ·  后续工作展望"],
        title_size=8.9,
        sub_size=7.4,
    )
    c.arrow((cx, y5 + h5 + 0.5), (cx, y6 - 0.5), color=arrow_col, lw=1.05)

    return c.export("fig1_01_thesis_roadmap", repo_root=REPO_ROOT, output_dirs=OUT_DIRS, qa_dir=QA_DIR)


def build_fig2_01_spatial_kinematics() -> dict[str, object]:
    """图 2-1：端—空—地自然覆盖几何、感知嵌套与连续线段防碰安全修复机制。"""
    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.263))
    gs = fig.add_gridspec(1, 2, left=0.088, right=0.980, bottom=0.145, top=0.840, wspace=0.27)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # --- Panel (a): 700x700 m spatial natural association geometry ---
    style_redesign_axis(ax_a, grid="none", box=True)
    ax_a.set_xlim(0, 700)
    ax_a.set_ylim(0, 700)
    ax_a.set_xticks([0, 175, 350, 525, 700])
    ax_a.set_yticks([0, 175, 350, 525, 700])
    ax_a.set_xlabel("水平坐标 $x$ ($\\mathrm{m}$)")
    ax_a.set_ylabel("水平坐标 $y$ ($\\mathrm{m}$)")
    add_panel_tag(ax_a, "(a)", y_offset_pt=24.0)

    # Inset feasible domain Q = [50, 650]^2
    ax_a.add_patch(
        Rectangle((50, 50), 600, 600, facecolor="none", edgecolor=P["prior"], lw=0.85, ls="--", zorder=1)
    )
    ax_a.text(72, 628, "$\\mathcal{Q}=[50,650]^2$", fontsize=7.5, color=P["ink"], ha="left", va="top")

    # 5 UAV positions satisfying d_min >= 200 m (U1 and U2 tangent at exact 200 m = 2 R_c)
    uavs = np.array(
        [
            [180.0, 455.0],
            [380.0, 455.0],
            [550.0, 300.0],
            [185.0, 185.0],
            [415.0, 145.0],
        ]
    )
    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            assert np.linalg.norm(uavs[i] - uavs[j]) >= 200.0 - 1e-6

    # Sensing arc R_s = 460 m around UAV 1 computed explicitly inside [55, 645]^2 (no unclipped PDF bbox!)
    ang = np.linspace(-0.42 * np.pi, 0.14 * np.pi, 180)
    sx = uavs[0, 0] + 460.0 * np.cos(ang)
    sy = uavs[0, 1] + 460.0 * np.sin(ang)
    valid = (sx >= 55.0) & (sx <= 642.0) & (sy >= 55.0) & (sy <= 642.0)
    ax_a.plot(sx[valid], sy[valid], color=P["distilled"], lw=0.95, ls="-.", zorder=1.5)
    ax_a.text(535, 622, "$R_s=460\\,\\mathrm{m}$", fontsize=7.5, color=P["distilled"], ha="center", va="top")

    # Disjoint coverage discs R_c = 100 m; put U1..U5 labels cleanly outside markers
    for idx, (ux, uy) in enumerate(uavs, 1):
        ax_a.add_patch(
            Circle((ux, uy), radius=100.0, facecolor=P["hero_light"], edgecolor=P["hero"], lw=0.90, alpha=0.62, zorder=2)
        )
        ax_a.scatter([ux], [uy], s=20, color=P["hero"], marker="^", zorder=4)
        ax_a.text(ux, uy + 34, f"U{idx}", fontsize=7.5, fontweight="bold", color=P["hero"], ha="center", va="bottom", zorder=5)

    # Highlight d_min = 2 R_c = 200 m between U1 and U2 in clear gap below centers
    ax_a.plot([188, 372], [430, 430], color=P["crimson"], lw=0.9, ls=":", zorder=3.5)
    ax_a.text(280, 332, "$d_{\\min}=2R_c$", fontsize=7.5, color=P["crimson"], ha="center", va="top")

    # Deterministic UEs avoiding all text callouts and circle strokes
    rng = np.random.default_rng(20261112)
    raw_pts = np.vstack(
        [
            rng.normal([165, 440], [32, 24], size=(10, 2)),
            rng.normal([545, 255], [32, 24], size=(10, 2)),
            rng.uniform(75, 625, size=(30, 2)),
        ]
    )
    raw_pts = np.clip(raw_pts, 75, 585)
    keep = []
    for px, py in raw_pts:
        if 55 <= px <= 290 and 575 <= py <= 650:
            continue
        if 430 <= px <= 645 and 575 <= py <= 650:
            continue
        if 205 <= px <= 355 and 285 <= py <= 445:
            continue
        if any(abs(px - ux) < 42 and abs(py - (uy + 45)) < 32 for ux, uy in uavs):
            continue
        if any(abs(np.hypot(px - ux, py - uy) - 100.0) < 12.0 for ux, uy in uavs):
            continue
        if abs(px - 350) < 32 and abs(py - 350) < 32:
            continue
        keep.append((px, py))
    ue_pts = np.array(keep)
    dists = np.linalg.norm(ue_pts[:, None, :] - uavs[None, :, :], axis=-1)
    covered = dists.min(axis=1) <= 100.0
    ax_a.scatter(ue_pts[covered, 0], ue_pts[covered, 1], s=10, color=P["success"], label="已覆盖", zorder=3)
    ax_a.scatter(
        ue_pts[~covered, 0],
        ue_pts[~covered, 1],
        s=12,
        facecolors="white",
        edgecolors=P["comparator"],
        linewidths=0.8,
        label="未覆盖",
        zorder=3,
    )
    ax_a.scatter([350], [350], s=32, color=P["ink"], marker="s", zorder=5, label="MBS")
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=3,
        fontsize=7.5,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
    )

    # --- Panel (b): Safety distance trajectory d(tau) under raw proposal vs repair E(.) ---
    style_redesign_axis(ax_b, grid="none", box=True)
    add_panel_tag(ax_b, "(b)", y_offset_pt=24.0)

    tau = np.linspace(0.0, 1.0, 201)
    q1_0 = np.array([200.0, 300.0])
    q2_0 = np.array([404.0, 300.0])
    q1_raw = q1_0 + np.array([14.0, 3.0])
    q2_raw = q2_0 + np.array([-14.0, -3.0])

    mid = 0.5 * (q1_raw + q2_raw)
    unit_vec = (q2_raw - q1_raw) / np.linalg.norm(q2_raw - q1_raw)
    q1_rep = mid - 100.0 * unit_vec
    q2_rep = mid + 100.0 * unit_vec

    d_raw = np.linalg.norm(((1 - tau)[:, None] * q1_0 + tau[:, None] * q1_raw) - ((1 - tau)[:, None] * q2_0 + tau[:, None] * q2_raw), axis=1)
    d_rep = np.linalg.norm(((1 - tau)[:, None] * q1_0 + tau[:, None] * q1_rep) - ((1 - tau)[:, None] * q2_0 + tau[:, None] * q2_rep), axis=1)
    d_hov = np.full_like(tau, float(np.linalg.norm(q2_0 - q1_0)))

    ax_b.set_xlim(0.0, 1.06)
    ax_b.set_ylim(172.0, 214.0)
    # Keep shaded danger zone strictly inside ylim [172, 200] so raw path bbox never touches tick labels
    ax_b.add_patch(
        Rectangle((0.0, 172.0), 1.06, 28.0, facecolor="#FADBD8", edgecolor="none", alpha=0.40, zorder=0.5)
    )
    ax_b.plot([0.0, 1.06], [200.0, 200.0], color=P["crimson"], lw=1.05, ls="--", zorder=2)
    ax_b.plot(tau, d_raw, color=P["comparator"], lw=1.35, ls="-.", label="原始提案", zorder=3)
    ax_b.plot(tau, d_rep, color=P["hero"], lw=1.55, ls="-", label="对称推离 $\\mathcal{E}$", zorder=4)
    ax_b.plot(tau, d_hov, color=P["distilled"], lw=1.2, ls=":", label="悬停回退", zorder=3.5)

    ax_b.scatter([1.0], [d_raw[-1]], s=18, color=P["comparator"], zorder=5)
    ax_b.scatter([1.0], [d_rep[-1]], s=18, color=P["hero"], zorder=5)

    ax_b.text(0.05, 183.0, "碰撞禁区 $d(\\tau)<200\\,\\mathrm{m}$", fontsize=7.5, color=P["crimson"], ha="left", va="center")
    ax_b.text(0.92, 174.5, "$176.1\\,\\mathrm{m}$", fontsize=7.5, color=P["comparator"], ha="right", va="bottom")
    ax_b.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_b.set_yticks([175, 185, 200, 210])
    ax_b.set_xlabel("宏时隙归一化运动进度 $\\tau\\in[0,1]$")
    ax_b.set_ylabel("机间水平距离 $d_{uv}(\\tau)$ ($\\mathrm{m}$)")
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=3,
        fontsize=7.5,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.25,
        columnspacing=0.65,
    )

    return export_redesign_figure(
        fig,
        "fig2_01_spatial_kinematics",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b],
    )


def build_fig2_02_execution_contention() -> dict[str, object]:
    """图 2-2：请求接入后三级执行路径分流与通信—计算资源竞争耦合关系。"""
    c = SchematicCanvas(height_mm=79.0)

    c.container(
        8.0,
        6.0,
        106.0,
        188.0,
        "1. 自然接入与上行",
        stroke=P["border"],
        fill=P["panel_bg"],
    )
    c.box(
        14.0,
        32.0,
        94.0,
        44.0,
        "已覆盖服务请求",
        "$(L_m, C_m, D_m, s_m)$",
        stroke=P["ink"],
    )
    c.box(
        14.0,
        93.0,
        94.0,
        45.0,
        "空地上行接入",
        "等时分 $B^{\\mathrm{edge}}/N_{\\mathrm{act}}$",
        fill=P["hero_light"],
        stroke=P["hero"],
    )
    c.box(
        14.0,
        150.0,
        94.0,
        36.0,
        "接入无人机 $u\\in\\mathcal{U}$",
        "路由指定 $b_m\\in\\mathcal{J}$",
        stroke=P["hero"],
        bold_title=True,
    )
    c.arrow((61.0, 76.5), (61.0, 92.5), color=P["hero"], lw=1.1)
    c.arrow((61.0, 138.5), (61.0, 149.5), color=P["hero"], lw=1.1)

    # Middle column: 3 execution paths (w=160 pt -> inner 150 pt -> max 18 CJK chars)
    c.container(
        124.0,
        6.0,
        162.0,
        188.0,
        "2. 三类异构执行路径",
        stroke=P["hero"],
        fill="#F8FBFF",
        title_color=P["hero"],
    )
    c.box(
        131.0,
        30.0,
        148.0,
        46.0,
        "路径 I：本地执行 ($b_m=u$)",
        "未命中时回源拉取镜像 $B_s$",
        stroke=P["hero"],
    )
    c.box(
        131.0,
        85.0,
        148.0,
        46.0,
        "路径 II：邻机中继 ($b_m=j$)",
        "机间链路 $B^{\\mathrm{inter}}$ ($d_{uj}\\leq R_s$)",
        stroke=P["distilled"],
    )
    c.box(
        131.0,
        140.0,
        148.0,
        45.0,
        "路径 III：宏站回传 ($b_m=\\mathrm{MBS}$)",
        "经回传链路 $B^{\\mathrm{bh}}$ 至全库宏站",
        stroke=P["comparator"],
    )

    c.ortho_arrow([(108.5, 169.5), (118.0, 169.5), (118.0, 53.0), (130.5, 53.0)], color=P["hero"], lw=1.05)
    c.ortho_arrow([(108.5, 169.5), (118.0, 169.5), (118.0, 108.0), (130.5, 108.0)], color=P["distilled"], lw=1.05)
    c.arrow((108.5, 169.5), (130.5, 169.5), color=P["comparator"], lw=1.05)

    # Right column: Resource contention & deadline settlement (w=111.5 pt -> box w=97.5 pt)
    c.container(
        296.0,
        6.0,
        111.5,
        188.0,
        "3. 竞争与截止期",
        stroke=P["border"],
        fill=P["panel_bg"],
    )
    c.box(
        303.0,
        30.0,
        97.5,
        46.0,
        "缓存容量竞争",
        "$\\sum h_{j,s}B_s\\leq S_j^{\\mathrm{cache}}$",
        stroke=P["distilled"],
    )
    c.box(
        303.0,
        85.0,
        97.5,
        46.0,
        "节点算力竞争",
        "$F_j/N_j$ 或 EDF",
        fill=P["comp_light"],
        stroke=P["comparator"],
    )
    c.box(
        303.0,
        140.0,
        97.5,
        45.0,
        "截止期满足判定",
        "$T_m\\leq D_m\\Rightarrow S+1$",
        fill=P["dist_light"],
        stroke=P["success"],
        bold_title=True,
    )

    c.arrow((279.5, 53.0), (302.5, 53.0), color=P["hero"], lw=1.0)
    c.arrow((279.5, 108.0), (302.5, 108.0), color=P["distilled"], lw=1.0)
    c.arrow((279.5, 162.5), (302.5, 162.5), color=P["comparator"], lw=1.0)
    c.arrow((351.8, 76.5), (351.8, 84.5), color=P["ink"], lw=1.0)
    c.arrow((351.8, 131.5), (351.8, 139.5), color=P["ink"], lw=1.0)

    c.legend_row(
        209.0,
        [
            ("box:" + P["hero_light"], P["hero"], "通信带宽共享"),
            ("box:" + P["comp_light"], P["comparator"], "节点算力竞争"),
            ("box:" + P["dist_light"], P["success"], "按时完成结算"),
        ],
        x_start=26.0,
        x_end=395.0,
    )
    return c.export("fig2_02_execution_contention", repo_root=REPO_ROOT, output_dirs=OUT_DIRS, qa_dir=QA_DIR)


def build_fig2_03_dual_protocol_accounting() -> dict[str, object]:
    """图 2-3：单时隙批量结算协议与跨时隙持续排队协议对照及全量 DSR 归因口径。"""
    c = SchematicCanvas(height_mm=93.0)

    c.panel_tag(6.0, 5.0, "(a)")
    c.container(
        24.0,
        5.0,
        184.0,
        134.0,
        "协议 I：单时隙批量共享（第 3–4 章）",
        stroke=P["hero"],
        fill="#F8FBFF",
        title_color=P["hero"],
    )
    c.box(
        31.0,
        27.0,
        80.0,
        35.0,
        "时隙 $t$ 到达",
        "未覆盖即失败",
        stroke=P["crimson"],
    )
    c.box(
        120.0,
        27.0,
        81.0,
        35.0,
        "当步等份共享",
        "$f_{m,j}=F_j/N_j$",
        fill=P["hero_light"],
        stroke=P["hero"],
    )
    c.arrow((111.5, 44.5), (119.5, 44.5), color=P["hero"], lw=1.05)

    c.line((31.0, 70.0), (201.0, 70.0), color=P["crimson"], lw=0.90, ls="--")
    c.text(116.0, 79.5, "时隙末队列清零（剥离跨步积压干扰）", fontsize=7.5, color=P["crimson"])

    c.box(
        31.0,
        91.0,
        170.0,
        40.0,
        "时隙 $t+1$ 独立批次重新结算",
        "聚焦前端盲区与同批算力外部性",
        stroke=P["border"],
    )

    # Right protocol container: Protocol II
    c.container(
        218.0,
        5.0,
        189.5,
        134.0,
        "协议 II：持续 EDF 排队（第 5 章）",
        stroke=P["distilled"],
        fill="#F6FCF9",
        title_color=P["distilled"],
    )
    c.box(
        225.0,
        27.0,
        82.0,
        35.0,
        "未覆盖等待池",
        "入圈首次接纳",
        stroke=P["amber"],
    )
    c.box(
        316.0,
        27.0,
        84.5,
        35.0,
        "不可变关联传输",
        "$10\\times 0.1\\,\\mathrm{s}$ 微步",
        stroke=P["distilled"],
    )
    c.arrow((307.5, 44.5), (315.5, 44.5), color=P["distilled"], lw=1.05)

    c.arrow((386.0, 62.5), (386.0, 90.5), color=P["distilled"], lw=1.15)
    c.text(301.0, 76.5, "剩余传输/计算进度跨步保留", fontsize=7.5, color=P["distilled"], bold=True)

    c.box(
        225.0,
        91.0,
        175.5,
        40.0,
        "节点抢占式 EDF 队列 + 排空步",
        "$200$ 到达步 + 至多 $4$ 排空步",
        fill=P["dist_light"],
        stroke=P["distilled"],
    )

    # Panel (b): Full-request conservation & paired cluster inference
    c.panel_tag(6.0, 148.0, "(b)")
    c.container(
        24.0,
        147.0,
        383.5,
        108.0,
        "全量请求守恒分解与同源工作负载簇配对推断（引理 2.3.1）",
        stroke=P["ink"],
        fill=P["panel_bg"],
    )
    c.box(
        32.0,
        171.0,
        88.0,
        38.0,
        "全部生成请求 $G_b$",
        "同源外生工作负载流",
        stroke=P["ink"],
        bold_title=True,
    )
    c.box(
        134.0,
        167.0,
        94.0,
        24.0,
        "自然接纳 $A=\\alpha G$",
        fill=P["hero_light"],
        stroke=P["hero"],
    )
    c.box(
        134.0,
        197.0,
        94.0,
        24.0,
        "未接纳失败 $G-A$",
        fill="#FCE8E6",
        stroke=P["crimson"],
    )
    c.arrow((120.5, 183.0), (133.5, 179.0), color=P["hero"], lw=1.05)
    c.arrow((120.5, 196.0), (133.5, 209.0), color=P["crimson"], lw=1.05)

    c.box(
        242.0,
        167.0,
        157.5,
        24.0,
        "按时完成 $S=\\eta A$ ($\\mathrm{DSR}=\\alpha\\eta$)",
        fill=P["dist_light"],
        stroke=P["success"],
    )
    c.box(
        242.0,
        197.0,
        157.5,
        24.0,
        "接纳后超时 $A-S=(1-\\eta)A$",
        fill=P["comp_light"],
        stroke=P["comparator"],
    )
    c.arrow((228.5, 179.0), (241.5, 179.0), color=P["success"], lw=1.05)
    c.arrow((228.5, 179.0), (241.5, 209.0), color=P["comparator"], lw=1.05)

    c.text(
        215.8,
        238.5,
        "精确对称配对差分：$d_b = \\Delta\\mathrm{DSR}_b = \\bar{\\eta}_b\\Delta\\alpha_b + \\bar{\\alpha}_b\\Delta\\eta_b$  |  按独立种子/工作负载簇构造 $95\\%$ 置信区间",
        fontsize=7.5,
    )
    return c.export("fig2_03_dual_protocol_accounting", repo_root=REPO_ROOT, output_dirs=OUT_DIRS, qa_dir=QA_DIR)


def main() -> None:
    results = [
        build_fig1_01_roadmap(),
        build_fig2_01_spatial_kinematics(),
        build_fig2_02_execution_contention(),
        build_fig2_03_dual_protocol_accounting(),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
