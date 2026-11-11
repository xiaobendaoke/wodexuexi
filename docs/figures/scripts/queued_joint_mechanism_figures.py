"""Mechanism schematics and confirmation queue-evolution plot for Chapter 5 (Protocol II).

All mechanism diagrams (ch5_mech_01..05) are pure vector architecture/operator schematics
aligned strictly with Chapter 5 formulas and notation.tex.
The quantitative process plot (ch5_proc_01_peak_queue_evolution) reads only frozen
confirmation episode records (parent=0, workload=955000, replica=0) without retraining.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_figure_style import SEMANTIC_PALETTE, resolve_qa_scripts_dir, setup_thesis_style

SKILL = resolve_qa_scripts_dir(REPO_ROOT)
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = REPO_ROOT / "docs" / "figures"
DATA_DIR = REPO_ROOT / "docs" / "figures" / "data" / "queued_joint_rl_20261106"
CONF_DIR = REPO_ROOT / "results" / "chapter5_dynamic_service_20261028" / "queued_joint_rl_v2_20261106" / "confirmation"
QA = REPO_ROOT / "output" / "thesis_figure_upgrade_20261111" / "validation"

BLUE = SEMANTIC_PALETTE["module_blue"]
TEAL = SEMANTIC_PALETTE["module_teal"]
GREEN = SEMANTIC_PALETTE["covered_ok"]
ORANGE = SEMANTIC_PALETTE["admitted_miss"]
RED = SEMANTIC_PALETTE["uncovered_expired"]
PURPLE = SEMANTIC_PALETTE["module_purple"]
GRAY = SEMANTIC_PALETTE["neutral_slate"]
DARK = SEMANTIC_PALETTE["text_dark"]


def card(ax, x, y, w, h, text, color=BLUE, fontsize=8.5, lw=0.95, alpha_hex="15", bold=False, linespacing=1.56):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008",
            facecolor=color + alpha_hex,
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
        linespacing=linespacing,
    )


def flow_arrow(ax, a, b, color=GRAY, ls="-", lw=1.1, scale=10):
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


def render_ch5_framework() -> None:
    """5.3 节初：第五章候选库构造、两轮 DAgger 蒸馏、PPO+CRN 微调与双模式在线部署总框架图。"""
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.10, top=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Stage containers
    stages = [
        (0.01, 0.225, "阶段 I：候选库与解析特征", BLUE),
        (0.25, 0.235, "阶段 II：两轮 DAgger 蒸馏", TEAL),
        (0.50, 0.24, "阶段 III：PPO + CRN 微调", ORANGE),
        (0.755, 0.235, "阶段 IV：在线双模式部署", GREEN),
    ]
    for x, w, title, col in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.08), w, 0.86, boxstyle="round,pad=0.008", facecolor="#F9FBFD", edgecolor=col, lw=0.95
            )
        )
        ax.text(x + w / 2, 0.895, title, ha="center", va="center", fontsize=8.6, fontweight="bold", color=col)

    # Stage I boxes
    card(ax, 0.022, 0.62, 0.201, 0.21, "公开排队状态 $s_t$\n分配库 $\\mathcal{K}(s_t)$（含 $k_{\\mathrm{safe}}$）", BLUE, 8.5)
    card(ax, 0.022, 0.36, 0.201, 0.21, "联合运动库 $\\mathcal{D}_{\\mathrm{mot}}(s_t)$\n悬停 $\\mathbf{0}$ + 救援位移", TEAL, 8.5)
    card(ax, 0.022, 0.11, 0.201, 0.20, "两阶段排队近似算子\n特征 $(\\mathbf{a}_k,\\mathbf{m}_{km},\\mathbf{c})$", PURPLE, 8.5)

    # Stage II boxes (dashed teacher supervision)
    card(ax, 0.262, 0.60, 0.211, 0.23, "第 1 轮：先验与均匀探索\n第 2 轮：首轮学生闭环采样", TEAL, 8.5)
    card(ax, 0.262, 0.34, 0.211, 0.21, "三步教师规划软分布\n$q_T(k,m\\mid s)$（$\\tau_T=0.015$）", ORANGE, 8.5)
    card(ax, 0.262, 0.11, 0.211, 0.18, "最小化 $D_{\\mathrm{KL}}(q_T\\|\\pi_\\theta)$\n冻结蒸馏学生 $\\pi_{\\mathrm{ref}}$", BLUE, 8.5)
    flow_arrow(ax, (0.368, 0.59), (0.368, 0.56), color=TEAL, ls="-", lw=1.05)
    flow_arrow(ax, (0.368, 0.33), (0.368, 0.30), color=ORANGE, ls="--", lw=1.15)

    # Stage III boxes (route top->bottom arrow along right corridor x=0.724 outside middle card)
    card(ax, 0.510, 0.60, 0.198, 0.23, "真实轨迹（含排空步）\n留一优势 $\\tilde{A}$ 与 $V_\\psi$", BLUE, 8.5)
    card(ax, 0.510, 0.34, 0.198, 0.21, "每 4 步共享到达 $\\xi_t$\nCRN 相对优势 $\\hat{A}_{\\mathrm{cf}}$", ORANGE, 8.5)
    card(ax, 0.510, 0.10, 0.218, 0.19, "合并 $\\tilde{A}$ 与 $1.25\\hat{A}_{\\mathrm{cf}}$\nPPO 裁剪得微调策略 $\\pi_\\theta^\\star$", GREEN, 8.5)
    ax.plot([0.710, 0.725, 0.725], [0.715, 0.715, 0.31], color=BLUE, ls="--", lw=1.05)
    flow_arrow(ax, (0.725, 0.315), (0.705, 0.295), color=BLUE, ls="--", lw=1.05)
    flow_arrow(ax, (0.585, 0.335), (0.585, 0.298), color=ORANGE, ls="--", lw=1.15)

    # Stage IV boxes (online deployment modes)
    card(ax, 0.768, 0.53, 0.21, 0.31, "模式 A：零在线前瞻\n温度 $1.0$ 直接采样\n0 次克隆（$0.520\\,\\mathrm{s}$）", GREEN, 8.5)
    card(ax, 0.768, 0.12, 0.21, 0.34, "模式 B：神经引导 MPC\n先验校准 $\\alpha_{\\mathrm{cal}}\\in[0.25,1.0]$\nTop-6 择优 $J(s,k,m)$", PURPLE, 8.5)

    # Inter-stage arrows
    flow_arrow(ax, (0.228, 0.71), (0.258, 0.71), color=BLUE, ls="-", lw=1.15)
    flow_arrow(ax, (0.228, 0.21), (0.258, 0.21), color=PURPLE, ls="-", lw=1.15)
    flow_arrow(ax, (0.478, 0.20), (0.508, 0.20), color=BLUE, ls="-", lw=1.15)
    flow_arrow(ax, (0.732, 0.20), (0.764, 0.66), color=GREEN, ls="-", lw=1.2)
    flow_arrow(ax, (0.732, 0.18), (0.764, 0.29), color=PURPLE, ls="-", lw=1.2)

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=1.4, ls="-", label="实线：前向特征构造与在线执行"),
        Line2D([0], [0], color=ORANGE, lw=1.4, ls="--", label="虚线：离线教师蒸馏与策略梯度更新"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.02),
        ncol=2,
        fontsize=8.5,
        frameon=False,
    )

    _export_and_audit(fig, [ax], "ch5_mech_01_framework")


def render_ch5_tandem_gantt() -> None:
    """5.3.1 节：两阶段释放时间串联排队甘特图 vs 未完成传输提前占位阻塞 CPU 反例。"""
    fig = plt.figure(figsize=(9.2, 5.1))
    gs = fig.add_gridspec(2, 1, left=0.16, right=0.98, bottom=0.13, top=0.92, hspace=0.36)
    ax_good = fig.add_subplot(gs[0, 0])
    ax_bad = fig.add_subplot(gs[1, 0])

    # ---- Top Panel (a): Two-stage release-time work-conserving EDF operator ----
    ax_good.set_xlim(0.0, 2.35)
    ax_good.set_ylim(-0.5, 2.6)
    ax_good.set_yticks([0.0, 1.0, 2.0])
    ax_good.set_yticklabels(["待接纳任务 $C$", "跨机紧迫任务 $B$", "本地就绪任务 $A$"], fontsize=8.5)
    ax_good.set_xticks([0.0, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0])
    ax_good.set_title("(a) 本文两阶段释放时间算子：无线传输完成后释放至节点先占式 EDF 队列", fontsize=8.8, pad=6)

    # Task A (y=2.0): ready at tau=0, d_A-t=1.20 s, total compute 0.70 s; preempted at [0.60, 0.95] by B!
    ax_good.add_patch(Rectangle((0.0, 1.76), 0.60, 0.46, facecolor=GREEN, alpha=0.82, edgecolor=DARK, lw=0.7))
    ax_good.add_patch(Rectangle((0.60, 1.76), 0.35, 0.46, facecolor="#E5E9F0", edgecolor=GRAY, ls="--", lw=0.7))
    ax_good.add_patch(Rectangle((0.95, 1.76), 0.10, 0.46, facecolor=GREEN, alpha=0.82, edgecolor=DARK, lw=0.7))
    ax_good.plot([1.20, 1.20], [1.68, 2.32], color=DARK, ls=":", lw=1.2)
    ax_good.text(0.30, 1.99, "EDF 计算", ha="center", va="center", fontsize=8.4, color="white")
    ax_good.text(1.26, 1.99, "按时完成 ($\\leq s_A$)", ha="left", va="center", fontsize=8.5, color=GREEN)

    # Task B (y=1.0): upload [0,0.35] -> relay [0.35,0.60] -> release tau_rel=0.60 -> preempts CPU [0.60,0.95] < s_B=1.05
    ax_good.add_patch(Rectangle((0.0, 0.76), 0.35, 0.46, facecolor=BLUE, alpha=0.75, edgecolor=DARK, lw=0.7))
    ax_good.add_patch(Rectangle((0.35, 0.76), 0.25, 0.46, facecolor=TEAL, alpha=0.82, edgecolor=DARK, lw=0.7))
    ax_good.add_patch(Rectangle((0.60, 0.76), 0.35, 0.46, facecolor=GREEN, alpha=0.88, edgecolor=DARK, lw=0.7))
    ax_good.plot([0.60, 0.60], [0.62, 2.32], color=BLUE, ls="--", lw=1.0)
    ax_good.plot([1.05, 1.05], [0.68, 1.32], color=DARK, ls=":", lw=1.2)
    ax_good.text(0.175, 0.99, "上传", ha="center", va="center", fontsize=8.4, color="white")
    ax_good.text(0.475, 0.99, "转发", ha="center", va="center", fontsize=8.4, color="white")
    ax_good.text(0.775, 0.99, "抢占", ha="center", va="center", fontsize=8.4, color="white")
    ax_good.text(1.12, 0.99, "${\\tilde{\\varsigma}_{n,j}}^{\\,\\mathrm{rel}}=0.60\\,\\mathrm{s}$ 后抢占完成", ha="left", va="center", fontsize=8.6, color=GREEN)

    # Task C (y=0.0): wait coverage [0,1.0] -> upload [1.0,1.45] -> compute [1.45,1.85]
    ax_good.add_patch(Rectangle((0.0, -0.24), 1.00, 0.46, facecolor="#F2F4F8", edgecolor=GRAY, ls="--", lw=0.75))
    ax_good.add_patch(Rectangle((1.00, -0.24), 0.42, 0.46, facecolor=BLUE, alpha=0.75, edgecolor=DARK, lw=0.7))
    ax_good.add_patch(Rectangle((1.42, -0.24), 0.40, 0.46, facecolor=GREEN, alpha=0.85, edgecolor=DARK, lw=0.7))
    ax_good.text(0.50, -0.01, "等待机动覆盖", ha="center", va="center", fontsize=8.4, color=GRAY)
    ax_good.text(1.21, -0.01, "上传", ha="center", va="center", fontsize=8.4, color="white")
    ax_good.text(1.62, -0.01, "计算", ha="center", va="center", fontsize=8.4, color="white")

    # ---- Bottom Panel (b): Counterexample without release-time gating ----
    ax_bad.set_xlim(0.0, 2.35)
    ax_bad.set_ylim(-0.4, 1.7)
    ax_bad.set_yticks([0.0, 1.0])
    ax_bad.set_yticklabels(["跨机紧迫任务 $B$", "本地就绪任务 $A$"], fontsize=8.5)
    ax_bad.set_xticks([0.0, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0])
    ax_bad.set_xlabel("相对当前时刻的推进时间 $\\varsigma$ / s", fontsize=8.6)
    ax_bad.set_title("(b) 忽略通信释放时刻的单阶段串行预约反例：未传完任务提前占位阻塞就绪 CPU", fontsize=8.8, pad=6)

    # Counterexample: B has smaller deadline s_B=1.05 < s_A=1.20, reserves CPU from 0..0.95 (idle block [0,0.60]!)
    ax_bad.add_patch(Rectangle((0.0, -0.22), 0.60, 0.46, facecolor=RED, alpha=0.25, edgecolor=RED, ls="--", lw=0.9))
    ax_bad.add_patch(Rectangle((0.60, -0.22), 0.35, 0.46, facecolor=GREEN, alpha=0.85, edgecolor=DARK, lw=0.7))
    ax_bad.text(0.30, 0.01, "未传完占 CPU", ha="center", va="center", fontsize=8.4, color=RED)
    ax_bad.text(0.775, 0.01, "计算", ha="center", va="center", fontsize=8.4, color="white")

    # Task A gets pushed to [0.95, 1.65] > s_A=1.20 -> false deadline miss!
    ax_bad.add_patch(Rectangle((0.0, 0.78), 0.95, 0.46, facecolor="#FCEFE8", edgecolor=ORANGE, ls="--", lw=0.8))
    ax_bad.add_patch(Rectangle((0.95, 0.78), 0.70, 0.46, facecolor=ORANGE, alpha=0.88, edgecolor=DARK, lw=0.7))
    ax_bad.plot([1.20, 1.20], [0.68, 1.36], color=RED, ls=":", lw=1.35)
    ax_bad.text(0.475, 1.01, "被未就绪任务 $B$ 阻塞等待", ha="center", va="center", fontsize=8.4, color=DARK)
    ax_bad.text(1.42, 1.01, "推迟计算", ha="center", va="center", fontsize=8.4, color="white")
    ax_bad.text(1.70, 1.01, "超过 $s_A$ 误判超时", ha="left", va="center", fontsize=8.5, color=RED)

    for ax in (ax_good, ax_bad):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _export_and_audit(fig, [ax_good, ax_bad], "ch5_mech_02_tandem_gantt")


def render_ch5_dual_branch() -> None:
    """5.3.2 节：候选等变双分支联合策略网络与期望运动反馈双向闭环图。"""
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.10, top=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Left input feature column (10.3 pt ensures 2nd-level sub-subscripts p_{\theta_m} >= 5.05 pt)
    card(ax, 0.012, 0.71, 0.228, 0.22, "条件运动特征\n$\\mathbf{m}_{km}\\in\\mathbb{R}^{16}$\n（含 $\\boldsymbol{\\phi}_{km}\\in\\mathbb{R}^4$）", TEAL, 10.3)
    card(ax, 0.012, 0.40, 0.228, 0.21, "全局排队上下文\n$\\mathbf{c}(s)\\in\\mathbb{R}^{14}$", PURPLE, 10.3)
    card(ax, 0.012, 0.09, 0.228, 0.21, "下层分配特征\n$\\mathbf{a}_k\\in\\mathbb{R}^{14}$", BLUE, 10.3)

    # Upper motion branch (top)
    card(
        ax, 0.292, 0.68, 0.295, 0.25,
        "上层运动打分网络 $g_m$\n$[\\mathbf{m}_{km},\\mathbf{a}_k,\\mathbf{c}]\\in\\mathbb{R}^{44}$\n跳连 + 48 维隐层 + $\\log p_m^0$",
        TEAL, 10.3
    )
    card(
        ax, 0.630, 0.70, 0.205, 0.21,
        "条件运动分布\n$p_{\\theta_m}(m\\mid s,k)$\n（置换同变）",
        TEAL, 10.3
    )

    # Center expectation coupling bridge
    card(
        ax, 0.395, 0.39, 0.40, 0.21,
        "上层期望运动反馈聚合\n$\\mathbf{C}_{\\mathrm{upper}}(s,k)=\\sum_{m}p_{\\theta_m}(m\\mid s,k)\\boldsymbol{\\phi}_{km}\\in\\mathbb{R}^4$",
        ORANGE, 10.3
    )

    # Lower allocation branch (bottom)
    card(
        ax, 0.292, 0.07, 0.295, 0.25,
        "下层分配打分网络 $g_a$\n$[\\mathbf{a}_k,\\mathbf{C}_{\\mathrm{upper}},\\mathbf{c}]\\in\\mathbb{R}^{32}$\n跳连 + 48 维隐层 + $\\log p_a^0$",
        BLUE, 10.3
    )
    card(
        ax, 0.630, 0.09, 0.205, 0.21,
        "候选分配分布\n$p_{\\theta_a}(k\\mid s)$",
        BLUE, 10.3
    )

    # Joint output on far right
    card(
        ax, 0.855, 0.30, 0.135, 0.36,
        "联合策略\n$\\pi_\\theta(k,m\\mid s)$\n$=p_{\\theta_a}p_{\\theta_m}$",
        GREEN, 10.3, linespacing=1.78
    )

    # Solid forward arrows
    flow_arrow(ax, (0.245, 0.82), (0.287, 0.82), color=TEAL, ls="-", lw=1.2)
    flow_arrow(ax, (0.245, 0.53), (0.287, 0.75), color=PURPLE, ls="-", lw=1.1)
    flow_arrow(ax, (0.245, 0.24), (0.287, 0.70), color=BLUE, ls="-", lw=1.05)
    flow_arrow(ax, (0.592, 0.805), (0.625, 0.805), color=TEAL, ls="-", lw=1.2)

    # Coupling arrows into C_upper
    flow_arrow(ax, (0.245, 0.72), (0.390, 0.50), color=TEAL, ls="-", lw=1.1)
    flow_arrow(ax, (0.705, 0.695), (0.645, 0.605), color=ORANGE, ls="-", lw=1.2)

    # C_upper + a_k + c(s) -> lower branch g_a
    flow_arrow(ax, (0.495, 0.385), (0.445, 0.325), color=ORANGE, ls="-", lw=1.25)
    flow_arrow(ax, (0.245, 0.47), (0.287, 0.25), color=PURPLE, ls="-", lw=1.1)
    flow_arrow(ax, (0.245, 0.18), (0.287, 0.18), color=BLUE, ls="-", lw=1.2)
    flow_arrow(ax, (0.592, 0.195), (0.625, 0.195), color=BLUE, ls="-", lw=1.2)

    # Joint policy product arrows (route cleanly below red gradient box)
    flow_arrow(ax, (0.790, 0.695), (0.875, 0.665), color=GREEN, ls="-", lw=1.2)
    flow_arrow(ax, (0.840, 0.21), (0.895, 0.295), color=GREEN, ls="-", lw=1.2)

    # Dashed backward gradient path from lower loss through C_upper to theta_m
    flow_arrow(ax, (0.535, 0.325), (0.585, 0.385), color=RED, ls="--", lw=1.3)
    flow_arrow(ax, (0.735, 0.605), (0.755, 0.695), color=RED, ls="--", lw=1.3)
    card(
        ax, 0.855, 0.72, 0.135, 0.19,
        "反向梯度\n$\\nabla_{\\theta_m}\\mathcal{L}$",
        RED, 10.3
    )

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=1.5, ls="-", label="实线：前向特征与条件概率耦合"),
        Line2D([0], [0], color=RED, lw=1.5, ls="--", label="虚线：下层分配损失经 $\\mathbf{C}_{\\mathrm{upper}}$ 向上层 $\\theta_m$ 回传梯度"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.02),
        ncol=2,
        fontsize=9.5,
        frameon=False,
    )

    _export_and_audit(fig, [ax], "ch5_mech_03_dual_branch")


def render_ch5_crn_tree() -> None:
    """5.4.2 节：CRN 反事实四分支克隆树、组内中心化优势合并与 Critic 严格隔离示意图。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.10, top=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Top real-trajectory track
    ax.add_patch(
        FancyBboxPatch(
            (0.012, 0.68), 0.976, 0.26, boxstyle="round,pad=0.008", facecolor="#F7FAFD", edgecolor=BLUE, lw=0.95
        )
    )
    ax.text(
        0.028, 0.898,
        "真实环境交互轨迹（共享外生工作负载的双动作流，含活跃步与非空排空步）",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=BLUE
    )

    card(ax, 0.028, 0.70, 0.215, 0.155, "公开排队状态 $s_t$\n采样执行真实动作", BLUE, 8.5)
    flow_arrow(ax, (0.248, 0.775), (0.292, 0.775), color=BLUE, ls="-", lw=1.2)
    card(ax, 0.298, 0.70, 0.295, 0.155, "一步真实奖励 $r_t$ 与 GAE\n留一优势 $\\tilde{A}_{i,t}$、回报 $\\hat{R}_t$", TEAL, 8.5)
    flow_arrow(ax, (0.598, 0.775), (0.642, 0.775), color=TEAL, ls="--", lw=1.2)
    card(ax, 0.648, 0.70, 0.285, 0.155, "Critic 价值网络 $V_\\psi(\\mathbf{x}_V)$\n$\\mathbf{x}_V\\in\\mathbb{R}^{34}$，仅拟合真实 $\\hat{R}_t$", PURPLE, 8.5)

    # Bottom CRN counterfactual tree (every 4 active steps)
    ax.add_patch(
        FancyBboxPatch(
            (0.012, 0.07), 0.976, 0.56, boxstyle="round,pad=0.008", facecolor="#FFFBF6", edgecolor=ORANGE, lw=0.95
        )
    )
    ax.text(
        0.165, 0.588,
        "每 4 个活跃步触发：共同随机数（CRN）短视界反事实克隆树",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=ORANGE
    )

    # Clone arrow from s_t center down to CRN root
    flow_arrow(ax, (0.135, 0.695), (0.135, 0.520), color=ORANGE, ls="--", lw=1.25)
    card(
        ax, 0.028, 0.31, 0.215, 0.205,
        "同源克隆至多 4 分支\n$\\mathcal{B}_t$：已执行 $\\cup$ Top-3\n共享预测到达流 $\\xi_t$",
        ORANGE, 8.5
    )
    card(
        ax, 0.028, 0.10, 0.215, 0.165,
        "消去公共到达波动\n$\\eta(s_t,\\xi_t)$（定理 5.4.1）",
        TEAL, 8.5
    )

    # 4 cloned rollout branches
    branch_y = [0.435, 0.315, 0.195, 0.085]
    branch_labels = [
        "分支 $b_1$：首步 $(k_1,m_1)$ + 两步 $(k_{\\mathrm{safe}},\\mathbf{0})$",
        "分支 $b_2$：首步 $(k_2,m_2)$ + 两步 $(k_{\\mathrm{safe}},\\mathbf{0})$",
        "分支 $b_3$：首步 $(k_3,m_3)$ + 两步 $(k_{\\mathrm{safe}},\\mathbf{0})$",
        "分支 $b_4$：首步 $(k_4,m_4)$ + 两步 $(k_{\\mathrm{safe}},\\mathbf{0})$",
    ]
    for yb, lbl in zip(branch_y, branch_labels):
        card(ax, 0.275, yb, 0.355, 0.095, lbl, TEAL, 8.5)
        flow_arrow(ax, (0.248, 0.40), (0.270, yb + 0.048), color=ORANGE, ls="-", lw=0.95)
        flow_arrow(ax, (0.635, yb + 0.048), (0.662, 0.37), color=TEAL, ls="-", lw=0.95)

    # Group centering -> Actor update (isolated from Critic!)
    card(
        ax, 0.665, 0.285, 0.155, 0.19,
        "组内中心化\n$\\hat{A}_{\\mathrm{cf}}(s_t,b)$\n$w_{\\mathrm{cf}}=1.25$",
        ORANGE, 8.5
    )
    flow_arrow(ax, (0.825, 0.38), (0.848, 0.38), color=GREEN, ls="--", lw=1.2)
    # Route LOO-GAE advantage from (0.45, 0.70) along y=0.655 corridor -> right margin x=0.955 -> Actor top!
    ax.plot([0.45, 0.45, 0.955, 0.955], [0.70, 0.655, 0.655, 0.485], color=BLUE, ls="--", lw=1.15)
    flow_arrow(ax, (0.955, 0.495), (0.955, 0.480), color=BLUE, ls="--", lw=1.15)

    card(
        ax, 0.852, 0.285, 0.122, 0.19,
        "联合 Actor\nPPO 裁剪\n合并更新",
        GREEN, 8.5
    )
    card(
        ax, 0.665, 0.095, 0.309, 0.145,
        "严格隔离：反事实评分 $Q_{\\mathrm{cf}}$\n绝不进入 Critic 价值回归标签",
        RED, 8.5
    )

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=1.4, ls="-", label="实线：真实执行与克隆物理推演"),
        Line2D([0], [0], color=ORANGE, lw=1.4, ls="--", label="虚线：优势聚合与 Actor/Critic 参数更新"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.02),
        ncol=2,
        fontsize=8.5,
        frameon=False,
    )

    _export_and_audit(fig, [ax], "ch5_mech_04_crn_tree")


def render_ch5_online_mpc() -> None:
    """5.4.4 节：零在线前瞻直接采样 vs 神经引导 Top-6 MPC 在线推理上下双通道对照图。"""
    fig = plt.figure(figsize=(9.3, 5.0))
    gs = fig.add_gridspec(2, 1, left=0.015, right=0.985, bottom=0.04, top=0.95, hspace=0.22)
    ax_fast = fig.add_subplot(gs[0, 0])
    ax_mpc = fig.add_subplot(gs[1, 0])

    # ---- Top Channel (a): Zero online lookahead ----
    ax_fast.set_xlim(0.0, 1.0)
    ax_fast.set_ylim(0.0, 1.0)
    ax_fast.axis("off")
    ax_fast.add_patch(
        FancyBboxPatch(
            (0.01, 0.05), 0.98, 0.88, boxstyle="round,pad=0.01", facecolor="#F7FBF8", edgecolor=GREEN, lw=0.95
        )
    )
    ax_fast.text(
        0.03, 0.84,
        "(a) 通道一：零在线仿真前瞻推理（联合微调策略，均值耗时 $0.520\\,\\mathrm{s}$，确认集 DSR $48.74\\%$）",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=GREEN
    )

    card(ax_fast, 0.03, 0.21, 0.225, 0.49, "公开状态 $s_t$\n构建合法候选库\n$(\\mathcal{K}(s_t),\\mathcal{D}_{\\mathrm{mot}}(s_t))$", BLUE, 8.5)
    flow_arrow(ax_fast, (0.26, 0.455), (0.295, 0.455), color=BLUE, ls="-", lw=1.2)
    card(ax_fast, 0.30, 0.21, 0.24, 0.49, "双分支联合提议分布\n$\\pi_\\theta(k,m\\mid s_t)$\n两阶段排队近似特征", TEAL, 8.5)
    flow_arrow(ax_fast, (0.545, 0.455), (0.585, 0.455), color=GREEN, ls="-", lw=1.2)
    card(ax_fast, 0.59, 0.21, 0.20, 0.49, "温度 $1.0$ 直接采样\n0 次在线克隆推演\n不查询离线教师", GREEN, 8.5)
    flow_arrow(ax_fast, (0.795, 0.455), (0.825, 0.455), color=GREEN, ls="-", lw=1.2)
    card(ax_fast, 0.83, 0.21, 0.145, 0.49, "真实环境\n执行本步\n$(k_t,m_t)$", BLUE, 8.5)

    # ---- Bottom Channel (b): Neural-guided Top-6 MPC ----
    ax_mpc.set_xlim(0.0, 1.0)
    ax_mpc.set_ylim(0.0, 1.0)
    ax_mpc.axis("off")
    ax_mpc.add_patch(
        FancyBboxPatch(
            (0.01, 0.05), 0.98, 0.88, boxstyle="round,pad=0.01", facecolor="#F8F7FC", edgecolor=PURPLE, lw=0.95
        )
    )
    ax_mpc.text(
        0.03, 0.84,
        "(b) 通道二：神经引导 Top-6 MPC 在线规划（微调 Top-6 MPC，均值耗时 $0.722\\,\\mathrm{s}$，确认集 DSR $49.31\\%$）",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=PURPLE
    )

    card(
        ax_mpc, 0.025, 0.16, 0.205, 0.58,
        "联合分布熵自适应校准\n$\\alpha_{\\mathrm{cal}}\\in[0.25,1.0]$\n先验项 $0.65\\alpha_{\\mathrm{cal}}\\log p$",
        BLUE, 8.5, linespacing=1.88
    )
    flow_arrow(ax_mpc, (0.235, 0.45), (0.262, 0.45), color=BLUE, ls="-", lw=1.15)
    card(
        ax_mpc, 0.267, 0.16, 0.185, 0.58,
        "严格保序筛选\n保留 Top-6 候选\n（含保底安全项）",
        TEAL, 8.5, linespacing=1.72
    )
    flow_arrow(ax_mpc, (0.457, 0.45), (0.483, 0.45), color=TEAL, ls="-", lw=1.15)
    card(
        ax_mpc, 0.488, 0.16, 0.215, 0.58,
        "共享两步预测到达\n三步克隆队列仿真\n首步候选+两步悬停",
        ORANGE, 8.5, linespacing=1.72
    )
    flow_arrow(ax_mpc, (0.708, 0.45), (0.735, 0.45), color=ORANGE, ls="-", lw=1.15)
    card(
        ax_mpc, 0.74, 0.16, 0.235, 0.58,
        "含先验综合评分择优\n$J(s,k,m)=S^{(3)}-\\eta_{\\mathrm{pen}} U^{(3)}$\n$+\\zeta R_{\\mathrm{rescue}}+0.65\\log\\tilde{p}$",
        PURPLE, 8.5, linespacing=1.85
    )

    _export_and_audit(fig, [ax_fast, ax_mpc], "ch5_mech_05_online_mpc")


def export_peak_queue_data() -> dict:
    """Deterministically export paired confirmation episode (0_955000_0) queue evolution JSON."""
    from scripts.analysis import queued_waiting_rescue as prototype

    t_mod = prototype.t
    workload = 955000
    parent = 0
    replica = 0
    tape = t_mod.RecoveryScenario(workload, "burst", "normal", 0.6)
    arrivals = [len(tape.slot(s)) for s in range(200)]

    arms = ("uncoordinated_init", "joint_rl", "joint_rl_finetuned", "joint_rl_mpc")
    out_records = {}
    for arm in arms:
        p = CONF_DIR / arm / f"{parent}_{workload}_{replica}.json"
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert rec["audit"]["status"] == "PASS"
        traj = rec["trajectory"]
        succ = rec["successes"]

        times, waiting_backlog, cum_waiting_expired, cum_unserved_and_expired = [], [], [], []
        w_prev = 0
        ew_cum = 0
        sc_cum = 0
        arr_cum = 0
        for step_row, s_cnt in zip(traj, succ):
            now = int(step_row["time"])
            arr = arrivals[now] if now < 200 else 0
            arr_cum += arr
            adm = int(step_row["admitted"])
            w_cur = int(step_row["metadata"]["waiting"])
            exp_w = w_prev + arr - adm - w_cur
            assert exp_w >= 0
            ew_cum += exp_w
            sc_cum += int(s_cnt)

            times.append(now)
            waiting_backlog.append(w_cur)
            cum_waiting_expired.append(ew_cum)
            # Exact conservation: G_<=t - S_<=t = |T_active(t+1)| + E(t+1); equals exact total expired after drain!
            cum_unserved_and_expired.append(arr_cum - sc_cum)
            w_prev = w_cur

        final_waiting_expired = ew_cum + w_prev
        assert final_waiting_expired == rec["counts"]["generated"] - rec["counts"]["admitted"]
        assert cum_unserved_and_expired[-1] == rec["counts"]["expired"]

        out_records[arm] = {
            "source_file": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_sha256": sha,
            "counts": rec["counts"],
            "times": times,
            "waiting_backlog": waiting_backlog,
            "cum_waiting_expired": cum_waiting_expired,
            "cum_unserved_and_expired": cum_unserved_and_expired,
            "final_waiting_expired": final_waiting_expired,
            "total_expired": int(rec["counts"]["expired"]),
        }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "peak_queue_evolution_955000.json"
    payload = {
        "description": "Paired confirmation episode queue evolution (parent=0, workload=955000, replica=0).",
        "parent": parent,
        "workload": workload,
        "replica": replica,
        "generated_total": sum(arrivals),
        "peak_window_s": [50, 100],
        "drain_window_s": [200, 204],
        "arms": out_records,
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def render_ch5_peak_queue_evolution() -> None:
    """5.5.1 节：确认集配对回合（0~204 s，含高峰与排空段）队列积压与累计过期演化对比图。"""
    data_file = DATA_DIR / "peak_queue_evolution_955000.json"
    payload = export_peak_queue_data() if not data_file.exists() else json.loads(data_file.read_text(encoding="utf-8"))
    if "cum_unserved_and_expired" not in next(iter(payload["arms"].values())):
        payload = export_peak_queue_data()

    fig = plt.figure(figsize=(8.0, 5.0))
    gs = fig.add_gridspec(2, 1, left=0.11, right=0.84, bottom=0.11, top=0.88, hspace=0.35)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    arm_specs = [
        ("uncoordinated_init", "初始先验选择器", GRAY, "--", 1.25),
        ("joint_rl", "联合冷启动 RL", ORANGE, "-.", 1.3),
        ("joint_rl_finetuned", "联合微调", BLUE, "-", 1.5),
        ("joint_rl_mpc", "微调 Top-6 MPC", GREEN, "-", 1.55),
    ]

    for ax in (ax_top, ax_bot):
        ax.axvspan(50.0, 100.0, color=ORANGE, alpha=0.10, zorder=0)
        ax.axvspan(200.0, 204.0, color=PURPLE, alpha=0.12, zorder=0)
        ax.set_xlim(0.0, 204.0)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.grid(axis="y", color="#DCE2EA", lw=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for arm, label, col, ls, lw in arm_specs:
        rec = payload["arms"][arm]
        t_arr = np.asarray(rec["times"], dtype=float)
        w_arr = np.asarray(rec["waiting_backlog"], dtype=float)
        # Extend drain flat to 204 s (0 remaining active tasks after step 201)
        t_ext = np.append(t_arr, [202.0, 204.0])
        w_ext = np.append(w_arr, [0.0, 0.0])
        e_ext = np.append(np.asarray(rec["cum_unserved_and_expired"], dtype=float), [rec["total_expired"], rec["total_expired"]])

        ax_top.plot(t_ext, w_ext, color=col, ls=ls, lw=lw, label=label)
        ax_bot.plot(t_ext, e_ext, color=col, ls=ls, lw=lw, label=label)

    ax_top.set_ylim(-4.0, 148.0)
    ax_top.set_ylabel("待接纳积压 $|\\mathcal{W}_t|$ / 个", fontsize=8.6)
    ax_top.set_title("(a) 活跃待接纳队列积压长度随时间演化", fontsize=8.8, pad=5)
    ax_top.text(75.0, 137.0, "高峰窗口 $[50,100)\\,\\mathrm{s}$", ha="center", va="center", fontsize=8.4, color=ORANGE)
    ax_top.text(196.0, 137.0, "排空段", ha="right", va="center", fontsize=8.4, color=PURPLE)

    ax_bot.set_ylim(-80.0, 3280.0)
    ax_bot.set_xlabel("物理时间 $t$ / s", fontsize=8.6)
    ax_bot.set_ylabel("在途积压与累计过期 / 个", fontsize=8.6)
    ax_bot.set_title("(b) 累计未完成与过期任务数（排空后精确收敛至总过期数 $E_{\\mathrm{tot}}$）", fontsize=8.8, pad=5)

    # Non-overlapping right-margin endpoint callouts for exact total expired counts
    endpoint_offsets = {
        "uncoordinated_init": 3100.0,
        "joint_rl": 2840.0,
        "joint_rl_finetuned": 2580.0,
        "joint_rl_mpc": 2320.0,
    }
    for arm, label, col, _, _ in arm_specs:
        tot_e = payload["arms"][arm]["total_expired"]
        ax_bot.scatter([204.0], [tot_e], s=16, color=col, zorder=5, clip_on=False)
        ax_bot.text(
            206.5,
            endpoint_offsets[arm],
            f"{tot_e} 个",
            ha="left",
            va="center",
            fontsize=8.4,
            color=col,
            clip_on=False,
        )

    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.99),
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )

    _export_and_audit(fig, [ax_top, ax_bot], "ch5_proc_01_peak_queue_evolution")


def render_all_chapter5_additions() -> None:
    family = setup_thesis_style(8.5)
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
    render_ch5_framework()
    render_ch5_tandem_gantt()
    render_ch5_dual_branch()
    render_ch5_crn_tree()
    render_ch5_online_mpc()
    render_ch5_peak_queue_evolution()


if __name__ == "__main__":
    render_all_chapter5_additions()
