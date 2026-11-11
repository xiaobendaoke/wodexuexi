#!/usr/bin/env python3
"""Generate the 2026-10-02 long-form figures for thesis chapters 3 and 4.

Data sources are archived result files only; no experiment is rerun:
  - results/canonical_all_20260920/chapter4/{analysis.json,evaluation_units.csv} (main worktree)
  - results/bridge_20260928/full/analysis.json (main worktree)
  - <coverage worktree>/results/coverage_observation_confirmation_20260927/evaluation_units.csv

Schematic figures (3-1, 3-3, 3-4, 3-5, 4-1, 4-2) draw no experimental data;
the constructed example in 图4-1 uses formulas verified against
environment/uavs.py::_get_computing_latency (latency = C/(F/n))
and is labelled as a mechanism example.

Outputs vector PDF/SVG and 600 dpi PNG into the formal docs/figures/ directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_figure_style import (
    PALETTE,
    SEMANTIC_PALETTE,
    panel_label,
    resolve_qa_scripts_dir,
    save_pub,
    setup_thesis_style,
    soften_axes,
)

SKILL_SCRIPTS = resolve_qa_scripts_dir(REPO_ROOT)
if SKILL_SCRIPTS.is_dir():
    sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUTPUT_DIRS = [
    REPO_ROOT / "docs" / "figures",
]
QA_REPORT = REPO_ROOT / "output" / "thesis_writing_20261004" / "validation" / "figure_text_qa.json"
ALIGN_DIR = REPO_ROOT / "output" / "thesis_writing_20261004" / "validation" / "figures"
UPGRADE_QA_DIR = REPO_ROOT / "output" / "thesis_figure_upgrade_20261111" / "validation"

T95_DF9 = 2.2622  # t(0.975, 9), 10 paired units

CH4_METHODS = [
    ("heuristic_control", "Heuristic-Control", "#9AA0A6"),
    ("greedy_single_pass", "Greedy-Single-Pass", "#D8A448"),
    ("iterative_base", "Iterative-Base", "#53A6A6"),
    ("mc_itr", "MC-ITR", "#3D6FB6"),
]
CH4_METRIC_IDX = {"misses": 0, "latency": 1, "time": 2}

CANDIDATE = "lower_mlp_upper_uncovered_obs"
STRONG_BASELINE = "lower_no_attention"

ALIGN_REPORTS: list[dict] = []
SOURCE_DIR = REPO_ROOT / "docs" / "figures" / "data" / "chapter34_20261004"

def source_file(name: str, fallback: Path) -> Path:
    local = SOURCE_DIR / name
    return local if local.exists() else fallback

def clipped_circle(ax, center, radius, *, bounds, **kwargs):
    """Render only circle segments in bounds; PDF clipping alone is not audited."""
    theta = np.linspace(0, 2*np.pi, 2401)
    pts = np.asarray(center) + radius*np.column_stack((np.cos(theta), np.sin(theta)))
    x0,x1,y0,y1 = bounds
    visible = (pts[:,0]>=x0)&(pts[:,0]<=x1)&(pts[:,1]>=y0)&(pts[:,1]<=y1)
    points = pts.copy(); points[~visible] = np.nan
    ax.plot(points[:,0],points[:,1], **kwargs)

def method_ticks(ax, *, online=False, exact=False):
    labels = (["O"] if online else []) + ["H", "G", "I", "MC"] + (["Ex"] if exact else [])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)

def method_key(fig, y=-0.02, *, online=False, exact=False):
    text = "H：Heuristic-Control；G：Greedy-Single-Pass；I：Iterative-Base；MC：MC-ITR"
    if online: text = "O：在线学习下层；" + text
    if exact: text += "；Ex：穷举精确解"
    fig.text(.5,y,text,ha="center",fontsize=8,color=PALETTE["baseline"])


def save(fig: plt.Figure, name: str, *, multi_panel: bool = False) -> None:
    from matplotlib.text import Text
    for item in fig.findobj(match=Text):
        if item.get_fontsize() < 8:
            item.set_fontsize(8)
    for ax in fig.axes:
        ax.grid(False)
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    alignment = require_matplotlib_panel_alignment(
        fig,
        json_out=ALIGN_DIR / f"{name}.alignment.json",
        overlay_svg=ALIGN_DIR / f"{name}.alignment.svg",
        strict=True,
    )
    ALIGN_REPORTS.append({"figure": name, "verdict": alignment["verdict"],
                          "applicable": alignment["applicable"]})
    save_pub(fig, name, OUTPUT_DIRS, formats=("pdf", "svg", "png"), png_dpi=600, qa_report=QA_REPORT)
    for exported in (OUTPUT_DIRS[0] / f"{name}.pdf", OUTPUT_DIRS[0] / f"{name}.svg",
                     OUTPUT_DIRS[0] / f"{name}.png"):
        if not exported.exists():
            raise RuntimeError(f"Required export missing: {exported}")
    subprocess.run([sys.executable, str(SKILL_SCRIPTS / "audit_figure_collisions.py"),
                    str(OUTPUT_DIRS[0] / f"{name}.pdf"),
                    "--json-out", str(ALIGN_DIR / f"{name}.collision.json")],
                   check=True, capture_output=True, text=True)
    if name in ("图3-4_Attention-MAPPO网络与训练结构", "ch4_mc_itr_iteration_trace"):
        UPGRADE_QA_DIR.mkdir(parents=True, exist_ok=True)
        (UPGRADE_QA_DIR / f"{name}.alignment.json").write_text(
            (ALIGN_DIR / f"{name}.alignment.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        (UPGRADE_QA_DIR / f"{name}.collision.json").write_text(
            (ALIGN_DIR / f"{name}.collision.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        res_txt = subprocess.run(
            [
                sys.executable,
                str(SKILL_SCRIPTS / "audit_pdf_text.py"),
                str(OUTPUT_DIRS[0] / f"{name}.pdf"),
                "--min-pt",
                "5",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (UPGRADE_QA_DIR / f"{name}.pdf_text.json").write_text(res_txt.stdout, encoding="utf-8")
    print(f"saved {name}")


# ---------------------------------------------------------------- ch3 schematics


def fig_system_overview() -> None:
    """Natural coverage admission; illustrative positions, not recorded data."""
    fig, ax = plt.subplots(figsize=(5.8,5.6))
    rng = np.random.default_rng(11)
    hotspots = [np.array([210.,500.]),np.array([520.,180.])]
    uavs = [np.array(p) for p in [(200,480),(500,200),(400,600),(150,170),(580,420)]]
    assert min(np.linalg.norm(a-b) for i,a in enumerate(uavs) for b in uavs[i+1:]) >= 200
    for h in hotspots:
        ax.add_patch(Circle(h,100,facecolor=PALETTE["orange"],alpha=.10,edgecolor="none"))
    for u in uavs:
        ax.add_patch(Circle(u,100,facecolor=PALETTE["full"],alpha=.13,edgecolor=PALETTE["full"],lw=.8))
        ax.plot(*u,marker="^",color=PALETTE["red"],markersize=10,zorder=5)
    clipped_circle(ax,uavs[0],460,bounds=(0,700,0,700),color=PALETTE["full"],ls="--",lw=1)
    def cloud(center,r,n):
        ang=rng.uniform(0,2*np.pi,n);rad=r*np.sqrt(rng.uniform(0,1,n))
        return center+np.column_stack((rad*np.cos(ang),rad*np.sin(ang)))
    pts=np.vstack([cloud(hotspots[0],95,24),cloud(hotspots[1],95,22),cloud(np.array([350.,350.]),320,8)])
    pts=pts[(pts[:,0]>0)&(pts[:,0]<700)&(pts[:,1]>0)&(pts[:,1]<700)]
    covered=np.array([any(np.linalg.norm(p-u)<=100 for u in uavs) for p in pts])
    ax.scatter(*pts[covered].T,s=14,color=PALETTE["green"],zorder=4,label="已覆盖 UE：接纳")
    ax.scatter(*pts[~covered].T,s=16,color=PALETTE["red"],marker="x",zorder=4,label="未覆盖 UE：不接纳")
    ax.plot(350,350,marker="*",color=PALETTE["purple"],markersize=15,zorder=5,label="MBS")
    ax.plot([],[],marker="^",color=PALETTE["red"],ls="",label="UAV")
    ax.set(xlim=(0,700),ylim=(0,700),xlabel="$x$ / m",ylabel="$y$ / m")
    ax.set_aspect("equal")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2,fontsize=8,frameon=False)
    fig.subplots_adjust(bottom=.18)
    save(fig,"图3-1_系统与自然覆盖示意图")


def fig_coverage_geometry() -> None:
    """Chap3: coverage radius vs sensing radius geometry (schematic, no data)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    uav_pos = np.array([350.0, 435.0])
    rng = np.random.default_rng(7)
    clipped_circle(ax, uav_pos, 460, bounds=(0, 700, 0, 700), color="#3D6FB6", ls="--", lw=1.2)
    ax.add_patch(Circle(uav_pos, 100.0, fill=True, facecolor="#3D6FB6", alpha=0.22,
                        edgecolor="#3D6FB6", linewidth=1.2))

    def sample_separated(r_min: float, r_max: float, n: int, min_sep: float) -> np.ndarray:
        accepted: list[np.ndarray] = []
        while len(accepted) < n:
            ang = float(rng.uniform(0, 2 * np.pi))
            rad = float(np.sqrt(rng.uniform(r_min**2, r_max**2)))
            cand = uav_pos + np.array([rad * np.cos(ang), rad * np.sin(ang)])
            if not (28.0 <= cand[0] <= 672.0 and 28.0 <= cand[1] <= 672.0):
                continue
            if np.linalg.norm(cand - uav_pos) < 22.0:
                continue
            if all(np.linalg.norm(cand - q) >= min_sep for q in accepted):
                accepted.append(cand)
        return np.asarray(accepted)

    cov_pts = sample_separated(24.0, 86.0, 7, 26.0)
    sense_pts = sample_separated(135.0, 425.0, 9, 65.0)
    outside_pts = np.array(
        [[42.0, 42.0], [90.0, 30.0], [30.0, 80.0], [610.0, 30.0], [658.0, 48.0]],
        dtype=float,
    )
    assert np.all(np.linalg.norm(outside_pts - uav_pos, axis=1) >= 475.0)

    ax.scatter(cov_pts[:, 0], cov_pts[:, 1], s=24, zorder=3, marker="o", color="#3A8D5A", label="已覆盖 UE（接纳）")
    ax.scatter(sense_pts[:, 0], sense_pts[:, 1], s=24, zorder=3, marker="s", color="#D8A448", label="感知范围内未覆盖 UE")
    ax.scatter(outside_pts[:, 0], outside_pts[:, 1], s=32, zorder=4, marker="x", linewidths=1.35, color="#4E5965", label="感知范围外 UE")
    ax.plot(*uav_pos, marker="^", color="#C75E5A", markersize=12, zorder=5)
    ax.plot([], [], marker="^", color="#C75E5A", ls="", label="UAV")
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 700)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$y$ / m")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=.20)
    save(fig, "图3-2_覆盖与感知几何示意图")

def fig_slot_timeline() -> None:
    """Current-request service precedes application of the prepared motion."""
    fig,ax=plt.subplots(figsize=(7.2,3.6));ax.axis("off")
    entries=[("S0 当前状态","请求已生成\n自然覆盖关联"),("S1 两层准备","当前观测计算\n轨迹与卸载动作"),("S2 请求执行","冻结移动前几何\n按最终负载均分"),("S3 应用轨迹","速度与边界约束\n联合冲突消解"),("S4 奖励指标","DSR/公平性/时延\n回传与违例惩罚"),("S5 下一状态","缓存提交/UE移动\n新请求与新观测")]
    positions=[(.04,.57),(.37,.57),(.70,.57),(.70,.08),(.37,.08),(.04,.08)]
    w,h=.26,.34
    for i,((title,body),(x,y)) in enumerate(zip(entries,positions)):
        color=PALETTE["full"] if i in (1,3) else PALETTE["lower"]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.007",facecolor=color,alpha=.12,edgecolor=color,transform=ax.transAxes))
        ax.text(x+w/2,y+h*.78,title,ha="center",va="center",fontsize=8,transform=ax.transAxes,fontweight="bold")
        ax.text(x+w/2,y+h*.35,body,ha="center",va="center",fontsize=8,transform=ax.transAxes)
        if i:
            px,py=positions[i-1]
            if py==y:
                start=(px+w+.005,py+h/2) if x>px else (px-.005,py+h/2)
                end=(x-.005,y+h/2) if x>px else (x+w+.005,y+h/2)
            else:start=(px+w/2,py-.015);end=(x+w/2,y+h+.015)
            ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=9,transform=ax.transAxes))
    save(fig,"图3-3_单时隙决策时序图")

def fig_network_structure() -> None:
    """3.3.3 节（编译图 3-5）：上层 295 维 Attention-MAPPO 与下层 1405 维受约束卸载的协同闭环结构图。"""
    fig, ax = plt.subplots(figsize=(10.6, 6.5))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.08, top=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    blue = SEMANTIC_PALETTE["module_blue"]
    teal = SEMANTIC_PALETTE["module_teal"]
    green = SEMANTIC_PALETTE["covered_ok"]
    orange = SEMANTIC_PALETTE["admitted_miss"]
    red = SEMANTIC_PALETTE["uncovered_expired"]
    purple = SEMANTIC_PALETTE["module_purple"]
    gray = SEMANTIC_PALETTE["neutral_slate"]
    dark = SEMANTIC_PALETTE["text_dark"]

    def box(x, y, w, h, text, color, fs=8.8, lw=0.95, ls_line=1.50):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.007",
                facecolor=color + "16",
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
            fontsize=fs,
            color=dark,
            linespacing=ls_line,
        )

    def arrow(a, b, color=gray, ls="-", lw=1.15):
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=10, lw=lw, linestyle=ls, color=color
            )
        )

    # ---- Outer group containers ----
    # 1. Upper-layer trajectory module
    ax.add_patch(
        FancyBboxPatch(
            (0.022, 0.53), 0.753, 0.43, boxstyle="round,pad=0.008", facecolor="#F8FAFD", edgecolor=blue, lw=1.0
        )
    )
    ax.text(
        0.036, 0.928,
        "上层轨迹规划模块：Attention-MAPPO（预接纳扩展局部观测 $o_{u,t}\\in\\mathbb{R}^{295}$，执行分散、训练集中评价）",
        ha="left", va="center", fontsize=9.0, fontweight="bold", color=blue,
    )

    # 2. Lower-layer constrained offloading module
    ax.add_patch(
        FancyBboxPatch(
            (0.022, 0.06), 0.753, 0.43, boxstyle="round,pad=0.008", facecolor="#F7FBFA", edgecolor=teal, lw=1.0
        )
    )
    ax.text(
        0.036, 0.458,
        "下层请求级卸载模块：受约束 MAPPO（仅处理真实自然接纳请求 $\\mathbf{o}_u^{\\mathrm{offload}}\\in\\mathbb{R}^{1405}$，两层参数独立）",
        ha="left", va="center", fontsize=9.0, fontweight="bold", color=teal,
    )

    # 3. Right-hand Protocol I physical environment closed loop
    ax.add_patch(
        FancyBboxPatch(
            (0.798, 0.06), 0.192, 0.90, boxstyle="round,pad=0.008", facecolor="#F8FBF8", edgecolor=green, lw=1.0
        )
    )
    ax.text(
        0.894, 0.928,
        "协议 I 物理环境闭环",
        ha="center", va="center", fontsize=9.0, fontweight="bold", color=green,
    )

    # ---- Upper Layer Cards ----
    box(0.034, 0.805, 0.186, 0.088, "自身状态 77 维\n坐标 2 + 缓存 75", blue, 8.8)
    box(0.034, 0.700, 0.186, 0.088, "邻机槽 4 × 2 维\n感知内相对位置", teal, 8.8)
    box(0.034, 0.595, 0.186, 0.088, "扩展 UE 槽 30 × 7 维\n已覆盖 + 感知内未覆盖", orange, 8.8)

    box(0.244, 0.675, 0.142, 0.215, "分组特征嵌入\nLinear / LN / ReLU\n各输出 64 维", gray, 8.8)
    box(0.410, 0.675, 0.184, 0.215, "8 头交叉注意力\n$d_h=8$，缩放 $1/\\sqrt{8}$\n融合 $192\\to 128\\to 64$", purple, 8.8)
    box(0.618, 0.675, 0.145, 0.215, "上层共享 Actor\n二维正态 $\\mathbf{z}_u$\n$\\boldsymbol{\\mu}_\\theta=\\tanh(\\cdot)$", green, 8.8)

    # Upper Critic (bottom strip of upper container)
    box(
        0.244, 0.548, 0.519, 0.098,
        "五机联合局部观测集中注意力 Critic（$5\\times 295=1475$ 维 $\\to V_{\\phi,u}$）与 PPO 裁剪更新",
        red, 8.8,
    )

    # Upper forward & training arrows
    arrow((0.224, 0.848), (0.240, 0.825), blue, "-", 1.1)
    arrow((0.224, 0.744), (0.240, 0.765), teal, "-", 1.1)
    arrow((0.224, 0.639), (0.240, 0.710), orange, "-", 1.1)
    arrow((0.390, 0.782), (0.406, 0.782), purple, "-", 1.15)
    arrow((0.598, 0.782), (0.614, 0.782), green, "-", 1.15)
    arrow((0.224, 0.605), (0.240, 0.597), red, "--", 1.1)
    arrow((0.690, 0.648), (0.690, 0.671), red, "--", 1.1)

    # ---- Lower Layer Cards ----
    box(0.034, 0.305, 0.190, 0.115, "来源特征 $\\mathbf{o}_u^{\\mathrm{low}}\\in\\mathbb{R}^5$\n坐标 / 负载 / 邻机 / 回传", blue, 8.8)
    box(0.034, 0.165, 0.190, 0.120, "请求槽 $100\\times 14$ 维\n$\\mathbf{r}_{u,m}$（预测时延比/份额）", teal, 8.8)

    box(0.248, 0.215, 0.182, 0.200, "请求级特征编码\n无注意力 MLP（候选）\n或 8 头请求自注意力", purple, 8.8)
    box(
        0.454, 0.215, 0.309, 0.200,
        "7 维离散卸载动作与质量掩码\n本地 0 / MBS 1 / 协作 $2+v$（屏蔽 $2+u$）\n预测时延 $\\leq 1.50 D_m$、算力份额比 $\\geq 0.25$",
        green, 8.8,
    )

    # Lower Critic & adaptive violation multipliers (bottom strip of lower container)
    box(
        0.248, 0.080, 0.515, 0.105,
        "下层 Critic（局部/集中）与违例乘子：$r_t^{\\mathrm{low}}$ 含单向乘子 $(\\lambda_{\\mathrm{dsr}},\\lambda_{\\mathrm{mbs}})\\in[0,10]^2$",
        orange, 8.8,
    )

    # Lower forward & training arrows
    arrow((0.228, 0.355), (0.244, 0.340), blue, "-", 1.1)
    arrow((0.228, 0.225), (0.244, 0.265), teal, "-", 1.1)
    arrow((0.434, 0.315), (0.450, 0.315), green, "-", 1.15)
    arrow((0.228, 0.180), (0.244, 0.138), orange, "--", 1.1)
    arrow((0.605, 0.187), (0.605, 0.211), orange, "--", 1.1)

    # ---- Right Environment Closed-Loop Cards ----
    box(0.810, 0.675, 0.168, 0.215, "S3 时隙末轨迹推进\n裁剪与防碰修复 $\\mathcal{E}(\\cdot)$\n改变 $t+1$ 覆盖几何", blue, 8.8)
    box(0.810, 0.395, 0.168, 0.225, "S0 自然覆盖接纳\n生成已接纳集 $\\mathcal{A}_u^{\\mathrm{srv}}$\n输出上下层观测", green, 8.8)
    box(0.810, 0.095, 0.168, 0.245, "S2 当步批量共享结算\n等份算力 $F_j/N_j(\\mathbf{b})$\n计算 $r_u, r_t^{\\mathrm{low}}$ 与违例", orange, 8.8)

    # Cross-module closed-loop arrows (strictly in empty corridors)
    # Upper Actor -> S3 trajectory
    arrow((0.767, 0.782), (0.806, 0.782), green, "-", 1.25)
    # S3 trajectory -> S0 next-slot natural coverage
    arrow((0.894, 0.670), (0.894, 0.625), blue, "-", 1.2)
    # Lower 7D action -> S2 current-slot execution
    arrow((0.767, 0.285), (0.806, 0.245), green, "-", 1.25)
    # S0 natural coverage -> Upper obs & Lower obs (via central horizontal corridor y=0.508 & left margin x=0.008)
    ax.plot([0.810, 0.008, 0.008], [0.508, 0.508, 0.635], color=green, ls="-", lw=1.15)
    arrow((0.008, 0.635), (0.030, 0.635), green, "-", 1.15)
    ax.plot([0.008, 0.008], [0.508, 0.365], color=teal, ls="-", lw=1.15)
    arrow((0.008, 0.365), (0.030, 0.365), teal, "-", 1.15)
    # S2 rewards/violations -> Upper Critic & Lower Multipliers (dashed)
    arrow((0.806, 0.132), (0.767, 0.132), orange, "--", 1.2)
    ax.plot([0.786, 0.786], [0.180, 0.597], color=red, ls="--", lw=1.15)
    arrow((0.786, 0.597), (0.767, 0.597), red, "--", 1.15)

    legend_handles = [
        Line2D([0], [0], color=green, lw=1.5, ls="-", label="实线：前向决策推理与单时隙环境物理闭环"),
        Line2D([0], [0], color=orange, lw=1.5, ls="--", label="虚线：价值 Critic 评价与自适应违例乘子更新"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.015),
        ncol=2,
        fontsize=8.8,
        frameon=False,
    )

    save(fig, "图3-4_Attention-MAPPO网络与训练结构")

def fig_observation_compare() -> None:
    """图3-5: original vs extended UE observation slot filling (schematic)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    titles = ["(a) 原观测：仅已覆盖 UE", "(b) 扩展观测：补入未覆盖 UE"]
    rng = np.random.default_rng(17)
    n_cov, n_unc = 6, 8
    cov_list: list[np.ndarray] = []
    while len(cov_list) < n_cov:
        ang = float(rng.uniform(0, 2 * np.pi))
        rad = float(np.sqrt(rng.uniform(0.18, 0.82)) * 86.0)
        cand = np.array([rad * np.cos(ang), rad * np.sin(ang)])
        if np.linalg.norm(cand) >= 24.0 and all(np.linalg.norm(cand - q) >= 34.0 for q in cov_list):
            cov_list.append(cand)
    unc_list: list[np.ndarray] = []
    while len(unc_list) < n_unc:
        ang = float(rng.uniform(0, 2 * np.pi))
        rad = float(145.0 + np.sqrt(rng.uniform(0.05, 0.92)) * 270.0)
        cand = np.array([rad * np.cos(ang), rad * np.sin(ang)])
        if all(np.linalg.norm(cand - q) >= 75.0 for q in unc_list):
            unc_list.append(cand)
    cov = np.asarray(cov_list)
    unc = np.asarray(unc_list)

    for k, ax in enumerate(axes):
        uav = np.array([0.0, 0.0])
        ax.add_patch(Circle(uav, 460, fill=False, edgecolor=PALETTE["full"], linestyle="--", lw=1.0))
        ax.add_patch(Circle(uav, 100, fill=True, facecolor=PALETTE["full"], alpha=0.15,
                            edgecolor=PALETTE["full"], lw=1.0))
        ax.plot(*uav, marker="^", color=PALETTE["red"], markersize=11, zorder=5)
        ax.scatter(cov[:, 0], cov[:, 1], s=24, color=PALETTE["green"], zorder=4, label="已覆盖 UE")
        ax.scatter(unc[:, 0], unc[:, 1], s=24, color=PALETTE["red"], marker="x", linewidths=1.15, zorder=4,
                   label="感知内未覆盖 UE")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-560, 500)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[k], fontsize=8.4)
        n_slots = 30
        filled = n_cov if k == 0 else n_cov + n_unc
        for i in range(n_slots):
            x = -470 + i * (940 / n_slots)
            color = PALETTE["green"] if i < n_cov else (
                PALETTE["red"] if (k == 1 and i < filled) else "#E4E6EB")
            ax.add_patch(Rectangle((x, -545), 940 / n_slots - 3, 52,
                                   facecolor=color, edgecolor="white", lw=0.3))
        ax.text(0, -625, f"UE 槽位（共 30 个）：填充 {filled} 个，其余补零",
                ha="center", fontsize=8, color=PALETTE["text"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=2, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.06, wspace=0.08)
    save(fig, "图3-5_原观测与扩展观测对照示意", multi_panel=True)


# ---------------------------------------------------------------- ch3 data figures


def fig_bottleneck_composition() -> None:
    """图3-6: outcome composition of generated service requests on bottleneck review batch."""
    review = json.loads((SOURCE_DIR / "bottleneck_review.json").read_text(encoding="utf-8"))["full_evaluation"]
    f_ok = float(review["success_fraction"])
    f_miss = float(review["admitted_deadline_miss_fraction"])
    f_unc = float(review["uncovered_fraction"])
    f_fail_unc = float(review["fraction_of_failures_uncovered"])

    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    segments = [
        ("按时完成", f_ok, PALETTE["green"]),
        ("已接纳但超时", f_miss, PALETTE["orange"]),
        ("未覆盖", f_unc, PALETTE["red"]),
    ]
    left = 0.0
    for label, val, col in segments:
        ax.barh([0], [val], left=left, height=0.58, color=col, edgecolor="white", lw=1.0, label=label)
        ax.text(left + val / 2, 0.0, f"{100.0 * val:.2f}%", ha="center", va="center", fontsize=9.0, color="white", fontweight="bold")
        left += val
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("占全部实际生成服务请求的比例")
    ax.set_title(f"未覆盖请求占全部失败请求的 {100.0 * f_fail_unc:.2f}%", fontsize=9.0, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=8.4, frameon=False)
    soften_axes(ax, axis="x")
    fig.subplots_adjust(bottom=0.28, top=0.84)
    save(fig, "图3-6_主场景服务请求结果构成")


def fig_confirmation_training() -> None:
    """图3-7: independent confirmation training curves (10 training seeds mean ± SD)."""
    path = SOURCE_DIR / "confirmation_training.csv"
    by_method: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["variant"] == "main":
                by_method[row["method"]][int(row["episode"])].append(
                    (float(row["training_DSR"]), float(row["mean_training_reward"]))
                )

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    specs = [
        (CANDIDATE, "候选（扩展观测）", PALETTE["full"]),
        (STRONG_BASELINE, "下层无注意力强基线", PALETTE["baseline"]),
    ]
    panels = [
        (0, "(a) 训练 DSR", "训练 DSR（每回合）"),
        (1, "(b) 平均训练奖励", "平均训练奖励（每回合）"),
    ]
    for ax, (idx, title, ylabel) in zip(axes, panels):
        for mkey, label, col in specs:
            eps = sorted(by_method[mkey])
            xs = np.array([e + 1 for e in eps], dtype=float)
            arr = np.array([[pair[idx] for pair in by_method[mkey][e]] for e in eps], dtype=float)
            mean = arr.mean(axis=1)
            sd = arr.std(axis=1, ddof=1)
            ax.plot(xs, mean, color=col, lw=1.25, label=label)
            ax.fill_between(xs, mean - sd, mean + sd, color=col, alpha=0.16, lw=0)
        ax.set_xlim(0, 200)
        ax.set_xlabel("训练回合")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8.6, pad=6)
        soften_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=8.4, frameon=False)
    fig.subplots_adjust(bottom=0.18, wspace=0.28)
    save(fig, "图3-7_确认实验训练曲线", multi_panel=True)


def _load_confirmation(cov_root: Path) -> dict[str, dict[int, dict[str, list[float]]]]:
    path = source_file("confirmation_units.csv", cov_root / "results" / "coverage_observation_confirmation_20260927" / "evaluation_units.csv")
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

    n_pos = sum(1 for d in diffs if d > 0)
    ax.text(0.02, 1.10,
            f"均值 +{mean:.2f} 个百分点，配对 $t$ 95% CI [{mean-ci:.2f}, {mean+ci:.2f}]；{n_pos}/{len(diffs)} 种子为正",
            transform=ax.transAxes, fontsize=8, color=PALETTE["red"], va="bottom")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
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
        ax.set_xlim(-0.25, 1.25)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["强基线", "候选"], fontsize=8)
        ax.set_title(f"{title}\n均值：{mb:.3f} → {mc:.3f}", fontsize=8,pad=8)
        vals = base + cand
        span = max(vals) - min(vals)
        ax.set_ylim(min(vals) - 0.25 * span - 0.02, max(vals) + 0.30 * span + 0.03)
        soften_axes(ax)
    pair_handles = [
        Line2D([0], [0], color=PALETTE["baseline"], lw=0.9, alpha=0.65, label="各训练种子配对连线（5 工作负载均值）"),
        Line2D([0], [0], color=PALETTE["full"], lw=1.8, marker="o", markersize=4.8, label="10 个训练种子均值"),
    ]
    fig.legend(handles=pair_handles, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2, fontsize=8, frameon=False)
    save(fig, "图3-9_接纳率与接纳后质量配对比较", multi_panel=True)


# ---------------------------------------------------------------- ch4 figures


def fig_sharing_externality() -> None:
    """图4-1: constructed mechanism example of batch compute sharing (formula-verified)."""
    L_m = 2e6          # bytes
    omega = 2000.0     # cycles / byte
    F_u = 60e9         # Hz
    D_m = 0.25         # s (reference deadline)
    C = omega * L_m
    res = {}
    for n in (2, 4):
        f = F_u / n
        t_cmp = C / f
        res[n] = (t_cmp, t_cmp / D_m)
    assert abs(res[2][0] - 0.13333) < 1e-3 and abs(res[4][0] - 0.26667) < 1e-3

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.65))
    ax = axes[0]
    for row, n in enumerate((2, 4)):
        seg_w = F_u / n / 1e9
        for i in range(n):
            ax.add_patch(Rectangle((i * seg_w, 1 - row), seg_w, 0.72,
                                   facecolor=PALETTE["full"] if row == 0 else PALETTE["orange"],
                                   alpha=0.25 + 0.12 * (i % 2), edgecolor=PALETTE["axis"], lw=0.6))
            ax.text(i * seg_w + seg_w / 2, 1 - row + 0.36, f"{seg_w:.0f} GHz", ha="center",
                    va="center", fontsize=8)
        ax.text(-3.5, 1 - row + 0.36, f"$n={n}$", ha="right", va="center", fontsize=8)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.15, 2.05)
    ax.set_yticks([])
    ax.set_xlabel("UAV 算力 $F_u=60$ GHz 在批内均分", fontsize=8.5)
    ax.set_title("(a) 批量算力均分", fontsize=8.5)

    for ax, (idx, title, unit, fmt) in zip(axes[1:], [
            (0, "(b) 每任务计算时延", "s", "%.3f"),
            (1, "(c) 归一化时延比 $T^{cmp}/D_m$", "比值", "%.3f")]):
        vals = [res[2][idx], res[4][idx]]
        bars = ax.bar(["$n=2$", "$n=4$"], vals, width=0.55,
                      color=[PALETTE["full"], PALETTE["orange"]])
        if idx == 1:
            ax.axhline(1.0, color=PALETTE["red"], lw=0.9, ls="--")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals)*.05, fmt % v, ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel(unit)
        ax.margins(y=0.20)
        soften_axes(ax)
    fig.subplots_adjust(bottom=0.18, wspace=0.30)
    save(fig, "图4-1_批量算力共享与迁移外部影响示意", multi_panel=True)


def fig_mc_itr_flowchart() -> None:
    """Finite search with distinct verified-local and budget exits."""
    fig,ax=plt.subplots(figsize=(6.4,5.5));ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.005",facecolor=color,alpha=.13,edgecolor=color,transform=ax.transAxes))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8,transform=ax.transAxes)
    def arrow(a,b,color="axis"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=.8,color=PALETTE[color],transform=ax.transAxes))
    x,w,cx=.35,.62,.66
    box(x,.84,w,.13,"冻结完整快照与硬候选域\n共享 Greedy 初值，评价系统目标",PALETTE["baseline"])
    box(x,.63,w,.13,"检查轮数预算 $R_{\\max} = 1000$\n有预算：完整枚举单请求迁移",PALETTE["orange"])
    box(x,.42,w,.13,"生产执行器整体重评估\n严格比较：超时数→平均归一化时延",PALETTE["full"])
    box(x,.21,w,.13,"选择最佳严格改善迁移\n重复评价，确认确定且严格下降",PALETTE["green"])
    box(.01,.63,.25,.13,"预算耗尽返回\n不认证局部最优",PALETTE["orange"])
    box(.01,.42,.25,.13,"完整邻域无改善\n验证单迁局部最优",PALETTE["red"])
    arrow((cx,.835),(cx,.765));arrow((cx,.625),(cx,.555));arrow((cx,.415),(cx,.345))
    arrow((x-.006,.695),(.265,.695),"orange");arrow((x-.006,.485),(.265,.485),"red")
    arrow((cx,.205),(cx,.11));arrow((cx,.11),(.30,.11));arrow((.30,.11),(.30,.795));arrow((.30,.795),(cx,.795));arrow((cx,.795),(cx,.765))
    ax.text(.09,.16,"接受后继续\n（目标严格下降）",ha="left",fontsize=8,transform=ax.transAxes)
    save(fig,"图4-2_MC-ITR算法流程图")

def _load_ch4() -> tuple[dict, list[dict]]:
    analysis = json.loads(
        source_file("chapter4_analysis.json", REPO_ROOT / "results" / "canonical_all_20260920" / "chapter4" / "analysis.json").read_text())
    rows = list(csv.DictReader(
        source_file("chapter4_units.csv", REPO_ROOT / "results" / "canonical_all_20260920" / "chapter4" / "evaluation_units.csv").open()))
    return analysis, rows


def fig_main_comparison() -> None:
    """图4-3: main-config quality/latency/time with per-workload-seed scatter."""
    _, rows = _load_ch4()
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["variant"] == "main":
            sums[r["method"]]["misses"].append(float(r["deadline_misses"]))
            sums[r["method"]]["latency"].append(float(r["mean_normalized_latency"]))
            sums[r["method"]]["time"].append(float(r["solver_wall_seconds"]))

    panels = [("misses", "(a) 每快照平均超时数", "%.3f"),
              ("latency", "(b) 平均归一化时延", "%.3f"),
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

        method_ticks(ax)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.20)
        soften_axes(ax)
    method_key(fig, y=-.035)
    save(fig, "图4-3_主配置质量时延耗时对比", multi_panel=True)


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
        if mkey == "mc_itr" and comp["ci95"] is not None:  # archived bootstrap CI must be reproduced exactly
            assert abs(lo - comp["ci95"][0][0]) < 1e-9 and abs(hi - comp["ci95"][1][0]) < 1e-9
        ax.axhline(0, color=PALETTE["axis"], lw=0.8)
        ax.scatter(range(len(seeds)), diffs, s=16, color=color, zorder=3)
        ax.axhline(mean, color=color, lw=1.2, linestyle="--")
        ax.fill_between([-0.5, len(seeds) - 0.5], lo, hi, color=color, alpha=0.15)
        ax.set_title(f"{label}\n均值 {mean:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]",fontsize=8,pad=8)
        ax.set_xlabel("工作负载种子（排序）")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"S{i+1}" for i in range(len(seeds))], fontsize=8)
        soften_axes(ax)
    axes[0].set_ylabel("超时数差值（方法 $-$ Iterative-Base）")
    save(fig, "图4-4_相对Iterative-Base配对差值", multi_panel=True)


def _sensitivity(rows: list[dict], variant_map: list[tuple[str, str]], metric: str):
    series: dict[str, dict[str, tuple[float, float]]] = {}
    for mkey, label, color in CH4_METHODS:
        pts = {}
        for variant, xlabel in variant_map:
            vals = np.array([float(r[metric]) for r in rows
                             if r["variant"] == variant and r["method"] == mkey])
            mean = float(vals.mean())
            ci = T95_DF9 * float(vals.std(ddof=1)) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
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
    panels = [("deadline_misses", "(a) 每快照平均超时数"), ("mean_normalized_latency", "(b) 平均归一化时延")]
    x = np.arange(len(variant_map))
    for ax, (metric, title) in zip(axes, panels):
        series = _sensitivity(rows, variant_map, metric)
        for mkey, label, color in CH4_METHODS:
            means = [series[label][xl][0] for _, xl in variant_map]
            cis = [series[label][xl][1] for _, xl in variant_map]
            ax.errorbar(x, means, yerr=cis, color=color, lw=1.2, marker="o", markersize=3.5,
                        capsize=2, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([xl for _, xl in variant_map], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=8.5)
        soften_axes(ax)
    legend_ax = axes[0] if kind == "ue" else axes[1]
    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,1.08), ncol=4, fontsize=8, frameon=False)
    save(fig, name, multi_panel=True)


def fig_exact_small() -> None:
    """图4-7: Exact-Small snapshot composition and per-component agreement."""
    _, rows = _load_ch4()
    es = [r for r in rows if r["variant"] == "exact_small" and r["method"] == "mc_itr"]
    n_snap = sum(int(r["snapshots"]) for r in es)
    n_empty = sum(float(r["empty_snapshot_fraction"]) * int(r["snapshots"]) for r in es)
    n_req = sum(float(r["admitted_service_requests"]) * int(r["snapshots"]) for r in es)
    n_double = max(0, int(round(n_req - (n_snap - n_empty))))
    n_single = max(0, int(round((n_snap - n_empty) - n_double)))
    n_empty_int = int(round(n_empty))

    means: dict[str, list[float]] = {}
    for mkey in ["heuristic_control", "greedy_single_pass", "iterative_base", "mc_itr", "exact_small"]:
        sel = [r for r in rows if r["variant"] == "exact_small" and r["method"] == mkey]
        arr = np.array([[float(r["deadline_misses"]), float(r["mean_normalized_latency"])] for r in sel])
        means[mkey] = arr.mean(axis=0).tolist()

    fig = plt.figure(figsize=(7.6, 2.8))
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    ax = fig.add_subplot(gs[0])
    cats = ["空快照", "单请求", "多请求"]
    vals = [n_empty_int, n_single, n_double]
    colors = ["#C9CDD4", PALETTE["full"], PALETTE["orange"]]
    left = 0
    for c, v, col in zip(cats, vals, colors):
        ax.barh([0], [v], left=left, color=col, height=0.5, label=f"{c}：{v}（{100.0*v/max(n_snap,1):.1f}%）")
        left += v
    ax.set_xlim(0, n_snap)
    ax.set_yticks([])
    ax.set_xlabel(f"快照数（共 {n_snap}）")
    ax.set_title("(a) 快照构成", fontsize=8.5)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8, loc="upper center",
               bbox_to_anchor=(.5,1.12), ncol=3, frameon=False)
    soften_axes(ax, axis="x")

    labels = [("heuristic_control", "Heuristic"), ("greedy_single_pass", "Greedy"),
              ("iterative_base", "Iterative"), ("mc_itr", "MC-ITR"), ("exact_small", "穷举精确解")]
    comp_titles = ["(b) 超时数", "(c) 归一化时延"]
    fmts = ["%.3f", "%.4f"]
    for gi, (comp, title, fmt) in enumerate(zip(range(2), comp_titles, fmts)):
        ax = fig.add_subplot(gs[gi + 1])
        vals = [means[k][comp] for k, _ in labels]
        cols = ["#9AA0A6", "#D8A448", "#53A6A6", "#3D6FB6", PALETTE["purple"]]
        ax.bar(range(5), vals, color=cols, width=0.62)

        method_ticks(ax, exact=True)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.22)
        soften_axes(ax)
    method_key(fig, y=-.06, exact=True)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "图4-7_Exact-Small验证与快照构成", multi_panel=True)


def fig_bridge() -> None:
    """图4-8: shared-trajectory snapshot bridge (chapter 4 solver vs online lower layer)."""
    from decimal import Decimal, ROUND_HALF_UP

    def fmt_half_up_signed(val: float) -> str:
        q = Decimal(f"{val:.8f}").quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return f"{'+' if q >= 0 else ''}{q}"

    analysis = json.loads(source_file("bridge_analysis.json", REPO_ROOT / "results" / "bridge_20260928" / "full" / "analysis.json").read_text())
    units = analysis["units"]
    methods = [("online_lower", "在线学习下层", PALETTE["baseline"])] + CH4_METHODS

    fig, axes = plt.subplots(1, 3, figsize=(8.9, 2.95))
    fig.subplots_adjust(wspace=0.30, bottom=0.20)
    rng = np.random.default_rng(4)
    for ax, key, title, fmt in [
            (axes[0], "deadline_misses", "(a) 每快照平均超时数", "%.2f"),
            (axes[1], "mean_normalized_latency", "(b) 平均归一化时延", "%.2f")]:
        for i, (mkey, label, color) in enumerate(methods):
            vals = np.array([u[mkey][key] for u in units])
            mean = float(vals.mean())
            ax.bar(i, mean, width=0.62, color=color, alpha=0.85, zorder=2)
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(i + jitter, vals, s=5, color=PALETTE["text"], alpha=0.4, zorder=3)

        method_ticks(ax, online=True)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.18)
        soften_axes(ax)

    ax = axes[2]
    pair = analysis["mc_itr_minus_online_lower_deadline_misses"]
    diffs = np.array(pair["seed_mean_differences"])
    ci = pair["training_seed_t_ci95"] if pair["training_seed_t_ci95"][0] is not None else [float(diffs.mean()), float(diffs.mean())]
    ax.axhline(0, color=PALETTE["axis"], lw=0.8)
    ax.bar(range(len(diffs)), diffs, color=PALETTE["full"], width=0.62)
    ax.axhline(float(pair["mean_difference"]), color=PALETTE["red"], lw=1.0, linestyle="--")

    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([f"S{i+1}" for i in range(len(diffs))], fontsize=8)
    ax.set_xlabel("训练种子（排序）")
    ax.set_ylabel("超时数差值")
    ax.set_title(f"(c) MC-ITR - 在线下层\n均值 {fmt_half_up_signed(pair['mean_difference'])}\n95% CI [{fmt_half_up_signed(ci[0])}, {fmt_half_up_signed(ci[1])}]",fontsize=8,pad=8)
    ax.margins(y=0.25)
    soften_axes(ax)
    method_key(fig, y=-.04, online=True)
    save(fig, "图4-8_候选轨迹快照桥接对比", multi_panel=True)


def fig_ch4_mc_itr_trace() -> None:
    """4.3 节末：真实冻结快照上 MC-ITR 与 Iterative-Base 的共享迭代步双目标演化轨迹对比图。"""
    trace_path = SOURCE_DIR / "mc_itr_trace_example.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing frozen snapshot trace JSON: {trace_path}")
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    snap_a = payload["snapshots"]["lexicographic_tradeoff_s126_t200"]
    snap_b = payload["snapshots"]["self_interested_worsening_s168_t100"]

    fig = plt.figure(figsize=(8.8, 5.2))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.98, bottom=0.12, top=0.86, hspace=0.36, wspace=0.25)
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_a2 = fig.add_subplot(gs[1, 0], sharex=ax_a1)
    ax_b2 = fig.add_subplot(gs[1, 1], sharex=ax_b1)

    mc_color = SEMANTIC_PALETTE["module_blue"]
    ib_color = SEMANTIC_PALETTE["admitted_miss"]
    heur_color = SEMANTIC_PALETTE["neutral_slate"]
    red_color = SEMANTIC_PALETTE["uncovered_expired"]
    green_color = SEMANTIC_PALETTE["covered_ok"]

    panels = [
        (
            snap_a,
            ax_a1,
            ax_a2,
            f"(a) 快照 A 超时数（种子 {snap_a['workload_seed']}，步 {snap_a['step']}，$K={snap_a['num_requests']}$）",
            "(c) 快照 A 平均归一化时延 $F_2(\\mathbf{a}^r)$",
        ),
        (
            snap_b,
            ax_b1,
            ax_b2,
            f"(b) 快照 B 超时数（种子 {snap_b['workload_seed']}，步 {snap_b['step']}，$K={snap_b['num_requests']}$）",
            "(d) 快照 B 平均归一化时延 $F_2(\\mathbf{a}^r)$",
        ),
    ]

    for idx, (snap, ax1, ax2, title1, title2) in enumerate(panels):
        tr_mc = np.asarray(snap["methods"]["mc_itr"]["trace"], dtype=float)
        tr_ib = np.asarray(snap["methods"]["iterative_base"]["trace"], dtype=float)

        max_r = max(len(tr_mc) - 1, len(tr_ib) - 1, 2)
        x_mc = np.arange(len(tr_mc))
        x_ib = np.arange(len(tr_ib))

        # Extend shorter terminated trajectory horizontally with dotted hold line
        if len(tr_ib) - 1 < max_r:
            ax1.plot([x_ib[-1], max_r], [tr_ib[-1, 0], tr_ib[-1, 0]], color=ib_color, ls=":", lw=1.2, label="Iterative-Base 终止保持值")
            ax2.plot([x_ib[-1], max_r], [tr_ib[-1, 1], tr_ib[-1, 1]], color=ib_color, ls=":", lw=1.2)

        ax1.plot(x_ib, tr_ib[:, 0], color=ib_color, ls="--", lw=1.6, marker="s", markersize=5.8, label="Iterative-Base（自身代价判据）")
        ax1.plot(x_mc, tr_mc[:, 0], color=mc_color, ls="-", lw=1.75, marker="o", markersize=5.8, label="MC-ITR（全批次字典序判据）")

        ax2.plot(x_ib, tr_ib[:, 1], color=ib_color, ls="--", lw=1.6, marker="s", markersize=5.8)
        ax2.plot(x_mc, tr_mc[:, 1], color=mc_color, ls="-", lw=1.75, marker="o", markersize=5.8)

        ax1.set_xlim(-0.22, max_r + 0.32)
        ax1.set_xticks(list(range(max_r + 1)))
        ax2.set_xticks(list(range(max_r + 1)))
        ax1.set_title(title1, fontsize=8.6, pad=6)
        ax2.set_title(title2, fontsize=8.6, pad=6)
        ax2.set_xlabel("接受迁移迭代步数 $r$（$r=0$ 为共享贪心初值）", fontsize=8.6)
        soften_axes(ax1)
        soften_axes(ax2)

        if idx == 0:
            ax1.set_ylabel("超时请求数 $F_1(\\mathbf{a}^r)$ / 个", fontsize=8.6)
            ax2.set_ylabel("归一化时延 $F_2(\\mathbf{a}^r)$", fontsize=8.6)
            ax1.set_ylim(2.2, 6.7)
            ax1.set_yticks([3, 4, 5, 6])
            ax2.set_ylim(0.92, 1.13)
            ax1.text(0.02, 3.30, "$F_1$ 严格下降：$6\\to 5\\to 3$", fontsize=8.5, color=green_color)
            ax2.text(0.02, 1.075, "$F_1$ 下降时允许 $F_2$ 合理微升", fontsize=8.5, color=mc_color)
        else:
            ax1.set_ylim(8.2, 11.8)
            ax1.set_yticks([9, 10, 11])
            ax2.set_ylim(1.095, 1.148)
            ax1.text(0.65, 11.35, "自利迁移恶化全局超时 ($10\\to 11$)", fontsize=8.5, color=red_color)
            ax1.text(1.12, 8.42, "MC-ITR 单调不增 ($10\\to 9$)", fontsize=8.5, color=green_color)
            ax2.text(1.05, 1.134, "$F_1=9$ 平台期继续优化 $F_2$", fontsize=8.5, color=mc_color)

    handles, labels = ax_a1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )

    save(fig, "ch4_mc_itr_iteration_trace", multi_panel=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-root",
        type=Path,
        default=Path.home() / ".codex" / "worktrees" / "coverage-optimization" / "wodexuexi",
        help="coverage-optimization worktree root holding the confirmation batch",
    )
    parser.add_argument(
        "--only",
        choices=("all", "ch34_upgrade"),
        default="all",
        help="Render all Chapter 3-4 figures or only the upgraded/new mechanism & trace figures",
    )
    args = parser.parse_args()
    family = setup_thesis_style(font_size=8.5)
    # Explicit guarantees also visible to a source-only preflight; the shared
    # style chooses an installed Chinese font before these final settings.
    plt.rcParams.update({"font.family": "serif", "font.serif": [family],
                         "font.sans-serif": [family],
                         "svg.fonttype": "none", "pdf.fonttype": 42})

    if args.only == "ch34_upgrade":
        fig_network_structure()
        fig_ch4_mc_itr_trace()
    else:
        fig_system_overview()
        fig_coverage_geometry()
        fig_slot_timeline()
        fig_network_structure()
        fig_observation_compare()
        fig_bottleneck_composition()
        fig_confirmation_training()
        fig_confirmation_seed_diff(args.coverage_root)
        fig_admission_quality_paired(args.coverage_root)
        fig_sharing_externality()
        fig_mc_itr_flowchart()
        fig_ch4_mc_itr_trace()
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
        passed = sum(r["verdict"] == "PASS" for r in ALIGN_REPORTS)
        nonapp = sum(not r["applicable"] for r in ALIGN_REPORTS)
        print(f"alignment: {passed} multi-panel pass, {nonapp} single-panel not applicable")


if __name__ == "__main__":
    main()
