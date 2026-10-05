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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_figure_style import PALETTE, panel_label, save_pub, setup_thesis_style, soften_axes

BUNDLED_QA = REPO_ROOT / "tools" / "figure_qa"
SKILL_SCRIPTS = (BUNDLED_QA if BUNDLED_QA.is_dir() else
                 Path.home() / ".codex" / "skills" / "nature-figure" / "scripts")
if SKILL_SCRIPTS.is_dir():
    sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUTPUT_DIRS = [
    REPO_ROOT / "docs" / "figures",
]
QA_REPORT = REPO_ROOT / "output" / "thesis_writing_20261004" / "validation" / "figure_text_qa.json"
ALIGN_DIR = REPO_ROOT / "output" / "thesis_writing_20261004" / "validation" / "figures"

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
    if online: text = "O：历史在线下层；" + text
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
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.11),ncol=2,fontsize=8)
    fig.text(.5,.01,"实线圆：100 m服务范围；虚线弧：460 m感知范围；橙色区域：热点",ha="center",fontsize=8)
    fig.subplots_adjust(bottom=.23)
    save(fig,"图3-1_系统与自然覆盖示意图")

def fig_slot_timeline() -> None:
    """Current-request service precedes application of the prepared motion."""
    fig,ax=plt.subplots(figsize=(7.2,3.8));ax.axis("off")
    entries=[("S0 当前状态","请求已生成\n自然覆盖关联"),("S1 两层准备","当前观测计算\n轨迹与卸载动作"),("S2 请求执行","冻结移动前几何\n按最终负载均分"),("S3 应用轨迹","速度与边界约束\n联合冲突消解"),("S4 能耗结算","实际移动/悬停\n通信/计算/WPT"),("S5 奖励指标","DSR/公平性\n能耗与违例"),("S6 下一状态","缓存提交/UE移动\n新请求与新观测")]
    positions=[(.02,.59),(.27,.59),(.52,.59),(.77,.59),(.77,.10),(.52,.10),(.27,.10)]
    w,h=.21,.31
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
    ax.text(.02,.24,"下一时隙\n返回S0",ha="left",va="center",fontsize=8,transform=ax.transAxes)
    ax.text(.5,1.02,"1 s 时隙：当前请求先执行，轨迹改变后续覆盖",ha="center",fontsize=9,transform=ax.transAxes)
    save(fig,"图3-3_单时隙决策时序图")

def fig_network_structure() -> None:
    """Actual upper actor and joint-local-observation critic; no energy own feature."""
    fig,ax=plt.subplots(figsize=(7.2,4.2));ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.006",facecolor=color,alpha=.13,edgecolor=color,transform=ax.transAxes))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8,transform=ax.transAxes)
    def arrow(a,b,ls="-"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=.8,ls=ls,transform=ax.transAxes))
    box(.01,.72,.20,.20,"自身77维\n位置2 + 缓存75",PALETTE["full"])
    box(.01,.44,.20,.20,"邻机4 × 2维\n相对位置",PALETTE["lower"])
    box(.01,.16,.20,.20,"UE 30 × 8维\n位置/请求/电量",PALETTE["orange"])
    box(.28,.43,.18,.49,"分组嵌入\nLinear/LN/ReLU\n各64维",PALETTE["baseline"])
    for y in (.82,.54,.26):arrow((.215,y),(.275,.67 if y>.7 else .54 if y>.4 else .46))
    box(.53,.43,.20,.49,"自身为query\n邻机与UE交叉注意力\n8头，每头8维\n缩放1/√8\n融合192→128→64",PALETTE["purple"])
    arrow((.465,.68),(.525,.68))
    box(.79,.60,.20,.28,"共享Actor\n二维Normal\n均值tanh\n原始动作概率",PALETTE["green"])
    arrow((.735,.74),(.785,.74))
    box(.35,.03,.28,.22,"联合局部观测\n集中注意力Critic\n每UAV一个价值",PALETTE["red"])
    box(.72,.03,.27,.22,"CTDE / PPO\nclip .2，GAE .95\n两网络分别更新",PALETTE["baseline"])
    arrow((.635,.14),(.715,.14),ls="--")
    ax.text(.5,1.02,"上层 Attention-MAPPO：局部执行，联合观测评价",ha="center",fontsize=9,transform=ax.transAxes)
    save(fig,"图3-4_Attention-MAPPO网络与训练结构")

def fig_observation_compare() -> None:
    """图3-5: original vs extended UE observation slot filling (schematic)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    rng = np.random.default_rng(5)
    titles = ["(a) 原观测：仅已覆盖 UE", "(b) 扩展观测：补入未覆盖 UE"]
    for k, ax in enumerate(axes):
        rng = np.random.default_rng(5)  # same illustrative geometry in both panels
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
                ha="center", fontsize=8, color=PALETTE["text"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.10),ncol=2,fontsize=8,frameon=False)
    fig.text(0.5, 0.015, "示意图：槽位填充规则对照，几何与 UE 数为示意，并非注意力权重或真实观测记录",
             ha="center", fontsize=8, color=PALETTE["baseline"], style="italic")
    fig.subplots_adjust(bottom=0.10, wspace=0.08)
    save(fig, "图3-5_原观测与扩展观测对照示意", multi_panel=True)


# ---------------------------------------------------------------- ch3 data figures


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

    ax.text(0.02, 1.10,
            f"均值 +{mean:.2f} 个百分点，配对 $t$ 95% CI [{mean-ci:.2f}, {mean+ci:.2f}]；10/10 种子为正",
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
    axes[0].text(0.0, -0.30, "灰线：10 个训练种子各自的配对连接（每种取 5 个工作负载均值）；蓝线：种子均值",
                 transform=axes[0].transAxes, fontsize=8, color=PALETTE["baseline"])
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
                    va="center", fontsize=8)
        ax.text(-3.5, 1 - row + 0.36, f"$n={n}$", ha="right", va="center", fontsize=8)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.15, 2.05)
    ax.set_yticks([])
    ax.set_xlabel("UAV 算力 $F_u=60$ GHz 在批内最终负载间均分", fontsize=9)
    ax.set_title("(a) 批量算力均分", fontsize=8.5)

    for ax, (idx, title, unit, fmt) in zip(axes[1:], [
            (0, "(b) 每任务计算时延", "s", "%.3f"),
            (1, "(c) 每任务计算能耗", "J", "%.0f")]):
        vals = [res[2][idx], res[4][idx]]
        bars = ax.bar(["$n=2$", "$n=4$"], vals, width=0.55,
                      color=[PALETTE["full"], PALETTE["orange"]])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals)*.05, fmt % v, ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel(unit)
        ax.margins(y=0.20)
        soften_axes(ax)
    fig.text(0.5, -0.11,
             "机制构造例（非仿真数据）：$L_m=2$ MB，$\\omega_m=2000$ cycles/byte，$\\kappa=10^{-27}$\n"
             "$T^{cmp}=\\omega_m L_m/(F_u/n)$，$E^{cmp}=\\kappa\\,\\omega_m L_m\\,(F_u/n)^2$\n"
             "任务迁入使批内每任务时延上升、单任务计算能耗下降。",
             ha="center", fontsize=9, color=PALETTE["baseline"])
    fig.subplots_adjust(bottom=0.20, wspace=0.30)
    save(fig, "图4-1_批量算力共享与迁移外部影响示意", multi_panel=True)


def fig_mc_itr_flowchart() -> None:
    """Finite search with distinct verified-local and budget exits."""
    fig,ax=plt.subplots(figsize=(6.4,5.8));ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.005",facecolor=color,alpha=.13,edgecolor=color,transform=ax.transAxes))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8,transform=ax.transAxes)
    def arrow(a,b,color="axis"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=.8,color=PALETTE[color],transform=ax.transAxes))
    x,w,cx=.35,.62,.66
    box(x,.85,w,.12,"冻结完整快照与硬候选域\n共享 Greedy 初值，评价系统目标",PALETTE["baseline"])
    box(x,.64,w,.12,"检查轮数预算 Rmax = 1000\n有预算：完整枚举单请求迁移",PALETTE["orange"])
    box(x,.43,w,.12,"生产执行器整体重评估\n严格比较：超时数→归一化时延→能耗",PALETTE["full"])
    box(x,.22,w,.12,"选择最佳严格改善迁移\n重复评价，确认确定且严格下降",PALETTE["green"])
    box(.01,.64,.25,.12,"预算耗尽返回\n不认证局部最优",PALETTE["orange"])
    box(.01,.43,.25,.12,"完整邻域无改善\n验证单迁局部最优",PALETTE["red"])
    arrow((cx,.845),(cx,.765));arrow((cx,.635),(cx,.555));arrow((cx,.425),(cx,.345))
    arrow((x-.006,.70),(.265,.70),"orange");arrow((x-.006,.49),(.265,.49),"red")
    arrow((cx,.215),(cx,.12));arrow((cx,.12),(.30,.12));arrow((.30,.12),(.30,.80));arrow((.30,.80),(cx,.80));arrow((cx,.80),(cx,.765))
    ax.text(.09,.17,"接受后继续\n（目标严格下降）",ha="left",fontsize=8,transform=ax.transAxes)
    ax.text(.5,1.015,"MC-ITR：全批次重评估的最佳单请求迁移",ha="center",fontsize=9,transform=ax.transAxes)
    save(fig,"图4-2_MC-ITR算法流程图")

def _load_ch4() -> tuple[dict, list[dict]]:
    analysis = json.loads(
        source_file("chapter4_analysis.json", REPO_ROOT / "results" / "canonical_all_20260920" / "chapter4" / "analysis.json").read_text())
    rows = list(csv.DictReader(
        source_file("chapter4_units.csv", REPO_ROOT / "results" / "canonical_all_20260920" / "chapter4" / "evaluation_units.csv").open()))
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

        method_ticks(ax)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.20)
        soften_axes(ax)
    method_key(fig, y=-.025)
    fig.text(.5,-.095,"柱：均值；散点：10 工作负载；迭代计时不含共享贪心初值",ha="center",fontsize=8)
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
        ax.set_title(f"{label}\n均值 {mean:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]",fontsize=8,pad=8)
        ax.set_xlabel("工作负载种子（排序）")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"S{i+1}" for i in range(len(seeds))], fontsize=8)
        soften_axes(ax)
    axes[0].set_ylabel("超时数差值（方法 $-$ Iterative-Base）")
    axes[0].text(0.0, -0.38, "S1-S10：42、84、126、168、210、252、294、336、378、420；阴影：5000 次 bootstrap 的95%区间",
                 transform=axes[0].transAxes, fontsize=8, color=PALETTE["baseline"])
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
        ax.set_xticklabels([xl for _, xl in variant_map], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=8.5)
        soften_axes(ax)
    legend_ax = axes[0] if kind == "ue" else axes[1]
    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,1.08), ncol=4, fontsize=8, frameon=False)
    axes[0].text(0.0, -0.32, "误差线：10 个工作负载种子的逐项 95% $t$ 区间（非同时覆盖）\n每个设置独立采集快照并求解",
                 transform=axes[0].transAxes, fontsize=8, color=PALETTE["baseline"])
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
        ax.barh([0], [v], left=left, color=col, height=0.5, label=f"{c}：{v}（{v/6:.1f}%）")
        left += v
    ax.set_xlim(0, 600)
    ax.set_yticks([])
    ax.set_xlabel("快照数（共 600）")
    ax.set_title("(a) 快照构成", fontsize=8.5)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8, loc="upper center",
               bbox_to_anchor=(.5,1.12), ncol=3, frameon=False)
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

        method_ticks(ax, exact=True)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.22)
        soften_axes(ax)
    fig.text(0.5, -0.06, "三 UE 设置：MC-ITR、Greedy、Iterative 与精确解的逐分量均值相同\n实现一致性检查，非主配置全局最优性证明",
             ha="center", fontsize=8, color=PALETTE["baseline"])
    method_key(fig, y=-.19, exact=True)
    fig.subplots_adjust(bottom=0.30)
    save(fig, "图4-7_Exact-Small验证与快照构成", multi_panel=True)


def fig_bridge() -> None:
    """图4-8: shared-trajectory snapshot bridge (chapter 4 solver vs online lower layer)."""
    analysis = json.loads(source_file("bridge_analysis.json", REPO_ROOT / "results" / "bridge_20260928" / "full" / "analysis.json").read_text())
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

        method_ticks(ax, online=True)
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

    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([f"S{i+1}" for i in range(len(diffs))], fontsize=8)
    ax.set_xlabel("训练种子（排序）")
    ax.set_ylabel("超时数差值")
    ax.set_title(f"(c) MC-ITR - 在线下层\n均值 {pair['mean_difference']:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",fontsize=8,pad=8)
    ax.margins(y=0.25)
    soften_axes(ax)
    fig.text(.5,-.25,
             "历史 09-28 覆盖候选（非 SF-HLM）：50 评估任务 × 8 冻结快照 = 400 快照\n"
             "10 训练种子 × 5 工作负载；反事实分配未回写环境",
             ha="center", fontsize=8, color=PALETTE["baseline"])
    method_key(fig, y=-.14, online=True)
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
    family = setup_thesis_style(font_size=8)
    # Explicit guarantees also visible to a source-only preflight; the shared
    # style chooses an installed Chinese font before these final settings.
    plt.rcParams.update({"font.family": "serif", "font.serif": [family],
                         "svg.fonttype": "none", "pdf.fonttype": 42})

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
        passed = sum(r["verdict"] == "PASS" for r in ALIGN_REPORTS)
        nonapp = sum(not r["applicable"] for r in ALIGN_REPORTS)
        print(f"alignment: {passed} multi-panel pass, {nonapp} single-panel not applicable")


if __name__ == "__main__":
    main()
