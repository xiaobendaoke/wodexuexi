#!/usr/bin/env python3
"""Generate Chapter 3 redesigned figures (Figs 3-1 through 3-6) from frozen canonical data.

Reads only read-only data from docs/figures/data/chapter34_20261004/.
Canvas width is locked 1:1 to SYSU master's thesis \\textwidth = 146.6 mm (415.56 pt).
"""

from __future__ import annotations

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

DATA_DIR = REPO_ROOT / "docs" / "figures" / "data" / "chapter34_20261004"
OUT_DIRS = [
    REPO_ROOT / "docs" / "figures",
    REPO_ROOT / "output" / "thesis_figure_redesign_20261112" / "deliverables",
]
QA_DIR = REPO_ROOT / "output" / "thesis_figure_redesign_20261112" / "validation"

T95_DF9 = 2.2621571627409915


def build_fig3_01_observation_aliasing() -> dict[str, object]:
    """图 3-1：双热点自然覆盖下前端观测混叠盲区与固定 30 槽位扩展需求观测机制。"""
    c = SchematicCanvas(height_mm=88.0)

    # Panel (a): Spatial aliasing contrast (Theorem 3.2.1)
    c.panel_tag(6.0, 4.0, "(a)")
    c.container(
        22.0,
        4.0,
        186.0,
        216.0,
        "预接纳观测混叠盲区（定理 3.2.1）",
        stroke=P["comparator"],
        fill="#FFFBF8",
        title_color=P["comparator"],
    )
    c.box(
        30.0,
        28.0,
        80.0,
        42.0,
        "状态 $s^{(A)}(t)$",
        "西北热点未入 $R_c$",
        stroke=P["comparator"],
    )
    c.box(
        120.0,
        28.0,
        80.0,
        42.0,
        "状态 $s^{(B)}(t)$",
        "东南热点未入 $R_c$",
        stroke=P["comparator"],
    )
    c.arrow((70.0, 70.5), (96.0, 91.5), color=P["comparator"], lw=1.05)
    c.arrow((160.0, 70.5), (134.0, 91.5), color=P["comparator"], lw=1.05)

    c.box(
        30.0,
        92.0,
        170.0,
        44.0,
        "原关联观测编码完全混叠",
        "$\\Phi_{\\mathrm{assoc},u}(s^{(A)})=\\Phi_{\\mathrm{assoc},u}(s^{(B)})$",
        fill=P["comp_light"],
        stroke=P["comparator"],
        bold_title=True,
    )
    c.arrow((115.0, 136.5), (115.0, 152.5), color=P["crimson"], lw=1.1, ls="--")
    c.box(
        30.0,
        153.0,
        170.0,
        56.0,
        "结构性零雅可比截断",
        "$\\partial\\Phi_{\\mathrm{assoc},u}/\\partial\\mathbf{w}_m=\\mathbf{0}$\n策略无法辨识圈外热点方位",
        fill="#FCE8E6",
        stroke=P["crimson"],
    )

    # Panel (b): Fixed 30-slot extended demand observation (Lemma 3.3.1)
    c.panel_tag(214.0, 4.0, "(b)")
    c.container(
        230.0,
        4.0,
        178.0,
        216.0,
        "固定 30 槽扩展观测（引理 3.3.1）",
        stroke=P["hero"],
        fill="#F8FBFF",
        title_color=P["hero"],
    )
    c.box(
        238.0,
        28.0,
        162.0,
        36.0,
        "原观测 $\\Phi_{\\mathrm{assoc},u}\\in\\mathbb{R}^{295}$",
        "槽 $1..k$: 已关联 | 槽 $k+1..30$: 全零",
        stroke=P["prior"],
    )
    c.arrow((319.0, 64.5), (319.0, 80.5), color=P["hero"], lw=1.15)
    c.text(364.0, 72.5, "距离升序补槽", fontsize=7.5, color=P["hero"], bold=True)

    c.box(
        238.0,
        81.0,
        162.0,
        54.0,
        "扩展观测 $\\Phi_{\\mathrm{ext},u}\\in\\mathbb{R}^{295}$",
        "槽 $1..k$: 无损保留 $\\mathcal{U}_u(t)$\n槽 $k+1..30$: 补入 $R_s$ 内 $\\mathcal{V}_{\\mathrm{un},u}(t)$",
        fill=P["hero_light"],
        stroke=P["hero"],
        bold_title=True,
    )
    c.arrow((319.0, 135.5), (319.0, 152.5), color=P["success"], lw=1.15)
    c.box(
        238.0,
        153.0,
        162.0,
        56.0,
        "维数不变与满秩可辨识",
        "$77+4\\times 2+30\\times 7=295$ 维不变\n$\\mathrm{rank}(\\partial\\mathbf{x}_{u,m}/\\partial\\mathbf{w}_m)=2$",
        fill=P["dist_light"],
        stroke=P["success"],
    )

    c.legend_row(
        235.0,
        [
            ("box:" + P["comp_light"], P["comparator"], "原仅关联观测混叠盲区"),
            ("box:" + P["hero_light"], P["hero"], "感知域 $R_s=460\\,\\mathrm{m}$ 扩展补槽"),
            ("box:" + P["dist_light"], P["success"], "参数规模与 $R_c=100\\,\\mathrm{m}$ 物理边界不变"),
        ],
        x_start=14.0,
        x_end=406.0,
        fontsize=7.4,
    )
    return c.export("fig3_01_observation_aliasing", repo_root=REPO_ROOT, output_dirs=OUT_DIRS, qa_dir=QA_DIR)


def build_fig3_02_hierarchical_mappo() -> dict[str, object]:
    """图 3-2：协议 I 下分层 Attention-MAPPO 与精简下层卸载策略的张量耦合及因果训练时序。"""
    c = SchematicCanvas(height_mm=98.0)

    # Top container: Single-slot S0-S3 causal timeline
    c.container(
        8.0,
        5.0,
        399.5,
        64.0,
        "单时隙物理执行与分层决策因果时序（$\\tau=1.0\\,\\mathrm{s}$）",
        stroke=P["border"],
        fill=P["panel_bg"],
    )
    stages = [
        (15.0, 90.0, "S0 自然覆盖接纳", "最近邻生成 $\\mathcal{A}_{\\mathrm{srv},u}(t)$", P["ink"]),
        (114.0, 92.0, "S1 双层前向推理", "上层扩槽 / 下层真接纳", P["hero"]),
        (215.0, 92.0, "S2 批量共享结算", "$f_{m,j}=F_j/N_j$ 算 DSR", P["distilled"]),
        (316.0, 84.5, "S3 运动安全推进", "修复 $\\mathcal{E}(\\delta)$ 至 $t+1$", P["comparator"]),
    ]
    for x, w, t1, t2, col in stages:
        c.box(x, 26.0, w, 36.0, t1, t2, stroke=col)
    c.arrow((105.5, 44.0), (113.5, 44.0), color=P["ink"], lw=1.05)
    c.arrow((206.5, 44.0), (214.5, 44.0), color=P["hero"], lw=1.05)
    c.arrow((307.5, 44.0), (315.5, 44.0), color=P["distilled"], lw=1.05)

    # Bottom Left: Upper Attention-MAPPO
    c.container(
        8.0,
        77.0,
        196.0,
        172.0,
        "上层轨迹规划：扩展观测 Attention-MAPPO",
        stroke=P["hero"],
        fill="#F8FBFF",
        title_color=P["hero"],
    )
    c.box(
        15.0,
        100.0,
        182.0,
        38.0,
        "扩展局部观测 $\\Phi_{\\mathrm{ext},u}\\in\\mathbb{R}^{295}$",
        "自身 $77$ + 邻机 $4\\times 2$ + 终端 $30\\times 7$",
        fill=P["hero_light"],
        stroke=P["hero"],
    )
    c.box(
        15.0,
        148.0,
        182.0,
        42.0,
        "$8$ 头交叉注意力（$d_h=8$，掩码空槽）",
        "$192\\to 128\\to 64$ 输出高斯均值 $\\mu_\\theta\\in(-1,1)^2$",
        stroke=P["hero"],
    )
    c.box(
        15.0,
        200.0,
        182.0,
        41.0,
        "集中式自注意力 Critic（仅训练期）",
        "联合观测 $\\mathbf{o}_t\\in\\mathbb{R}^{5\\times 295}\\to V_{\\phi,u}(\\mathbf{o}_t)$",
        fill="#FFFFFF",
        stroke=P["crimson"],
        ls="--",
    )
    c.arrow((106.0, 138.5), (106.0, 147.5), color=P["hero"], lw=1.05)
    c.arrow((106.0, 199.5), (106.0, 190.5), color=P["crimson"], lw=1.05, ls="--")

    # Bottom Right: Slim Lower Constrained Offload-MAPPO
    c.container(
        212.0,
        77.0,
        195.5,
        172.0,
        "下层任务卸载：精简无注意力约束 MAPPO",
        stroke=P["distilled"],
        fill="#F6FCF9",
        title_color=P["distilled"],
    )
    c.box(
        219.0,
        100.0,
        181.5,
        38.0,
        "请求级观测 $\\mathbb{R}^{1405}$（仅含 $\\mathcal{A}_{\\mathrm{srv},u}(t)$）",
        "状态 $5$ + $100$ 请求槽 $\\times 14$（去请求自注意力）",
        fill=P["dist_light"],
        stroke=P["distilled"],
    )
    c.box(
        219.0,
        148.0,
        181.5,
        42.0,
        "精简 MLP Actor + $7$ 维质量掩码",
        "本地 ($0$) / 宏站 ($1$) / 邻机 ($2+v$) 路由采样",
        stroke=P["distilled"],
    )
    c.box(
        219.0,
        200.0,
        181.5,
        41.0,
        "局部观测 Critic + 自适应违例乘子",
        "$\\mathbb{R}^{1405}\\to V_{\\psi,u}$ 及 $(\\lambda_{\\mathrm{dsr}},\\lambda_{\\mathrm{mbs}})$ 单向更新",
        fill="#FFFFFF",
        stroke=P["crimson"],
        ls="--",
    )
    c.arrow((309.8, 138.5), (309.8, 147.5), color=P["distilled"], lw=1.05)
    c.arrow((309.8, 199.5), (309.8, 190.5), color=P["crimson"], lw=1.05, ls="--")

    # Environment coupling arrows from top timeline to upper/lower layers
    c.arrow((152.0, 69.5), (152.0, 76.5), color=P["hero"], lw=1.05)
    c.arrow((261.0, 69.5), (261.0, 76.5), color=P["distilled"], lw=1.05)

    c.legend_row(
        264.0,
        [
            ("-", P["hero"], "实线：单步在线推理与物理状态闭环"),
            ("--", P["crimson"], "虚线：训练期 Critic 优势估计与拉格朗日乘子更新"),
        ],
        x_start=22.0,
        x_end=398.0,
    )
    return c.export("fig3_02_hierarchical_mappo", repo_root=REPO_ROOT, output_dirs=OUT_DIRS, qa_dir=QA_DIR)


def build_fig3_03_architecture_bottleneck() -> dict[str, object]:
    """图 3-3：主场景分层架构基线受控筛选与自然覆盖下全量生成请求失败归因诊断。"""
    cba = json.loads((DATA_DIR / "canonical_baseline_analysis.json").read_text(encoding="utf-8"))
    bneck = json.loads((DATA_DIR / "bottleneck_review.json").read_text(encoding="utf-8"))

    # Aggregate 5 training seeds on main variant
    by_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in cba["units"]:
        if row["variant"] == "main":
            by_seed[(row["method"], int(row["training_seed"]))].append(float(row["DSR"]) * 100.0)
    seeds = sorted({s for (_, s) in by_seed})
    assert len(seeds) == 5
    dual_seed = np.array([np.mean(by_seed[("hierarchical", s)]) for s in seeds])
    slim_seed = np.array([np.mean(by_seed[("lower_no_attention", s)]) for s in seeds])
    assert abs(np.mean(dual_seed) - 23.9044) < 0.02
    assert abs(np.mean(slim_seed) - 25.0680) < 0.02

    fe = bneck["full_evaluation"]
    succ_pct = float(fe["success_fraction"]) * 100.0
    miss_pct = float(fe["admitted_deadline_miss_fraction"]) * 100.0
    uncov_pct = float(fe["uncovered_fraction"]) * 100.0
    assert abs((succ_pct + miss_pct + uncov_pct) - 100.0) < 1e-4
    uncov_fail_share = float(fe["fraction_of_failures_uncovered"]) * 100.0
    assert abs(uncov_fail_share - 68.5502) < 0.02

    st = bneck["snapshot_totals"]
    gen_snap = float(st["generated"])
    s_learn = float(st["learned_success"]) / gen_snap * 100.0
    s_qual = float(st["quality_search_success"]) / gen_snap * 100.0
    s_hard = float(st["hard_search_success"]) / gen_snap * 100.0

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.58))
    gs = fig.add_gridspec(1, 3, left=0.078, right=0.988, bottom=0.195, top=0.845, wspace=0.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): Paired 5-seed architecture comparison
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=8.0)
    for d_v, s_v in zip(dual_seed, slim_seed):
        ax_a.plot([0, 1], [d_v, s_v], color=P["muted"], lw=0.85, alpha=0.85, zorder=2)
    ax_a.scatter(np.zeros_like(dual_seed), dual_seed, s=16, color=P["ablation"], alpha=0.75, zorder=3)
    ax_a.scatter(np.ones_like(slim_seed), slim_seed, s=16, color=P["comparator"], alpha=0.75, zorder=3)
    ax_a.plot([0, 1], [np.mean(dual_seed), np.mean(slim_seed)], color=P["ink"], lw=1.6, zorder=4)
    ax_a.scatter([0, 1], [np.mean(dual_seed), np.mean(slim_seed)], s=34, color=[P["ablation"], P["comparator"]], zorder=5)
    ax_a.text(0.5, 29.1, "$+1.16$ pp $[+0.43,+1.99]$", fontsize=7.4, fontweight="bold", color=P["comparator"], ha="center", va="top")
    ax_a.set_xlim(-0.35, 1.35)
    ax_a.set_ylim(21.5, 29.6)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["双注意力", "下层精简"])
    ax_a.set_yticks([22, 25, 28])
    ax_a.set_ylabel("端到端满足率 DSR (%)")

    # Panel (b): Request outcome & failure source decomposition (68.55% uncovered!)
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=8.0)
    cats = ["按时\n完成", "接纳\n超时", "前端\n未覆盖"]
    vals = [succ_pct, miss_pct, uncov_pct]
    cols = [P["success"], P["amber"], P["crimson"]]
    bars = ax_b.bar([0, 1, 2], vals, width=0.54, color=cols, edgecolor=P["ink"], linewidth=0.65, zorder=3)
    for bar, val in zip(bars, vals):
        ax_b.text(bar.get_x() + bar.get_width() / 2.0, val + 1.8, f"{val:.1f}%", fontsize=7.4, ha="center", va="bottom")
    ax_b.text(0.52, 64.5, "未覆盖占失败\n$68.55\\%$", fontsize=7.4, fontweight="bold", color=P["crimson"], ha="center", va="top")
    ax_b.set_xlim(-0.55, 2.55)
    ax_b.set_ylim(0.0, 67.5)
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(cats, fontsize=7.4)
    ax_b.set_yticks([0, 20, 40, 60])
    ax_b.set_ylabel("占生成服务请求比例 (%)")

    # Panel (c): 50-snapshot fixed-coverage 2-step search headroom (zero baseline)
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=8.0)
    s_vals = [s_learn, s_qual, s_hard]
    s_cols = [P["comparator"], P["distilled"], P["hero"]]
    bars_c = ax_c.bar([0, 1, 2], s_vals, width=0.54, color=s_cols, edgecolor=P["ink"], linewidth=0.65, zorder=3)
    for bar, val in zip(bars_c, s_vals):
        ax_c.text(bar.get_x() + bar.get_width() / 2.0, val + 0.9, f"{val:.1f}%", fontsize=7.4, ha="center", va="bottom")
    ax_c.text(1.0, 35.8, "同批重分配 $+3.7\\sim+4.2$ pp", fontsize=7.3, color=P["hero"], ha="center", va="top")
    ax_c.set_xlim(-0.55, 2.55)
    ax_c.set_ylim(0.0, 37.5)
    ax_c.set_xticks([0, 1, 2])
    ax_c.set_xticklabels(["在线\n下层", "质量\n掩码", "硬可\n行域"], fontsize=7.4)
    ax_c.set_yticks([0, 10, 20, 30])
    ax_c.set_ylabel("冻结快照完成率 (%)")

    return export_redesign_figure(
        fig,
        "fig3_03_architecture_bottleneck",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig3_04_confirmation_convergence() -> dict[str, object]:
    """图 3-4：十组独立训练种子的收敛轨迹、逐种子配对斜率与确认实验核心指标森林图。"""
    with (DATA_DIR / "confirmation_training.csv").open(encoding="utf-8-sig") as f:
        tr_rows = list(csv.DictReader(f))
    with (DATA_DIR / "confirmation_units.csv").open(encoding="utf-8-sig") as f:
        unit_rows = list(csv.DictReader(f))
    comp = json.loads((DATA_DIR / "confirmation_paired_comparison.json").read_text(encoding="utf-8"))

    # Training curves smoothed with causal window W=10 across 200 episodes for 10 seeds
    curves: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in tr_rows:
        curves[r["method"]][int(r["training_seed"])].append((int(r["episode"]), float(r["training_DSR"]) * 100.0))

    def _agg_curve(method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seed_mats = []
        for s in sorted(curves[method]):
            pts = [v for _, v in sorted(curves[method][s])]
            kernel = np.ones(10) / 10.0
            sm = np.convolve(pts, kernel, mode="valid")
            seed_mats.append(sm)
        mat = np.asarray(seed_mats)
        eps = np.arange(10, 10 + mat.shape[1])
        mean = mat.mean(axis=0)
        se = mat.std(axis=0, ddof=1) / math.sqrt(mat.shape[0])
        return eps, mean, T95_DF9 * se

    eps, base_m, base_ci = _agg_curve("lower_no_attention")
    _, cand_m, cand_ci = _agg_curve("lower_mlp_upper_uncovered_obs")

    # Seed means on confirmation evaluation (10 seeds x 5 workloads)
    seed_dsr: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in unit_rows:
        seed_dsr[(r["method"], int(r["training_seed"]))].append(float(r["DSR"]) * 100.0)
    seeds = sorted(comp["training_seeds"])
    b_seeds = np.array([np.mean(seed_dsr[("lower_no_attention", s)]) for s in seeds])
    c_seeds = np.array([np.mean(seed_dsr[("lower_mlp_upper_uncovered_obs", s)]) for s in seeds])
    assert abs(np.mean(c_seeds - b_seeds) - 2.8509) < 0.01

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.68))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.988, bottom=0.185, top=0.825, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): 10-seed training DSR curves (mean +/- 95% t-band)
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=20.0)
    ax_a.fill_between(eps, base_m - base_ci, base_m + base_ci, color=P["comparator"], alpha=0.18, edgecolor="none", zorder=1)
    ax_a.fill_between(eps, cand_m - cand_ci, cand_m + cand_ci, color=P["hero"], alpha=0.20, edgecolor="none", zorder=1.5)
    ax_a.plot(eps, base_m, color=P["comparator"], lw=1.25, ls="--", label="原观测基线", zorder=3)
    ax_a.plot(eps, cand_m, color=P["hero"], lw=1.45, ls="-", label="扩展需求观测", zorder=4)
    ax_a.set_xlim(10, 200)
    ax_a.set_ylim(10.0, 31.0)
    ax_a.set_xticks([20, 80, 140, 200])
    ax_a.set_yticks([10, 15, 20, 25, 30])
    ax_a.set_xlabel("训练回合 (Episode)")
    ax_a.set_ylabel("训练期满足率 DSR (%)")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=7.4, frameon=False, labelspacing=0.15)

    # Panel (b): 10-seed paired slope plot on blind confirmation workloads
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=20.0)
    for bv, cv in zip(b_seeds, c_seeds):
        col = P["hero_mid"] if cv >= bv else P["crimson"]
        ax_b.plot([0, 1], [bv, cv], color=col, lw=0.85, alpha=0.72, zorder=2)
    ax_b.scatter(np.zeros_like(b_seeds), b_seeds, s=15, color=P["comparator"], alpha=0.8, zorder=3)
    ax_b.scatter(np.ones_like(c_seeds), c_seeds, s=15, color=P["hero"], alpha=0.8, zorder=3)
    ax_b.plot([0, 1], [b_seeds.mean(), c_seeds.mean()], color=P["ink"], lw=1.7, zorder=4)
    ax_b.scatter([0, 1], [b_seeds.mean(), c_seeds.mean()], s=34, color=[P["comparator"], P["hero"]], zorder=5)
    ax_b.text(0.5, 31.2, "$9/10$ 种子正增益\n均值 $+2.85$ pp", fontsize=7.5, fontweight="bold", color=P["hero"], ha="center", va="top")
    ax_b.set_xlim(-0.32, 1.32)
    ax_b.set_ylim(21.0, 31.8)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["原观测", "扩展观测"])
    ax_b.set_yticks([22, 25, 28, 31])
    ax_b.set_ylabel("确认集满足率 DSR (%)")

    # Panel (c): Multi-metric 95% t-CI forest (percentage points)
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=20.0)
    forest_specs = [
        ("端到端 DSR", comp["metrics"]["DSR"], P["hero"]),
        ("接纳率 $\\alpha$", comp["metrics"]["Admission fraction"], P["distilled"]),
        ("接纳完成 $\\eta$", comp["metrics"]["Deadline satisfaction among admitted"], P["comparator"]),
        ("公平性 $J$", comp["metrics"]["Fairness"], P["ablation"]),
    ]
    ax_c.axvline(0.0, color=P["prior"], lw=0.9, ls="--", zorder=1)
    y_pos = np.arange(len(forest_specs))[::-1]
    for y, (_, mdict, col) in zip(y_pos, forest_specs):
        est = float(mdict["mean_difference"]) * 100.0
        lo, hi = [float(x) * 100.0 for x in mdict["training_seed_t_ci95"]]
        ax_c.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=3)
        ax_c.scatter([est], [y], s=26, color=col, zorder=4)
    ax_c.set_xlim(-10.5, 16.5)
    ax_c.set_ylim(-0.55, 3.65)
    ax_c.set_xticks([-8, 0, 8, 15])
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([x[0] for x in forest_specs], fontsize=7.4)
    ax_c.set_xlabel("配对差值 (百分点, $95\\%$ CI)")

    return export_redesign_figure(
        fig,
        "fig3_04_confirmation_convergence",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig3_05_dsr_phase_waterfall() -> dict[str, object]:
    """图 3-5：确认实验接纳率—接纳后完成率等 DSR 相平面迁移、对称分解瀑布与全量请求构成。"""
    with (DATA_DIR / "confirmation_units.csv").open(encoding="utf-8-sig") as f:
        unit_rows = list(csv.DictReader(f))
    comp = json.loads((DATA_DIR / "confirmation_paired_comparison.json").read_text(encoding="utf-8"))

    # Exact 50-unit paired symmetric decomposition (AUD-08 verified: +4.8664 pp and -2.0155 pp)
    by_unit: dict[tuple[int, int], dict[str, tuple[float, float, float]]] = defaultdict(dict)
    by_seed: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for r in unit_rows:
        s = int(r["training_seed"])
        w = int(r["workload_seed"])
        m = r["method"]
        al = float(r["Admission fraction"])
        et = float(r["Deadline satisfaction among admitted"])
        ds = float(r["DSR"])
        by_unit[(s, w)][m] = (al, et, ds)
        by_seed[(m, s)].append((al * 100.0, et * 100.0))

    adm_terms, qual_terms = [], []
    for (_, _), pair in sorted(by_unit.items()):
        a0, e0, d0 = pair["lower_no_attention"]
        a1, e1, d1 = pair["lower_mlp_upper_uncovered_obs"]
        adm_terms.append(0.5 * (e0 + e1) * (a1 - a0) * 100.0)
        qual_terms.append(0.5 * (a0 + a1) * (e1 - e0) * 100.0)
        assert abs((a1 * e1 - a0 * e0) - (d1 - d0)) < 1e-6
    adm_gain = float(np.mean(adm_terms))
    qual_loss = float(np.mean(qual_terms))
    base_dsr = float(comp["metrics"]["DSR"]["baseline_mean"]) * 100.0
    cand_dsr = float(comp["metrics"]["DSR"]["candidate_mean"]) * 100.0
    assert abs(adm_gain - 4.8664) < 0.01 and abs(qual_loss - (-2.0155)) < 0.01

    # Pooled counts conservation check (G = 9,805,180)
    pt_b = comp["pooled_totals"]["lower_no_attention"]
    pt_c = comp["pooled_totals"]["lower_mlp_upper_uncovered_obs"]
    assert pt_b["generated_service_count"] == pt_c["generated_service_count"] == 9805180

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.68))
    gs = fig.add_gridspec(1, 3, left=0.076, right=0.988, bottom=0.190, top=0.840, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): (alpha, eta) phase plane with iso-DSR hyperbolas
    style_redesign_axis(ax_a, grid="none", box=True)
    add_panel_tag(ax_a, "(a)", y_offset_pt=14.0)
    a_grid = np.linspace(38.0, 68.0, 200)
    for iso in (22.0, 25.0, 28.0, 31.0):
        e_curve = (iso / a_grid) * 100.0
        mask = (e_curve >= 38.0) & (e_curve <= 61.0)
        ax_a.plot(a_grid[mask], e_curve[mask], color=P["border"], lw=0.75, ls=":", zorder=1)
    for s in sorted(comp["training_seeds"]):
        a0, e0 = np.mean(by_seed[("lower_no_attention", s)], axis=0)
        a1, e1 = np.mean(by_seed[("lower_mlp_upper_uncovered_obs", s)], axis=0)
        ax_a.plot([a0, a1], [e0, e1], color=P["muted"], lw=0.75, alpha=0.75, zorder=2)
        ax_a.scatter([a0], [e0], s=14, color=P["comparator"], alpha=0.8, zorder=3)
        ax_a.scatter([a1], [e1], s=14, color=P["hero"], alpha=0.8, zorder=3)
    ma0 = float(comp["metrics"]["Admission fraction"]["baseline_mean"]) * 100.0
    me0 = float(comp["metrics"]["Deadline satisfaction among admitted"]["baseline_mean"]) * 100.0
    ma1 = float(comp["metrics"]["Admission fraction"]["candidate_mean"]) * 100.0
    me1 = float(comp["metrics"]["Deadline satisfaction among admitted"]["candidate_mean"]) * 100.0
    ax_a.annotate(
        "",
        xy=(ma1, me1),
        xytext=(ma0, me0),
        arrowprops=dict(arrowstyle="-|>", color=P["ink"], lw=1.5, mutation_scale=9),
        zorder=5,
    )
    ax_a.plot([], [], color=P["border"], lw=0.95, ls=":", label="等 DSR 线")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=1, fontsize=7.2, frameon=False)
    ax_a.set_xlim(38.0, 68.0)
    ax_a.set_ylim(38.0, 61.5)
    ax_a.set_xticks([40, 50, 60])
    ax_a.set_yticks([40, 50, 60])
    ax_a.set_xlabel("自然接纳率 $\\alpha$ (%)")
    ax_a.set_ylabel("接纳后完成率 $\\eta$ (%)")

    # Panel (b): Symmetric decomposition waterfall (+4.87 - 2.02 = +2.85 pp)
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=14.0)
    ax_b.bar([0], [base_dsr], width=0.48, color=P["comparator"], edgecolor=P["ink"], lw=0.65, zorder=3)
    ax_b.bar([1], [adm_gain], bottom=[base_dsr], width=0.48, color=P["success"], edgecolor=P["ink"], lw=0.65, zorder=3)
    ax_b.bar([2], [abs(qual_loss)], bottom=[cand_dsr], width=0.48, color=P["amber"], edgecolor=P["ink"], lw=0.65, zorder=3)
    ax_b.bar([3], [cand_dsr], width=0.48, color=P["hero"], edgecolor=P["ink"], lw=0.65, zorder=3)
    ax_b.text(0, base_dsr + 0.8, f"{base_dsr:.1f}", fontsize=7.3, ha="center", va="bottom")
    ax_b.text(1, base_dsr + adm_gain + 0.8, f"+{adm_gain:.2f}", fontsize=7.3, color=P["success"], ha="center", va="bottom")
    ax_b.text(2, base_dsr + adm_gain + 3.8, f"{qual_loss:.2f}", fontsize=7.3, color=P["comparator"], ha="center", va="bottom")
    ax_b.text(3, cand_dsr + 0.8, f"{cand_dsr:.1f}", fontsize=7.3, fontweight="bold", color=P["hero"], ha="center", va="bottom")
    ax_b.set_xlim(-0.55, 3.55)
    ax_b.set_ylim(0.0, 36.5)
    ax_b.set_xticks([0, 1, 2, 3])
    ax_b.set_xticklabels(["基线", "+接纳", "-稀释", "扩展"], fontsize=7.3)
    ax_b.set_yticks([0, 10, 20, 30])
    ax_b.set_ylabel("满足率拆解 (%)")

    # Panel (c): 9.805M pooled request conservation stacked bar
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=14.0)
    for idx, pt in enumerate([pt_b, pt_c]):
        s_m = pt["deadline_satisfied_count"] / 1e6
        m_m = (pt["admitted_service_count"] - pt["deadline_satisfied_count"]) / 1e6
        u_m = pt["uncovered_service_count"] / 1e6
        assert abs((s_m + m_m + u_m) - 9.80518) < 1e-4
        ax_c.bar([idx], [s_m], width=0.48, color=P["success"], edgecolor=P["ink"], lw=0.6, label="完成 $S$" if idx == 0 else "")
        ax_c.bar([idx], [m_m], bottom=[s_m], width=0.48, color=P["amber"], edgecolor=P["ink"], lw=0.6, label="超时 $A-S$" if idx == 0 else "")
        ax_c.bar([idx], [u_m], bottom=[s_m + m_m], width=0.48, color=P["muted"], edgecolor=P["ink"], lw=0.6, label="未覆盖" if idx == 0 else "")
    ax_c.set_xlim(-0.55, 1.55)
    ax_c.set_ylim(0.0, 10.8)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["原观测", "扩展观测"])
    ax_c.set_yticks([0, 3, 6, 9.8])
    ax_c.set_ylabel("累计服务请求 ($\\times 10^6$)")
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=2, fontsize=7.2, frameon=False, handlelength=1.0, handletextpad=0.25, columnspacing=0.5)

    return export_redesign_figure(
        fig,
        "fig3_05_dsr_phase_waterfall",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig3_06_spatial_trajectory_replay() -> dict[str, object]:
    """图 3-6：中位确认种子固定配对回合的千步空间飞行轨迹与热点覆盖时序响应对比。"""
    rep_dir = DATA_DIR / "ch3_trajectory_replay"
    rep_sum = json.loads((rep_dir / "replay_summary.json").read_text(encoding="utf-8"))
    dens = np.loadtxt(rep_dir / "shared_ue_density.csv", delimiter=",")

    traj: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    with (rep_dir / "trajectory_positions.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            traj[(r["method"], int(r["uav_id"]))].append((int(r["state_index"]), float(r["x_m"]), float(r["y_m"])))

    steps: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with (rep_dir / "step_counts.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            steps[r["method"]].append((int(r["step"]), float(r["hotspot_generated"]), float(r["hotspot_admitted"])))

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.38))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.972, bottom=0.205, top=0.810, wspace=0.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    uav_cols = ["#005A9C", "#D55E00", "#009E73", "#6A51A3", "#C43C39"]

    for ax, tag, mkey, mtitle in [
        (ax_a, "(a)", "lower_no_attention", "原观测 $\\alpha=46.7\\%$"),
        (ax_b, "(b)", "lower_mlp_upper_uncovered_obs", "扩展观测 $\\alpha=61.3\\%$"),
    ]:
        style_redesign_axis(ax, grid="none", box=True)
        add_panel_tag(ax, tag, y_offset_pt=19.0)
        ax.imshow(
            dens,
            origin="lower",
            extent=(0, 700, 0, 700),
            cmap="Blues",
            alpha=0.45,
            aspect="auto",
            zorder=1,
        )
        for uid in range(1, 6):
            pts = sorted(traj[(mkey, uid)])
            xs = [p[1] for p in pts[::5]]
            ys = [p[2] for p in pts[::5]]
            ax.plot(xs, ys, color=uav_cols[uid - 1], lw=0.75, alpha=0.82, zorder=2.5)
            # Slot 500 coverage circle computed strictly inside [2, 698]^2
            _, cx, cy = pts[499]
            ang = np.linspace(0, 2 * np.pi, 90)
            rx = np.clip(cx + 100.0 * np.cos(ang), 2.0, 698.0)
            ry = np.clip(cy + 100.0 * np.sin(ang), 2.0, 698.0)
            ax.plot(rx, ry, color=uav_cols[uid - 1], lw=0.95, ls="--", zorder=3.5)
            ax.scatter([cx], [cy], s=14, color=uav_cols[uid - 1], marker="^", zorder=4)
        ax.set_xlim(0, 700)
        ax.set_ylim(0, 700)
        ax.set_xticks([0, 350, 700])
        ax.set_yticks([0, 350, 700])
        ax.set_xlabel("坐标 $x$ ($\\mathrm{m}$)")
        if ax is ax_a:
            ax.set_ylabel("坐标 $y$ ($\\mathrm{m}$)")
        ax.set_title(mtitle, fontsize=7.8, pad=3.5)

    # Panel (c): Rolling 50-slot hotspot admission rate over 1000 slots
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=19.0)
    for mkey, col, ls, lbl in [
        ("lower_no_attention", P["comparator"], "--", "原观测 ($49.0\\%$)"),
        ("lower_mlp_upper_uncovered_obs", P["hero"], "-", "扩展观测 ($65.5\\%$)"),
    ]:
        arr = sorted(steps[mkey])
        gen = np.array([x[1] for x in arr])
        adm = np.array([x[2] for x in arr])
        win = 60
        g_w = np.convolve(gen, np.ones(win), mode="valid")
        a_w = np.convolve(adm, np.ones(win), mode="valid")
        rate = np.divide(a_w, np.maximum(g_w, 1.0)) * 100.0
        tx = np.arange(win, win + len(rate))
        ax_c.plot(tx, rate, color=col, lw=1.30, ls=ls, label=lbl, zorder=3)
    assert abs(rep_sum["methods"]["lower_mlp_upper_uncovered_obs"]["hotspot_admission"] * 100.0 - 65.5091) < 0.02
    ax_c.set_xlim(60, 1000)
    ax_c.set_ylim(25.0, 85.0)
    ax_c.set_xticks([100, 400, 700, 1000])
    ax_c.set_yticks([30, 50, 70])
    ax_c.set_xlabel("运行宏时隙 $t$")
    ax_c.set_ylabel("热点窗口接纳率 (%)")
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.57, 1.005), ncol=1, fontsize=7.2, frameon=False, handlelength=1.1, labelspacing=0.12)

    return export_redesign_figure(
        fig,
        "fig3_06_spatial_trajectory_replay",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def main() -> None:
    results = [
        build_fig3_01_observation_aliasing(),
        build_fig3_02_hierarchical_mappo(),
        build_fig3_03_architecture_bottleneck(),
        build_fig3_04_confirmation_convergence(),
        build_fig3_05_dsr_phase_waterfall(),
        build_fig3_06_spatial_trajectory_replay(),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
