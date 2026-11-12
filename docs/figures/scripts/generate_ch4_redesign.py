#!/usr/bin/env python3
"""Generate Chapter 4 redesigned figures (Figs 4-1 through 4-6) from frozen canonical data.

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
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def build_fig4_01_batch_externality() -> dict[str, object]:
    """图 4-1：批量等份算力稀释越界曲线与单请求迁移三方截止期外部性反转构造。"""
    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.58))
    gs = fig.add_gridspec(1, 2, left=0.105, right=0.970, bottom=0.195, top=0.820, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel (a): Equal-share compute dilution T_cmp / D_m vs node concurrency N_j
    # Exact Chap 4 Sec 4.2.2 parameters: F_u = 60 GHz, C_m = 4e9 cycles, D_m = 0.25 s
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=18.0)
    n_vals = np.array([1, 2, 3, 4, 5])
    norm_cmp = (4.0e9 * n_vals / 60.0e9) / 0.25  # 0.2667 * N_j
    bar_cols = [P["hero_mid"] if v <= 1.0 else P["crimson"] for v in norm_cmp]
    bars_a = ax_a.bar(n_vals, norm_cmp, width=0.52, color=bar_cols, edgecolor=P["ink"], lw=0.65, zorder=3)
    ax_a.axhline(1.0, color=P["crimson"], lw=1.0, ls="--", zorder=2, label="截止期阈值 $T/D=1$")
    for n_v, val, bar in zip(n_vals, norm_cmp, bars_a):
        y_txt = val + 0.05 if abs(val + 0.05 - 1.0) > 0.12 else val + 0.22
        ax_a.text(bar.get_x() + bar.get_width() / 2.0, y_txt, f"{val:.2f}", fontsize=7.2, ha="center", va="bottom")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=1, fontsize=7.3, frameon=False)
    ax_a.set_xlim(0.35, 5.65)
    ax_a.set_ylim(0.0, 1.58)
    ax_a.set_xticks(n_vals)
    ax_a.set_xlabel("同节点最终并发任务数 $N_j(\\mathbf{a})$")
    ax_a.set_ylabel("归一化计算时延 $T_{\\mathrm{cmp}}/D_m$")

    # Panel (b): Three-party migration externality reversal (Table 4-1 exact numbers)
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=18.0)
    x = np.arange(3)
    w = 0.30
    before = np.array([1.30 / 1.25, 1.30 / 1.70, 0.90 / 1.00])
    after = np.array([1.22 / 1.25, 1.10 / 1.70, 1.10 / 1.00])
    ax_b.bar(x - w / 2, before, width=w, color=P["muted"], edgecolor=P["ink"], lw=0.65, label="迁移前 ($F_1=1$)", zorder=3)
    ax_b.bar(
        x + w / 2,
        after,
        width=w,
        color=[P["success"], P["hero_mid"], P["crimson"]],
        edgecolor=P["ink"],
        lw=0.65,
        label="自利迁移后 ($F_1=3$)",
        zorder=3,
    )
    ax_b.axhline(1.0, color=P["crimson"], lw=1.0, ls="--", zorder=2)
    ax_b.text(0.0, 1.15, "$1.04\\to 0.98$", fontsize=7.2, color=P["success"], ha="center", va="bottom")
    ax_b.text(1.0, 0.80, "$0.76\\to 0.65$", fontsize=7.2, color=P["ink"], ha="center", va="bottom")
    ax_b.text(2.0, 1.16, "$0.90\\to 1.10$ (×3)", fontsize=7.2, fontweight="bold", color=P["crimson"], ha="center", va="bottom")
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=2, fontsize=7.2, frameon=False, columnspacing=0.7, handlelength=1.1)
    ax_b.set_xlim(-0.55, 2.62)
    ax_b.set_ylim(0.0, 1.42)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["待迁请求\n(1 项)", "源余留\n(4 项)", "目标既有\n(3 项)"], fontsize=7.3)
    ax_b.set_ylabel("组内归一化总时延 $T_r/D_r$")

    return export_redesign_figure(
        fig,
        "fig4_01_batch_externality",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b],
    )


def build_fig4_02_mc_itr_mechanism_trace() -> dict[str, object]:
    """图 4-2：MC-ITR 克隆全批次邻域搜索流程与真实冻结快照二元字典序轨迹对比。"""
    trace_data = json.loads((DATA_DIR / "mc_itr_trace_example.json").read_text(encoding="utf-8"))
    snap_a = trace_data["snapshots"]["lexicographic_tradeoff_s126_t200"]
    snap_b = trace_data["snapshots"]["self_interested_worsening_s168_t100"]

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.68))
    gs = fig.add_gridspec(1, 3, left=0.092, right=0.968, bottom=0.190, top=0.805, wspace=0.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): Compact MC-ITR cloned batch neighborhood search loop schematic inside ax_a
    ax_a.set_xlim(0.0, 100.0)
    ax_a.set_ylim(0.0, 100.0)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    add_panel_tag(ax_a, "(a)", x_offset_pt=-14.0, y_offset_pt=21.0)

    def _node(y_lo: float, h: float, title: str, fill: str, stroke: str) -> None:
        patch = FancyBboxPatch(
            (4.0, y_lo),
            92.0,
            h,
            boxstyle="round,pad=0.0,rounding_size=3.0",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=0.85,
            zorder=2,
        )
        ax_a.add_patch(patch)
        ax_a.text(50.0, y_lo + h / 2.0, title, fontsize=7.2, color=P["ink"], ha="center", va="center", zorder=4)

    _node(80.0, 17.0, "共享初值 $\\mathbf{a}^0$ (贪心初始化)", P["panel_bg"], P["ink"])
    _node(54.0, 19.0, "克隆快照遍历 $\\mathcal{N}_1(\\mathbf{a})$\n全批评估 $F=(F_1,F_2)$", P["hero_light"], P["hero"])
    _node(28.0, 19.0, "选最小 $\\mathbf{v}'<_{\\mathrm{lex}}\\mathbf{v}$\n独立复核后更新 $\\mathbf{a}$", P["dist_light"], P["distilled"])
    _node(2.0, 18.0, "无改善退出 $\\Rightarrow$ 邻域最优", "#EEF8F2", P["success"])

    for y1, y2 in [(79.5, 73.5), (53.5, 47.5), (27.5, 20.5)]:
        ax_a.annotate(
            "",
            xy=(50.0, y2),
            xytext=(50.0, y1),
            arrowprops=dict(arrowstyle="-|>", color=P["hero"], lw=0.95, mutation_scale=7.5),
            zorder=5,
        )

    # Panel (b): Snapshot A (s126, t200, K=9) 2D lexicographic trajectory (F_2, F_1)
    style_redesign_axis(ax_b, grid="none", box=True)
    add_panel_tag(ax_b, "(b)", y_offset_pt=21.0)
    tr_mc_a = np.array(snap_a["methods"]["mc_itr"]["trace"])  # cols: F1, F2
    tr_ib_a = np.array(snap_a["methods"]["iterative_base"]["trace"])
    ax_b.plot(tr_mc_a[:, 1], tr_mc_a[:, 0], color=P["hero"], lw=1.45, marker="o", ms=4.5, label="MC-ITR", zorder=4)
    ax_b.scatter(tr_ib_a[:, 1], tr_ib_a[:, 0], color=P["comparator"], marker="s", s=34, label="Iter-Base (停滞)", zorder=5)
    ax_b.text(tr_mc_a[0, 1] + 0.014, tr_mc_a[0, 0] + 0.08, "$r=0$", fontsize=7.2, color=P["ink"], ha="left", va="bottom")
    ax_b.text(tr_mc_a[1, 1] + 0.016, tr_mc_a[1, 0] + 0.14, "$r=1$", fontsize=7.2, color=P["hero"], ha="left", va="bottom")
    ax_b.text(tr_mc_a[2, 1] - 0.004, tr_mc_a[2, 0] - 0.25, "$r=2$ $(3,1.084)$", fontsize=7.2, fontweight="bold", color=P["hero"], ha="right", va="top")
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.58, 1.005), ncol=1, fontsize=7.2, frameon=False, handlelength=1.1, labelspacing=0.10)
    ax_b.set_xlim(0.925, 1.115)
    ax_b.set_ylim(2.15, 6.75)
    ax_b.set_xticks([0.95, 1.02, 1.09])
    ax_b.set_yticks([3, 4, 5, 6])
    ax_b.set_xlabel("快照 A 归一化时延 $F_2$")
    ax_b.set_ylabel("超时请求数 $F_1\\downarrow$")

    # Panel (c): Snapshot B (s168, t100, K=21) self-interested timeout worsening vs MC-ITR descent
    style_redesign_axis(ax_c, grid="none", box=True)
    add_panel_tag(ax_c, "(c)", y_offset_pt=21.0)
    tr_mc_b = np.array(snap_b["methods"]["mc_itr"]["trace"])
    tr_ib_b = np.array(snap_b["methods"]["iterative_base"]["trace"])
    ax_c.plot(tr_ib_b[:, 1], tr_ib_b[:, 0], color=P["comparator"], lw=1.30, ls="--", marker="s", ms=3.8, label="Iter-Base ($10\\to 11$)", zorder=3)
    ax_c.plot(tr_mc_b[:, 1], tr_mc_b[:, 0], color=P["hero"], lw=1.45, ls="-", marker="o", ms=4.2, label="MC-ITR ($10\\to 9$)", zorder=4)
    ax_c.text(1.133, 10.18, "$r=0$", fontsize=7.2, color=P["ink"], ha="right", va="bottom")
    ax_c.text(1.102, 11.20, "自利恶化 $F_1=11$", fontsize=7.2, fontweight="bold", color=P["comparator"], ha="left", va="bottom")
    ax_c.text(1.102, 8.68, "平台降 $F_2$ $(9,1.106)$", fontsize=7.2, fontweight="bold", color=P["hero"], ha="left", va="top")
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.58, 1.005), ncol=1, fontsize=7.2, frameon=False, handlelength=1.1, labelspacing=0.10)
    ax_c.set_xlim(1.098, 1.140)
    ax_c.set_ylim(8.3, 11.65)
    ax_c.set_xticks([1.10, 1.12, 1.14])
    ax_c.set_yticks([9, 10, 11])
    ax_c.set_xlabel("快照 B 归一化时延 $F_2$")
    ax_c.set_ylabel("超时请求数 $F_1\\downarrow$")

    return export_redesign_figure(
        fig,
        "fig4_02_mc_itr_mechanism_trace",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig4_03_main_lexicographic_gains() -> dict[str, object]:
    """图 4-3：主配置四种卸载算法的超时数与时延表现、配对超时差值及搜索耗时构成。"""
    with (DATA_DIR / "chapter4_units.csv").open(encoding="utf-8-sig") as f:
        u_rows = [r for r in csv.DictReader(f) if r["variant"] == "main"]
    c4 = json.loads((DATA_DIR / "chapter4_analysis.json").read_text(encoding="utf-8"))

    methods = ["heuristic_control", "greedy_single_pass", "iterative_base", "mc_itr"]
    m_labels = ["Heuristic", "Greedy", "Iter-Base", "MC-ITR"]
    m_cols = [P["muted"], P["ablation"], P["comparator"], P["hero"]]

    by_m_f1: dict[str, list[float]] = defaultdict(list)
    by_m_f2: dict[str, list[float]] = defaultdict(list)
    by_m_t: dict[str, list[float]] = defaultdict(list)
    by_seed_f1: dict[int, dict[str, float]] = defaultdict(dict)
    for r in u_rows:
        m = r["method"]
        s = int(r["workload_seed"])
        by_m_f1[m].append(float(r["deadline_misses"]))
        by_m_f2[m].append(float(r["mean_normalized_latency"]))
        by_m_t[m].append(float(r["solver_wall_seconds"]))
        by_seed_f1[s][m] = float(r["deadline_misses"])

    f1_means = [float(np.mean(by_m_f1[m])) for m in methods]
    f2_means = [float(np.mean(by_m_f2[m])) for m in methods]
    t_means = [float(np.mean(by_m_t[m])) for m in methods]
    assert abs(f1_means[3] - 2.045) < 0.002 and abs(f1_means[2] - 2.215) < 0.002

    # Exact bootstrap CIs for MC-ITR minus each baseline from chapter4_analysis.json
    comp_map = {
        c["reference"]: (c["mean_difference"][0], c["ci95"][0][0], c["ci95"][1][0])
        for c in c4["comparisons"]
        if c["variant"] == "main"
    }

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.66))
    gs = fig.add_gridspec(1, 3, left=0.092, right=0.970, bottom=0.195, top=0.835, wspace=0.43)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): Main setting F1 bars + 10 seed dots + F2 labels
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=14.0)
    x = np.arange(4)
    ax_a.bar(x, f1_means, width=0.54, color=m_cols, edgecolor=P["ink"], lw=0.65, alpha=0.88, zorder=2)
    rng = np.random.default_rng(20261112)
    for idx, m in enumerate(methods):
        vals = np.array(by_m_f1[m])
        jit = rng.uniform(-0.11, 0.11, size=len(vals))
        ax_a.scatter(np.full_like(vals, idx) + jit, vals, s=10, color=P["ink"], alpha=0.55, zorder=3)
        ax_a.text(idx, 4.82, f"{f1_means[idx]:.2f}", fontsize=7.2, fontweight="bold" if idx == 3 else "normal", color=m_cols[idx] if idx == 3 else P["ink"], ha="center", va="bottom")
    ax_a.set_xlim(-0.55, 3.55)
    ax_a.set_ylim(0.0, 5.45)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["启发", "贪心", "自利", "MC-ITR"], fontsize=7.2)
    ax_a.set_yticks([0, 1, 2, 3, 4, 5])
    ax_a.set_ylabel("每快照超时数 $F_1\\downarrow$")

    # Panel (b): Paired Delta F1 (MC-ITR minus each baseline) + 10-seed dots + 95% Bootstrap CI
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=14.0)
    refs = [
        ("启发式", "heuristic_control", P["muted"]),
        ("单贪心", "greedy_single_pass", P["ablation"]),
        ("自利法", "iterative_base", P["hero"]),
    ]
    ax_b.axvline(0.0, color=P["crimson"], lw=0.9, ls="--", zorder=1)
    y_pos = np.array([2, 1, 0])
    for y, (_, rkey, col) in zip(y_pos, refs):
        est, lo, hi = comp_map[rkey]
        seed_diffs = [by_seed_f1[s]["mc_itr"] - by_seed_f1[s][rkey] for s in sorted(by_seed_f1)]
        ax_b.scatter(seed_diffs, np.full(len(seed_diffs), y + 0.18), s=9, color=col, alpha=0.55, zorder=2)
        ax_b.plot([lo, hi], [y - 0.08, y - 0.08], color=col, lw=1.7, zorder=4)
        ax_b.scatter([est], [y - 0.08], s=26, color=col, zorder=5)
        ax_b.text(est - 0.04, y - 0.24, f"{est:.3f}", fontsize=7.2, fontweight="bold" if rkey == "iterative_base" else "normal", color=col if rkey == "iterative_base" else P["ink"], ha="center", va="top")
    ax_b.set_xlim(-1.42, 0.16)
    ax_b.set_ylim(-0.85, 2.55)
    ax_b.set_xticks([-1.2, -0.6, 0.0])
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([r[0] for r in refs], fontsize=7.3)
    ax_b.set_xlabel("MC-ITR 配对超时差 $\\Delta F_1$")

    # Panel (c): Stacked wall-time breakdown (shared init vs search kernel)
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=14.0)
    t_h, t_g, t_ib, t_mc = t_means
    init_shared = np.array([0.0, t_h, t_h + t_g, t_h + t_g])
    kernel_own = np.array([t_h, t_g, t_ib, t_mc])
    ax_c.bar(x, init_shared, width=0.52, color=P["border"], edgecolor=P["ink"], lw=0.6, label="共享初值", zorder=2)
    ax_c.bar(x, kernel_own, bottom=init_shared, width=0.52, color=m_cols, edgecolor=P["ink"], lw=0.6, label="自身搜索核", zorder=3)
    totals = init_shared + kernel_own
    for idx, tot in enumerate(totals):
        ax_c.text(idx, tot + 0.14, f"{tot:.2f}", fontsize=7.2, ha="center", va="bottom")
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.52, 1.01), ncol=1, fontsize=7.2, frameon=False, labelspacing=0.12)
    ax_c.set_xlim(-0.55, 3.55)
    ax_c.set_ylim(0.0, 4.75)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["启发", "贪心", "自利", "MC-ITR"], fontsize=7.2)
    ax_c.set_yticks([0, 1, 2, 3, 4])
    ax_c.set_ylabel("单快照墙钟耗时 ($\\mathrm{s}$)")

    return export_redesign_figure(
        fig,
        "fig4_03_main_lexicographic_gains",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig4_04_scale_compute_sensitivity() -> dict[str, object]:
    """图 4-4：终端规模与机载算力倍率双维竞争强度下四种方法超时表现及配对差值森林图。"""
    with (DATA_DIR / "chapter4_units.csv").open(encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    c4 = json.loads((DATA_DIR / "chapter4_analysis.json").read_text(encoding="utf-8"))

    vals_map: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in all_rows:
        vals_map[(r["variant"], r["method"])].append(float(r["deadline_misses"]))

    comp_iter = {
        c["variant"]: (c["mean_difference"][0], c["ci95"][0][0], c["ci95"][1][0])
        for c in c4["comparisons"]
        if c["reference"] == "iterative_base"
    }

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.68))
    gs = fig.add_gridspec(1, 3, left=0.092, right=0.970, bottom=0.195, top=0.815, wspace=0.43)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    m_specs = [
        ("heuristic_control", "启发式", P["muted"], ":"),
        ("greedy_single_pass", "单贪心", P["ablation"], "-."),
        ("iterative_base", "自利法", P["comparator"], "--"),
        ("mc_itr", "MC-ITR", P["hero"], "-"),
    ]

    # Panel (a): UE count sensitivity (60..140)
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=20.0)
    ue_vars = [("ue_60", 60), ("ue_80", 80), ("main", 100), ("ue_120", 120), ("ue_140", 140)]
    xs_ue = [u[1] for u in ue_vars]
    for mkey, mlbl, col, ls in m_specs:
        ys = [float(np.mean(vals_map[(v, mkey)])) for v, _ in ue_vars]
        ax_a.plot(xs_ue, ys, color=col, lw=1.35 if mkey == "mc_itr" else 1.1, ls=ls, marker="o", ms=3.5, label=mlbl, zorder=4 if mkey == "mc_itr" else 3)
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.14, 1.01), ncol=4, fontsize=7.2, frameon=False, columnspacing=0.9, handlelength=1.2, handletextpad=0.3)
    ax_a.set_xlim(54, 146)
    ax_a.set_ylim(0.5, 5.2)
    ax_a.set_xticks([60, 80, 100, 120, 140])
    ax_a.set_yticks([1, 2, 3, 4, 5])
    ax_a.set_xlabel("地面终端规模 $M$")
    ax_a.set_ylabel("每快照平均超时数 $F_1\\downarrow$")

    # Panel (b): UAV compute scale sensitivity (0.6..1.4x)
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=20.0)
    cp_vars = [("compute_0.6", 0.6), ("compute_0.8", 0.8), ("main", 1.0), ("compute_1.2", 1.2), ("compute_1.4", 1.4)]
    xs_cp = [c[1] for c in cp_vars]
    for mkey, mlbl, col, ls in m_specs:
        ys = [float(np.mean(vals_map[(v, mkey)])) for v, _ in cp_vars]
        ax_b.plot(xs_cp, ys, color=col, lw=1.35 if mkey == "mc_itr" else 1.1, ls=ls, marker="o", ms=3.5, label=mlbl, zorder=4 if mkey == "mc_itr" else 3)
    ax_b.set_xlim(0.54, 1.46)
    ax_b.set_ylim(1.2, 4.0)
    ax_b.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax_b.set_yticks([1.5, 2.5, 3.5])
    ax_b.set_xlabel("无人机算力倍率 $\\kappa_F$")
    ax_b.set_ylabel("每快照平均超时数 $F_1\\downarrow$")

    # Panel (c): Forest plot of MC-ITR minus Iterative-Base across all 9 variants
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=20.0)
    order_vars = [
        ("ue_60", "M=60", P["hero_mid"]),
        ("ue_80", "M=80", P["hero_mid"]),
        ("ue_120", "M=120", P["hero_mid"]),
        ("ue_140", "M=140", P["hero_mid"]),
        ("main", "主配置", P["hero"]),
        ("compute_1.4", "1.4×", P["distilled"]),
        ("compute_1.2", "1.2×", P["distilled"]),
        ("compute_0.8", "0.8×", P["distilled"]),
        ("compute_0.6", "0.6×", P["distilled"]),
    ]
    ax_c.axvline(0.0, color=P["crimson"], lw=0.9, ls="--", zorder=1)
    y_idx = np.arange(len(order_vars))[::-1]
    for y, (vkey, _, col) in zip(y_idx, order_vars):
        est, lo, hi = comp_iter[vkey]
        ax_c.plot([lo, hi], [y, y], color=col, lw=1.35, zorder=3)
        ax_c.scatter([est], [y], s=18 if vkey != "main" else 28, color=col, zorder=4)
    ax_c.set_xlim(-0.34, 0.04)
    ax_c.set_ylim(-0.6, 8.6)
    ax_c.set_xticks([-0.3, -0.15, 0.0])
    ax_c.set_yticks(y_idx)
    ax_c.set_yticklabels([x[1] for x in order_vars], fontsize=7.2)
    ax_c.set_xlabel("较自利迭代 $\\Delta F_1$ ($95\\%$ CI)")

    return export_redesign_figure(
        fig,
        "fig4_04_scale_compute_sensitivity",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig4_05_exact_competitive_gap() -> dict[str, object]:
    """图 4-5：主配置 354 个可穷举多请求竞争快照的请求构成、平均超时与全局最优命中率。"""
    ex = json.loads((DATA_DIR / "exact_competitive_verification.json").read_text(encoding="utf-8"))
    assert ex["competitive_tractable_snapshots"] == 354
    assert ex["same_source_uav_snapshots"] == 321

    # Stratify K=2..10 by same-source UAV contention vs cross-UAV only
    k_same: dict[int, int] = defaultdict(int)
    k_other: dict[int, int] = defaultdict(int)
    for rec in ex["records"]:
        k = int(rec["num_requests"])
        if rec["has_same_source_uav"]:
            k_same[k] += 1
        else:
            k_other[k] += 1
    ks = list(range(2, 11))
    same_arr = np.array([k_same[k] for k in ks])
    other_arr = np.array([k_other[k] for k in ks])
    assert int(same_arr.sum() + other_arr.sum()) == 354

    ps = ex["pooled_summary"]
    m_keys = ["heuristic_control", "greedy_single_pass", "iterative_base", "mc_itr", "exact_small"]
    f1_vals = [float(ps[k]["mean_deadline_misses"]) for k in m_keys]

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.72))
    gs = fig.add_gridspec(1, 3, left=0.092, right=0.970, bottom=0.195, top=0.790, wspace=0.39)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): K in [2, 10] distribution across 354 competitive snapshots
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=24.0)
    ax_a.bar(ks, same_arr, width=0.64, color=P["hero"], edgecolor=P["ink"], lw=0.55, label="同源竞争 (90.7%)", zorder=3)
    ax_a.bar(ks, other_arr, bottom=same_arr, width=0.64, color=P["border"], edgecolor=P["ink"], lw=0.55, label="异源并发 (9.3%)", zorder=3)
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.60, 1.01), ncol=1, fontsize=7.2, frameon=False, labelspacing=0.12, handlelength=1.1)
    ax_a.set_xlim(1.3, 10.7)
    ax_a.set_ylim(0, 68)
    ax_a.set_xticks([2, 4, 6, 8, 10])
    ax_a.set_yticks([0, 20, 40, 60])
    ax_a.set_xlabel("活跃服务请求数 $K$")
    ax_a.set_ylabel("可穷举竞争快照数")

    # Panel (b): 5-method mean timeout count F_1 vs Exact-Global (0.963 vs 0.890)
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=24.0)
    cols_b = [P["muted"], P["ablation"], P["comparator"], P["hero"], P["success"]]
    xb = np.arange(5)
    ax_b.bar(xb, f1_vals, width=0.48, color=cols_b, edgecolor=P["ink"], lw=0.6, zorder=3)
    ax_b.text(0.0, f1_vals[0] + 0.05, f"{f1_vals[0]:.2f}", fontsize=7.2, ha="center", va="bottom")
    ax_b.text(2.0, f1_vals[2] + 0.06, f"{f1_vals[2]:.2f}", fontsize=7.2, color=P["comparator"], ha="center", va="bottom")
    ax_b.text(3.0, f1_vals[3] + 0.05, f"{f1_vals[3]:.2f}", fontsize=7.2, fontweight="bold", color=P["hero"], ha="center", va="bottom")
    ax_b.plot([4.0, 4.0], [f1_vals[4] + 0.02, 1.33], color=P["success"], lw=0.75, zorder=3)
    ax_b.text(4.0, 1.35, f"{f1_vals[4]:.2f}", fontsize=7.2, fontweight="bold", color=P["success"], ha="center", va="bottom")
    ax_b.text(2.45, 1.80, "消除 67.1% 间隙", fontsize=7.2, fontweight="bold", color=P["hero"], ha="center", va="top")
    ax_b.set_xlim(-0.58, 4.62)
    ax_b.set_ylim(0.0, 1.88)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["启发", "贪心", "自利", "MC", "全局"], fontsize=7.2)
    ax_b.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax_b.set_ylabel("每快照平均超时数 $F_1\\downarrow$")

    # Panel (c): Global F_1 minimum hit rate vs exact lexicographic (F_1, F_2) hit rate
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=24.0)
    sub_keys = m_keys[:4]
    f1_hit = [float(ps[k]["exact_f1_hit_rate"]) * 100.0 for k in sub_keys]
    lex_hit = [float(ps[k]["exact_lex_hit_rate"]) * 100.0 for k in sub_keys]
    xc = np.arange(4)
    ax_c.bar(xc - 0.18, f1_hit, width=0.28, color=P["hero"], edgecolor=P["ink"], lw=0.55, label="$F_1$ 最优命中", zorder=3)
    ax_c.bar(xc + 0.18, lex_hit, width=0.28, color=P["distilled"], edgecolor=P["ink"], lw=0.55, label="字典序全命中", zorder=3)
    ax_c.text(3 - 0.18, f1_hit[3] + 2.5, "92.9%", fontsize=7.2, fontweight="bold", color=P["hero"], ha="center", va="bottom")
    ax_c.text(3.06, lex_hit[3] + 2.5, "58.5%", fontsize=7.2, color=P["distilled"], ha="left", va="bottom")
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.60, 1.01), ncol=1, fontsize=7.2, frameon=False, labelspacing=0.12, handlelength=1.1)
    ax_c.set_xlim(-0.55, 4.18)
    ax_c.set_ylim(0.0, 110.0)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(["启发", "贪心", "自利", "MC-ITR"], fontsize=7.2)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.set_ylabel("全局最优命中率 (%)")

    return export_redesign_figure(
        fig,
        "fig4_05_exact_competitive_gap",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig4_06_bridge_attribution() -> dict[str, object]:
    """图 4-6：第三章扩展观测轨迹 400 个冻结快照的五方法桥接对比、两阶归因瀑布与耗时折中。"""
    br = json.loads((DATA_DIR / "bridge_analysis.json").read_text(encoding="utf-8"))
    assert br["snapshots"] == 400 and br["evaluation_jobs"] == 50

    m_keys = ["online_lower", "heuristic_control", "greedy_single_pass", "iterative_base", "mc_itr"]
    seed_f1: dict[tuple[str, int], list[float]] = defaultdict(list)
    seed_t: dict[str, list[float]] = defaultdict(list)
    for u in br["units"]:
        s = int(u["training_seed"])
        for mk in m_keys:
            seed_f1[(mk, s)].append(float(u[mk]["deadline_misses"]))
            if mk == "mc_itr":
                seed_t[mk].append(float(u[mk]["full_path_wall_seconds"]))
            else:
                seed_t[mk].append(float(u[mk]["wall_seconds"]))
    t_seeds = sorted(br["training_seeds"])
    means_f1 = {mk: float(np.mean([np.mean(seed_f1[(mk, s)]) for s in t_seeds])) for mk in m_keys}
    means_t = {mk: float(np.mean(seed_t[mk])) for mk in m_keys}
    # Add shared init to Iterative-Base for full-path runtime consistency
    init_t = means_t["heuristic_control"] + means_t["greedy_single_pass"]
    means_t["iterative_base_full"] = means_t["iterative_base"] + init_t
    assert abs(means_f1["online_lower"] - 9.2375) < 0.005
    assert abs(means_f1["iterative_base"] - 7.4775) < 0.005
    assert abs(means_f1["mc_itr"] - 6.6050) < 0.005

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.74))
    gs = fig.add_gridspec(1, 3, left=0.092, right=0.970, bottom=0.220, top=0.835, wspace=0.39)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): 5-method timeout ladder + 10 training-seed scatter
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=14.0)
    cols_a = [P["ablation"], P["crimson"], P["amber"], P["comparator"], P["hero"]]
    xa = np.arange(5)
    f1_arr = [means_f1[mk] for mk in m_keys]
    ax_a.bar(xa, f1_arr, width=0.46, color=cols_a, edgecolor=P["ink"], lw=0.6, alpha=0.88, zorder=2)
    rng = np.random.default_rng(20261113)
    top_seed: dict[int, float] = {}
    for idx, mk in enumerate(m_keys):
        s_vals = np.array([np.mean(seed_f1[(mk, s)]) for s in t_seeds])
        top_seed[idx] = max(f1_arr[idx], float(np.max(s_vals)))
        ax_a.scatter(np.full_like(s_vals, idx) + rng.uniform(-0.08, 0.08, len(s_vals)), s_vals, s=8, color=P["ink"], alpha=0.5, zorder=3)
    ax_a.text(0, top_seed[0] + 0.55, "9.24", fontsize=7.2, ha="center", va="bottom")
    ax_a.text(1, top_seed[1] + 0.55, "11.98", fontsize=7.2, color=P["crimson"], ha="center", va="bottom")
    ax_a.text(4, top_seed[4] + 0.55, "6.61", fontsize=7.2, fontweight="bold", color=P["hero"], ha="center", va="bottom")
    ax_a.set_xlim(-0.58, 4.58)
    ax_a.set_ylim(0.0, max(top_seed.values()) + 2.5)
    ax_a.set_xticks(xa)
    ax_a.set_xticklabels(["在线", "启发", "贪心", "自利", "MC"], fontsize=7.2)
    ax_a.set_yticks([0, 5, 10, 15])
    ax_a.set_ylabel("桥接快照超时数 $F_1\\downarrow$")

    # Panel (b): Exact 2-step attribution waterfall (-1.760 [66.9%] and -0.873 [33.1%])
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=14.0)
    v_on = means_f1["online_lower"]
    v_ib = means_f1["iterative_base"]
    v_mc = means_f1["mc_itr"]
    d_step1 = v_on - v_ib  # 1.760
    d_step2 = v_ib - v_mc  # 0.8725
    ax_b.bar([0], [v_on], width=0.46, color=P["ablation"], edgecolor=P["ink"], lw=0.6, zorder=3)
    ax_b.bar([1], [d_step1], bottom=[v_ib], width=0.46, color=P["comparator"], edgecolor=P["ink"], lw=0.6, zorder=3)
    ax_b.bar([2], [d_step2], bottom=[v_mc], width=0.46, color=P["success"], edgecolor=P["ink"], lw=0.6, zorder=3)
    ax_b.bar([3], [v_mc], width=0.46, color=P["hero"], edgecolor=P["ink"], lw=0.6, zorder=3)
    ax_b.text(0, v_on + 0.30, "9.24", fontsize=7.2, ha="center", va="bottom")
    ax_b.text(1, v_on + 0.95, "-1.76\n(66.9%)", fontsize=7.2, color=P["comparator"], ha="center", va="bottom")
    ax_b.text(2, v_ib + 0.35, "-0.87\n(33.1%)", fontsize=7.2, fontweight="bold", color=P["success"], ha="center", va="bottom")
    ax_b.text(3, v_mc + 0.30, "6.61", fontsize=7.2, fontweight="bold", color=P["hero"], ha="center", va="bottom")
    ax_b.set_xlim(-0.55, 3.55)
    ax_b.set_ylim(0.0, 13.5)
    ax_b.set_xticks([0, 1, 2, 3])
    ax_b.set_xticklabels(["在线", "放宽\n自利", "协同\n净增", "MC"], fontsize=7.2)
    ax_b.set_yticks([0, 4, 8, 12])
    ax_b.set_ylabel("超时数两阶拆解 $F_1$")

    # Panel (c): Log10 decision wall-time vs timeout count F_1 (0.0056 s vs 21.88 s)
    style_redesign_axis(ax_c, grid="none", box=True)
    add_panel_tag(ax_c, "(c)", y_offset_pt=14.0)
    pts = [
        ("在线下层", means_t["online_lower"], means_f1["online_lower"], P["ablation"], (6.0, -4.0), "left"),
        ("启发式", means_t["heuristic_control"], means_f1["heuristic_control"], P["crimson"], (6.0, -1.0), "left"),
        ("单遍贪心", means_t["greedy_single_pass"], means_f1["greedy_single_pass"], P["amber"], (-8.0, -1.0), "right"),
        ("自利全径", means_t["iterative_base_full"], means_f1["iterative_base"], P["comparator"], (2.0, 11.0), "right"),
        ("MC-ITR", means_t["mc_itr"], means_f1["mc_itr"], P["hero"], (-8.0, -4.0), "right"),
    ]
    for lbl, tx, fy, col, off, ha in pts:
        ax_c.scatter([tx], [fy], s=26, color=col, edgecolor=P["ink"], linewidth=0.5, zorder=4)
        ax_c.annotate(lbl, xy=(tx, fy), xytext=off, textcoords="offset points", fontsize=7.2, color=col if col == P["hero"] else P["ink"], fontweight="bold" if col == P["hero"] else "normal", ha=ha, va="center", zorder=5)
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.0018, 65.0)
    ax_c.set_ylim(5.1, 13.4)
    ax_c.set_xticks([0.01, 1.0, 20.0])
    ax_c.set_xticklabels(["0.01", "1", "20"])
    ax_c.set_yticks([6, 8, 10, 12])
    ax_c.set_xlabel("单快照全路径耗时 ($\\mathrm{s}$, 对数)")
    ax_c.set_ylabel("平均超时数 $F_1\\downarrow$")

    return export_redesign_figure(
        fig,
        "fig4_06_bridge_attribution",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def main() -> None:
    results = [
        build_fig4_01_batch_externality(),
        build_fig4_02_mc_itr_mechanism_trace(),
        build_fig4_03_main_lexicographic_gains(),
        build_fig4_04_scale_compute_sensitivity(),
        build_fig4_05_exact_competitive_gap(),
        build_fig4_06_bridge_attribution(),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
