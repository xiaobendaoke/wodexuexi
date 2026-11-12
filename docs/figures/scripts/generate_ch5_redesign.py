"""第五章（跨时隙持续 EDF 排队、等变双分支蒸馏与 CRN 反事实微调）全新重设计图体系生成脚本（7 幅）。

严格基于冻结盲测确认数据源（只读，不启动任何本地训练或重评测）：
- docs/figures/data/queued_joint_rl_20261106/confirmation_summary.csv
- docs/figures/data/queued_joint_rl_20261106/confirmation_units.csv
- docs/figures/data/queued_joint_rl_20261106/paired_dsr_effects.csv
- docs/figures/data/queued_joint_rl_20261106/validation_curves.csv
- docs/figures/data/queued_joint_rl_20261106/cf0_ablation_summary.csv
- docs/figures/data/queued_joint_rl_20261106/arrival_cohorts.csv
- docs/figures/data/queued_joint_rl_20261106/stage_decision_cost.csv
- docs/figures/data/queued_joint_rl_20261106/peak_queue_evolution_955000.json
- results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/tandem_approximation_audit.json
- results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/decomposed_independence_audit.json

输出清单（1:1 物理线宽 146.6 mm = 415.56 pt）：
1. fig5_01_burst_tandem_edf
2. fig5_02_equivariant_dual_branch
3. fig5_03_distill_crn_mpc_framework
4. fig5_04_main_confirmation_effects
5. fig5_05_training_counterfactual_coupling
6. fig5_06_cohort_peak_queue_dynamics
7. fig5_07_decision_cost_pareto
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paths import REPO_ROOT as CANONICAL_ROOT  # noqa: E402
from svg_schematic_builder import SchematicCanvas  # noqa: E402
from thesis_figure_style import (  # noqa: E402
    REDESIGN_PALETTE as P,
    TEXTWIDTH_IN,
    add_panel_tag,
    export_redesign_figure,
    setup_redesign_style,
    style_redesign_axis,
)

assert REPO_ROOT == CANONICAL_ROOT

DATA_DIR = REPO_ROOT / "docs" / "figures" / "data" / "queued_joint_rl_20261106"
AUDIT_DIR = REPO_ROOT / "results" / "chapter5_dynamic_service_20261028" / "queued_joint_rl_v2_20261106"
OUT_DIRS = [
    REPO_ROOT / "docs" / "figures",
    REPO_ROOT / "output" / "thesis_figure_redesign_20261112" / "deliverables",
]
QA_DIR = REPO_ROOT / "output" / "thesis_figure_redesign_20261112" / "validation"

SHORT_LABELS = {
    "uncoordinated_init": "初始先验选择器",
    "lower_only_rl": "仅下层 RL",
    "upper_only_rl": "仅上层 RL",
    "decomposed_composed_rl": "单层训练后拼接",
    "decomposed_rl": "去耦冷启动 RL",
    "joint_no_expectation_rl": "无期望冷启动 RL",
    "joint_rl": "联合冷启动 RL",
    "decomposed_distilled": "去耦规划蒸馏",
    "joint_no_expectation_distilled": "无期望规划蒸馏",
    "joint_distilled": "联合规划蒸馏",
    "decomposed_rl_finetuned": "去耦反事实微调",
    "joint_no_expectation_rl_finetuned": "无期望反事实微调",
    "joint_rl_finetuned": "联合反事实微调",
    "quality_amortized": "在线规划基线",
    "prior_top6_mpc": "先验 Top-6 MPC",
    "distilled_top6_mpc": "蒸馏 Top-6 MPC",
    "joint_rl_mpc": "微调 Top-6 MPC",
}

ARM_ORDER_17 = list(SHORT_LABELS.keys())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_fig5_01_burst_tandem_edf() -> dict[str, object]:
    """图 5-1：非平稳突发到达剖面、两阶段释放时间先占 EDF 调度与 932 对解析近似审计。"""
    audit = json.loads((AUDIT_DIR / "tandem_approximation_audit.json").read_text(encoding="utf-8"))
    assert int(audit["evaluated_pairs"]) == 932

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.74))
    gs = fig.add_gridspec(1, 3, left=0.092, right=0.970, bottom=0.198, top=0.815, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): Non-stationary arrival profile lambda(t) over [0, 204] s
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=16.0)
    ax_a.add_patch(Rectangle((50.0, 0.2), 50.0, 71.0, facecolor="#FDECEC", edgecolor="none", zorder=1))
    ax_a.add_patch(Rectangle((198.0, 0.2), 6.0, 71.0, facecolor=P["dist_light"], edgecolor="none", zorder=1))
    t_pts = [0.0, 50.0, 50.0, 100.0, 100.0, 200.0, 200.0, 204.0]
    lam_pts = [15.0, 15.0, 60.0, 60.0, 15.0, 15.0, 0.0, 0.0]
    ax_a.plot(t_pts, lam_pts, color=P["hero"], lw=1.6, zorder=4)
    ax_a.text(75.0, 63.0, "4倍突发\n60/s", fontsize=7.2, fontweight="bold", color=P["crimson"], ha="center", va="bottom")
    ax_a.text(24.0, 19.5, "15/s", fontsize=7.2, color=P["ink"], ha="center", va="bottom")
    ax_a.text(152.0, 19.5, "恢复 15/s", fontsize=7.2, color=P["ink"], ha="center", va="bottom")
    ax_a.text(196.0, 44.0, "尾排空", fontsize=7.2, color=P["success"], ha="right", va="center")
    ax_a.set_xlim(0.0, 204.0)
    ax_a.set_ylim(0.0, 78.0)
    ax_a.set_xticks([0, 50, 100, 200])
    ax_a.set_yticks([0, 20, 40, 60])
    ax_a.set_xlabel("仿真时刻 $t$ ($\\mathrm{s}$)")
    ax_a.set_ylabel("系统期望到达率 $\\lambda(t)$ ($\\mathrm{s}^{-1}$)")

    # Panel (b): Stage-1 release time + Stage-2 preemptive EDF vs static Lindley false-blocking
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=16.0)
    # Row y=2.45: Stage 1 communication release
    ax_b.add_patch(Rectangle((0.05, 2.25), 0.18, 0.42, facecolor=P["hero_light"], edgecolor=P["hero"], lw=0.75, zorder=3))
    ax_b.add_patch(Rectangle((0.18, 2.72), 0.14, 0.36, facecolor=P["dist_light"], edgecolor=P["success"], lw=0.75, zorder=3))
    ax_b.text(1.18, 2.66, "释放 $\\tilde{\\varsigma}_{\\mathrm{rel}}$", fontsize=7.2, color=P["ink"], ha="center", va="center")
    # Row y=1.40: Static Lindley false-blocking (r_1 blocks urgent r_2 -> miss!)
    ax_b.add_patch(Rectangle((0.23, 1.22), 0.88, 0.44, facecolor=P["comp_light"], edgecolor=P["comparator"], lw=0.75, zorder=3))
    ax_b.add_patch(Rectangle((1.11, 1.22), 0.36, 0.44, facecolor="#FDECEC", edgecolor=P["crimson"], lw=0.85, zorder=3))
    ax_b.text(0.67, 1.44, "$r_1$", fontsize=7.2, color=P["ink"], ha="center", va="center")
    ax_b.text(1.80, 1.44, "$r_2$ 超时", fontsize=7.2, fontweight="bold", color=P["crimson"], ha="center", va="center")
    # Row y=0.45: Preemptive EDF (r_2 preempts & finishes at 0.68 <= 0.95, r_1 finishes at 1.47 <= 1.65)
    ax_b.add_patch(Rectangle((0.23, 0.25), 0.09, 0.44, facecolor=P["hero_light"], edgecolor=P["hero"], lw=0.75, zorder=3))
    ax_b.add_patch(Rectangle((0.32, 0.25), 0.36, 0.44, facecolor=P["dist_light"], edgecolor=P["success"], lw=0.85, zorder=3))
    ax_b.add_patch(Rectangle((0.68, 0.25), 0.79, 0.44, facecolor=P["hero_light"], edgecolor=P["hero"], lw=0.75, zorder=3))
    ax_b.text(0.50, 0.47, "$r_2$", fontsize=7.2, fontweight="bold", color=P["success"], ha="center", va="center")
    ax_b.text(1.08, 0.47, "$r_1$ 续算", fontsize=7.2, color=P["hero"], ha="center", va="center")
    ax_b.text(1.80, 0.47, "全按时", fontsize=7.2, fontweight="bold", color=P["success"], ha="center", va="center")
    ax_b.set_xlim(0.0, 2.16)
    ax_b.set_ylim(0.0, 3.30)
    ax_b.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_b.set_yticks([0.47, 1.44, 2.60])
    ax_b.set_yticklabels(["先占\nEDF", "静态\n阻塞", "通信\n释放"], fontsize=7.2)
    ax_b.set_xlabel("节点局部时间轴 ($\\mathrm{s}$)")

    # Panel (c): Tandem EDF vs Static Lindley approximation audit (932 pairs)
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=22.0)
    rel_mae = float(audit["release_time_edf"]["mae"])
    rel_rmse = float(audit["release_time_edf"]["rmse"])
    rel_corr = float(audit["release_time_edf"]["correlation"])
    lin_mae = float(audit["static_single_pass_lindley"]["mae"])
    lin_rmse = float(audit["static_single_pass_lindley"]["rmse"])
    xc = np.arange(2)
    ax_c.bar(xc - 0.17, [rel_mae, rel_rmse], width=0.28, color=P["hero"], edgecolor=P["ink"], lw=0.6, label="释放时间 EDF", zorder=3)
    ax_c.bar(xc + 0.17, [lin_mae, lin_rmse], width=0.28, color=P["border"], edgecolor=P["ink"], lw=0.6, label="静态单遍近似", zorder=3)
    ax_c.text(0 - 0.18, rel_mae + 0.03, f"{rel_mae:.2f}", fontsize=7.2, fontweight="bold", color=P["hero"], ha="center", va="bottom")
    ax_c.text(1 - 0.18, rel_rmse + 0.03, f"{rel_rmse:.2f}", fontsize=7.2, fontweight="bold", color=P["hero"], ha="center", va="bottom")
    ax_c.text(0.50, 0.84, f"932对核验 $r={rel_corr:.4f}$", fontsize=7.2, fontweight="bold", color=P["success"], ha="center", va="top")
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.58, 1.01), ncol=1, fontsize=7.2, frameon=False, labelspacing=0.12, handlelength=1.1)
    ax_c.set_xlim(-0.55, 1.55)
    ax_c.set_ylim(0.0, 0.90)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(["MAE", "RMSE"])
    ax_c.set_yticks([0.0, 0.3, 0.6, 0.9])
    ax_c.set_ylabel("按时完成数预测误差")

    return export_redesign_figure(
        fig,
        "fig5_01_burst_tandem_edf",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig5_02_equivariant_dual_branch() -> dict[str, object]:
    """图 5-2：槽置换等变双分支联合策略网络与期望运动反馈 C_upper 耦合架构图。"""
    c = SchematicCanvas(height_mm=74.0)

    # Outer containers with 22 pt inter-layer corridor (y = 92..114)
    c.container(
        6.0, 6.0, 403.5, 86.0,
        "(a) 上层条件运动分支与期望物理反馈向量聚合",
        fill="#FBFDFF", stroke=P["border"], lw=0.85, title_color=P["hero"], title_size=7.8,
    )
    c.container(
        6.0, 114.0, 403.5, 88.0,
        "(b) 下层候选分配分支、自回归联合策略与价值网络",
        fill="#FBFDFF", stroke=P["border"], lw=0.85, title_color=P["ink"], title_size=7.8,
    )

    # Row (a) boxes
    c.box(
        14.0, 30.0, 112.0, 54.0,
        "运动分支特征输入",
        "全局 $\\mathbf{c}(s)$ + 候选 $\\mathbf{a}_k$\n运动特征 $\\mathbf{m}_{km}$ (44维)",
        fill=P["panel_bg"], stroke=P["prior"], lw=0.85,
        bold_title=True, title_color=P["ink"], sub_color=P["prior"], title_size=7.4, sub_size=7.2,
    )
    c.box(
        144.0, 30.0, 120.0, 54.0,
        "共享运动评分器 $g_m$",
        "$44\\to 48\\to 1$ 跳连感知机\n分布 $p_m(m\\mid s,k)$",
        fill=P["hero_light"], stroke=P["hero"], lw=1.0,
        bold_title=True, title_color=P["hero"], sub_color=P["ink"], title_size=7.4, sub_size=7.2,
    )
    c.box(
        282.0, 30.0, 119.0, 54.0,
        "期望反馈 $\\mathbf{C}_{\\mathrm{upper}}$",
        "救援潜力 + 预测可服务\n综合溢出 + 净吞吐 (4维)",
        fill=P["dist_light"], stroke=P["success"], lw=1.05,
        bold_title=True, title_color=P["success"], sub_color=P["ink"], title_size=7.4, sub_size=7.2,
    )

    c.arrow((128.0, 57.0), (142.0, 57.0), color=P["hero"], lw=1.05)
    c.arrow((266.0, 57.0), (280.0, 57.0), color=P["success"], lw=1.05)

    # Row (b) boxes
    c.box(
        14.0, 139.0, 118.0, 55.0,
        "状态价值网络 $V_\\psi(s)$",
        "输入 $\\mathbf{x}_V(s)$ (34维聚合)\n仅由真实轨迹 GAE 更新",
        fill=P["abl_light"], stroke=P["ablation"], lw=0.95,
        bold_title=True, title_color=P["ablation"], sub_color=P["ink"], title_size=7.4, sub_size=7.2,
    )
    c.box(
        150.0, 139.0, 122.0, 55.0,
        "共享分配评分器 $g_a$",
        "$32\\to 48\\to 1$ ($\\mathbf{a}_k\\oplus\\mathbf{C}_{\\mathrm{upper}}\\oplus\\mathbf{c}$)\n输出分布 $p_a(k\\mid s)$",
        fill=P["hero_light"], stroke=P["hero"], lw=1.05,
        bold_title=True, title_color=P["hero"], sub_color=P["ink"], title_size=7.4, sub_size=7.2,
    )
    c.box(
        290.0, 139.0, 111.0, 55.0,
        "自回归联合分布",
        "$\\pi_\\theta(k,m\\mid s)$\n$= p_a(k\\mid s)\\,p_m(m\\mid s,k)$",
        fill=P["comp_light"], stroke=P["comparator"], lw=1.05,
        bold_title=True, title_color=P["comparator"], sub_color=P["ink"], title_size=7.4, sub_size=7.2,
    )

    # Coupling & gradient paths strictly right of Row (b) title (x > 218 pt)
    c.arrow((341.5, 85.5), (254.0, 137.0), color=P["success"], lw=1.15)
    c.arrow((274.0, 166.5), (288.0, 166.5), color=P["comparator"], lw=1.05)
    c.arrow((232.0, 137.0), (232.0, 85.5), color=P["crimson"], lw=1.0, ls="--")
    c.text(132.0, 103.0, "下层损失回传 $\\partial \\mathcal{L}/\\partial \\mathbf{C}_{\\mathrm{upper}}$", fontsize=7.2, bold=True, color=P["crimson"], ha="center")

    return c.export(
        "fig5_02_equivariant_dual_branch",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
    )


def build_fig5_03_distill_crn_mpc_framework() -> dict[str, object]:
    """图 5-3：两轮 DAgger 规划蒸馏、双流 CRN 三步反事实 PPO 微调与双模式部署全流程架构图。"""
    c = SchematicCanvas(height_mm=72.0)

    # Stage I column
    c.container(
        6.0, 8.0, 125.0, 188.0,
        "阶段 I：两轮规划蒸馏",
        fill="#FBFDFF", stroke=P["distilled"], lw=0.95, title_color=P["distilled"], title_size=7.6, title_align="center",
    )
    c.box(
        14.0, 32.0, 109.0, 65.0,
        "离线规划教师 $Q_T(s,k,m)$",
        "温度 $\\tau_T = 0.015$ 软分布\n双分支交叉熵暖启动",
        fill=P["panel_bg"], stroke=P["distilled"], lw=0.85,
        bold_title=True, title_color=P["ink"], sub_color=P["prior"], title_size=7.3, sub_size=7.2,
    )
    c.box(
        14.0, 114.0, 109.0, 72.0,
        "两轮 DAgger 分布纠偏",
        "第1轮教师 + 第2轮混合\n确认 DSR = 48.04%\n(训练后退出在线环路)",
        fill=P["dist_light"], stroke=P["distilled"], lw=0.95,
        bold_title=True, title_color=P["distilled"], sub_color=P["ink"], title_size=7.3, sub_size=7.2,
    )
    c.arrow((68.5, 99.0), (68.5, 112.0), color=P["distilled"], lw=1.0)

    # Stage II column
    c.container(
        142.0, 8.0, 134.0, 188.0,
        "阶段 II：反事实辅助 PPO",
        fill="#FBFDFF", stroke=P["hero"], lw=1.05, title_color=P["hero"], title_size=7.6, title_align="center",
    )
    c.box(
        150.0, 32.0, 118.0, 65.0,
        "双流真实采样 ($N_t=2$)",
        "阻尼留一优势 $\\tilde{A}$ 更新策略\n独占更新价值网络 $V_\\psi$",
        fill=P["hero_light"], stroke=P["hero"], lw=0.95,
        bold_title=True, title_color=P["hero"], sub_color=P["ink"], title_size=7.4, sub_size=7.3,
    )
    c.box(
        150.0, 114.0, 118.0, 72.0,
        "CRN 三步反事实展开树",
        "$|\\mathcal{B}_t|\\leq 4$ 共享随机源 $\\xi_t$\n组中心化 $\\hat{A}_{\\mathrm{cf}}$ 消去 $\\zeta_{\\mathrm{com}}$\n净增益 +0.69 pp (+1742次)",
        fill=P["comp_light"], stroke=P["comparator"], lw=0.95,
        bold_title=True, title_color=P["comparator"], sub_color=P["ink"], title_size=7.4, sub_size=7.3,
    )
    c.arrow((209.0, 99.0), (209.0, 112.0), color=P["hero"], lw=1.0)

    # Stage III column
    c.container(
        287.0, 8.0, 122.5, 188.0,
        "阶段 III：双模式部署",
        fill="#FBFDFF", stroke=P["success"], lw=1.05, title_color=P["success"], title_size=7.6, title_align="center",
    )
    c.box(
        295.0, 32.0, 106.5, 65.0,
        "模式 A：零前瞻策略",
        "DSR = 48.74% (0次前瞻)\n单步均值 0.520 s",
        fill=P["hero_light"], stroke=P["hero"], lw=1.0,
        bold_title=True, title_color=P["hero"], sub_color=P["ink"], title_size=7.3, sub_size=7.2,
    )
    c.box(
        295.0, 114.0, 106.5, 72.0,
        "模式 B：神经引导 MPC",
        "熵校准 Top-6 三步推演\nDSR = 49.31% (+0.75 pp)\n0.722 s (降耗时 54.9%)",
        fill=P["dist_light"], stroke=P["success"], lw=1.05,
        bold_title=True, title_color=P["success"], sub_color=P["ink"], title_size=7.3, sub_size=7.2,
    )

    # Inter-stage connectors (strictly outside inner box boundaries)
    c.arrow((132.5, 150.0), (140.5, 65.0), color=P["hero"], lw=1.1)
    c.arrow((277.5, 64.5), (285.5, 64.5), color=P["hero"], lw=1.1)
    c.arrow((277.5, 150.0), (285.5, 150.0), color=P["success"], lw=1.1)

    return c.export(
        "fig5_03_distill_crn_mpc_framework",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
    )


def build_fig5_04_main_confirmation_effects() -> dict[str, object]:
    """图 5-4：816 回合盲测确认集全部 17 方案 DSR 阶梯、(alpha, eta) 前沿与 Bonferroni 配对效应。"""
    sum_rows = {r["arm"]: r for r in _read_csv(DATA_DIR / "confirmation_summary.csv")}
    unit_rows = _read_csv(DATA_DIR / "confirmation_units.csv")
    eff_rows = {r["pair"]: r for r in _read_csv(DATA_DIR / "paired_dsr_effects.csv")}
    assert len(sum_rows) == 17 and len(unit_rows) == 816

    parent_succ: dict[tuple[str, int], int] = defaultdict(int)
    parent_gen: dict[tuple[str, int], int] = defaultdict(int)
    for r in unit_rows:
        key = (r["arm"], int(r["parent"]))
        parent_succ[key] += int(r["success"])
        parent_gen[key] += int(r["generated"])

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 4.68))
    gs = fig.add_gridspec(2, 2, left=0.228, right=0.972, bottom=0.100, top=0.910, wspace=0.38, hspace=0.56, width_ratios=[1.10, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # Panel (a): All 17 schemes Cleveland dot-and-stem ladder + 3 upstream parent lineage markers
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", x_offset_pt=-78.0, y_offset_pt=14.0)
    ya = np.arange(len(ARM_ORDER_17))
    x_base = 43.30
    for idx, arm in enumerate(ARM_ORDER_17):
        dsr = float(sum_rows[arm]["dsr"]) * 100.0
        if arm == "joint_rl_mpc":
            col, fw = P["success"], "bold"
        elif arm == "joint_rl_finetuned":
            col, fw = P["hero"], "bold"
        elif arm in ("joint_distilled", "distilled_top6_mpc"):
            col, fw = P["distilled"], "normal"
        elif arm == "quality_amortized":
            col, fw = P["comparator"], "normal"
        else:
            col, fw = P["muted"], "normal"
        ax_a.plot([x_base, dsr], [idx, idx], color=col, lw=1.25 if fw == "bold" else 0.90, zorder=2)
        p_vals = [parent_succ[(arm, p)] / parent_gen[(arm, p)] * 100.0 for p in range(3)]
        ax_a.scatter(p_vals, [idx] * 3, s=9, color=P["border"], edgecolor=P["ink"], linewidth=0.35, zorder=3)
        ax_a.scatter([dsr], [idx], s=26 if fw == "bold" else 18, color=col, edgecolor=P["ink"], linewidth=0.45, zorder=4)
        if arm in ("uncoordinated_init", "joint_rl", "joint_distilled", "joint_rl_finetuned", "quality_amortized", "joint_rl_mpc"):
            ax_a.text(max(dsr, max(p_vals)) + 0.16, idx, f"{dsr:.2f}", fontsize=7.2, fontweight=fw, color=col if fw == "bold" else P["ink"], ha="left", va="center")
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([SHORT_LABELS[a] for a in ARM_ORDER_17], fontsize=7.2)
    ax_a.invert_yaxis()
    ax_a.set_xlim(x_base, 51.45)
    ax_a.set_ylim(16.65, -0.65)
    ax_a.set_xticks([44, 46, 48, 50])
    ax_a.set_xlabel("盲测确认集 DSR (%)")

    # Panel (b): (alpha, eta) 2D frontier
    style_redesign_axis(ax_b, grid="none", box=True)
    add_panel_tag(ax_b, "(b)", y_offset_pt=12.0)
    frontier_specs = [
        ("uncoordinated_init", "初始先验", P["muted"], "o", (6.0, 1.0), "left"),
        ("joint_distilled", "联合蒸馏", P["distilled"], "o", (-6.0, -4.5), "right"),
        ("joint_rl_finetuned", "联合微调", P["hero"], "o", (6.0, -1.5), "left"),
        ("quality_amortized", "在线规划", P["comparator"], "^", (-6.0, -0.5), "right"),
        ("distilled_top6_mpc", "蒸馏MPC", P["ablation"], "s", (-6.0, -2.5), "right"),
        ("joint_rl_mpc", "微调MPC", P["success"], "D", (6.0, 2.5), "left"),
    ]
    alp_all = [float(sum_rows[a]["admission"]) * 100.0 for a, *_ in frontier_specs]
    eta_all = [float(sum_rows[a]["conditional"]) * 100.0 for a, *_ in frontier_specs]
    for arm, lbl, col, mk, off, ha in frontier_specs:
        alp = float(sum_rows[arm]["admission"]) * 100.0
        eta = float(sum_rows[arm]["conditional"]) * 100.0
        fw = "bold" if arm in ("joint_rl_finetuned", "joint_rl_mpc") else "normal"
        ax_b.scatter([alp], [eta], marker=mk, s=24, color=col, edgecolor=P["ink"], linewidth=0.45, zorder=4)
        ax_b.annotate(lbl, xy=(alp, eta), xytext=off, textcoords="offset points", fontsize=7.2, fontweight=fw, color=col if fw == "bold" else P["ink"], ha=ha, va="center", zorder=5)
    ax_b.set_xlim(min(alp_all) - 1.85, max(alp_all) + 2.05)
    ax_b.set_ylim(min(eta_all) - 2.2, max(eta_all) + 2.0)
    ax_b.set_yticks([86, 90, 94])
    ax_b.set_xlabel("接纳率 $\\alpha$ (%)")
    ax_b.set_ylabel("条件完成率 $\\eta$ (%)")

    # Panel (c): 6 key paired DSR contrasts with 95% CI + 31-Bonferroni simultaneous CI
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=10.0)
    focal_pairs = [
        ("joint_rl_finetuned--joint_distilled", "微调-蒸馏", P["hero"], True),
        ("joint_rl_finetuned--decomposed_rl_finetuned", "微调-去耦", P["distilled"], False),
        ("joint_rl_finetuned--joint_no_expectation_rl_finetuned", "微调-无期望", P["muted"], False),
        ("joint_rl_mpc--quality_amortized", "MPC-规划", P["success"], True),
        ("joint_rl_mpc--distilled_top6_mpc", "微调-蒸馏M", P["ablation"], False),
        ("joint_rl_finetuned--quality_amortized", "微调-规划", P["comparator"], False),
    ]
    ax_c.axvline(0.0, color=P["crimson"], lw=0.90, ls="--", zorder=1)
    yc = np.arange(len(focal_pairs))
    for idx, (pk, lbl, col, sig_bonf) in enumerate(focal_pairs):
        r = eff_rows[pk]
        est = float(r["effect_pp"])
        lo95, hi95 = float(r["ci95_low_pp"]), float(r["ci95_high_pp"])
        lob, hib = float(r["bonf31_low_pp"]), float(r["bonf31_high_pp"])
        ax_c.plot([lob, hib], [idx, idx], color=col, lw=0.85, zorder=2)
        ax_c.plot([lo95, hi95], [idx, idx], color=col, lw=2.3, zorder=3)
        ax_c.scatter([est], [idx], s=22, color=col, edgecolor=P["ink"], linewidth=0.45, zorder=4)
        if sig_bonf:
            ax_c.text(hib + 0.06, idx, f"+{est:.2f}*", fontsize=7.2, fontweight="bold", color=col, ha="left", va="center")
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([fp[1] for fp in focal_pairs], fontsize=7.2)
    ax_c.invert_yaxis()
    ax_c.set_xlim(-0.38, 2.02)
    ax_c.set_ylim(5.65, -0.65)
    ax_c.set_xticks([0.0, 0.7, 1.4])
    ax_c.set_xlabel("配对 $\\Delta\\mathrm{DSR}$ (pp, 粗95%/细Bonf)")

    return export_redesign_figure(
        fig,
        "fig5_04_main_confirmation_effects",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig5_05_training_counterfactual_coupling() -> dict[str, object]:
    """图 5-5：PPO 验证收敛轨迹、CRN 反事实开关受控消融与去耦策略零泄露审计。"""
    val_rows = _read_csv(DATA_DIR / "validation_curves.csv")
    cf_rows = {r["arm"]: r for r in _read_csv(DATA_DIR / "cf0_ablation_summary.csv")}
    indep = json.loads((AUDIT_DIR / "decomposed_independence_audit.json").read_text(encoding="utf-8"))
    assert indep["verified"] is True and float(indep["max_plan_dependence_uncoupled"]) == 0.0

    curves: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in val_rows:
        curves[r["arm"]].append((int(r["iteration"]), float(r["validation_dsr"]) * 100.0))
    for k in curves:
        curves[k].sort(key=lambda t: t[0])

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.82))
    gs = fig.add_gridspec(1, 3, left=0.092, right=0.970, bottom=0.205, top=0.775, wspace=0.41)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): Validation DSR curves (warm-start fine-tuned vs cold-start scratch)
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", y_offset_pt=26.0)
    line_specs = [
        ("joint_rl_finetuned", "联合微调", P["hero"], "-", "o", 1.55),
        ("decomposed_rl_finetuned", "去耦微调", P["distilled"], "--", "s", 1.15),
        ("joint_rl", "联合冷启动", P["comparator"], "-", "^", 1.25),
        ("decomposed_rl", "去耦冷启动", P["muted"], ":", "d", 1.10),
    ]
    for arm, lbl, col, ls, mk, lw in line_specs:
        xs = [pt[0] for pt in curves[arm]]
        ys = [pt[1] for pt in curves[arm]]
        ax_a.plot(xs, ys, color=col, ls=ls, lw=lw, marker=mk, markersize=3.8, label=lbl, zorder=3)
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.56, 1.01), ncol=2, fontsize=7.2, frameon=False, columnspacing=0.6, handlelength=1.1, handletextpad=0.25, labelspacing=0.26)
    ax_a.set_xlim(-0.25, 4.25)
    ax_a.set_ylim(43.4, 49.3)
    ax_a.set_xticks([0, 1, 2, 3, 4])
    ax_a.set_yticks([44, 46, 48])
    ax_a.set_xlabel("PPO 策略更新轮次")
    ax_a.set_ylabel("验证集 DSR (%)")

    # Panel (b): Controlled w_cf=0 vs w_cf=1.25 dumbbell ablation
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=26.0)
    cf_order = [
        ("decomposed_rl_finetuned", "去耦\n微调"),
        ("joint_no_expectation_rl_finetuned", "无期望\n微调"),
        ("joint_rl_finetuned", "联合\n微调"),
    ]
    xb = np.arange(3)
    for idx, (arm, _) in enumerate(cf_order):
        r = cf_rows[arm]
        v0 = float(r["cf0_dsr"]) * 100.0
        v1 = float(r["cf035_dsr"]) * 100.0
        dv = float(r["delta_dsr_pp"])
        ax_b.plot([idx, idx], [v0, v1], color=P["hero"] if idx == 2 else P["distilled"], lw=1.8, zorder=2)
        ax_b.scatter([idx], [v0], s=26, color=P["panel_bg"], edgecolor=P["prior"], linewidth=1.0, zorder=3, label="纯在策略 ($w_{\\mathrm{cf}}=0$)" if idx == 0 else None)
        ax_b.scatter([idx], [v1], s=30, color=P["hero"] if idx == 2 else P["distilled"], edgecolor=P["ink"], linewidth=0.5, zorder=4, label="反事实 ($w_{\\mathrm{cf}}=1.25$)" if idx == 0 else None)
        ax_b.text(idx, v1 + 0.10, f"+{dv:.2f}", fontsize=7.2, fontweight="bold" if idx == 2 else "normal", color=P["hero"] if idx == 2 else P["ink"], ha="center", va="bottom")
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.60, 1.01), ncol=1, fontsize=7.2, frameon=False, labelspacing=0.40, handletextpad=0.25)
    ax_b.set_xlim(-0.55, 2.55)
    ax_b.set_ylim(47.65, 49.18)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([item[1] for item in cf_order], fontsize=7.2)
    ax_b.set_yticks([47.8, 48.4, 49.0])
    ax_b.set_ylabel("盲测确认集 DSR (%)")

    # Panel (c): Plan-dependence L_1 audit (exact 0.0 on uncoupled vs ~0.95 on joint)
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=26.0)
    dep = indep["max_plan_dependence_by_arm"]
    c_items = [
        ("去耦6臂", 0.0, P["muted"]),
        ("联合冷启", float(dep["joint_rl"]), P["comparator"]),
        ("联合蒸馏", float(dep["joint_distilled"]), P["distilled"]),
        ("联合微调", float(dep["joint_rl_finetuned"]), P["hero"]),
    ]
    xc = np.arange(4)
    vals_c = [it[1] for it in c_items]
    cols_c = [it[2] for it in c_items]
    ax_c.bar(xc, vals_c, width=0.48, color=cols_c, edgecolor=P["ink"], lw=0.6, zorder=3)
    ax_c.text(0, 0.04, "0.00", fontsize=7.2, fontweight="bold", color=P["success"], ha="center", va="bottom")
    ax_c.text(1, vals_c[1] + 0.04, f"{vals_c[1]:.2f}", fontsize=7.2, ha="center", va="bottom")
    ax_c.text(3, vals_c[3] + 0.04, f"{vals_c[3]:.2f}", fontsize=7.2, fontweight="bold", color=P["hero"], ha="center", va="bottom")
    ax_c.set_xlim(-0.55, 3.55)
    ax_c.set_ylim(0.0, 1.16)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(["去耦", "冷启", "蒸馏", "微调"], fontsize=7.2)
    ax_c.set_yticks([0.0, 0.5, 1.0])
    ax_c.set_ylabel("运动分支方案依赖度 $L_1$")

    return export_redesign_figure(
        fig,
        "fig5_05_training_counterfactual_coupling",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig5_06_cohort_peak_queue_dynamics() -> dict[str, object]:
    """图 5-6：请求三结局比例、三到达阶段（平峰/突发/恢复）DSR 与工作负载 955000 排队积压演化。"""
    sum_rows = {r["arm"]: r for r in _read_csv(DATA_DIR / "confirmation_summary.csv")}
    coh_rows = {(r["arm"], r["phase"]): r for r in _read_csv(DATA_DIR / "arrival_cohorts.csv")}
    pq = json.loads((DATA_DIR / "peak_queue_evolution_955000.json").read_text(encoding="utf-8"))

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.80))
    gs = fig.add_gridspec(1, 3, left=0.125, right=0.970, bottom=0.200, top=0.790, wspace=0.41)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): 6 key schemes 100% stacked horizontal request outcomes
    style_redesign_axis(ax_a, grid="none")
    add_panel_tag(ax_a, "(a)", x_offset_pt=-36.0, y_offset_pt=24.0)
    arms_a = [
        ("uncoordinated_init", "初始先验"),
        ("joint_rl", "联合冷启"),
        ("joint_distilled", "联合蒸馏"),
        ("quality_amortized", "在线规划"),
        ("joint_rl_finetuned", "联合微调"),
        ("joint_rl_mpc", "微调MPC"),
    ]
    ya = np.arange(len(arms_a))
    s_pct = np.array([float(sum_rows[a]["dsr"]) * 100.0 for a, _ in arms_a])
    adm_pct = np.array([float(sum_rows[a]["admission"]) * 100.0 for a, _ in arms_a])
    fail_adm = adm_pct - s_pct
    unadm = 100.0 - adm_pct
    ax_a.barh(ya, s_pct, height=0.58, color=P["hero"], edgecolor=P["ink"], lw=0.45, label="按时完成", zorder=3)
    ax_a.barh(ya, fail_adm, left=s_pct, height=0.58, color=P["comparator"], edgecolor=P["ink"], lw=0.45, label="接纳超时", zorder=3)
    ax_a.barh(ya, unadm, left=adm_pct, height=0.58, color=P["panel_bg"], edgecolor=P["ink"], lw=0.45, label="未接纳过期", zorder=3)
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=1, fontsize=7.2, frameon=False, labelspacing=0.10, handlelength=1.1)
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([lbl for _, lbl in arms_a], fontsize=7.2)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.0, 100.0)
    ax_a.set_xticks([0, 50, 100])
    ax_a.set_xlabel("生成请求结局占比 (%)")

    # Panel (b): Arrival cohort DSR (pre, peak, post)
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=24.0)
    phases = ["pre", "peak", "post"]
    xb = np.arange(3)
    coh_specs = [
        ("uncoordinated_init", "初始先验", P["muted"], ":", "o", 1.1),
        ("quality_amortized", "在线规划", P["comparator"], "--", "^", 1.2),
        ("joint_rl_finetuned", "联合微调", P["hero"], "-", "o", 1.45),
        ("joint_rl_mpc", "微调MPC", P["success"], "-", "D", 1.50),
    ]
    for arm, lbl, col, ls, mk, lw in coh_specs:
        ys = [float(coh_rows[(arm, ph)]["dsr"]) * 100.0 for ph in phases]
        ax_b.plot(xb, ys, color=col, ls=ls, lw=lw, marker=mk, markersize=4.0, label=lbl, zorder=3)
    ax_b.text(1.0, 63.8, "高峰 +1.30 pp", fontsize=7.2, fontweight="bold", color=P["success"], ha="center", va="top")
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.56, 1.01), ncol=2, fontsize=7.2, frameon=False, columnspacing=0.55, handlelength=1.05, handletextpad=0.25, labelspacing=0.24)
    ax_b.set_xlim(-0.30, 2.30)
    ax_b.set_ylim(36.0, 66.2)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["平峰前", "突发峰", "恢复期"], fontsize=7.2)
    ax_b.set_yticks([40, 50, 60])
    ax_b.set_ylabel("分阶段到达 DSR (%)")

    # Panel (c): Second-by-second cumulative unserved & expired requests on workload 955000
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=24.0)
    pq_arms = pq["arms"]
    max_exp = max(max(v["cum_unserved_and_expired"]) for v in pq_arms.values())
    y_top = float(max_exp) * 1.12
    ax_c.add_patch(Rectangle((50.0, 10.0), 50.0, y_top - 20.0, facecolor="#FDECEC", edgecolor="none", zorder=1))
    trace_specs = [
        ("uncoordinated_init", "先验", P["muted"], ":", 1.15),
        ("quality_amortized", "规划", P["comparator"], "--", 1.15),
        ("joint_rl_finetuned", "微调", P["hero"], "-", 1.35),
        ("joint_rl_mpc", "MPC", P["success"], "-", 1.40),
    ]
    for arm, lbl, col, ls, lw in trace_specs:
        if arm in pq_arms:
            ts = pq_arms[arm]["times"]
            ce = pq_arms[arm]["cum_unserved_and_expired"]
            ax_c.plot(ts, ce, color=col, ls=ls, lw=lw, label=lbl, zorder=3)
    ax_c.text(138.0, 1120.0, "较先验\n-443次", fontsize=7.2, fontweight="bold", color=P["success"], ha="center", va="center")
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.56, 1.01), ncol=2, fontsize=7.2, frameon=False, columnspacing=0.6, handlelength=1.1, handletextpad=0.25, labelspacing=0.24)
    ax_c.set_xlim(0.0, 205.0)
    ax_c.set_ylim(0.0, y_top)
    ax_c.set_xticks([0, 50, 100, 200])
    ax_c.set_yticks([0, 1000, 2000, 3000])
    ax_c.set_xlabel("决策时隙 $t$ ($\\mathrm{s}$)")
    ax_c.set_ylabel("累计未结与过期请求数")

    return export_redesign_figure(
        fig,
        "fig5_06_cohort_peak_queue_dynamics",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def build_fig5_07_decision_cost_pareto() -> dict[str, object]:
    """图 5-7：服务质量-耗时经验帕累托前沿、运行阶段耗时拆解与零前瞻内部四模块占比。"""
    sum_rows = {r["arm"]: r for r in _read_csv(DATA_DIR / "confirmation_summary.csv")}
    stg_rows = {r["arm"]: r for r in _read_csv(DATA_DIR / "stage_decision_cost.csv")}

    setup_redesign_style(8.0)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.78))
    gs = fig.add_gridspec(1, 3, left=0.092, right=0.970, bottom=0.205, top=0.790, wspace=0.41)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel (a): Empirical DSR vs Mean Decision Time Pareto frontier
    style_redesign_axis(ax_a, grid="none", box=True)
    add_panel_tag(ax_a, "(a)", y_offset_pt=24.0)
    pareto_arms = ["joint_distilled", "joint_rl_finetuned", "distilled_top6_mpc", "joint_rl_mpc"]
    px = [float(sum_rows[a]["decision_mean_s"]) for a in pareto_arms]
    py = [float(sum_rows[a]["dsr"]) * 100.0 for a in pareto_arms]
    ax_a.plot(px, py, color=P["hero"], lw=1.35, zorder=2)

    pts = [
        ("uncoordinated_init", "初始先验", P["muted"], "o", (6.0, -1.0), "left"),
        ("prior_top6_mpc", "先验MPC", P["muted"], "s", (6.0, -1.0), "left"),
        ("joint_distilled", "联合蒸馏", P["distilled"], "o", (6.0, -7.5), "left"),
        ("joint_rl_finetuned", "联合微调", P["hero"], "o", (-11.0, 18.0), "left"),
        ("distilled_top6_mpc", "蒸馏MPC", P["ablation"], "s", (7.0, -5.5), "left"),
        ("joint_rl_mpc", "微调MPC", P["success"], "D", (8.0, 3.0), "left"),
        ("quality_amortized", "在线规划", P["comparator"], "^", (-6.0, -7.0), "right"),
    ]
    for arm, lbl, col, mk, off, ha in pts:
        tx = float(sum_rows[arm]["decision_mean_s"])
        ty = float(sum_rows[arm]["dsr"]) * 100.0
        fw = "bold" if arm in ("joint_rl_finetuned", "joint_rl_mpc") else "normal"
        ax_a.scatter([tx], [ty], marker=mk, s=24, color=col, edgecolor=P["ink"], linewidth=0.45, zorder=4)
        ap = dict(arrowstyle="-", color=col, lw=0.65, shrinkA=1.0, shrinkB=3.5) if arm == "joint_rl_finetuned" else None
        ax_a.annotate(lbl, xy=(tx, ty), xytext=off, textcoords="offset points", fontsize=7.2, fontweight=fw, color=col if fw == "bold" else P["ink"], ha=ha, va="center", arrowprops=ap, zorder=5)
    ax_a.set_xlim(0.28, 1.82)
    ax_a.set_ylim(43.6, 51.45)
    ax_a.set_xticks([0.5, 1.0, 1.5])
    ax_a.set_yticks([44, 46, 48, 50])
    ax_a.set_xlabel("单步平均决策耗时 ($\\mathrm{s}$)")
    ax_a.set_ylabel("盲测确认集 DSR (%)")

    # Panel (b): Operating stage decision cost (pre, peak, post)
    style_redesign_axis(ax_b, grid="none")
    add_panel_tag(ax_b, "(b)", y_offset_pt=24.0)
    bar_arms = [
        ("joint_distilled", "蒸馏", P["distilled"]),
        ("joint_rl_finetuned", "微调", P["hero"]),
        ("joint_rl_mpc", "MPC", P["success"]),
        ("quality_amortized", "规划", P["comparator"]),
    ]
    xb = np.arange(3)
    bw = 0.18
    st_keys = ["pre_mean_s", "peak_mean_s", "post_mean_s"]
    for idx_b, (arm, lbl, col) in enumerate(bar_arms):
        offs = xb + (idx_b - 1.5) * bw
        vals = [float(stg_rows[arm][sk]) for sk in st_keys]
        ax_b.bar(offs, vals, width=bw * 0.90, color=col, edgecolor=P["ink"], lw=0.45, label=lbl, zorder=3)
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.56, 1.01), ncol=2, fontsize=7.2, frameon=False, columnspacing=0.6, handlelength=1.05, handletextpad=0.25, labelspacing=0.24)
    ax_b.set_xlim(-0.52, 2.52)
    ax_b.set_ylim(0.0, 4.35)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["平峰\n<40s", "拥塞\n40-105s", "恢复\n≥105s"], fontsize=7.2)
    ax_b.set_yticks([0, 1, 2, 3, 4])
    ax_b.set_ylabel("分阶段单步耗时 ($\\mathrm{s}$)")

    # Panel (c): 606-step internal zero-lookahead 4-module runtime breakdown (chap05 Sec 5.5.3)
    style_redesign_axis(ax_c, grid="none")
    add_panel_tag(ax_c, "(c)", y_offset_pt=24.0)
    comp_labels = ["上游候选", "串联排队", "安全几何", "网络前向"]
    comp_pcts = [73.7, 22.9, 3.2, 0.21]
    comp_cols = [P["distilled"], P["hero"], P["ablation"], P["success"]]
    yc = np.arange(4)
    ax_c.barh(yc, comp_pcts, height=0.54, color=comp_cols, edgecolor=P["ink"], lw=0.55, zorder=3)
    ax_c.text(73.7 + 2.5, 0, "73.7%", fontsize=7.2, color=P["ink"], ha="left", va="center")
    ax_c.text(22.9 + 2.5, 1, "22.9%", fontsize=7.2, color=P["hero"], ha="left", va="center")
    ax_c.text(3.2 + 2.5, 2, "3.2%", fontsize=7.2, color=P["ink"], ha="left", va="center")
    ax_c.text(0.21 + 2.5, 3, "0.21% (1.1ms)", fontsize=7.2, fontweight="bold", color=P["success"], ha="left", va="center")
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(comp_labels, fontsize=7.2)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0.0, 106.0)
    ax_c.set_xticks([0, 40, 80])
    ax_c.set_xlabel("零前瞻耗时占比 (%)")

    return export_redesign_figure(
        fig,
        "fig5_07_decision_cost_pareto",
        repo_root=REPO_ROOT,
        output_dirs=OUT_DIRS,
        qa_dir=QA_DIR,
        axes=[ax_a, ax_b, ax_c],
    )


def main() -> None:
    results = [
        build_fig5_01_burst_tandem_edf(),
        build_fig5_02_equivariant_dual_branch(),
        build_fig5_03_distill_crn_mpc_framework(),
        build_fig5_04_main_confirmation_effects(),
        build_fig5_05_training_counterfactual_coupling(),
        build_fig5_06_cohort_peak_queue_dynamics(),
        build_fig5_07_decision_cost_pareto(),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
