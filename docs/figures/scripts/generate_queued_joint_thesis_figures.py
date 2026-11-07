"""Rebuild Chapter 5 tables/figures from the frozen dynamic-queue confirmation.

This is a publication-data exporter, not a training or evaluation entry point.
The additional content hashes are post-hoc provenance, not preregistration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paths import REPO_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from thesis_figure_style import setup_thesis_style

SOURCE = REPO_ROOT / "results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106"
DATA = REPO_ROOT / "docs/figures/data/queued_joint_rl_20261106"
FIGURES = REPO_ROOT / "docs/figures"
TABLES = REPO_ROOT / "latex/docs/tables"
QA = REPO_ROOT / "output/thesis_writing_20261004/validation/queued_chapter5_20261107"

LABELS = {
    "uncoordinated_init": "初始先验选择器",
    "lower_only_rl": "仅下层 RL",
    "upper_only_rl": "仅上层 RL",
    "decomposed_composed_rl": "单层训练后拼接",
    "decomposed_rl": "去耦冷启动 RL",
    "joint_no_expectation_rl": "无期望向量冷启动 RL",
    "joint_rl": "联合冷启动 RL",
    "decomposed_distilled": "去耦蒸馏",
    "joint_no_expectation_distilled": "无期望向量蒸馏",
    "joint_distilled": "联合蒸馏",
    "decomposed_rl_finetuned": "去耦微调",
    "joint_no_expectation_rl_finetuned": "无期望向量微调",
    "joint_rl_finetuned": "联合微调",
    "quality_amortized": "在线规划基线",
    "prior_top6_mpc": "先验 Top-6 MPC",
    "distilled_top6_mpc": "蒸馏 Top-6 MPC",
    "joint_rl_mpc": "微调 Top-6 MPC",
}
BLUE, ORANGE, GRAY, PURPLE = "#306BA0", "#C58232", "#7A818A", "#8660A5"
COLORS = {key: GRAY for key in LABELS}
COLORS.update(joint_distilled=ORANGE, joint_rl_finetuned=BLUE, joint_rl_mpc=PURPLE)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_and_export():
    analysis = read_json(SOURCE / "confirmation_analysis.json")
    curves = read_json(SOURCE / "rl_training_curves.json")
    manifest_path = SOURCE / "freeze_manifest.json"
    manifest = read_json(manifest_path)
    freeze_hash = digest(manifest_path)
    source_files = [SOURCE / name for name in (
        "confirmation_analysis.json", "rl_training_curves.json", "freeze_manifest.json",
        "protocol.json", "tandem_approximation_audit.json",
        "rollout_replay_audit.json", "decomposed_independence_audit.json",
    )]
    rows, parents, seen = [], {}, set()
    for path in sorted((SOURCE / "confirmation").glob("*/*.json")):
        record = read_json(path)
        key = tuple(record[k] for k in ("arm", "parent", "workload", "replica"))
        assert key not in seen
        seen.add(key)
        assert record["audit"]["status"] == "PASS"
        assert record["freeze_manifest_sha256"] == freeze_hash
        assert record["started_at_utc"] >= manifest["frozen_at_utc"]
        counts = record["counts"]
        assert counts["generated"] == counts.get("success", 0) + counts.get("expired", 0)
        parents.setdefault(record["arm"], {}).setdefault(record["parent"], [0, 0])
        parent_counts = parents[record["arm"]][record["parent"]]
        parent_counts[0] += counts.get("success", 0)
        parent_counts[1] += counts["generated"]
        rows.append(dict(arm=record["arm"], parent=record["parent"], workload=record["workload"],
                         replica=record["replica"], generated=counts["generated"],
                         admitted=counts["admitted"], success=counts.get("success", 0)))
        source_files.append(path)
    expected = {(arm, p, w, r) for arm in LABELS for p in range(3)
                for w in range(955000, 955008) for r in range(2)}
    assert seen == expected and len(rows) == 816
    for arm in LABELS:
        own = [row for row in rows if row["arm"] == arm]
        for metric in ("generated", "admitted", "success"):
            assert sum(row[metric] for row in own) == analysis["summary"][arm]["all"][metric]
    for relative, expected_hash in manifest["source_sha256"].items():
        assert digest(REPO_ROOT / relative) == expected_hash, relative
    for relative, expected_hash in manifest["model_sha256"].items():
        assert digest(SOURCE / relative) == expected_hash, relative
    csv_write(DATA / "confirmation_units.csv", rows)
    totals = []
    phases = []
    for arm, item in analysis["summary"].items():
        totals.append(dict(arm=arm, label=LABELS[arm], **item["all"],
                           decision_mean_s=item["decision_mean"], decision_p95_s=item["decision_p95"],
                           over_1s=item["over_second"], engine_steps=item["deterministic_engine_steps"],
                           forecast_calls=item["forecast_calls"]))
        for phase in ("pre", "peak", "post", "nonpeak"):
            phases.append(dict(arm=arm, phase=phase, **item[phase]))
    csv_write(DATA / "confirmation_summary.csv", totals)
    csv_write(DATA / "arrival_cohorts.csv", phases)
    contrast_rows = []
    for pair, item in analysis["comparisons"].items():
        value = item["all"]["dsr"]
        contrast_rows.append(dict(pair=pair, primary=item["is_primary_family"],
                                 effect_pp=value["mean"] * 100,
                                 ci95_low_pp=value["ci95"][0] * 100,
                                 ci95_high_pp=value["ci95"][1] * 100,
                                 bonf31_low_pp=value["bonferroni_all_comparisons95"][0] * 100,
                                 bonf31_high_pp=value["bonferroni_all_comparisons95"][1] * 100,
                                 parent0_pp=value["by_parent"][0] * 100,
                                 parent1_pp=value["by_parent"][1] * 100,
                                 parent2_pp=value["by_parent"][2] * 100))
    csv_write(DATA / "paired_dsr_effects.csv", contrast_rows)
    csv_write(DATA / "validation_curves.csv", [dict(arm=arm, **row) for arm, points in curves["curves"].items()
              for row in points])
    provenance = dict(
        status="PASS", data_role="post_hoc_manuscript_source_data", freeze_manifest_sha256=freeze_hash,
        inference_scope="Conditional on 3 fixed parent lineages; 8 paired workloads x 2 action streams",
        raw_confirmation_records=816, sources_checked=15, experiment_models_checked=51,
        parent_checkpoint_scope="3 scorer files locally available; 3 proposer .pt files absent",
        training_replay_scope="Existing audit records only; original rl_rollouts absent locally",
        selected_display_scope="All 17 arms in main table/DSR figure; focused secondary displays explicitly identified",
        sources={str(path.relative_to(REPO_ROOT)).replace("\\", "/"): digest(path) for path in source_files},
    )
    (DATA / "source_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return analysis, curves, parents


def table(caption, english, label, columns, header, rows, note=""):
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             rf"\bicaption{{{caption}}}{{{english}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\vspace{0.4em}\begin{minipage}{0.96\textwidth}\footnotesize " + note + r"\end{minipage}"]
    lines += [r"\end{table}"]
    return "\n".join(lines) + "\n"


def export_tables(analysis):
    summary = analysis["summary"]
    rows = []
    for arm in LABELS:
        c = summary[arm]["all"]
        rows.append([LABELS[arm], f"{c['success']:,}", f"{100*c['dsr']:.2f}",
                     f"{100*c['admission']:.2f}", f"{100*c['conditional']:.2f}"])
    text = table("动态排队确认集的全部十七个评估臂", "All seventeen evaluation arms on the dynamic-queue confirmation set",
                 "tab:ch5-dynamic-main", "lrrrr", r"方法 & 成功数 & DSR/\% & 接纳率/\% & 条件完成率/\%", rows,
                 "每臂为三个固定父谱系、八个工作负载和两个动作流，共48回合、252024个生成请求。表内为池化请求计数比率；请求与回合不作为独立训练重复。")
    (TABLES / "ch5_dynamic_main.tex").write_text(text, encoding="utf-8")
    pairs = [
        ("joint_rl_finetuned--joint_distilled", "联合微调相对联合蒸馏"),
        ("joint_rl_finetuned--decomposed_rl_finetuned", "联合微调相对去耦微调"),
        ("joint_rl_finetuned--joint_no_expectation_rl_finetuned", "联合微调相对无期望微调"),
        ("joint_rl_mpc--quality_amortized", "微调 MPC 相对在线规划"),
        ("joint_rl_finetuned--quality_amortized", "联合微调相对在线规划"),
        ("joint_rl_mpc--distilled_top6_mpc", "微调 MPC 相对蒸馏 MPC"),
    ]
    rows = []
    for key, label in pairs:
        comp = analysis["comparisons"][key]
        value = comp["all"]["dsr"]
        interval = lambda k: f"[{value[k][0]*100:+.2f}, {value[k][1]*100:+.2f}]"
        rows.append([label, "主" if comp["is_primary_family"] else "补", f"{value['mean']*100:+.2f}", interval("ci95"), interval("bonferroni_all_comparisons95")])
    text = table("关键比较的全程 DSR 配对差与置信区间", "Paired full-horizon DSR differences and confidence intervals",
                 "tab:ch5-dynamic-effects", "lcrrr", r"比较 & 检验族 & 差值 & 单比较95\%区间 & $M=31$区间", rows,
                 "单位均为百分点。主表示原八项主比较之一，补表示补充比较；最后一列采用全31项DSR比较的Bonferroni区间。Bootstrap为10000次工作负载与动作流配对重采样，条件于三个固定父谱系。")
    (TABLES / "ch5_dynamic_effects.tex").write_text(text, encoding="utf-8")
    arms = ["uncoordinated_init", "joint_distilled", "joint_rl_finetuned", "prior_top6_mpc", "distilled_top6_mpc", "joint_rl_mpc", "quality_amortized"]
    rows = [[LABELS[a], f"{summary[a]['decision_mean']:.3f}", f"{summary[a]['decision_p95']:.3f}", f"{100*summary[a]['over_second']:.1f}",
             str(summary[a]["deterministic_engine_steps"]), str(summary[a]["forecast_calls"])] for a in arms]
    text = table("主要控制方式的平均耗时、尾延迟与前瞻计数", "Decision time, tail latency and lookahead counters",
                 "tab:ch5-dynamic-cost", "lrrrrr", r"方法 & 均值/s & p95/s & $>1$s/\% & 确定性步数 & 预测分支数", rows,
                 "计时来自本轮评估记录，不是隔离硬件基准。预测分支使用Engine克隆推进，但记在预测分支计数中，不能把其确定性计数为零理解为没有仿真。物理环境未计入墙钟等待。")
    (TABLES / "ch5_dynamic_cost.tex").write_text(text, encoding="utf-8")


def axes_style(ax, axis="x"):
    ax.grid(axis=axis, color="#E1E5E9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)


def save(fig, name, require_matplotlib_panel_alignment):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=QA / f"{name}.alignment.json", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(FIGURES / f"{name}.pdf", facecolor="white")
    fig.savefig(FIGURES / f"{name}.svg", facecolor="white")
    fig.savefig(FIGURES / f"{name}.png", dpi=600, facecolor="white")
    plt.close(fig)


def render(analysis, curves, parents, alignment):
    s = analysis["summary"]
    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    fig.subplots_adjust(left=.10, right=.97, bottom=.24, top=.88)
    ax.axvspan(50, 100, color="#E7EDF4", zorder=0)
    ax.axvspan(200, 204, color="#E8E8E8", zorder=0)
    ax.step([0, 50, 100, 200, 204], [15, 60, 15, 0, 0], where="post", color=BLUE, lw=1.8)
    ax.text(24, 31, "突发前", ha="center")
    ax.text(75, 31, "高峰", ha="center")
    ax.text(150, 31, "突发后", ha="center")
    ax.text(195, 67, "排空", ha="right", va="bottom")
    ax.set(xlim=(0, 205), ylim=(-3, 74), xlabel="物理时间 / s", ylabel="期望到达率 / 请求每秒")
    ax.set_xticks([0, 50, 100, 150, 200])
    axes_style(ax, "y")
    save(fig, "ch5_dynamic_01_protocol", alignment)

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    fig.subplots_adjust(left=.11, right=.97, bottom=.20, top=.78)
    chosen = ["joint_rl", "decomposed_rl_finetuned", "joint_no_expectation_rl_finetuned", "joint_rl_finetuned"]
    colors = [GRAY, "#4B8C82", ORANGE, BLUE]
    for arm, color in zip(chosen, colors):
        points = curves["curves"][arm]
        ax.plot([v["iteration"] for v in points], [100*v["validation_dsr"] for v in points],
                marker="o", markersize=3.5, lw=1.4, color=color, label=LABELS[arm])
    ax.set(xlabel="更新轮次（0 为各自初值）", ylabel="验证集 DSR / %", xticks=range(5), ylim=(43, 49.0))
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=2, fontsize=7.6)
    axes_style(ax, "y")
    save(fig, "ch5_dynamic_02_validation", alignment)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    fig.subplots_adjust(left=.29, right=.94, bottom=.12, top=.91)
    arms = list(LABELS)
    for y, arm in enumerate(arms):
        for p, marker in enumerate(("o", "s", "^")):
            suc, gen = parents[arm][p]
            ax.scatter(suc/gen*100, y + (p-1)*.12, s=17, marker=marker, color="#B9BEC5", zorder=2)
        ax.scatter(s[arm]["all"]["dsr"]*100, y, s=38, marker="D", color=COLORS[arm], zorder=3)
    ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
    ax.invert_yaxis()
    ax.set(xlabel="全程 DSR / %", xlim=(42.5, 50.5))
    ax.text(.5, 1.04, "菱形：总体请求比率    灰点：三个固定父谱系", transform=ax.transAxes, ha="center")
    axes_style(ax)
    save(fig, "ch5_dynamic_03_all_arms", alignment)

    keys = list(analysis["primary_comparison_pairs"]) + ["joint_rl_finetuned--quality_amortized", "joint_rl_mpc--distilled_top6_mpc"]
    short = {
        "joint_rl--uncoordinated_init": "冷启动相对初始先验",
        "joint_rl--decomposed_rl": "联合冷启动相对去耦冷启动",
        "joint_rl--joint_distilled": "冷启动相对联合蒸馏",
        "joint_distilled--decomposed_distilled": "联合蒸馏相对去耦蒸馏",
        "joint_rl_finetuned--joint_distilled": "联合微调相对联合蒸馏",
        "joint_rl_finetuned--decomposed_rl_finetuned": "联合微调相对去耦微调",
        "joint_rl_finetuned--joint_no_expectation_rl_finetuned": "联合微调相对无期望微调",
        "joint_rl_mpc--quality_amortized": "微调 MPC 相对在线规划",
        "joint_rl_finetuned--quality_amortized": "联合微调相对在线规划（补）",
        "joint_rl_mpc--distilled_top6_mpc": "微调 MPC 相对蒸馏 MPC（补）",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    fig.subplots_adjust(left=.39, right=.96, bottom=.16, top=.90)
    for y, key in enumerate(keys):
        value = analysis["comparisons"][key]["all"]["dsr"]
        lo, hi = np.asarray(value["bonferroni_all_comparisons95"]) * 100
        color = BLUE if lo > 0 else GRAY
        ax.plot([lo, hi], [y, y], lw=1.7, color=color)
        ax.scatter(value["mean"]*100, y, s=24, color=color, zorder=3)
    ax.axvline(0, color="#333333", ls="--", lw=.7)
    ax.set_yticks(range(len(keys)), [short[k] for k in keys])
    ax.invert_yaxis()
    ax.set(xlabel="全程 DSR 配对差 / 百分点", xlim=(-5.2, 2.1))
    ax.text(.5, 1.04, "点：配对差    线：全部 31 项比较的 Bonferroni 区间", transform=ax.transAxes, ha="center", fontsize=7.5)
    axes_style(ax)
    save(fig, "ch5_dynamic_04_paired_effects", alignment)

    arms = ["uncoordinated_init", "joint_distilled", "decomposed_rl_finetuned", "joint_rl_finetuned", "quality_amortized", "joint_rl_mpc"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.2, 3.4))
    fig.subplots_adjust(left=.23, right=.97, bottom=.25, top=.76, wspace=.26)
    y = np.arange(len(arms))
    success = np.array([s[a]["all"]["dsr"]*100 for a in arms])
    admitted = np.array([s[a]["all"]["admission"]*100 for a in arms])
    axes[0].barh(y, success, color=BLUE, height=.55, label="按时完成")
    axes[0].barh(y, admitted-success, left=success, color=ORANGE, height=.55, label="接纳后失败")
    axes[0].barh(y, 100-admitted, left=admitted, color="#DDE1E6", height=.55, label="未接纳")
    axes[0].set_yticks(y, [LABELS[a] for a in arms])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 100), xlabel="生成请求的结局 / %")
    axes[0].legend(loc="lower left", bbox_to_anchor=(-.01, 1.03), ncol=1, fontsize=7.3)
    for phase, marker, color, label in [("nonpeak", "o", GRAY, "非高峰到达"), ("peak", "s", PURPLE, "高峰到达")]:
        axes[1].scatter([s[a][phase]["dsr"]*100 for a in arms], y, marker=marker, s=25, color=color, label=label)
    axes[1].set(xlim=(30, 65), xlabel="按到达阶段分组的 DSR / %")
    axes[1].legend(loc="lower left", bbox_to_anchor=(-.01, 1.03), fontsize=7.3)
    for index, ax in enumerate(axes):
        ax.text(0, 1.32, chr(97+index), transform=ax.transAxes, weight="bold", fontsize=9)
        axes_style(ax)
    save(fig, "ch5_dynamic_05_service_outcomes", alignment)

    arms = ["uncoordinated_init", "joint_distilled", "joint_rl_finetuned", "prior_top6_mpc", "distilled_top6_mpc", "joint_rl_mpc", "quality_amortized"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.2, 3.65))
    fig.subplots_adjust(left=.24, right=.97, bottom=.23, top=.80, wspace=.26)
    y = np.arange(len(arms))
    for i, arm in enumerate(arms):
        axes[0].plot([s[arm]["decision_mean"], s[arm]["decision_p95"]], [i, i], color="#BBC1C9", lw=1.0)
    axes[0].scatter([s[a]["decision_mean"] for a in arms], y, s=23, color=BLUE, label="均值")
    axes[0].scatter([s[a]["decision_p95"] for a in arms], y, s=27, marker="s", color=ORANGE, label="p95")
    axes[0].set_yticks(y, [LABELS[a] for a in arms])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 6.4), xlabel="每步决策耗时 / s")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2)
    axes[1].barh(y, [s[a]["over_second"]*100 for a in arms], color="#8798AF", height=.55)
    axes[1].set(xlim=(0, 40), xlabel="耗时超过 1 s 的决策占比 / %")
    for index, ax in enumerate(axes):
        ax.text(0, 1.19, chr(97+index), transform=ax.transAxes, weight="bold", fontsize=9)
        axes_style(ax)
    save(fig, "ch5_dynamic_06_decision_cost", alignment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-scripts", type=Path, required=True, help="External nature-figure QA scripts directory")
    args = parser.parse_args()
    sys.path.insert(0, str(args.skill_scripts.resolve()))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    for folder in (DATA, TABLES, QA):
        folder.mkdir(parents=True, exist_ok=True)
    setup_thesis_style(font_size=8.2)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
                        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 600})
    analysis, curves, parents = load_and_export()
    export_tables(analysis)
    render(analysis, curves, parents, require_matplotlib_panel_alignment)
    print("Exported 6 figures, 3 LaTeX tables and traceable data for 816 confirmation episodes.")


if __name__ == "__main__":
    main()
