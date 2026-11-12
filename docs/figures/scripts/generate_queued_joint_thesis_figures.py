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
from matplotlib.lines import Line2D
import numpy as np
from thesis_figure_style import resolve_qa_scripts_dir, setup_thesis_style
from queued_joint_mechanism_figures import render_all_chapter5_additions

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


def matches_frozen_digest(path: Path, expected_hash: str) -> bool:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected_hash:
        return True
    if path.suffix in {".py", ".json", ".csv", ".md", ".txt"}:
        lf = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(lf).hexdigest() == expected_hash:
            return True
        crlf = lf.replace(b"\n", b"\r\n")
        if hashlib.sha256(crlf).hexdigest() == expected_hash:
            return True
    return False


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
    stage_buckets = {arm: {"pre": [], "peak": [], "post": []} for arm in LABELS}
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
        arm_buckets = stage_buckets[record["arm"]]
        for step_rec in record["trajectory"]:
            t_slot = int(step_rec["time"])
            dt = float(step_rec["decision_seconds"])
            if t_slot < 40:
                arm_buckets["pre"].append(dt)
            elif t_slot < 105:
                arm_buckets["peak"].append(dt)
            else:
                arm_buckets["post"].append(dt)
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
        assert matches_frozen_digest(REPO_ROOT / relative, expected_hash), relative
    for relative, expected_hash in manifest["model_sha256"].items():
        assert matches_frozen_digest(SOURCE / relative, expected_hash), relative
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
    stage_times = {
        arm: {st: float(np.mean(vals)) for st, vals in buckets.items()}
        for arm, buckets in stage_buckets.items()
    }
    csv_write(
        DATA / "stage_decision_cost.csv",
        [
            dict(
                arm=arm,
                label=LABELS[arm],
                pre_mean_s=stage_times[arm]["pre"],
                peak_mean_s=stage_times[arm]["peak"],
                post_mean_s=stage_times[arm]["post"],
            )
            for arm in LABELS
        ],
    )
    supplemental_files = [
        DATA / name
        for name in (
            "cf0_ablation_confirmation_analysis.json",
            "cf0_ablation_summary.csv",
            "peak_queue_evolution_955000.json",
            "stage_decision_cost.csv",
        )
        if (DATA / name).exists()
    ]
    provenance = dict(
        status="PASS", data_role="post_hoc_manuscript_source_data", freeze_manifest_sha256=freeze_hash,
        inference_scope="Conditional on 3 fixed upstream pretrained model groups; 8 paired workloads x 2 action streams",
        raw_confirmation_records=816, sources_checked=15, experiment_models_checked=51,
        parent_checkpoint_scope="3 scorer files locally available; 3 proposer .pt files absent",
        training_replay_scope="Existing audit records only; original rl_rollouts absent locally",
        selected_display_scope="All 17 control schemes in main table/DSR figure; focused secondary displays explicitly identified",
        cf0_ablation_provenance="Table 5-7 w_cf=0 row archived from git commit 97b34d7 (queued_joint_rl_20261106 pre-v2 run on identical 816-episode confirmation workloads 955000-955007)",
        sources={str(path.relative_to(REPO_ROOT)).replace("\\", "/"): digest(path) for path in source_files},
        supplemental_data_sha256={
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): digest(path) for path in supplemental_files
        },
    )
    (DATA / "source_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return analysis, curves, parents, stage_times


def table(caption, english, label, columns, header, rows, note=""):
    lines = [r"\begin{table}[!htbp]", r"\centering",
             rf"\bicaption{{{caption}}}{{{english}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\sysutabnote{" + note + r"}"]
    lines += [r"\end{table}"]
    return "\n".join(lines) + "\n"


def export_tables(analysis):
    summary = analysis["summary"]
    rows = []
    for arm in LABELS:
        c = summary[arm]["all"]
        rows.append([LABELS[arm], f"${c['success']:,}$", f"${100*c['dsr']:.2f}$",
                     f"${100*c['admission']:.2f}$", f"${100*c['conditional']:.2f}$"])
    text = table("动态排队盲测确认集上全部十七种控制方案的服务性能对比", "Service performance of all seventeen control schemes on the dynamic-queue confirmation set",
                 "tab:ch5-dynamic-main", "lrrrr", r"控制方案 & 按时完成数 $S$ & DSR/\% & 接纳率 $\alpha$/\% & 条件完成率 $\eta$/\%", rows,
                 r"每种控制方案均在三组固定上游预训练模型、八个盲测工作负载和两个动作流（共 $48$ 回合、全集累计生成 $G=252{,}024$ 个请求）上配对评估。表内数值为全集累计请求计数比率（满足精确计数恒等式 $\mathrm{DSR}=\alpha\eta=(A/G)\times(S/A)$，例如联合微调 $122{,}842/252{,}024 = (132{,}524/252{,}024)\times(122{,}842/132{,}524) = 48.7422\%$，两位小数显示下 $52.58\%\times 92.69\%=48.736\%$ 的千分位尾差仅来自四舍五入显示）；单条请求与回合不作为独立训练重复样本。")
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
        interval = lambda k: f"$[{value[k][0]*100:+.2f}, {value[k][1]*100:+.2f}]$"
        rows.append([label, "主" if comp["is_primary_family"] else "补", f"${value['mean']*100:+.2f}$", interval("ci95"), interval("bonferroni_all_comparisons95")])
    text = table("关键比较的全程 DSR 配对差与置信区间", "Paired full-horizon DSR differences and confidence intervals",
                 "tab:ch5-dynamic-effects", "lcrrr", r"比较对象 & 检验族 & 差值 & 单比较 $95\%$ 区间 & $N_{\mathrm{comp}}=31$ 区间", rows,
                 r"单位均为百分点（pp）。“主”表示预设八项主比较族之一，“补”表示补充机制比较；最后一列采用全部 $N_{\mathrm{comp}}=31$ 项 DSR 比较的 Bonferroni 同时置信区间。置信区间由 $10{,}000$ 次以工作负载为聚类单位的分块 Bootstrap 重采样计算，条件于三组固定上游预训练模型。")
    (TABLES / "ch5_dynamic_effects.tex").write_text(text, encoding="utf-8")
    arms = ["uncoordinated_init", "joint_distilled", "joint_rl_finetuned", "prior_top6_mpc", "distilled_top6_mpc", "joint_rl_mpc", "quality_amortized"]
    rows = [[LABELS[a], f"${summary[a]['decision_mean']:.3f}$", f"${summary[a]['decision_p95']:.3f}$", f"${100*summary[a]['over_second']:.1f}$",
             f"${summary[a]['deterministic_engine_steps']:,}$", f"${summary[a]['forecast_calls']:,}$"] for a in arms]
    text = table("主要控制方案的平均决策耗时、尾延迟与在线前瞻计数", "Mean decision time, tail latency and online lookahead counters",
                 "tab:ch5-dynamic-cost", "lrrrrr", r"控制方案 & 均值/$\mathrm{s}$ & $\mathrm{P95}/\mathrm{s}$ & $>1\,\mathrm{s}$/\% & 确定性步数 & 预测分支数", rows,
                 r"计时记录自单线程 CPU 限制下的多进程评估回合，非独立实时操作系统硬件基准。Top-6 MPC 的预测分支同样调用离散事件队列仿真器克隆推进，但单独记录于“预测分支数”列，故“确定性步数”为零不代表未执行在线仿真推演。物理队列推进未叠加决策计算等待时延。")
    (TABLES / "ch5_dynamic_cost.tex").write_text(text, encoding="utf-8")


def axes_style(ax, axis="both"):
    if axis in ("x", "y", "both"):
        ax.grid(True, axis=axis, color="#D5D9E2", linewidth=0.55, alpha=0.82)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.95)
        ax.spines[side].set_edgecolor("#22252A")
    has_x = len(ax.get_xticks()) > 0
    has_y = len(ax.get_yticks()) > 0
    ax.tick_params(axis="x", direction="in", length=3.0, width=0.8, top=has_x, bottom=has_x)
    ax.tick_params(axis="y", direction="in", length=3.0, width=0.8, left=has_y, right=has_y)


def save(fig, name, require_matplotlib_panel_alignment, skill_scripts: Path):
    import io
    import subprocess
    import time

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=QA / f"{name}.alignment.json", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True,
    )
    pdf_path = FIGURES / f"{name}.pdf"
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 600)):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, dpi=dpi, facecolor="white")
        payload = buf.getvalue()
        target = FIGURES / f"{name}.{ext}"
        for attempt in range(4):
            try:
                target.write_bytes(payload)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(0.2 * (attempt + 1))
    plt.close(fig)
    res_txt = subprocess.run(
        [sys.executable, str(skill_scripts / "audit_pdf_text.py"), str(pdf_path), "--min-pt", "5", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    (QA / f"{name}.pdf_text.json").write_text(res_txt.stdout, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(skill_scripts / "audit_figure_collisions.py"), str(pdf_path), "--json-out", str(QA / f"{name}.collision.json")],
        check=True,
        capture_output=True,
        text=True,
    )


def render(analysis, curves, parents, stage_times, alignment, skill_scripts: Path):
    s = analysis["summary"]
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    fig.subplots_adjust(left=.10, right=.97, bottom=.24, top=.88)
    ax.axvspan(50, 100, color="#E7EDF4", zorder=0)
    ax.axvspan(200, 204, color="#E8E8E8", zorder=0)
    ax.step([0, 50, 100, 200, 204], [15, 60, 15, 0, 0], where="post", color=BLUE, lw=1.8)
    ax.text(24, 31, "突发前", ha="center", fontsize=8.0)
    ax.text(75, 31, "高峰", ha="center", fontsize=8.0)
    ax.text(150, 31, "突发后", ha="center", fontsize=8.0)
    ax.text(199, 64, "末尾排空 [200, 204] s", ha="right", va="bottom", fontsize=7.8)
    ax.set(xlim=(0, 205), ylim=(-3, 74), xlabel="物理时间 / s", ylabel="期望到达率 / 请求每秒")
    ax.set_xticks([0, 50, 100, 150, 200])
    axes_style(ax, "none")
    save(fig, "ch5_dynamic_01_protocol", alignment, skill_scripts)

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    fig.subplots_adjust(left=.11, right=.97, bottom=.20, top=.76)
    chosen = ["joint_rl", "decomposed_rl_finetuned", "joint_no_expectation_rl_finetuned", "joint_rl_finetuned"]
    colors = [GRAY, "#4B8C82", ORANGE, BLUE]
    for arm, color in zip(chosen, colors):
        points = curves["curves"][arm]
        ax.plot([v["iteration"] for v in points], [100*v["validation_dsr"] for v in points],
                marker="o", markersize=3.5, lw=1.4, color=color, label=LABELS[arm])
    ax.set(xlabel="更新轮次（0 为各自初值）", ylabel="验证集 DSR / %", xticks=range(5), ylim=(43, 49.0))
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=2, fontsize=7.6, frameon=True, fancybox=False, edgecolor="#4A525D")
    axes_style(ax, "both")
    save(fig, "ch5_dynamic_02_validation", alignment, skill_scripts)

    fig, ax = plt.subplots(figsize=(6.8, 5.85))
    fig.subplots_adjust(left=.29, right=.95, bottom=.10, top=.85)
    arms = list(LABELS)
    for y, arm in enumerate(arms):
        for p, marker in enumerate(("o", "s", "^")):
            suc, gen = parents[arm][p]
            ax.scatter(suc/gen*100, y + (p-1)*.12, s=17, marker=marker, color="#B9BEC5", zorder=2)
        ax.scatter(s[arm]["all"]["dsr"]*100, y, s=38, marker="D", color=COLORS[arm], zorder=3)
    ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
    ax.invert_yaxis()
    ax.set(xlabel="全程 DSR / %", xlim=(42.5, 50.5))
    arm_handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PURPLE, markersize=5.8, label="微调 Top-6 MPC"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=BLUE, markersize=5.8, label="联合微调"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=ORANGE, markersize=5.8, label="联合蒸馏"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=GRAY, markersize=5.8, label="其余控制方案"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B9BEC5", markersize=5.0, label="三组上游模型（圆/方/三角）"),
    ]
    ax.legend(handles=arm_handles, loc="lower center", bbox_to_anchor=(0.45, 1.025), ncol=3, fontsize=7.3, frameon=True, fancybox=False, edgecolor="#4A525D", columnspacing=0.9, handletextpad=0.3)
    axes_style(ax, "both")
    save(fig, "ch5_dynamic_03_all_arms", alignment, skill_scripts)

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
    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    fig.subplots_adjust(left=.39, right=.96, bottom=.15, top=.86)
    for y, key in enumerate(keys):
        value = analysis["comparisons"][key]["all"]["dsr"]
        lo, hi = np.asarray(value["bonferroni_all_comparisons95"]) * 100
        color = BLUE if lo > 0 else GRAY
        ax.plot([lo, hi], [y, y], lw=1.7, color=color, zorder=2)
        ax.scatter(value["mean"]*100, y, s=24, color=color, zorder=3)
    ax.axvline(0, color="#333333", ls="--", lw=.8, zorder=2)
    ax.set_yticks(range(len(keys)), [short[k] for k in keys])
    ax.invert_yaxis()
    ax.set(xlabel="全程 DSR 配对差 / 百分点", xlim=(-5.2, 2.1))
    eff_handles = [
        Line2D([0], [0], color=BLUE, marker="o", markersize=4.8, lw=1.6, label="显著正增益（$N_{\\mathrm{comp}}=31$ 下界 $>0$）"),
        Line2D([0], [0], color=GRAY, marker="o", markersize=4.8, lw=1.6, label="其余比较（$N_{\\mathrm{comp}}=31$ Bonferroni 区间）"),
    ]
    ax.legend(handles=eff_handles, loc="lower center", bbox_to_anchor=(0.45, 1.025), ncol=2, fontsize=7.3, frameon=True, fancybox=False, edgecolor="#4A525D", columnspacing=1.0)
    axes_style(ax, "both")
    save(fig, "ch5_dynamic_04_paired_effects", alignment, skill_scripts)

    arms = ["uncoordinated_init", "joint_distilled", "decomposed_rl_finetuned", "joint_rl_finetuned", "quality_amortized", "joint_rl_mpc"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.2, 3.45))
    fig.subplots_adjust(left=.23, right=.97, bottom=.23, top=.78, wspace=.26)
    y = np.arange(len(arms))
    success = np.array([s[a]["all"]["dsr"]*100 for a in arms])
    admitted = np.array([s[a]["all"]["admission"]*100 for a in arms])
    axes[0].barh(y, success, color=BLUE, height=.55, label="按时完成", zorder=2)
    axes[0].barh(y, admitted-success, left=success, color=ORANGE, height=.55, label="接纳后失败", zorder=2)
    axes[0].barh(y, 100-admitted, left=admitted, color="#DDE1E6", height=.55, label="未接纳", zorder=2)
    axes[0].set_yticks(y, [LABELS[a] for a in arms])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 100), xlabel="生成请求的结局 / %")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.035), ncol=3, fontsize=7.0, frameon=True, fancybox=False, edgecolor="#4A525D", columnspacing=0.55, handlelength=1.0, handletextpad=0.3)
    for phase, marker, color, label in [("nonpeak", "o", GRAY, "非高峰到达"), ("peak", "s", PURPLE, "高峰到达")]:
        axes[1].scatter([s[a][phase]["dsr"]*100 for a in arms], y, marker=marker, s=25, color=color, label=label, zorder=3)
    axes[1].set(xlim=(30, 65), xlabel="按到达阶段分组的 DSR / %")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.035), ncol=2, fontsize=7.2, frameon=True, fancybox=False, edgecolor="#4A525D", columnspacing=0.75, handletextpad=0.3)
    for index, ax in enumerate(axes):
        ax.text(0, 1.23, f"({chr(97+index)})", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
        axes_style(ax, "both")
    save(fig, "ch5_dynamic_05_service_outcomes", alignment, skill_scripts)

    pareto_arms = ["joint_distilled", "joint_rl_finetuned", "distilled_top6_mpc", "joint_rl_mpc"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55))
    fig.subplots_adjust(left=.09, right=.98, bottom=.21, top=.78, wspace=.28)

    ax0 = axes[0]
    ax0.axvline(1.0, color="#6B7280", ls="--", lw=0.85, zorder=1)
    ax0.text(1.03, 44.05, "1.0 s 时隙参考", fontsize=6.9, color="#4A525D", ha="left", va="bottom")
    px = [s[a]["decision_mean"] for a in pareto_arms]
    py = [s[a]["all"]["dsr"] * 100 for a in pareto_arms]
    ax0.plot(px, py, color=BLUE, lw=1.25, ls="-", zorder=2)

    point_specs = [
        ("uncoordinated_init", "o", GRAY, 26, (0.50, 44.30), "left", "center", "初始先验"),
        ("prior_top6_mpc", "s", GRAY, 26, (0.65, 45.46), "right", "center", "先验 Top-6 MPC"),
        ("joint_distilled", "o", ORANGE, 34, (0.47, 47.74), "left", "top", "联合蒸馏"),
        ("joint_rl_finetuned", "o", BLUE, 38, (0.54, 48.46), "left", "top", "联合微调"),
        ("distilled_top6_mpc", "s", "#4B8C82", 32, (0.97, 48.78), "right", "top", "蒸馏 Top-6 MPC"),
        ("joint_rl_mpc", "D", PURPLE, 42, (0.97, 49.58), "right", "bottom", "微调 Top-6 MPC"),
        ("quality_amortized", "^", "#B5524C", 36, (1.55, 48.56), "right", "center", "在线规划基线"),
    ]
    for arm, marker, col, sz, (tx, ty), ha, va, lbl in point_specs:
        x_val = s[arm]["decision_mean"]
        y_val = s[arm]["all"]["dsr"] * 100
        ax0.scatter(x_val, y_val, marker=marker, s=sz, color=col, zorder=4)
        ax0.text(tx, ty, lbl, fontsize=6.9, color="#22252A", ha=ha, va=va, zorder=5)
    ax0.set(xlim=(0.28, 1.74), ylim=(43.7, 50.22), xlabel="每步平均决策耗时 / s", ylabel="全程 DSR / %")
    p_handles = [
        Line2D([0], [0], color=BLUE, lw=1.25, label="经验帕累托前沿"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=5.2, label="零前瞻神经策略"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PURPLE, markersize=5.2, label="神经引导 MPC"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#B5524C", markersize=5.4, label="在线规划基线"),
    ]
    ax0.legend(handles=p_handles, loc="lower center", bbox_to_anchor=(0.5, 1.035), ncol=2, fontsize=6.8,
               frameon=True, fancybox=False, edgecolor="#4A525D", columnspacing=0.65, handletextpad=0.3)

    ax1 = axes[1]
    stage_keys = ["pre", "peak", "post"]
    stage_labels = ["平峰期\n[0, 40) s", "拥塞窗口\n[40, 105) s", "恢复排空\n≥105 s"]
    bar_arms = [
        ("joint_distilled", "联合蒸馏", ORANGE),
        ("joint_rl_finetuned", "联合微调", BLUE),
        ("joint_rl_mpc", "微调 Top-6 MPC", PURPLE),
        ("quality_amortized", "在线规划基线", "#B5524C"),
    ]
    xg = np.arange(len(stage_keys))
    bw = 0.18
    ax1.axhline(1.0, color="#6B7280", ls="--", lw=0.85, zorder=1)
    for idx_b, (arm, lbl, col) in enumerate(bar_arms):
        offsets = xg + (idx_b - 1.5) * bw
        vals = [stage_times[arm][sk] for sk in stage_keys]
        ax1.bar(offsets, vals, width=bw * 0.92, color=col, label=lbl, zorder=3)
    ax1.set_xticks(xg, stage_labels)
    ax1.set(ylim=(0, 4.45), xlabel="运行阶段", ylabel="分阶段每步平均耗时 / s")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.035), ncol=2, fontsize=6.8,
               frameon=True, fancybox=False, edgecolor="#4A525D", columnspacing=0.65, handletextpad=0.3)

    for index, ax in enumerate(axes):
        ax.text(-0.12, 1.24, f"({chr(97+index)})", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    axes_style(ax0, "none")
    axes_style(ax1, "y")
    save(fig, "ch5_dynamic_06_decision_cost", alignment, skill_scripts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-scripts",
        type=Path,
        default=resolve_qa_scripts_dir(REPO_ROOT),
        help="External nature-figure QA scripts directory",
    )
    parser.add_argument(
        "--include-additions",
        action="store_true",
        help="Also render optional supplemental mechanism figures not included in main.tex",
    )
    args = parser.parse_args()
    skill_scripts = args.skill_scripts.resolve()
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    for folder in (DATA, TABLES, QA):
        folder.mkdir(parents=True, exist_ok=True)
    family = setup_thesis_style(font_size=8.2)
    plt.rcParams.update({"font.sans-serif": [family], "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 600})
    analysis, curves, parents, stage_times = load_and_export()
    export_tables(analysis)
    render(analysis, curves, parents, stage_times, require_matplotlib_panel_alignment, skill_scripts)
    if args.include_additions:
        render_all_chapter5_additions()
    print("Exported 6 active Chapter 5 figures, 3 LaTeX tables and traceable data for 816 confirmation episodes.")


if __name__ == "__main__":
    main()
