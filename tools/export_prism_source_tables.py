#!/usr/bin/env python3
"""Export thesis figure summaries to tab-delimited tables for GraphPad Prism.

Each table preserves the statistic shown in the current thesis figures.  Prism
can import these files directly and use the adjacent *_SD / CI columns for
error bars.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "prism_source_data"
FINAL = ROOT / "results" / "final_force_admission" / "evaluation_20260630" / "variant_statistics.json"
BASELINE = ROOT / "results" / "baseline_matrix_force_admission_20260626_resume" / "statistics_workload_balanced.json"
UE = ROOT / "results" / "sensitivity_force_admission" / "ue_count" / "summary.json"
CPU = ROOT / "results" / "sensitivity_force_admission" / "uav_cpu_scale" / "summary.json"
FIG33 = ROOT / "docs" / "figures" / "data" / "figure_3_3_main_configuration.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, rows: list[list[object]]) -> None:
    path = OUT / name.replace(".tsv", ".csv")
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def ms(item: dict, scale: float = 1.0) -> tuple[float, float]:
    return item["mean"] * scale, item.get("std", 0.0) * scale


def figure_3_3() -> None:
    data = load(FIG33)["configurations"]
    rows = [["Configuration", "Energy (10^6 J)", "Energy SD", "Fairness", "Fairness SD", "DSR", "DSR SD", "MBS load (%)", "MBS load SD"]]
    for item in data:
        rows.append([
            item["label"], item["energy_mj"]["mean"], item["energy_mj"]["std"],
            item["fairness"]["mean"], item["fairness"]["std"],
            item["dsr"]["mean"], item["dsr"]["std"],
            item["mbs_load_pct"]["mean"], item["mbs_load_pct"]["std"],
        ])
    write("图3-3_主配置性能工作点.tsv", rows)


def summary_table(name: str, data: dict, methods: list[str], labels: list[str], metrics: list[tuple[str, str, float]]) -> None:
    rows = [["Method"] + [v for _, label, _ in metrics for v in (label, f"{label} SD")]]
    for method, label in zip(methods, labels):
        row = [label]
        for key, _, scale in metrics:
            mean, std = ms(data["methods"][method]["metrics"][key], scale)
            row.extend((mean, std))
        rows.append(row)
    write(name, rows)


def final_figures() -> None:
    data = load(FINAL)
    summary_table(
        "图5-8_卸载与接入分布.tsv", data,
        ["main_upper_only", "main_lower_only_fixed_upper", "main_full_hierarchical"],
        ["上层+启发式", "固定上层+下层", "完整双层"],
        [("offloading_ratio_local_processed", "本地执行 (%)", 100), ("offloading_ratio_cooperative_processed", "协作执行 (%)", 100), ("offloading_ratio_mbs_processed", "MBS执行 (%)", 100), ("natural_coverage_service_ratio", "自然覆盖 (%)", 100), ("forced_admission_ratio", "兜底接入 (%)", 100)],
    )
    summary_table(
        "图5-9_下层消融实验.tsv", data,
        ["ablation_full", "ablation_no_mask", "ablation_no_lagrange", "ablation_no_attention"],
        ["完整下层", "无Mask", "无Lagrange", "无Attention"],
        [("reward", "系统奖励", 1), ("energy", "能耗 (10^6 J)", 1e-6), ("dsr_request_weighted", "DSR", 1), ("fairness_final", "服务公平性", 1), ("mbs_load_ratio_generated", "MBS负载 (%)", 100), ("offloading_ratio_cooperative_processed", "协作执行 (%)", 100)],
    )
    summary_table(
        "图5-10_DSR优先配置.tsv", data,
        ["main_full_hierarchical", "dsr_priority_full"], ["完整双层", "DSR优先"],
        [("reward", "系统奖励", 1), ("energy", "能耗 (10^6 J)", 1e-6), ("dsr_request_weighted", "DSR", 1), ("mbs_load_ratio_generated", "MBS负载 (%)", 100)],
    )


def baseline_figures() -> None:
    data = load(BASELINE)
    methods = ["random", "uniform", "ippo", "vanilla_mappo", "joint_mappo", "proposed"]
    labels = ["随机", "均匀", "IPPO", "Vanilla MAPPO", "Joint MAPPO", "本文方法"]
    metrics = [("reward", "系统奖励", 1), ("latency", "时延 (10^6)", 1e-6), ("energy", "能耗 (10^6 J)", 1e-6), ("fairness_final", "公平性", 1), ("offline_rate_final", "离线率 (%)", 100), ("dsr_request_weighted", "请求加权DSR", 1), ("offloading_ratio_local_processed", "本地执行 (%)", 100), ("offloading_ratio_cooperative_processed", "协作执行 (%)", 100), ("offloading_ratio_mbs_processed", "MBS执行 (%)", 100), ("mbs_load_ratio_generated", "MBS负载 (%)", 100), ("processed_request_ratio", "处理/生成 (%)", 100), ("deadline_satisfied_per_processed", "达期/处理 (%)", 100)]
    summary_table("图5-14_5-15_扩展基线与可靠性.tsv", data, methods, labels, metrics)


def sensitivity(name: str, data_path: Path, x_label: str) -> None:
    data = load(data_path)
    variables = sorted(data["variables"], key=float)
    rows = [[x_label, "启发式 DSR", "启发式 DSR SD", "普通MAPPO DSR", "普通MAPPO DSR SD", "本文方法 DSR", "本文方法 DSR SD", "启发式 EEE", "启发式 EEE SD", "普通MAPPO EEE", "普通MAPPO EEE SD", "本文方法 EEE", "本文方法 EEE SD", "启发式 Fairness", "启发式 Fairness SD", "普通MAPPO Fairness", "普通MAPPO Fairness SD", "本文方法 Fairness", "本文方法 Fairness SD"]]
    methods = [("heuristic", "启发式"), ("vanilla_mappo", "普通MAPPO"), ("proposed", "本文方法")]
    for variable in variables:
        row: list[object] = [float(variable)]
        for metric in ("deadline_satisfaction_rate", "energy_efficiency", "fairness"):
            for method, _ in methods:
                item = data["variables"][variable][method]["stats"][metric]
                row.extend((item["mean"], item["std"]))
        rows.append(row)
    write(name, rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_3_3()
    final_figures()
    baseline_figures()
    sensitivity("图5-16_地面节点数量敏感性.tsv", UE, "UE数量")
    sensitivity("图5-17_UAV算力敏感性.tsv", CPU, "UAV算力缩放")
    print(f"Wrote {len(list(OUT.glob('*.csv')))} Prism source tables to {OUT}")


if __name__ == "__main__":
    main()
