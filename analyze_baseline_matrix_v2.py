#!/usr/bin/env python3
"""
Baseline Matrix v2 statistical analysis.

The statistical unit is not an individual eval episode. For learning methods,
one unit is a (training_seed, workload_seed) aggregate over eval_episodes. For
non-learning methods, one unit is a workload_seed aggregate over eval_episodes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

DEFAULT_INPUT_DIR = "results/baseline_matrix_v2"
LEARNING_METHODS = ["ippo", "vanilla_mappo", "joint_mappo", "proposed"]
NON_LEARNING_METHODS = ["random", "uniform"]

TABLE_METRICS = [
    "reward",
    "latency",
    "energy",
    "energy_efficiency_global",
    "fairness_final",
    "fairness_step_mean",
    "offline_rate_final",
    "offline_rate_step_mean",
    "dsr_request_weighted",
    "offloading_ratio_local_processed",
    "offloading_ratio_cooperative_processed",
    "offloading_ratio_mbs_processed",
    "mbs_load_ratio_generated",
    "processed_request_ratio",
    "deadline_satisfied_per_processed",
]

METRIC_DIRECTION = {
    "reward": "up",
    "latency": "down",
    "energy": "down",
    "energy_efficiency_global": "up",
    "fairness_final": "up",
    "fairness_step_mean": "up",
    "offline_rate_final": "down",
    "offline_rate_step_mean": "down",
    "dsr_request_weighted": "up",
    "offloading_ratio_local_processed": "neutral",
    "offloading_ratio_cooperative_processed": "neutral",
    "offloading_ratio_mbs_processed": "neutral",
    "mbs_load_ratio_generated": "down",
    "processed_request_ratio": "up",
    "deadline_satisfied_per_processed": "up",
}

ADDITIVE_FIELDS = [
    "reward",
    "latency",
    "energy",
    "service_requests_generated",
    "service_requests_processed",
    "deadline_satisfied_service_requests",
    "service_offloads_local",
    "service_offloads_cooperative",
    "service_offloads_mbs",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_data(input_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {method: {} for method in LEARNING_METHODS + NON_LEARNING_METHODS}

    for method in LEARNING_METHODS:
        method_dir = input_dir / method
        if not method_dir.exists():
            continue
        for seed_dir in method_dir.glob("seed_*"):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            data[method].setdefault(seed, {})
            for metrics_file in seed_dir.glob("workload_*_metrics.json"):
                try:
                    workload_seed = int(metrics_file.stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                data[method][seed][workload_seed] = read_json(metrics_file)

        # Backward compatibility for old nested output_dir/method/method layout.
        nested = method_dir / method
        if nested.exists():
            for metrics_file in nested.glob("metrics_seed_*_workload_*.json"):
                parts = metrics_file.stem.split("_")
                try:
                    seed = int(parts[2])
                    workload_seed = int(parts[4])
                except (IndexError, ValueError):
                    continue
                data[method].setdefault(seed, {})[workload_seed] = read_json(metrics_file)

    for method in NON_LEARNING_METHODS:
        method_dir = input_dir / method
        if not method_dir.exists():
            continue
        for metrics_file in method_dir.glob("workload_*_metrics.json"):
            try:
                workload_seed = int(metrics_file.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            data[method][workload_seed] = read_json(metrics_file)

        nested = method_dir / method
        if nested.exists():
            for metrics_file in nested.glob("metrics_seed_*_workload_*.json"):
                parts = metrics_file.stem.split("_")
                try:
                    workload_seed = int(parts[4])
                except (IndexError, ValueError):
                    continue
                data[method][workload_seed] = read_json(metrics_file)

    return {method: units for method, units in data.items() if units}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def mean_episode_field(episodes: list[dict[str, Any]], field: str) -> float:
    vals = [float(ep[field]) for ep in episodes if field in ep]
    return float(np.mean(vals)) if vals else 0.0


def unit_from_metrics(metrics: dict[str, Any], *, method: str, seed: int | None, workload_seed: int) -> dict[str, Any]:
    episodes = metrics.get("per_episode", [])
    if not episodes:
        aggregate = metrics.get("aggregate", {})
        unit_metrics = {metric: float(aggregate.get(metric, {}).get("mean", 0.0)) for metric in TABLE_METRICS}
    else:
        sums = {field: float(sum(float(ep.get(field, 0.0)) for ep in episodes)) for field in ADDITIVE_FIELDS}
        processed = sums["service_requests_processed"]
        generated = sums["service_requests_generated"]
        satisfied = sums["deadline_satisfied_service_requests"]
        energy = sums["energy"]

        unit_metrics = {
            "reward": mean_episode_field(episodes, "reward"),
            "latency": mean_episode_field(episodes, "latency"),
            "energy": mean_episode_field(episodes, "energy"),
            "energy_efficiency_global": safe_div(satisfied, energy),
            "fairness_final": mean_episode_field(episodes, "fairness_final"),
            "fairness_step_mean": mean_episode_field(episodes, "fairness_step_mean"),
            "offline_rate_final": mean_episode_field(episodes, "offline_rate_final"),
            "offline_rate_step_mean": mean_episode_field(episodes, "offline_rate_step_mean"),
            "dsr_request_weighted": safe_div(satisfied, generated),
            "offloading_ratio_local_processed": safe_div(sums["service_offloads_local"], processed),
            "offloading_ratio_cooperative_processed": safe_div(sums["service_offloads_cooperative"], processed),
            "offloading_ratio_mbs_processed": safe_div(sums["service_offloads_mbs"], processed),
            "mbs_load_ratio_generated": safe_div(sums["service_offloads_mbs"], generated),
            "processed_request_ratio": safe_div(processed, generated),
            "deadline_satisfied_per_processed": safe_div(satisfied, processed),
        }

    return {
        "method": method,
        "training_seed": seed,
        "workload_seed": workload_seed,
        "unit_id": f"seed{seed}_workload{workload_seed}" if seed is not None else f"workload{workload_seed}",
        "num_episodes": int(metrics.get("num_episodes", len(metrics.get("per_episode", [])))),
        **unit_metrics,
    }


def build_units(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    units: dict[str, list[dict[str, Any]]] = {}

    for method in LEARNING_METHODS:
        if method not in data:
            continue
        units[method] = []
        for seed in sorted(data[method]):
            for workload_seed in sorted(data[method][seed]):
                units[method].append(unit_from_metrics(data[method][seed][workload_seed], method=method, seed=seed, workload_seed=workload_seed))

    for method in NON_LEARNING_METHODS:
        if method not in data:
            continue
        units[method] = []
        for workload_seed in sorted(data[method]):
            units[method].append(unit_from_metrics(data[method][workload_seed], method=method, seed=None, workload_seed=workload_seed))

    return units


def compute_statistics(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = int(finite.size)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0, "n": 0}
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half_width = float(stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
        ci_low = mean - half_width
        ci_high = mean + half_width
    else:
        ci_low = ci_high = mean
    return {"mean": mean, "std": std, "ci_95_lower": ci_low, "ci_95_upper": ci_high, "n": n}


def unit_map(units: list[dict[str, Any]], metric: str, *, key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], float]:
    out: dict[tuple[Any, ...], float] = {}
    for unit in units:
        if metric not in unit:
            continue
        key = tuple(unit[field] for field in key_fields)
        out[key] = float(unit[metric])
    return out


def paired_comparison(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    n = min(len(reference), len(candidate))
    if n < 2:
        return {"n": n, "mean_delta": None, "ci_95_lower": None, "ci_95_upper": None, "t_statistic": None, "t_p_value": None, "wilcoxon_statistic": None, "wilcoxon_p_value": None}
    ref = np.asarray(reference[:n], dtype=np.float64)
    cand = np.asarray(candidate[:n], dtype=np.float64)
    delta = ref - cand
    delta_stats = compute_statistics(delta.tolist())
    t_stat, t_p = stats.ttest_rel(ref, cand)
    try:
        w_stat, w_p = stats.wilcoxon(ref, cand)
    except ValueError:
        w_stat, w_p = None, None
    return {
        "n": n,
        "mean_delta": float(delta_stats["mean"]),
        "ci_95_lower": float(delta_stats["ci_95_lower"]),
        "ci_95_upper": float(delta_stats["ci_95_upper"]),
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else None,
        "t_p_value": float(t_p) if np.isfinite(t_p) else None,
        "wilcoxon_statistic": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p_value": float(w_p) if w_p is not None else None,
    }


def proposed_workload_mean(units: list[dict[str, Any]], metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for unit in units:
        grouped[int(unit["workload_seed"])].append(float(unit[metric]))
    return {workload: float(np.mean(vals)) for workload, vals in grouped.items() if vals}


def compute_all_statistics(units: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "unit_note": "Learning: one unit=(training_seed, workload_seed). Non-learning: one unit=workload_seed.",
        "metric_note": "Global ratios are recomputed from per-episode request/offload counts within each unit.",
        "methods": {},
        "comparisons": {},
        "units": units,
    }

    for method, method_units in units.items():
        method_type = "learning" if method in LEARNING_METHODS else "non_learning"
        result["methods"][method] = {"type": method_type, "n": len(method_units), "metrics": {}}
        for metric in TABLE_METRICS:
            values = [float(unit[metric]) for unit in method_units if metric in unit]
            result["methods"][method]["metrics"][metric] = compute_statistics(values)

    if "proposed" in units:
        proposed_by_unit = {metric: unit_map(units["proposed"], metric, key_fields=("training_seed", "workload_seed")) for metric in TABLE_METRICS}
        for method in ["ippo", "vanilla_mappo", "joint_mappo"]:
            if method not in units:
                continue
            key = f"proposed_vs_{method}"
            result["comparisons"][key] = {"type": "learning_paired", "metrics": {}}
            for metric in TABLE_METRICS:
                other_map = unit_map(units[method], metric, key_fields=("training_seed", "workload_seed"))
                common = sorted(set(proposed_by_unit[metric]).intersection(other_map))
                proposed_vals = [proposed_by_unit[metric][k] for k in common]
                other_vals = [other_map[k] for k in common]
                result["comparisons"][key]["metrics"][metric] = paired_comparison(proposed_vals, other_vals)

        for method in NON_LEARNING_METHODS:
            if method not in units:
                continue
            key = f"proposed_vs_{method}"
            result["comparisons"][key] = {"type": "workload_paired", "metrics": {}}
            for metric in TABLE_METRICS:
                proposed_map = proposed_workload_mean(units["proposed"], metric)
                other_map = unit_map(units[method], metric, key_fields=("workload_seed",))
                common = sorted(set(proposed_map).intersection(k[0] for k in other_map))
                proposed_vals = [proposed_map[w] for w in common]
                other_vals = [other_map[(w,)] for w in common]
                result["comparisons"][key]["metrics"][metric] = paired_comparison(proposed_vals, other_vals)

    return result


def fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:.2f}"
    return f"{value:.4f}"


def generate_summary_table(stats_result: dict[str, Any]) -> str:
    header = ["Method", "Type", "N"] + [f"{metric} ({METRIC_DIRECTION.get(metric, 'neutral')})" for metric in TABLE_METRICS]
    lines = ["\t".join(header)]
    for method in LEARNING_METHODS + NON_LEARNING_METHODS:
        if method not in stats_result["methods"]:
            continue
        method_stats = stats_result["methods"][method]
        row = [method, method_stats["type"], str(method_stats["n"])]
        for metric in TABLE_METRICS:
            s = method_stats["metrics"][metric]
            row.append(f"{fmt(s['mean'])} +/- {fmt(s['std'])} [{fmt(s['ci_95_lower'])}, {fmt(s['ci_95_upper'])}]")
        lines.append("\t".join(row))
    return "\n".join(lines)


def generate_report(stats_result: dict[str, Any]) -> str:
    lines = [
        "# Baseline Matrix v2 Report",
        "",
        f"Generated at: {stats_result['timestamp']}",
        "",
        "## Notes",
        "",
        f"- {stats_result['unit_note']}",
        f"- {stats_result['metric_note']}",
        "- MBS offload ratio uses processed service requests as denominator.",
        "- MBS load ratio uses all generated service requests as denominator.",
        "",
        "## Method Summary",
        "",
    ]
    for method in LEARNING_METHODS + NON_LEARNING_METHODS:
        if method not in stats_result["methods"]:
            continue
        method_stats = stats_result["methods"][method]
        lines.append(f"### {method} ({method_stats['type']}, N={method_stats['n']})")
        for metric in TABLE_METRICS:
            s = method_stats["metrics"][metric]
            lines.append(f"- {metric}: {fmt(s['mean'])} +/- {fmt(s['std'])} [95% CI {fmt(s['ci_95_lower'])}, {fmt(s['ci_95_upper'])}]")
        lines.append("")

    lines.append("## Paired Comparisons")
    lines.append("")
    for comparison, details in stats_result["comparisons"].items():
        lines.append(f"### {comparison} ({details['type']})")
        for metric in TABLE_METRICS:
            c = details["metrics"][metric]
            lines.append(f"- {metric}: delta={fmt(c['mean_delta'])}, p_t={fmt(c['t_p_value'])}, p_wilcoxon={fmt(c['wilcoxon_p_value'])}, N={c['n']}")
        lines.append("")
    return "\n".join(lines)


def validate_units(units: dict[str, list[dict[str, Any]]]) -> list[str]:
    warnings: list[str] = []
    ratio_metrics = [m for m in TABLE_METRICS if "ratio" in m or "rate" in m or m.startswith("fairness") or m.startswith("dsr") or m.endswith("processed")]
    for method, method_units in units.items():
        for unit in method_units:
            off_sum = (
                float(unit.get("offloading_ratio_local_processed", 0.0))
                + float(unit.get("offloading_ratio_cooperative_processed", 0.0))
                + float(unit.get("offloading_ratio_mbs_processed", 0.0))
            )
            if float(unit.get("processed_request_ratio", 0.0)) > 0 and not np.isclose(off_sum, 1.0, atol=1e-5):
                warnings.append(f"{method} {unit['unit_id']}: processed offloading ratios sum to {off_sum:.6f}")
            for metric in ratio_metrics:
                value = float(unit.get(metric, 0.0))
                if value < -1e-8 or value > 1.0 + 1e-8:
                    warnings.append(f"{method} {unit['unit_id']}: {metric}={value:.6f} outside [0,1]")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Baseline Matrix v2 results")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    data = load_experiment_data(input_dir)
    units = build_units(data)
    warnings = validate_units(units)
    stats_result = compute_all_statistics(units)
    stats_result["validation_warnings"] = warnings

    stats_path = input_dir / "statistics.json"
    stats_path.write_text(json.dumps(stats_result, indent=2, ensure_ascii=False), encoding="utf-8")
    table_path = input_dir / "summary_table.tsv"
    table_path.write_text(generate_summary_table(stats_result), encoding="utf-8")
    report_path = input_dir / "report.md"
    report_path.write_text(generate_report(stats_result), encoding="utf-8")

    print(f"Loaded methods: {', '.join(units)}")
    for method, method_units in units.items():
        print(f"  {method}: N={len(method_units)}")
    if warnings:
        print("Validation warnings:")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more")
    print(f"Statistics saved to {stats_path}")
    print(f"Summary table saved to {table_path}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
