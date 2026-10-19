from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRICS: tuple[str, ...] = (
    "reward",
    "latency",
    "energy",
    "fairness",
    "offline_rate",
    "deadline_satisfaction_rate",
    "offloading_ratio_local",
    "offloading_ratio_cooperative",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
    "service_fallback_count",
    "service_predict_exception_fallback_count",
)

LOWER_IS_BETTER = {
    "latency",
    "energy",
    "offline_rate",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
    "service_fallback_count",
    "service_predict_exception_fallback_count",
}

RATIO_METRICS = {
    "fairness",
    "offline_rate",
    "deadline_satisfaction_rate",
    "offloading_ratio_local",
    "offloading_ratio_cooperative",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def available_metric_names(policy_units: dict[str, list[dict[str, float]]]) -> list[str]:
    names: set[str] = set()
    for units in policy_units.values():
        for unit in units:
            names.update(unit)
    return [metric for metric in DEFAULT_METRICS if metric in names] + sorted(names.difference(DEFAULT_METRICS))


def extract_joint_units(data: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    per_policy = data.get("per_combo") or data.get("per_policy")
    if not isinstance(per_policy, dict):
        return {}

    policy_units: dict[str, list[dict[str, float]]] = {}
    for policy_name, details in per_policy.items():
        if not isinstance(details, dict):
            continue
        units = []
        for entry in details.get("per_seed", []):
            if not isinstance(entry, dict):
                continue
            unit_mean = entry.get("unit_mean") or entry.get("per_seed_mean")
            if isinstance(unit_mean, dict):
                units.append({key: float(value) for key, value in unit_mean.items()})
        if units:
            policy_units[str(policy_name)] = units
    return policy_units


def extract_runtime_units(data: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    runtime = data.get("runtime_comparison", data)
    per_scenario = runtime.get("per_scenario") if isinstance(runtime, dict) else None
    if not isinstance(per_scenario, dict):
        return {}

    by_policy_seed: dict[str, dict[int, list[dict[str, float]]]] = {}
    for scenario_details in per_scenario.values():
        if not isinstance(scenario_details, dict):
            continue
        for policy_name, policy_details in scenario_details.items():
            if not isinstance(policy_details, dict):
                continue
            for seed_entry in policy_details.get("per_seed", []):
                if not isinstance(seed_entry, dict) or not isinstance(seed_entry.get("per_seed_mean"), dict):
                    continue
                seed = int(seed_entry.get("seed", len(by_policy_seed.get(str(policy_name), {}))))
                by_policy_seed.setdefault(str(policy_name), {}).setdefault(seed, []).append(
                    {key: float(value) for key, value in seed_entry["per_seed_mean"].items()}
                )

    policy_units: dict[str, list[dict[str, float]]] = {}
    for policy_name, seed_groups in by_policy_seed.items():
        units = []
        for seed in sorted(seed_groups):
            scenario_units = seed_groups[seed]
            metric_names = sorted({metric for unit in scenario_units for metric in unit})
            units.append(
                {
                    metric: float(np.mean([unit[metric] for unit in scenario_units if metric in unit]))
                    for metric in metric_names
                }
            )
        if units:
            policy_units[policy_name] = units
    return policy_units


def extract_policy_units(data: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    policy_units = extract_joint_units(data)
    if policy_units:
        return policy_units
    return extract_runtime_units(data)


def mean_ci(values: np.ndarray, confidence: float, bootstrap_samples: int, rng: np.random.Generator) -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n == 1:
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean, "n": n}

    try:
        from scipy import stats  # type: ignore

        alpha = 1.0 - confidence
        half_width = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1) * std / math.sqrt(n))
        return {"mean": mean, "std": std, "ci_low": mean - half_width, "ci_high": mean + half_width, "n": n}
    except Exception:
        samples = rng.choice(values, size=(bootstrap_samples, n), replace=True)
        means = np.mean(samples, axis=1)
        alpha = 1.0 - confidence
        return {
            "mean": mean,
            "std": std,
            "ci_low": float(np.quantile(means, alpha / 2.0)),
            "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
            "n": n,
        }


def paired_tests(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    n = min(reference.size, candidate.size)
    if n < 2:
        return {"paired_t_p": None, "wilcoxon_p": None}
    reference = reference[:n]
    candidate = candidate[:n]
    try:
        from scipy import stats  # type: ignore

        t_result = stats.ttest_rel(candidate, reference, nan_policy="omit")
        try:
            w_result = stats.wilcoxon(candidate, reference, zero_method="wilcox", alternative="two-sided")
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_p = None
        return {"paired_t_p": float(t_result.pvalue), "wilcoxon_p": wilcoxon_p}
    except Exception:
        return {"paired_t_p": None, "wilcoxon_p": None}


def effect_direction(metric: str, mean_delta: float) -> str:
    if not math.isfinite(mean_delta) or abs(mean_delta) < 1e-12:
        return "tie"
    improves = mean_delta < 0.0 if metric in LOWER_IS_BETTER else mean_delta > 0.0
    return "improves" if improves else "regresses"


def build_statistics(
    policy_units: dict[str, list[dict[str, float]]],
    *,
    reference: str,
    confidence: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if reference not in policy_units:
        raise ValueError(f"Reference policy '{reference}' not found. Available: {', '.join(policy_units)}")

    rng = np.random.default_rng(20260428)
    metrics = available_metric_names(policy_units)
    summary: dict[str, Any] = {
        "metadata": {
            "reference": reference,
            "confidence": confidence,
            "ci_method": "t interval when scipy is available, otherwise bootstrap",
            "unit_note": "Joint H-MARL summaries use paired (training_seed, workload_seed) episode means when training_seed is present; older summaries fall back to workload-seed means.",
            "ratio_metric_note": "Ratio metrics are treated as paired unit means; use request-level counts for exact binomial intervals when available.",
        },
        "metrics": {},
        "paired_vs_reference": {},
    }

    for policy_name, units in policy_units.items():
        summary["metrics"][policy_name] = {}
        for metric in metrics:
            values = np.asarray([unit[metric] for unit in units if metric in unit], dtype=np.float64)
            if values.size:
                stat = mean_ci(values, confidence, bootstrap_samples, rng)
                stat["is_ratio_metric"] = metric in RATIO_METRICS
                summary["metrics"][policy_name][metric] = stat

    reference_units = policy_units[reference]
    for policy_name, units in policy_units.items():
        if policy_name == reference:
            continue
        summary["paired_vs_reference"][policy_name] = {}
        for metric in metrics:
            ref_values = np.asarray([unit[metric] for unit in reference_units if metric in unit], dtype=np.float64)
            cand_values = np.asarray([unit[metric] for unit in units if metric in unit], dtype=np.float64)
            n = min(ref_values.size, cand_values.size)
            if n == 0:
                continue
            delta = cand_values[:n] - ref_values[:n]
            delta_stat = mean_ci(delta, confidence, bootstrap_samples, rng)
            tests = paired_tests(ref_values, cand_values)
            summary["paired_vs_reference"][policy_name][metric] = {
                **delta_stat,
                **tests,
                "effect": effect_direction(metric, float(delta_stat["mean"])),
            }
    return summary


def format_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:.2f}"
    return f"{value:.4g}"


def markdown_report(stats: dict[str, Any]) -> str:
    lines = [
        "# Experiment Statistics",
        "",
        f"Reference: `{stats['metadata']['reference']}`",
        "",
        "## Metric Summary",
        "",
        "| Policy | Metric | Mean | Std | CI Low | CI High | N |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy_name, metrics in stats["metrics"].items():
        for metric, stat in metrics.items():
            lines.append(
                f"| {policy_name} | {metric} | {format_float(stat['mean'])} | {format_float(stat['std'])} | "
                f"{format_float(stat['ci_low'])} | {format_float(stat['ci_high'])} | {stat['n']} |"
            )

    lines.extend(
        [
            "",
            "## Paired Comparison",
            "",
            "| Policy | Metric | Mean Delta | CI Low | CI High | paired t p | Wilcoxon p | Effect |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for policy_name, metrics in stats["paired_vs_reference"].items():
        for metric, stat in metrics.items():
            lines.append(
                f"| {policy_name} | {metric} | {format_float(stat['mean'])} | {format_float(stat['ci_low'])} | "
                f"{format_float(stat['ci_high'])} | {format_float(stat['paired_t_p'])} | "
                f"{format_float(stat['wilcoxon_p'])} | {stat['effect']} |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired-unit confidence intervals and tests from experiment summaries.")
    parser.add_argument("summary_json", type=str, help="Path to joint_experiment_summary.json or experiment_summary.json.")
    parser.add_argument("--reference", type=str, default=None, help="Reference policy/combo. Defaults to the first policy in the summary.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--output_md", type=str, default=None, help="Optional Markdown output path.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for intervals.")
    parser.add_argument("--bootstrap_samples", type=int, default=10000, help="Bootstrap samples used when scipy is unavailable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    data = load_json(summary_path)
    policy_units = extract_policy_units(data)
    if not policy_units:
        raise ValueError("No paired policy units were found in the summary JSON.")

    reference = args.reference or next(iter(policy_units))
    stats = build_statistics(
        policy_units,
        reference=reference,
        confidence=float(args.confidence),
        bootstrap_samples=int(args.bootstrap_samples),
    )

    json_text = json.dumps(stats, indent=2, ensure_ascii=False)
    md_text = markdown_report(stats)
    if args.output_json:
        Path(args.output_json).write_text(json_text, encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(md_text, encoding="utf-8")
    print(md_text)


if __name__ == "__main__":
    main()
