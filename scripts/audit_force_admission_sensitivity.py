#!/usr/bin/env python3
"""Audit force-admission sensitivity results for thesis use."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "sensitivity_force_admission"
DEFAULT_OUTPUT = ROOT / "docs" / "defense_experiment_audit_force_admission_sensitivity.md"
EXPERIMENTS = {
    "ue_count": "UE 数量敏感性",
    "uav_cpu_scale": "UAV 算力敏感性",
}
ALGOS = ("heuristic", "vanilla_mappo", "proposed")


@dataclass
class AuditRow:
    experiment: str
    variable: str
    algorithm: str
    raw_records: int
    summary_samples: int | None
    generated: float
    processed: float
    min_processed_ratio: float
    mean_forced_ratio: float
    mean_natural_ratio: float
    min_dsr: float
    mean_dsr: float
    max_dsr: float
    mismatches: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audit_experiment(base_dir: Path, experiment: str) -> list[AuditRow]:
    raw_path = base_dir / experiment / "raw" / f"{experiment}_results.json"
    summary_path = base_dir / experiment / "summary.json"
    raw_data = load_json(raw_path)
    summary = load_json(summary_path)
    rows: list[AuditRow] = []

    for variable in sorted(raw_data, key=lambda x: float(x)):
        for algorithm in ALGOS:
            records = raw_data[variable].get(algorithm, [])
            if not records:
                rows.append(
                    AuditRow(
                        experiment=experiment,
                        variable=str(variable),
                        algorithm=algorithm,
                        raw_records=0,
                        summary_samples=None,
                        generated=0.0,
                        processed=0.0,
                        min_processed_ratio=0.0,
                        mean_forced_ratio=0.0,
                        mean_natural_ratio=0.0,
                        min_dsr=0.0,
                        mean_dsr=0.0,
                        max_dsr=0.0,
                        mismatches=0,
                    )
                )
                continue

            generated = sum(as_float(r.get("service_requests_generated")) for r in records)
            processed = sum(as_float(r.get("service_requests_processed")) for r in records)
            mismatches = sum(
                1
                for r in records
                if as_float(r.get("service_requests_generated")) != as_float(r.get("service_requests_processed"))
            )
            processed_ratios = [as_float(r.get("processed_request_ratio")) for r in records]
            forced = [as_float(r.get("forced_admission_ratio")) for r in records]
            natural = [as_float(r.get("natural_coverage_service_ratio")) for r in records]
            dsr = [as_float(r.get("dsr_request_weighted", r.get("deadline_satisfaction_rate"))) for r in records]
            summary_samples = (
                summary.get("variables", {})
                .get(str(variable), {})
                .get(algorithm, {})
                .get("num_samples")
            )
            rows.append(
                AuditRow(
                    experiment=experiment,
                    variable=str(variable),
                    algorithm=algorithm,
                    raw_records=len(records),
                    summary_samples=int(summary_samples) if summary_samples is not None else None,
                    generated=generated,
                    processed=processed,
                    min_processed_ratio=min(processed_ratios),
                    mean_forced_ratio=mean(forced),
                    mean_natural_ratio=mean(natural),
                    min_dsr=min(dsr),
                    mean_dsr=mean(dsr),
                    max_dsr=max(dsr),
                    mismatches=mismatches,
                )
            )

    return rows


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_report(rows: list[AuditRow], output: Path) -> None:
    total_mismatches = sum(row.mismatches for row in rows)
    min_processed = min(row.min_processed_ratio for row in rows)
    lines = [
        "# Force-Admission Sensitivity Audit",
        "",
        "## Summary",
        "",
        f"- Audited experiments: {', '.join(EXPERIMENTS.values())}.",
        f"- Minimum processed request ratio across all raw records: {fmt_pct(min_processed)}.",
        f"- Generated/processed mismatch records: {total_mismatches}.",
        "- DSR is audited as request-weighted DSR when available; it is not treated as service admission rate.",
        "",
        "## Detailed Checks",
        "",
        "| Experiment | Variable | Algorithm | Raw records | Summary samples | Generated | Processed | Min processed | Forced admission | Natural coverage | Mean DSR | Mismatches |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {variable} | {algorithm} | {raw_records} | {summary_samples} | "
            "{generated:.0f} | {processed:.0f} | {min_processed} | {forced} | {natural} | {dsr:.4f} | {mismatches} |".format(
                experiment=EXPERIMENTS[row.experiment],
                variable=row.variable,
                algorithm=row.algorithm,
                raw_records=row.raw_records,
                summary_samples=row.summary_samples if row.summary_samples is not None else 0,
                generated=row.generated,
                processed=row.processed,
                min_processed=fmt_pct(row.min_processed_ratio),
                forced=fmt_pct(row.mean_forced_ratio),
                natural=fmt_pct(row.mean_natural_ratio),
                dsr=row.mean_dsr,
                mismatches=row.mismatches,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Force-admission guarantees that service requests enter the processing chain; it does not guarantee deadline satisfaction.",
            "- Higher forced admission ratio means more requests required nearest-UAV fallback access, not necessarily worse execution quality by itself.",
            "- Sensitivity figures should present DSR together with EEE and fairness to avoid overstating DSR-only performance.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows: list[AuditRow] = []
    for experiment in EXPERIMENTS:
        rows.extend(audit_experiment(args.input_dir, experiment))
    write_report(rows, args.output)
    print(f"Wrote audit report: {args.output}")


if __name__ == "__main__":
    main()
