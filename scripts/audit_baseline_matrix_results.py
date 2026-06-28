#!/usr/bin/env python3
"""Audit baseline matrix result files without modifying experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


CORE_FIELDS = [
    "service_requests_generated",
    "service_requests_processed",
    "service_requests_uncovered",
    "forced_service_admissions",
    "forced_admission_ratio",
    "natural_coverage_service_ratio",
    "deadline_satisfied_service_requests",
    "dsr_request_weighted",
    "processed_request_ratio",
    "deadline_satisfied_per_processed",
]


@dataclass
class MethodAudit:
    units: int = 0
    episodes: int = 0
    generated: float = 0.0
    processed: float = 0.0
    deadline_satisfied: float = 0.0
    uncovered: float = 0.0
    forced_admissions: float = 0.0
    natural_covered: float = 0.0
    local: float = 0.0
    cooperative: float = 0.0
    mbs: float = 0.0
    processed_ratio_values: list[float] = field(default_factory=list)
    dsr_values: list[float] = field(default_factory=list)
    satisfied_per_processed_values: list[float] = field(default_factory=list)
    mbs_processed_values: list[float] = field(default_factory=list)
    mbs_generated_values: list[float] = field(default_factory=list)
    missing_fields: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    denom_mismatch_count: int = 0


@dataclass
class TrajectoryAudit:
    files: int = 0
    episodes: int = 0
    distance_m: float = 0.0
    max_step_m: float = 0.0
    min_pair_m: float | None = None
    edge_positions: int = 0
    total_positions: int = 0
    below_min_sep_count: int = 0


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{100.0 * value:.2f}%"


def mean_std(values: list[float]) -> str:
    if not values:
        return "NA"
    if len(values) == 1:
        return fmt_float(values[0], 4)
    return f"{mean(values):.4f} +/- {pstdev(values):.4f}"


def metric_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*_metrics.json") if path.is_file())


def trajectory_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("trajectory*.json") if path.is_file())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def method_from_payload(path: Path, payload: dict[str, Any]) -> str:
    method = payload.get("method")
    if isinstance(method, str) and method:
        return method
    parts = path.relative_to(path.parents[2]).parts if len(path.parents) >= 3 else path.parts
    return parts[0] if parts else "unknown"


def collect_metrics(input_dir: Path) -> dict[str, MethodAudit]:
    audits: dict[str, MethodAudit] = defaultdict(MethodAudit)
    for path in metric_files(input_dir):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        method = str(payload.get("method") or path.parent.name)
        audit = audits[method]
        audit.units += 1
        per_episode = payload.get("per_episode", [])
        if not isinstance(per_episode, list):
            per_episode = []

        for episode in per_episode:
            if not isinstance(episode, dict):
                continue
            audit.episodes += 1
            for field_name in CORE_FIELDS:
                if field_name not in episode:
                    audit.missing_fields[field_name] += 1

            generated = float(episode.get("service_requests_generated", 0.0) or 0.0)
            processed = float(episode.get("service_requests_processed", 0.0) or 0.0)
            satisfied = float(episode.get("deadline_satisfied_service_requests", 0.0) or 0.0)
            uncovered = float(episode.get("service_requests_uncovered", 0.0) or 0.0)
            forced = float(episode.get("forced_service_admissions", 0.0) or 0.0)
            natural_ratio = float(episode.get("natural_coverage_service_ratio", 0.0) or 0.0)
            local = float(episode.get("service_offloads_local", 0.0) or 0.0)
            cooperative = float(episode.get("service_offloads_cooperative", 0.0) or 0.0)
            mbs = float(episode.get("service_offloads_mbs", 0.0) or 0.0)

            audit.generated += generated
            audit.processed += processed
            audit.deadline_satisfied += satisfied
            audit.uncovered += uncovered
            audit.forced_admissions += forced
            audit.natural_covered += natural_ratio * generated
            audit.local += local
            audit.cooperative += cooperative
            audit.mbs += mbs

            if generated > 0:
                audit.processed_ratio_values.append(processed / generated)
                audit.dsr_values.append(satisfied / generated)
                audit.mbs_generated_values.append(mbs / generated)
            if processed > 0:
                audit.satisfied_per_processed_values.append(satisfied / processed)
                audit.mbs_processed_values.append(mbs / processed)

            reported_mbs_generated = episode.get("mbs_load_ratio_generated")
            reported_mbs_processed = episode.get("offloading_ratio_mbs_processed")
            reported_processed_ratio = episode.get("processed_request_ratio")
            if (
                reported_mbs_generated is not None
                and reported_mbs_processed is not None
                and reported_processed_ratio is not None
            ):
                expected = float(reported_mbs_processed) * float(reported_processed_ratio)
                if not math.isclose(float(reported_mbs_generated), expected, rel_tol=1e-6, abs_tol=1e-6):
                    audit.denom_mismatch_count += 1
    return dict(audits)


def audit_trajectory_episode(frames: list[dict[str, Any]], audit: TrajectoryAudit) -> None:
    if not frames:
        return
    audit.episodes += 1
    positions_by_step: list[list[list[float]]] = []
    for frame in frames:
        positions = frame.get("uav_positions", [])
        if isinstance(positions, list):
            positions_by_step.append(positions)

    for step_idx, positions in enumerate(positions_by_step):
        audit.total_positions += len(positions)
        for x, y, *_ in positions:
            if x <= 60.0 or x >= 640.0 or y <= 60.0 or y >= 640.0:
                audit.edge_positions += 1
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = math.dist(positions[i][:2], positions[j][:2])
                audit.min_pair_m = dist if audit.min_pair_m is None else min(audit.min_pair_m, dist)
                if dist < 200.0:
                    audit.below_min_sep_count += 1
        if step_idx == 0:
            continue
        previous = positions_by_step[step_idx - 1]
        for current_pos, previous_pos in zip(positions, previous):
            step_distance = math.dist(current_pos[:2], previous_pos[:2])
            audit.distance_m += step_distance
            audit.max_step_m = max(audit.max_step_m, step_distance)


def collect_trajectories(input_dir: Path) -> dict[str, TrajectoryAudit]:
    audits: dict[str, TrajectoryAudit] = defaultdict(TrajectoryAudit)
    for path in trajectory_files(input_dir):
        payload = load_json(path)
        method = path.parent.name
        audit = audits[method]
        audit.files += 1
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    frames = item.get("trajectory", [])
                    if isinstance(frames, list):
                        audit_trajectory_episode(frames, audit)
        elif isinstance(payload, dict):
            frames = payload.get("trajectory", [])
            if isinstance(frames, list):
                audit_trajectory_episode(frames, audit)
    return dict(audits)


def render_report(input_dir: Path, metric_audits: dict[str, MethodAudit], trajectory_audits: dict[str, TrajectoryAudit]) -> str:
    lines: list[str] = []
    lines.append("# Baseline Matrix Audit Report")
    lines.append("")
    lines.append(f"- Input directory: `{input_dir}`")
    lines.append(f"- Metrics files: {len(metric_files(input_dir))}")
    lines.append(f"- Trajectory files: {len(trajectory_files(input_dir))}")
    lines.append("- This report is an audit artifact and should not be treated as a thesis result table by itself.")
    lines.append("")

    lines.append("## Service Coverage Chain")
    lines.append("")
    lines.append(
        "| Method | Units | Episodes | Generated | Processed | Unprocessed | Deadline satisfied | Processed / generated | DSR request-weighted | Satisfied / processed |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in sorted(metric_audits):
        audit = metric_audits[method]
        unprocessed = audit.generated - audit.processed
        processed_ratio = audit.processed / audit.generated if audit.generated > 0 else math.nan
        dsr = audit.deadline_satisfied / audit.generated if audit.generated > 0 else math.nan
        satisfied_per_processed = audit.deadline_satisfied / audit.processed if audit.processed > 0 else math.nan
        lines.append(
            f"| {method} | {audit.units} | {audit.episodes} | {audit.generated:.0f} | {audit.processed:.0f} | "
            f"{unprocessed:.0f} | {audit.deadline_satisfied:.0f} | {fmt_pct(processed_ratio)} | "
            f"{fmt_pct(dsr)} | {fmt_pct(satisfied_per_processed)} |"
        )
    lines.append("")

    lines.append("## Admission Audit")
    lines.append("")
    lines.append("| Method | Naturally covered service requests | Uncovered service requests | Forced service admissions | Forced admission ratio | Natural coverage ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for method in sorted(metric_audits):
        audit = metric_audits[method]
        forced_ratio = audit.forced_admissions / audit.generated if audit.generated > 0 else math.nan
        natural_ratio = audit.natural_covered / audit.generated if audit.generated > 0 else math.nan
        lines.append(
            f"| {method} | {audit.natural_covered:.0f} | {audit.uncovered:.0f} | {audit.forced_admissions:.0f} | "
            f"{fmt_pct(forced_ratio)} | {fmt_pct(natural_ratio)} |"
        )
    lines.append("")

    lines.append("## Ratio Field Audit")
    lines.append("")
    lines.append(
        "| Method | processed_request_ratio | dsr_request_weighted | deadline_satisfied_per_processed | offloading_ratio_mbs_processed | mbs_load_ratio_generated | MBS denominator mismatches |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in sorted(metric_audits):
        audit = metric_audits[method]
        lines.append(
            f"| {method} | {mean_std(audit.processed_ratio_values)} | {mean_std(audit.dsr_values)} | "
            f"{mean_std(audit.satisfied_per_processed_values)} | {mean_std(audit.mbs_processed_values)} | "
            f"{mean_std(audit.mbs_generated_values)} | {audit.denom_mismatch_count} |"
        )
    lines.append("")

    lines.append("## Red-Flag Checks")
    lines.append("")
    for method in sorted(metric_audits):
        audit = metric_audits[method]
        unprocessed = audit.generated - audit.processed
        processed_ratio = audit.processed / audit.generated if audit.generated > 0 else math.nan
        dsr = audit.deadline_satisfied / audit.generated if audit.generated > 0 else math.nan
        flags = [
            "has unprocessed service requests" if unprocessed > 0 else "all generated service requests processed",
            "DSR is lower than processed ratio" if dsr < processed_ratio else "DSR is not lower than processed ratio",
            "MBS ratio denominators consistent" if audit.denom_mismatch_count == 0 else "MBS denominator mismatch detected",
        ]
        if audit.missing_fields:
            missing = ", ".join(f"{name}:{count}" for name, count in sorted(audit.missing_fields.items()))
            flags.append(f"missing fields: {missing}")
        lines.append(f"- `{method}`: " + "; ".join(flags) + ".")
    lines.append("")

    lines.append("## Trajectory Audit")
    lines.append("")
    if not trajectory_audits:
        lines.append("- No trajectory JSON files were found under the input directory.")
        lines.append("- Matrix metrics files do not contain UAV positions, so trajectory distance/boundary/separation checks are unavailable for this run.")
    else:
        lines.append("| Method | Files | Episodes | Distance (km) | Edge-position ratio | Max step (m) | Min UAV pair (m) | Pair checks < 200m |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for method in sorted(trajectory_audits):
            audit = trajectory_audits[method]
            edge_ratio = audit.edge_positions / audit.total_positions if audit.total_positions > 0 else math.nan
            lines.append(
                f"| {method} | {audit.files} | {audit.episodes} | {audit.distance_m / 1000.0:.2f} | "
                f"{fmt_pct(edge_ratio)} | {fmt_float(audit.max_step_m, 2)} | {fmt_float(audit.min_pair_m, 2)} | "
                f"{audit.below_min_sep_count} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit baseline matrix metrics and optional trajectory JSON files.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Result directory to audit.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    metric_audits = collect_metrics(args.input_dir)
    trajectory_audits = collect_trajectories(args.input_dir)
    report = render_report(args.input_dir, metric_audits, trajectory_audits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote audit report to {args.output}")


if __name__ == "__main__":
    main()
