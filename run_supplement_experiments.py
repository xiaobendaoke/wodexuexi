from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import config
from analyze_experiment_statistics import build_statistics, extract_policy_units, markdown_report
from paths import REPO_ROOT, results_path
from run_full_offload_experiment import run_full_offload_experiment
from run_all_experiments import (
    configure_compile_backend,
    configure_fp32_precision,
    restore_config,
    run_single_rl_experiment,
    snapshot_config,
)
from train_offload_policy import parse_hidden_dims


MAIN_JOINT_REFERENCE = "uncoordinated_greedy__heuristic"
ATTENTION_HEURISTIC_REFERENCE = "attention_mappo__heuristic"
FULL_METHOD_COMBO = "attention_mappo__oracle_guided"

MAIN_METRICS = (
    "reward",
    "latency",
    "energy",
    "deadline_satisfaction_rate",
    "fairness",
    "offloading_ratio_local",
    "offloading_ratio_cooperative",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"[supplement] command: {command_text(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def latest_log_json(run_root: Path, model_name: str) -> Path | None:
    log_dir = run_root / "test_logs" / model_name
    candidates = sorted(log_dir.glob("log_data_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def mean_log_metrics(log_json_path: Path) -> dict[str, float]:
    rows = load_json(log_json_path)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No episode rows found in {log_json_path}")

    metric_names = sorted({key for row in rows if isinstance(row, dict) for key, value in row.items() if isinstance(value, (int, float))})
    return {
        metric: float(np.mean([float(row[metric]) for row in rows if isinstance(row, dict) and metric in row]))
        for metric in metric_names
    }


def build_and_write_stats(
    *,
    policy_units: dict[str, list[dict[str, float]]],
    reference: str,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    stats = build_statistics(policy_units, reference=reference, confidence=0.95, bootstrap_samples=10000)
    write_json(output_json, stats)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_report(stats), encoding="utf-8")
    return stats


def selected_comparison_markdown(stats_by_reference: dict[str, dict[str, Any]]) -> str:
    rows = [
        "# Key Supplement Comparisons",
        "",
        "| Comparison | Metric | Mean Delta | CI Low | CI High | paired t p | Wilcoxon p | Effect |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    comparisons = [
        ("upper: attention/heuristic vs uncoordinated/heuristic", MAIN_JOINT_REFERENCE, ATTENTION_HEURISTIC_REFERENCE),
        ("lower: uncoordinated/oracle-guided vs uncoordinated/heuristic", MAIN_JOINT_REFERENCE, "uncoordinated_greedy__oracle_guided"),
        ("full: attention/oracle-guided vs uncoordinated/heuristic", MAIN_JOINT_REFERENCE, FULL_METHOD_COMBO),
        ("full vs upper-only: attention/oracle-guided vs attention/heuristic", ATTENTION_HEURISTIC_REFERENCE, FULL_METHOD_COMBO),
    ]
    for label, reference, candidate in comparisons:
        stats = stats_by_reference.get(reference, {})
        paired = stats.get("paired_vs_reference", {}).get(candidate, {})
        for metric in MAIN_METRICS:
            stat = paired.get(metric)
            if not stat:
                continue
            rows.append(
                "| "
                + " | ".join(
                    [
                        label,
                        metric,
                        f"{float(stat['mean']):.6g}",
                        f"{float(stat['ci_low']):.6g}",
                        f"{float(stat['ci_high']):.6g}",
                        "NA" if stat.get("paired_t_p") is None else f"{float(stat['paired_t_p']):.6g}",
                        "NA" if stat.get("wilcoxon_p") is None else f"{float(stat['wilcoxon_p']):.6g}",
                        str(stat["effect"]),
                    ]
                )
                + " |"
            )
    return "\n".join(rows) + "\n"


def validate_joint_summary(summary_path: Path, *, expected_seeds: int, expected_episodes: int) -> dict[str, Any]:
    summary = load_json(summary_path)
    per_combo = summary.get("per_combo") or summary.get("per_policy") or {}
    expected_combos = {
        "uncoordinated_greedy__heuristic",
        "attention_mappo__heuristic",
        "uncoordinated_greedy__oracle_guided",
        "attention_mappo__oracle_guided",
    }
    validation: dict[str, Any] = {
        "summary_path": str(summary_path),
        "has_four_combos": expected_combos.issubset(set(per_combo)),
        "expected_seed_count": int(expected_seeds),
        "expected_episodes_per_seed": int(expected_episodes),
        "per_combo": {},
        "passed": True,
    }
    for combo_name in sorted(expected_combos):
        details = per_combo.get(combo_name)
        if not isinstance(details, dict):
            validation["per_combo"][combo_name] = {"present": False}
            validation["passed"] = False
            continue
        per_seed = details.get("per_seed", [])
        seed_count_ok = len(per_seed) == expected_seeds
        episode_counts = [len(entry.get("episodes", [])) for entry in per_seed if isinstance(entry, dict)]
        episode_count_ok = all(count == expected_episodes for count in episode_counts) and len(episode_counts) == expected_seeds
        learned_count = float(details.get("aggregate", {}).get("service_learned_decision_count", {}).get("mean", 0.0))
        exception_count = float(details.get("aggregate", {}).get("service_predict_exception_fallback_count", {}).get("mean", 0.0))
        learned_ok = learned_count > 0.0 if "oracle_guided" in combo_name else True
        exception_ok = exception_count == 0.0
        combo_ok = seed_count_ok and episode_count_ok and learned_ok and exception_ok
        validation["per_combo"][combo_name] = {
            "present": True,
            "seed_count": len(per_seed),
            "seed_count_ok": seed_count_ok,
            "episode_counts": episode_counts,
            "episode_count_ok": episode_count_ok,
            "service_learned_decision_count_mean": learned_count,
            "learned_count_ok": learned_ok,
            "service_predict_exception_fallback_count_mean": exception_count,
            "exception_count_ok": exception_ok,
            "passed": combo_ok,
        }
        validation["passed"] = bool(validation["passed"] and combo_ok)
    validation["passed"] = bool(validation["passed"] and validation["has_four_combos"])
    return validation


def run_joint_phase(args: argparse.Namespace) -> dict[str, Any]:
    joint_name = args.joint_name
    seeds = [int(seed) for seed in args.joint_eval_seeds]
    episodes_per_seed = int(args.joint_episodes_per_seed)
    steps_per_episode = args.joint_steps_per_episode
    if args.smoke:
        joint_name = f"{joint_name}_smoke"
        seeds = [seeds[0]]
        episodes_per_seed = 1
        steps_per_episode = 2

    command = [
        sys.executable,
        "run_joint_trajectory_offload_experiment.py",
        "--name",
        joint_name,
        "--trajectory_run_root",
        args.trajectory_run_root,
        "--offload_experiment_root",
        args.offload_experiment_root,
        "--four_way_ablation",
        "--seeds",
        *[str(seed) for seed in seeds],
        "--episodes_per_seed",
        str(episodes_per_seed),
    ]
    if steps_per_episode is not None:
        command.extend(["--steps_per_episode", str(int(steps_per_episode))])
    if args.record_spatial_trace:
        command.extend(["--record_spatial_trace", "--spatial_trace_interval", str(int(args.spatial_trace_interval))])

    run_command(command, dry_run=args.dry_run)
    run_root = results_path("joint_experiments", joint_name)
    summary_path = run_root / "joint_experiment_summary.json"
    if args.dry_run:
        return {"phase": "joint", "name": joint_name, "summary_path": str(summary_path), "dry_run": True}

    summary = load_json(summary_path)
    policy_units = extract_policy_units(summary)
    reports_dir = run_root / "reports"
    stats_baseline = build_and_write_stats(
        policy_units=policy_units,
        reference=MAIN_JOINT_REFERENCE,
        output_json=reports_dir / "joint_four_way_statistics_vs_uncoordinated_heuristic.json",
        output_md=reports_dir / "joint_four_way_statistics_vs_uncoordinated_heuristic.md",
    )
    stats_upper = build_and_write_stats(
        policy_units=policy_units,
        reference=ATTENTION_HEURISTIC_REFERENCE,
        output_json=reports_dir / "joint_four_way_statistics_vs_attention_heuristic.json",
        output_md=reports_dir / "joint_four_way_statistics_vs_attention_heuristic.md",
    )
    key_md = selected_comparison_markdown(
        {
            MAIN_JOINT_REFERENCE: stats_baseline,
            ATTENTION_HEURISTIC_REFERENCE: stats_upper,
        }
    )
    (reports_dir / "joint_four_way_key_comparisons.md").write_text(key_md, encoding="utf-8")

    validation = validate_joint_summary(summary_path, expected_seeds=len(seeds), expected_episodes=episodes_per_seed)
    write_json(reports_dir / "joint_four_way_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"Joint validation failed; see {reports_dir / 'joint_four_way_validation.json'}")

    return {
        "phase": "joint",
        "name": joint_name,
        "summary_path": str(summary_path),
        "reports_dir": str(reports_dir),
        "validation": validation,
    }


def seed42_existing_upper_entries() -> list[dict[str, Any]]:
    run_root = results_path("full_runs", "wpt_fix_thesis_run")
    entries: list[dict[str, Any]] = []
    for model_name, policy_name in [
        ("attention_mappo", "attention_mappo"),
        ("uncoordinated_greedy", "uncoordinated_greedy"),
    ]:
        log_path = latest_log_json(run_root, model_name)
        if log_path is not None:
            entries.append(
                {
                    "seed": 42,
                    "policy": policy_name,
                    "source": "existing_wpt_fix_thesis_run",
                    "test_log_json_path": str(log_path),
                    "test_unit": mean_log_metrics(log_path),
                }
            )
    return entries


def run_upper_phase(args: argparse.Namespace) -> dict[str, Any]:
    upper_run_name = f"{args.upper_run_name}_smoke" if args.smoke else args.upper_run_name
    seeds = [int(seed) for seed in args.upper_train_seeds]
    train_episodes = int(args.upper_train_episodes)
    test_episodes = int(args.upper_test_episodes)
    if args.smoke:
        seeds = [seeds[0]]
        train_episodes = 1
        test_episodes = 1

    run_root = results_path("full_runs", upper_run_name)
    reports_dir = run_root / "reports"
    manifest_path = run_root / "upper_multiseed_manifest.json"

    if args.dry_run:
        return {
            "phase": "upper",
            "run_root": str(run_root),
            "seeds": seeds,
            "dry_run": True,
        }

    configure_fp32_precision()
    configure_compile_backend()
    base_snapshot = snapshot_config()
    entries = [] if args.skip_existing_upper_seed42 else seed42_existing_upper_entries()
    try:
        for seed in seeds:
            print(f"[supplement] upper seed={seed} model=attention_mappo")
            restore_config(base_snapshot)
            config.SEED = int(seed)
            attention_summary = run_single_rl_experiment(
                model_name="attention_mappo",
                train_episodes=train_episodes,
                test_episodes=test_episodes,
                run_name=upper_run_name,
            )
            attention_log = Path(attention_summary["test"]["log_json_path"])
            entries.append(
                {
                    "seed": int(seed),
                    "policy": "attention_mappo",
                    "source": "supplement_train",
                    "summary": attention_summary,
                    "test_log_json_path": str(attention_log),
                    "test_unit": mean_log_metrics(attention_log),
                }
            )

            print(f"[supplement] upper seed={seed} model=uncoordinated_greedy")
            restore_config(base_snapshot)
            config.SEED = int(seed)
            baseline_summary = run_single_rl_experiment(
                model_name="uncoordinated_greedy",
                train_episodes=1,
                test_episodes=test_episodes,
                run_name=upper_run_name,
            )
            baseline_log = Path(baseline_summary["test"]["log_json_path"])
            entries.append(
                {
                    "seed": int(seed),
                    "policy": "uncoordinated_greedy",
                    "source": "supplement_eval",
                    "summary": baseline_summary,
                    "test_log_json_path": str(baseline_log),
                    "test_unit": mean_log_metrics(baseline_log),
                }
            )
    finally:
        restore_config(base_snapshot)

    policy_units: dict[str, list[dict[str, float]]] = {}
    for entry in entries:
        policy_units.setdefault(str(entry["policy"]), []).append(copy.deepcopy(entry["test_unit"]))

    stats = build_and_write_stats(
        policy_units=policy_units,
        reference="uncoordinated_greedy",
        output_json=reports_dir / "upper_multiseed_statistics.json",
        output_md=reports_dir / "upper_multiseed_statistics.md",
    )
    manifest = {
        "metadata": {
            "run_name": upper_run_name,
            "included_existing_seed42": not args.skip_existing_upper_seed42,
            "upper_train_seeds": seeds,
            "upper_train_episodes": train_episodes,
            "upper_test_episodes": test_episodes,
            "baseline_train_episodes": 1,
        },
        "entries": entries,
        "statistics_path": str(reports_dir / "upper_multiseed_statistics.json"),
        "statistics": stats,
    }
    write_json(manifest_path, manifest)
    return {
        "phase": "upper",
        "run_root": str(run_root),
        "manifest_path": str(manifest_path),
        "reports_dir": str(reports_dir),
    }


def safe_weight_name(weight: float) -> str:
    return f"{weight:.3f}".replace(".", "p").replace("-", "m")


def summarize_runtime_policy(summary: dict[str, Any], policy: str) -> dict[str, float]:
    units = extract_policy_units(summary)
    policy_units = units.get(policy, [])
    if not policy_units:
        return {}
    metric_names = sorted({metric for unit in policy_units for metric in unit})
    return {
        metric: float(np.mean([unit[metric] for unit in policy_units if metric in unit]))
        for metric in metric_names
    }


def run_sensitivity_phase(args: argparse.Namespace) -> dict[str, Any]:
    weights = [float(weight) for weight in args.mbs_penalty_weights]
    name_prefix = args.sensitivity_name_prefix
    runtime_seeds = [int(seed) for seed in args.sensitivity_runtime_seeds]
    per_class_target = int(args.sensitivity_per_class_target)
    procedural_train_per_class = int(args.sensitivity_procedural_train_per_class)
    max_attempts = int(args.sensitivity_max_attempts)
    epochs = int(args.sensitivity_epochs)
    episodes_per_seed = int(args.sensitivity_episodes_per_seed)
    steps_per_episode = int(args.sensitivity_steps_per_episode)
    if args.smoke:
        weights = [weights[0]]
        name_prefix = f"{name_prefix}_smoke"
        runtime_seeds = [runtime_seeds[0]]
        per_class_target = 5
        procedural_train_per_class = 5
        max_attempts = 1000
        epochs = 1
        episodes_per_seed = 1
        steps_per_episode = 2

    if args.dry_run:
        return {
            "phase": "sensitivity",
            "weights": weights,
            "dry_run": True,
        }

    base_snapshot = snapshot_config()
    runs: list[dict[str, Any]] = []
    try:
        for weight in weights:
            restore_config(base_snapshot)
            config.OFFLOAD_ORACLE_MBS_LOAD_WEIGHT = float(weight)
            experiment_name = f"{name_prefix}_w{safe_weight_name(weight)}"
            print(f"[supplement] sensitivity weight={weight:.3f} experiment={experiment_name}")
            summary = run_full_offload_experiment(
                experiment_name=experiment_name,
                per_class_target=per_class_target,
                procedural_train_per_class=procedural_train_per_class,
                max_attempts=max_attempts,
                dataset_seed=int(args.sensitivity_dataset_seed),
                train_seed=int(args.sensitivity_train_seed),
                runtime_seeds=runtime_seeds,
                epochs=epochs,
                batch_size=int(args.sensitivity_batch_size),
                learning_rate=float(args.sensitivity_lr),
                val_ratio=float(args.sensitivity_val_ratio),
                hidden_dims=parse_hidden_dims(args.sensitivity_hidden_dims),
                device=args.device,
                sampler_mode=args.sampler_mode,
                episodes_per_seed=episodes_per_seed,
                steps_per_episode=steps_per_episode,
                label_mode="enhanced_oracle",
            )
            output_root = results_path("full_offload_experiments", experiment_name)
            policy_units = extract_policy_units(summary)
            reports_dir = output_root / "reports"
            stats = build_and_write_stats(
                policy_units=policy_units,
                reference="heuristic_offloading",
                output_json=reports_dir / "runtime_statistics_vs_heuristic.json",
                output_md=reports_dir / "runtime_statistics_vs_heuristic.md",
            )
            runs.append(
                {
                    "weight": float(weight),
                    "experiment_name": experiment_name,
                    "experiment_summary_path": str(output_root / "experiment_summary.json"),
                    "statistics_path": str(reports_dir / "runtime_statistics_vs_heuristic.json"),
                    "surrogate_baseline": summarize_runtime_policy(summary, "surrogate_baseline"),
                    "rich_reduced_runtime_policy": summarize_runtime_policy(summary, "rich_reduced_runtime_policy"),
                    "heuristic_offloading": summarize_runtime_policy(summary, "heuristic_offloading"),
                    "statistics": stats,
                }
            )
    finally:
        restore_config(base_snapshot)

    group_summary = {
        "metadata": {
            "name_prefix": name_prefix,
            "weights": weights,
            "runtime_seeds": runtime_seeds,
            "episodes_per_seed": episodes_per_seed,
            "steps_per_episode": steps_per_episode,
            "note": "One-dimensional MBS penalty scan; other oracle weights remain at config defaults.",
        },
        "runs": runs,
    }
    summary_path = results_path("full_offload_experiments", f"{name_prefix}_summary.json")
    write_json(summary_path, group_summary)
    return {
        "phase": "sensitivity",
        "summary_path": str(summary_path),
        "runs": [{"weight": run["weight"], "experiment_name": run["experiment_name"]} for run in runs],
    }


def run_figures_phase(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        ".\\run_all_thesis_figures.ps1",
    ]
    run_command(command, dry_run=args.dry_run)
    return {"phase": "figures", "command": command, "dry_run": bool(args.dry_run)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run journal-oriented supplementary UAV-MEC experiments.")
    parser.add_argument("--phase", choices=["all", "joint", "upper", "sensitivity", "figures"], default="all")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and planned outputs without running experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke version of the selected phase where supported.")

    parser.add_argument("--trajectory_run_root", default=str(results_path("full_runs", "wpt_fix_thesis_run")))
    parser.add_argument("--offload_experiment_root", default=str(results_path("full_offload_experiments", "wpt_fix_thesis_run_offload")))
    parser.add_argument("--joint_name", default="joint_four_way_journal")
    parser.add_argument(
        "--joint_eval_seeds",
        type=int,
        nargs="+",
        default=[42, 84, 126, 168, 210, 252, 294, 336, 378, 420],
    )
    parser.add_argument("--joint_episodes_per_seed", type=int, default=6)
    parser.add_argument("--joint_steps_per_episode", type=int, default=None)
    parser.add_argument("--record_spatial_trace", action="store_true")
    parser.add_argument("--spatial_trace_interval", type=int, default=20)

    parser.add_argument("--upper_run_name", default="supplement_upper_multiseed")
    parser.add_argument("--upper_train_seeds", type=int, nargs="+", default=[84, 126, 168])
    parser.add_argument("--upper_train_episodes", type=int, default=500)
    parser.add_argument("--upper_test_episodes", type=int, default=60)
    parser.add_argument("--skip_existing_upper_seed42", action="store_true")

    parser.add_argument("--sensitivity_name_prefix", default="mbs_penalty_sensitivity")
    parser.add_argument("--mbs_penalty_weights", type=float, nargs="+", default=[0.0, 0.065, 0.13])
    parser.add_argument("--sensitivity_runtime_seeds", type=int, nargs="+", default=[42, 84, 126, 168])
    parser.add_argument("--sensitivity_per_class_target", type=int, default=800)
    parser.add_argument("--sensitivity_procedural_train_per_class", type=int, default=600)
    parser.add_argument("--sensitivity_max_attempts", type=int, default=30000)
    parser.add_argument("--sensitivity_dataset_seed", type=int, default=42)
    parser.add_argument("--sensitivity_train_seed", type=int, default=42)
    parser.add_argument("--sensitivity_epochs", type=int, default=15)
    parser.add_argument("--sensitivity_batch_size", type=int, default=256)
    parser.add_argument("--sensitivity_lr", type=float, default=1e-3)
    parser.add_argument("--sensitivity_val_ratio", type=float, default=0.2)
    parser.add_argument("--sensitivity_hidden_dims", default="64,64")
    parser.add_argument("--sensitivity_episodes_per_seed", type=int, default=3)
    parser.add_argument("--sensitivity_steps_per_episode", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampler_mode", choices=["auto", "weighted", "none"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_order = ["upper", "joint", "sensitivity", "figures"] if args.phase == "all" else [args.phase]
    results: list[dict[str, Any]] = []
    for phase in phase_order:
        if phase == "upper":
            results.append(run_upper_phase(args))
        elif phase == "joint":
            results.append(run_joint_phase(args))
        elif phase == "sensitivity":
            results.append(run_sensitivity_phase(args))
        elif phase == "figures":
            results.append(run_figures_phase(args))
        else:
            raise ValueError(f"Unsupported phase: {phase}")

    manifest = {
        "phase": args.phase,
        "smoke": bool(args.smoke),
        "dry_run": bool(args.dry_run),
        "results": results,
    }
    manifest_path = results_path("supplement_experiments", "latest_manifest.json")
    if not args.dry_run:
        write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
