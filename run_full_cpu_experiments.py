from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_SEEDS = [42, 84, 126]
DEFAULT_WORKLOAD_SEEDS = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420]
SUMMARY_METRICS = (
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
    "service_learned_decision_count",
    "service_heuristic_decision_count",
    "service_fallback_count",
    "service_predict_exception_fallback_count",
)


@dataclass(frozen=True)
class Task:
    name: str
    cmd: list[str]
    done_paths: tuple[Path, ...]
    log_path: Path
    cwd: Path = PROJECT_ROOT


def timestamp_prefix(run_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_name)
    return f"cpu_full_{safe}"


def upper_timestamp(run_name: str, seed: int, episodes: int) -> str:
    return f"{timestamp_prefix(run_name)}_upper_attention_seed{seed}_{episodes}ep"


def lower_attention_timestamp(run_name: str, seed: int, episodes: int) -> str:
    return f"{timestamp_prefix(run_name)}_lower_attention_seed{seed}_{episodes}ep"


def lower_uncoord_timestamp(run_name: str, seed: int, episodes: int) -> str:
    return f"{timestamp_prefix(run_name)}_lower_uncoord_seed{seed}_{episodes}ep"


def full_hmarl_timestamp(run_name: str, seed: int, episodes: int) -> str:
    return f"{timestamp_prefix(run_name)}_full_hmarl_seed{seed}_{episodes}ep"


def lower_ablation_timestamp(run_name: str, ablation: str, seed: int, episodes: int) -> str:
    if ablation == "full":
        return lower_attention_timestamp(run_name, seed, episodes)
    return f"{timestamp_prefix(run_name)}_lower_{ablation}_attention_seed{seed}_{episodes}ep"


def report_path(timestamp: str) -> Path:
    return PROJECT_ROOT / "results" / "reports" / f"hierarchical_mappo_summary_{timestamp}.json"


def upper_model_dir(timestamp: str) -> Path:
    return PROJECT_ROOT / "saved_models" / f"attention_mappo_{timestamp}" / "final"


def lower_model_dir(timestamp: str) -> Path:
    return PROJECT_ROOT / "saved_models" / f"offload_mappo_{timestamp}" / "final"


def upper_model_file(timestamp: str) -> Path:
    return upper_model_dir(timestamp) / "attention_mappo.pth"


def lower_model_file(timestamp: str) -> Path:
    return lower_model_dir(timestamp) / "offload_mappo.pth"


def config_path_from_training_timestamp(timestamp: str) -> Path:
    return PROJECT_ROOT / "train_logs" / "hierarchical_mappo" / f"config_{timestamp}.json"


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def all_done(paths: tuple[Path, ...]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def status_path(state_dir: Path, task_name: str, suffix: str) -> Path:
    safe = task_name.replace("/", "__").replace(" ", "_")
    return state_dir / f"{safe}.{suffix}.json"


def write_status(state_dir: Path, task: Task, suffix: str, payload: dict[str, Any]) -> None:
    path = status_path(state_dir, task.name, suffix)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def remove_statuses(state_dir: Path, task: Task, suffixes: tuple[str, ...]) -> None:
    for suffix in suffixes:
        path = status_path(state_dir, task.name, suffix)
        if path.exists():
            path.unlink()


def run_task(task: Task, state_dir: Path, retries: int, env: dict[str, str]) -> dict[str, Any]:
    if all_done(task.done_paths):
        payload = {
            "task": task.name,
            "status": "skipped",
            "reason": "done paths already exist",
            "done_paths": [str(path) for path in task.done_paths],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_status(state_dir, task, "done", payload)
        return payload

    ensure_dirs(task.log_path.parent, state_dir)
    remove_statuses(state_dir, task, ("done", "failed"))
    last_returncode: int | None = None
    started = time.time()
    for attempt in range(1, retries + 2):
        running_payload = {
            "task": task.name,
            "status": "running",
            "attempt": attempt,
            "cmd": task.cmd,
            "cmd_pretty": shell_join(task.cmd),
            "log_path": str(task.log_path),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "done_paths": [str(path) for path in task.done_paths],
        }
        write_status(state_dir, task, "running", running_payload)
        with task.log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 100 + "\n")
            log.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')} attempt={attempt}\n")
            log.write(shell_join(task.cmd) + "\n")
            log.flush()
            proc = subprocess.run(
                task.cmd,
                cwd=str(task.cwd),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            last_returncode = int(proc.returncode)
            log.write(f"\n[end] {time.strftime('%Y-%m-%d %H:%M:%S')} returncode={last_returncode}\n")
            log.flush()

        if last_returncode == 0 and all_done(task.done_paths):
            remove_statuses(state_dir, task, ("running", "failed"))
            payload = {
                "task": task.name,
                "status": "done",
                "attempt": attempt,
                "returncode": last_returncode,
                "elapsed_sec": round(time.time() - started, 3),
                "cmd": task.cmd,
                "log_path": str(task.log_path),
                "done_paths": [str(path) for path in task.done_paths],
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_status(state_dir, task, "done", payload)
            return payload

    remove_statuses(state_dir, task, ("running",))
    payload = {
        "task": task.name,
        "status": "failed",
        "returncode": last_returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "cmd": task.cmd,
        "log_path": str(task.log_path),
        "done_paths": [str(path) for path in task.done_paths],
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_status(state_dir, task, "failed", payload)
    return payload


def run_tasks(tasks: list[Task], state_dir: Path, max_workers: int, retries: int, env: dict[str, str]) -> None:
    pending = [task for task in tasks if not all_done(task.done_paths)]
    skipped = len(tasks) - len(pending)
    if skipped:
        print(f"[skip] {skipped} task(s) already complete.")
    if not pending:
        return

    workers = max(1, min(max_workers, len(pending)))
    print(f"[run] {len(pending)} task(s), workers={workers}")
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(run_task, task, state_dir, retries, env): task for task in pending}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"task": task.name, "status": "failed", "exception": repr(exc), "log_path": str(task.log_path)}
                failures.append(result)
                print(f"[failed] {task.name}: {exc!r}")
                continue
            status = result.get("status")
            if status == "done":
                print(f"[done] {task.name}")
            elif status == "skipped":
                print(f"[skip] {task.name}")
            else:
                failures.append(result)
                print(f"[failed] {task.name}; see {result.get('log_path')}")

    if failures:
        failed_names = ", ".join(str(item.get("task")) for item in failures)
        raise RuntimeError(f"{len(failures)} task(s) failed: {failed_names}")


def python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def make_smoke_tasks(run_dir: Path) -> list[Task]:
    done = run_dir / "state" / "smoke.ok"
    cmd = python_cmd("smoke_offloading_consistency.py")
    return [
        Task(
            name="smoke_offloading_consistency",
            cmd=[sys.executable, "-c", (
                "import pathlib, subprocess, sys; "
                "rc=subprocess.call([sys.executable, 'smoke_offloading_consistency.py']); "
                "pathlib.Path(sys.argv[1]).write_text('ok\\n') if rc == 0 else None; "
                "sys.exit(rc)"
            ), str(done)],
            done_paths=(done,),
            log_path=run_dir / "logs" / "smoke_offloading_consistency.log",
        )
    ]


def make_upper_tasks(args: argparse.Namespace, run_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    for seed in args.training_seeds:
        ts = upper_timestamp(args.run_name, seed, args.train_episodes)
        cmd = python_cmd(
            "run_hierarchical_mappo_experiment.py",
            "--mode", "upper_only",
            "--trajectory_model", "attention_mappo",
            "--seed", str(seed),
            "--num_episodes", str(args.train_episodes),
            "--timestamp", ts,
        )
        tasks.append(
            Task(
                name=f"train_upper_attention_seed{seed}",
                cmd=cmd,
                done_paths=(report_path(ts), upper_model_file(ts)),
                log_path=run_dir / "logs" / f"train_upper_attention_seed{seed}.log",
            )
        )
    return tasks


def make_main_training_tasks(args: argparse.Namespace, run_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    for seed in args.training_seeds:
        upper_ts = upper_timestamp(args.run_name, seed, args.train_episodes)

        lower_att_ts = lower_attention_timestamp(args.run_name, seed, args.train_episodes)
        tasks.append(
            Task(
                name=f"train_lower_attention_seed{seed}",
                cmd=python_cmd(
                    "run_hierarchical_mappo_experiment.py",
                    "--mode", "lower_only_fixed_upper",
                    "--trajectory_model", "attention_mappo",
                    "--trajectory_model_dir", str(upper_model_dir(upper_ts)),
                    "--seed", str(seed),
                    "--num_episodes", str(args.train_episodes),
                    "--lower_ablation", "full",
                    "--timestamp", lower_att_ts,
                ),
                done_paths=(report_path(lower_att_ts), lower_model_file(lower_att_ts)),
                log_path=run_dir / "logs" / f"train_lower_attention_seed{seed}.log",
            )
        )

        lower_uncoord_ts = lower_uncoord_timestamp(args.run_name, seed, args.train_episodes)
        tasks.append(
            Task(
                name=f"train_lower_uncoord_seed{seed}",
                cmd=python_cmd(
                    "run_hierarchical_mappo_experiment.py",
                    "--mode", "lower_only_fixed_upper",
                    "--trajectory_model", "uncoordinated_greedy",
                    "--seed", str(seed),
                    "--num_episodes", str(args.train_episodes),
                    "--lower_ablation", "full",
                    "--timestamp", lower_uncoord_ts,
                ),
                done_paths=(report_path(lower_uncoord_ts), lower_model_file(lower_uncoord_ts)),
                log_path=run_dir / "logs" / f"train_lower_uncoord_seed{seed}.log",
            )
        )

        full_ts = full_hmarl_timestamp(args.run_name, seed, args.train_episodes)
        tasks.append(
            Task(
                name=f"train_full_hmarl_seed{seed}",
                cmd=python_cmd(
                    "run_hierarchical_mappo_experiment.py",
                    "--mode", "full_hierarchical",
                    "--trajectory_model", "attention_mappo",
                    "--seed", str(seed),
                    "--num_episodes", str(args.train_episodes),
                    "--lower_ablation", "full",
                    "--timestamp", full_ts,
                ),
                done_paths=(report_path(full_ts), upper_model_file(full_ts), lower_model_file(full_ts)),
                log_path=run_dir / "logs" / f"train_full_hmarl_seed{seed}.log",
            )
        )
    return tasks


def make_ablation_training_tasks(args: argparse.Namespace, run_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    for ablation in args.lower_ablations:
        for seed in args.training_seeds:
            upper_ts = upper_timestamp(args.run_name, seed, args.train_episodes)
            ts = lower_ablation_timestamp(args.run_name, ablation, seed, args.train_episodes)
            tasks.append(
                Task(
                    name=f"train_lower_{ablation}_seed{seed}",
                    cmd=python_cmd(
                        "run_hierarchical_mappo_experiment.py",
                        "--mode", "lower_only_fixed_upper",
                        "--trajectory_model", "attention_mappo",
                        "--trajectory_model_dir", str(upper_model_dir(upper_ts)),
                        "--seed", str(seed),
                        "--num_episodes", str(args.train_episodes),
                        "--lower_ablation", ablation,
                        "--timestamp", ts,
                    ),
                    done_paths=(report_path(ts), lower_model_file(ts)),
                    log_path=run_dir / "logs" / f"train_lower_{ablation}_seed{seed}.log",
                )
            )
    return tasks


def seed_map_entries(label: str, values: dict[int, Path]) -> list[str]:
    return [f"{label}@{seed}={path}" for seed, path in sorted(values.items())]


def main_combo_specs(include_oracle: bool) -> list[str]:
    combos = [
        "uncoordinated_greedy__heuristic:uncoordinated_greedy:heuristic",
        "attention_mappo__heuristic:attention_mappo:heuristic",
        "uncoordinated_greedy__lower_mappo:uncoordinated_greedy:lower_mappo",
        "attention_mappo__lower_mappo:attention_mappo:lower_mappo",
        "full_hierarchical_marl:attention_mappo:lower_mappo",
    ]
    if include_oracle:
        combos.insert(2, "attention_mappo__oracle_guided:attention_mappo:oracle_guided")
    return combos


def make_eval_tasks(
    args: argparse.Namespace,
    run_dir: Path,
    suite: str,
    combos: list[str],
    lower_dirs_by_label: dict[str, dict[int, Path]],
    lower_names: dict[str, str] | None = None,
    trajectory_model_dirs_by_label: dict[str, dict[int, Path]] | None = None,
    trajectory_configs_by_label: dict[str, dict[int, Path]] | None = None,
) -> list[Task]:
    tasks: list[Task] = []
    lower_names = lower_names or {}
    trajectory_model_dirs_by_label = trajectory_model_dirs_by_label or {}
    trajectory_configs_by_label = trajectory_configs_by_label or {}
    for workload_seed in args.workload_seeds:
        name = f"{args.run_name}_{suite}_workload{workload_seed}"
        summary = PROJECT_ROOT / "results" / "joint_experiments" / name / "joint_experiment_summary.json"
        cmd = python_cmd(
            "run_joint_trajectory_offload_experiment.py",
            "--name", name,
            "--trajectory_run_root", ".",
            "--combos", *combos,
            "--training_seeds", *[str(seed) for seed in args.training_seeds],
            "--seeds", str(workload_seed),
            "--episodes_per_seed", str(args.episodes_per_seed),
            "--steps_per_episode", str(args.steps_per_episode),
            "--comparison_smoothing", str(args.comparison_smoothing),
        )
        lower_entries: list[str] = []
        for label, mapping in lower_dirs_by_label.items():
            lower_entries.extend(seed_map_entries(label, mapping))
        if lower_entries:
            cmd.extend(["--lower_model_dirs", *lower_entries])

        lower_name_entries = [f"{label}={model_name}" for label, model_name in sorted(lower_names.items())]
        if lower_name_entries:
            cmd.extend(["--lower_model_names", *lower_name_entries])

        traj_dir_entries: list[str] = []
        for label, mapping in trajectory_model_dirs_by_label.items():
            traj_dir_entries.extend(seed_map_entries(label, mapping))
        if traj_dir_entries:
            cmd.extend(["--trajectory_model_dirs", *traj_dir_entries])

        traj_config_entries: list[str] = []
        for label, mapping in trajectory_configs_by_label.items():
            traj_config_entries.extend(seed_map_entries(label, mapping))
        if traj_config_entries:
            cmd.extend(["--trajectory_configs", *traj_config_entries])

        if args.include_oracle:
            if not args.surrogate_checkpoint:
                raise ValueError("--include_oracle requires --surrogate_checkpoint PATH.")
            cmd.extend(["--surrogate_checkpoint", str(args.surrogate_checkpoint)])

        tasks.append(
            Task(
                name=f"eval_{suite}_workload{workload_seed}",
                cmd=cmd,
                done_paths=(summary,),
                log_path=run_dir / "logs" / f"eval_{suite}_workload{workload_seed}.log",
            )
        )
    return tasks


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {metric_name: {"mean": 0.0, "std": 0.0} for metric_name in SUMMARY_METRICS}
    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in SUMMARY_METRICS:
        values = [float(entry[metric_name]) for entry in metric_dicts if metric_name in entry]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        aggregate[metric_name] = {
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
        }
    return aggregate


def build_delta_vs_reference(results_by_policy: dict[str, dict[str, Any]], reference_policy: str) -> dict[str, dict[str, dict[str, float]]]:
    if reference_policy not in results_by_policy:
        return {}
    reference_units = [entry["per_seed_mean"] for entry in results_by_policy[reference_policy]["per_seed"]]
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for policy_label, details in results_by_policy.items():
        if policy_label == reference_policy:
            continue
        policy_units = [entry["per_seed_mean"] for entry in details["per_seed"]]
        metric_deltas: dict[str, dict[str, float]] = {}
        for metric_name in SUMMARY_METRICS:
            paired = [
                float(policy_unit[metric_name]) - float(reference_unit[metric_name])
                for policy_unit, reference_unit in zip(policy_units, reference_units)
                if metric_name in policy_unit and metric_name in reference_unit
            ]
            mean = sum(paired) / len(paired) if paired else 0.0
            variance = sum((value - mean) ** 2 for value in paired) / len(paired) if paired else 0.0
            metric_deltas[metric_name] = {
                "mean_delta": float(mean),
                "std_delta": float(math.sqrt(variance)),
            }
        deltas[policy_label] = metric_deltas
    return deltas


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_eval_summaries(args: argparse.Namespace, run_dir: Path, suite: str, reference: str) -> Path:
    shard_paths = [
        PROJECT_ROOT / "results" / "joint_experiments" / f"{args.run_name}_{suite}_workload{seed}" / "joint_experiment_summary.json"
        for seed in args.workload_seeds
    ]
    missing = [path for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot merge {suite}; missing shard summaries: {missing[:3]}")

    shard_summaries = [load_summary(path) for path in shard_paths]
    first = shard_summaries[0]
    combo_labels = list(first.get("per_combo", first.get("per_policy", {})).keys())
    merged_per_combo: dict[str, dict[str, Any]] = {}
    for label in combo_labels:
        pieces = [summary.get("per_combo", summary.get("per_policy", {}))[label] for summary in shard_summaries]
        per_seed = []
        for piece in pieces:
            per_seed.extend(piece.get("per_seed", []))
        aggregate_units = [entry["per_seed_mean"] for entry in per_seed]
        template = dict(pieces[0])
        template["per_seed"] = per_seed
        template["aggregate"] = aggregate_metric_dicts(aggregate_units)
        template["merged_from"] = [str(path) for path in shard_paths]
        merged_per_combo[label] = template

    merged = {
        "metadata": dict(first.get("metadata", {})),
        "per_policy": merged_per_combo,
        "per_combo": merged_per_combo,
        "delta_vs_reference": build_delta_vs_reference(merged_per_combo, reference),
        "merged_from": [str(path) for path in shard_paths],
    }
    merged["metadata"].update(
        {
            "experiment_name": f"{args.run_name}_{suite}_merged",
            "training_seeds": [int(seed) for seed in args.training_seeds],
            "workload_seeds": [int(seed) for seed in args.workload_seeds],
            "episodes_per_seed": int(args.episodes_per_seed),
            "steps_per_episode": int(args.steps_per_episode),
            "suite": suite,
        }
    )

    out_dir = PROJECT_ROOT / "results" / "joint_experiments" / f"{args.run_name}_{suite}_merged"
    ensure_dirs(out_dir)
    summary_path = out_dir / "joint_experiment_summary.json"
    summary_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    stats_json = out_dir / "statistics.json"
    stats_md = out_dir / "statistics.md"
    subprocess.run(
        [
            sys.executable,
            "analyze_experiment_statistics.py",
            str(summary_path),
            "--reference",
            reference,
            "--output_json",
            str(stats_json),
            "--output_md",
            str(stats_md),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    print(f"[merge] {suite}: {summary_path}")
    print(f"[stats] {suite}: {stats_md}")
    return summary_path


def create_main_eval_tasks(args: argparse.Namespace, run_dir: Path) -> list[Task]:
    trajectory_dirs = {
        "attention_mappo__heuristic": {
            seed: upper_model_dir(upper_timestamp(args.run_name, seed, args.train_episodes)) for seed in args.training_seeds
        },
        "attention_mappo__lower_mappo": {
            seed: upper_model_dir(upper_timestamp(args.run_name, seed, args.train_episodes)) for seed in args.training_seeds
        },
        "full_hierarchical_marl": {
            seed: upper_model_dir(full_hmarl_timestamp(args.run_name, seed, args.train_episodes)) for seed in args.training_seeds
        },
    }
    if args.include_oracle:
        trajectory_dirs["attention_mappo__oracle_guided"] = {
            seed: upper_model_dir(upper_timestamp(args.run_name, seed, args.train_episodes)) for seed in args.training_seeds
        }

    trajectory_configs = {
        label: {
            seed: config_path_from_training_timestamp(
                full_hmarl_timestamp(args.run_name, seed, args.train_episodes)
                if label == "full_hierarchical_marl"
                else upper_timestamp(args.run_name, seed, args.train_episodes)
            )
            for seed in args.training_seeds
        }
        for label in trajectory_dirs
    }

    lower_dirs = {
        "uncoordinated_greedy__lower_mappo": {
            seed: lower_model_dir(lower_uncoord_timestamp(args.run_name, seed, args.train_episodes)) for seed in args.training_seeds
        },
        "attention_mappo__lower_mappo": {
            seed: lower_model_dir(lower_attention_timestamp(args.run_name, seed, args.train_episodes)) for seed in args.training_seeds
        },
        "full_hierarchical_marl": {
            seed: lower_model_dir(full_hmarl_timestamp(args.run_name, seed, args.train_episodes)) for seed in args.training_seeds
        },
    }
    return make_eval_tasks(
        args,
        run_dir,
        suite="main",
        combos=main_combo_specs(args.include_oracle),
        lower_dirs_by_label=lower_dirs,
        trajectory_model_dirs_by_label=trajectory_dirs,
        trajectory_configs_by_label=trajectory_configs,
    )


def create_ablation_eval_tasks(args: argparse.Namespace, run_dir: Path) -> list[Task]:
    combos = [f"lower_{ablation}:attention_mappo:lower_mappo" for ablation in args.lower_ablations]
    trajectory_dirs = {
        f"lower_{ablation}": {
            seed: upper_model_dir(upper_timestamp(args.run_name, seed, args.train_episodes)) for seed in args.training_seeds
        }
        for ablation in args.lower_ablations
    }
    trajectory_configs = {
        f"lower_{ablation}": {
            seed: config_path_from_training_timestamp(upper_timestamp(args.run_name, seed, args.train_episodes))
            for seed in args.training_seeds
        }
        for ablation in args.lower_ablations
    }
    lower_dirs = {
        f"lower_{ablation}": {
            seed: lower_model_dir(lower_ablation_timestamp(args.run_name, ablation, seed, args.train_episodes))
            for seed in args.training_seeds
        }
        for ablation in args.lower_ablations
    }
    lower_names = {
        f"lower_{ablation}": ("no_attention_offload_mappo" if ablation == "no_attention" else "constrained_attention_offload_mappo")
        for ablation in args.lower_ablations
    }
    return make_eval_tasks(
        args,
        run_dir,
        suite="ablation",
        combos=combos,
        lower_dirs_by_label=lower_dirs,
        lower_names=lower_names,
        trajectory_model_dirs_by_label=trajectory_dirs,
        trajectory_configs_by_label=trajectory_configs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full CPU-parallel UAV-MEC experiments with resumable task state. "
            "Re-run the same command after interruption; completed tasks are skipped."
        )
    )
    parser.add_argument("--run_name", default=time.strftime("formula_fixed_%Y%m%d_%H%M%S"))
    parser.add_argument("--suite", choices=["main", "ablation", "all"], default="all")
    parser.add_argument("--training_seeds", type=int, nargs="+", default=DEFAULT_TRAINING_SEEDS)
    parser.add_argument("--workload_seeds", type=int, nargs="+", default=DEFAULT_WORKLOAD_SEEDS)
    parser.add_argument("--train_episodes", type=int, default=200)
    parser.add_argument("--episodes_per_seed", type=int, default=6)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--lower_ablations", nargs="+", default=["full", "no_mask", "no_lagrange", "no_attention"])
    parser.add_argument("--max_workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--torch_threads", type=int, default=None)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--comparison_smoothing", type=int, default=3)
    parser.add_argument("--include_oracle", action="store_true")
    parser.add_argument("--surrogate_checkpoint", type=Path, default=None)
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    cpu_count = os.cpu_count() or 1
    workers = max(1, int(args.max_workers))
    torch_threads = args.torch_threads
    if torch_threads is None:
        torch_threads = max(1, math.floor(cpu_count / workers))
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(torch_threads),
            "MKL_NUM_THREADS": str(torch_threads),
            "OPENBLAS_NUM_THREADS": str(torch_threads),
            "NUMEXPR_NUM_THREADS": str(torch_threads),
            "VECLIB_MAXIMUM_THREADS": str(torch_threads),
        }
    )
    return env


def print_plan(tasks: list[Task]) -> None:
    for task in tasks:
        state = "done" if all_done(task.done_paths) else "pending"
        print(f"[{state}] {task.name}")
        print(f"        {shell_join(task.cmd)}")
        print(f"        log: {task.log_path}")


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    run_dir = PROJECT_ROOT / "results" / "full_cpu_runs" / args.run_name
    state_dir = run_dir / "state"
    ensure_dirs(run_dir / "logs", state_dir)
    env = build_env(args)

    metadata = {
        "run_name": args.run_name,
        "suite": args.suite,
        "training_seeds": args.training_seeds,
        "workload_seeds": args.workload_seeds,
        "train_episodes": args.train_episodes,
        "episodes_per_seed": args.episodes_per_seed,
        "steps_per_episode": args.steps_per_episode,
        "max_workers": args.max_workers,
        "thread_env": {key: env[key] for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.include_oracle and not args.surrogate_checkpoint:
        raise ValueError("--include_oracle requires --surrogate_checkpoint PATH.")

    stages: list[tuple[str, list[Task]]] = []
    if not args.skip_smoke:
        stages.append(("smoke", make_smoke_tasks(run_dir)))
    stages.append(("upper", make_upper_tasks(args, run_dir)))

    if args.suite in {"main", "all"}:
        stages.append(("main_training", make_main_training_tasks(args, run_dir)))
        stages.append(("main_eval", create_main_eval_tasks(args, run_dir)))
    if args.suite in {"ablation", "all"}:
        stages.append(("ablation_training", make_ablation_training_tasks(args, run_dir)))
        stages.append(("ablation_eval", create_ablation_eval_tasks(args, run_dir)))

    if args.dry_run:
        print(f"Run directory: {run_dir}")
        for stage_name, tasks in stages:
            print(f"\n## {stage_name}")
            print_plan(tasks)
        return

    for stage_name, tasks in stages:
        print(f"\n## stage: {stage_name}")
        run_tasks(tasks, state_dir, int(args.max_workers), int(args.retries), env)
        if stage_name == "main_eval":
            merge_eval_summaries(args, run_dir, suite="main", reference="uncoordinated_greedy__heuristic")
        if stage_name == "ablation_eval":
            merge_eval_summaries(args, run_dir, suite="ablation", reference="lower_full")

    print(f"\n[complete] run_name={args.run_name}")
    print(f"[logs] {run_dir / 'logs'}")
    print(f"[state] {state_dir}")


if __name__ == "__main__":
    main()
