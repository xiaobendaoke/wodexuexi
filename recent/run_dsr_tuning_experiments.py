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
DEFAULT_WORKLOAD_SEEDS = [42, 84, 126, 168]
ALL_WORKLOAD_SEEDS = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420]


PROFILES: dict[str, dict[str, float | str | bool]] = {
    "current": {},
    "dsr_balanced": {
        "OFFLOAD_REWARD_DEADLINE_WEIGHT": 5.0,
        "OFFLOAD_REWARD_SUCCESS_BONUS": 1.0,
        "OFFLOAD_REWARD_LATENCY_WEIGHT": 0.55,
        "OFFLOAD_REWARD_ENERGY_WEIGHT": 0.06,
        "OFFLOAD_REWARD_MBS_WEIGHT": 0.35,
        "OFFLOAD_DSR_TARGET": 0.24,
        "OFFLOAD_LAGRANGE_LR": 0.25,
        "OFFLOAD_LAGRANGE_MAX": 80.0,
        "OFFLOAD_COOP_MAX_DEADLINE_RATIO": 1.20,
        "OFFLOAD_COOP_MAX_RELATIVE_LATENCY": 1.20,
    },
    "dsr_strong": {
        "OFFLOAD_REWARD_DEADLINE_WEIGHT": 7.0,
        "OFFLOAD_REWARD_SUCCESS_BONUS": 1.5,
        "OFFLOAD_REWARD_LATENCY_WEIGHT": 0.75,
        "OFFLOAD_REWARD_ENERGY_WEIGHT": 0.04,
        "OFFLOAD_REWARD_MBS_WEIGHT": 0.30,
        "OFFLOAD_DSR_TARGET": 0.25,
        "OFFLOAD_LAGRANGE_LR": 0.50,
        "OFFLOAD_LAGRANGE_MAX": 120.0,
        "OFFLOAD_COOP_MAX_DEADLINE_RATIO": 1.10,
        "OFFLOAD_COOP_MAX_RELATIVE_LATENCY": 1.10,
    },
    "latency_first": {
        "OFFLOAD_REWARD_DEADLINE_WEIGHT": 6.0,
        "OFFLOAD_REWARD_SUCCESS_BONUS": 1.2,
        "OFFLOAD_REWARD_LATENCY_WEIGHT": 1.0,
        "OFFLOAD_REWARD_ENERGY_WEIGHT": 0.03,
        "OFFLOAD_REWARD_MBS_WEIGHT": 0.25,
        "OFFLOAD_DSR_TARGET": 0.24,
        "OFFLOAD_LAGRANGE_LR": 0.35,
        "OFFLOAD_LAGRANGE_MAX": 100.0,
        "OFFLOAD_COOP_MAX_DEADLINE_RATIO": 1.05,
        "OFFLOAD_COOP_MAX_RELATIVE_LATENCY": 1.05,
    },
    "dsr_mbs_relaxed": {
        "OFFLOAD_REWARD_DEADLINE_WEIGHT": 6.0,
        "OFFLOAD_REWARD_SUCCESS_BONUS": 1.8,
        "OFFLOAD_REWARD_LATENCY_WEIGHT": 0.70,
        "OFFLOAD_REWARD_ENERGY_WEIGHT": 0.025,
        "OFFLOAD_REWARD_MBS_WEIGHT": 0.10,
        "OFFLOAD_DSR_TARGET": 0.245,
        "OFFLOAD_LAGRANGE_LR": 0.35,
        "OFFLOAD_LAGRANGE_MAX": 100.0,
        "OFFLOAD_COOP_MAX_DEADLINE_RATIO": 1.15,
        "OFFLOAD_COOP_MAX_RELATIVE_LATENCY": 1.15,
    },
    "dsr_energy_light": {
        "OFFLOAD_REWARD_DEADLINE_WEIGHT": 7.5,
        "OFFLOAD_REWARD_SUCCESS_BONUS": 2.2,
        "OFFLOAD_REWARD_LATENCY_WEIGHT": 0.80,
        "OFFLOAD_REWARD_ENERGY_WEIGHT": 0.01,
        "OFFLOAD_REWARD_MBS_WEIGHT": 0.15,
        "OFFLOAD_DSR_TARGET": 0.25,
        "OFFLOAD_LAGRANGE_LR": 0.40,
        "OFFLOAD_LAGRANGE_MAX": 120.0,
        "OFFLOAD_COOP_MAX_DEADLINE_RATIO": 1.12,
        "OFFLOAD_COOP_MAX_RELATIVE_LATENCY": 1.12,
    },
    "dsr_mbs_push": {
        "OFFLOAD_REWARD_DEADLINE_WEIGHT": 8.5,
        "OFFLOAD_REWARD_SUCCESS_BONUS": 2.8,
        "OFFLOAD_REWARD_LATENCY_WEIGHT": 0.90,
        "OFFLOAD_REWARD_ENERGY_WEIGHT": 0.015,
        "OFFLOAD_REWARD_MBS_WEIGHT": 0.05,
        "OFFLOAD_DSR_TARGET": 0.25,
        "OFFLOAD_LAGRANGE_LR": 0.45,
        "OFFLOAD_LAGRANGE_MAX": 140.0,
        "OFFLOAD_COOP_MAX_DEADLINE_RATIO": 1.08,
        "OFFLOAD_COOP_MAX_RELATIVE_LATENCY": 1.08,
    },
}


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


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def all_done(paths: tuple[Path, ...]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def safe_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)


def status_path(state_dir: Path, task_name: str, suffix: str) -> Path:
    return state_dir / f"{safe_name(task_name)}.{suffix}.json"


def write_status(state_dir: Path, task: Task, suffix: str, payload: dict[str, Any]) -> None:
    status_path(state_dir, task.name, suffix).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def remove_statuses(state_dir: Path, task: Task) -> None:
    for suffix in ("running", "done", "failed"):
        path = status_path(state_dir, task.name, suffix)
        if path.exists():
            path.unlink()


def run_task(task: Task, state_dir: Path, env: dict[str, str], retries: int) -> dict[str, Any]:
    if all_done(task.done_paths):
        payload = {"task": task.name, "status": "skipped", "done_paths": [str(path) for path in task.done_paths]}
        write_status(state_dir, task, "done", payload)
        return payload

    ensure_dirs(task.log_path.parent, state_dir)
    remove_statuses(state_dir, task)
    start = time.time()
    last_returncode: int | None = None
    for attempt in range(1, retries + 2):
        write_status(
            state_dir,
            task,
            "running",
            {
                "task": task.name,
                "status": "running",
                "attempt": attempt,
                "cmd": task.cmd,
                "log_path": str(task.log_path),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        with task.log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 100 + "\n")
            log.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')} attempt={attempt}\n")
            log.write(shell_join(task.cmd) + "\n")
            log.flush()
            proc = subprocess.run(task.cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            last_returncode = int(proc.returncode)
            log.write(f"\n[end] {time.strftime('%Y-%m-%d %H:%M:%S')} returncode={last_returncode}\n")
            log.flush()
        if last_returncode == 0 and all_done(task.done_paths):
            remove_statuses(state_dir, task)
            payload = {
                "task": task.name,
                "status": "done",
                "attempt": attempt,
                "elapsed_sec": round(time.time() - start, 3),
                "done_paths": [str(path) for path in task.done_paths],
                "log_path": str(task.log_path),
            }
            write_status(state_dir, task, "done", payload)
            return payload

    remove_statuses(state_dir, task)
    payload = {
        "task": task.name,
        "status": "failed",
        "returncode": last_returncode,
        "elapsed_sec": round(time.time() - start, 3),
        "done_paths": [str(path) for path in task.done_paths],
        "log_path": str(task.log_path),
    }
    write_status(state_dir, task, "failed", payload)
    return payload


def run_tasks(tasks: list[Task], state_dir: Path, max_workers: int, retries: int, env: dict[str, str]) -> None:
    pending = [task for task in tasks if not all_done(task.done_paths)]
    if len(tasks) != len(pending):
        print(f"[skip] {len(tasks) - len(pending)} task(s) already complete.")
    if not pending:
        return
    workers = max(1, min(int(max_workers), len(pending)))
    print(f"[run] {len(pending)} task(s), workers={workers}")
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {pool.submit(run_task, task, state_dir, env, retries): task for task in pending}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            if result.get("status") in {"done", "skipped"}:
                print(f"[{result.get('status')}] {task.name}")
            else:
                failures.append(result)
                print(f"[failed] {task.name}; see {task.log_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} task(s) failed: {', '.join(str(item['task']) for item in failures)}")


def source_upper_dir(source_run: str, source_episodes: int, seed: int) -> Path:
    timestamp = f"cpu_full_{source_run}_upper_attention_seed{seed}_{source_episodes}ep"
    return PROJECT_ROOT / "saved_models" / f"attention_mappo_{timestamp}" / "final"


def tuned_timestamp(run_name: str, profile: str, seed: int, episodes: int) -> str:
    return f"dsr_tune_{safe_name(run_name)}_{profile}_lower_attention_seed{seed}_{episodes}ep"


def tuned_lower_dir(run_name: str, profile: str, seed: int, episodes: int) -> Path:
    return PROJECT_ROOT / "saved_models" / f"offload_mappo_{tuned_timestamp(run_name, profile, seed, episodes)}" / "final"


def tuned_summary_path(run_name: str, profile: str, seed: int, episodes: int) -> Path:
    return PROJECT_ROOT / "results" / "reports" / f"hierarchical_mappo_summary_{tuned_timestamp(run_name, profile, seed, episodes)}.json"


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def train_inline_code() -> str:
    return r"""
import json
import sys

import config
from run_hierarchical_mappo_experiment import train_hierarchical_mappo

payload = json.loads(sys.argv[1])
for key, value in payload["overrides"].items():
    setattr(config, key, value)

summary = train_hierarchical_mappo(
    num_episodes=int(payload["num_episodes"]),
    timestamp=payload["timestamp"],
    mode="lower_only_fixed_upper",
    seed=int(payload["seed"]),
    trajectory_model_name="attention_mappo",
    trajectory_model_dir=payload["trajectory_model_dir"],
    lower_ablation="full",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
"""


def eval_inline_code() -> str:
    return r"""
import json
import sys
from pathlib import Path

import config
from run_joint_trajectory_offload_experiment import evaluate_joint_policy

payload = json.loads(sys.argv[1])
for key, value in payload["overrides"].items():
    setattr(config, key, value)

run_root = Path(payload["run_root"])
run_root.mkdir(parents=True, exist_ok=True)
result = evaluate_joint_policy(
    combo_label=payload["profile"],
    policy_label="lower_mappo",
    checkpoint_path=None,
    trajectory_model_name="attention_mappo",
    trajectory_model_dir={str(seed): path for seed, path in payload["trajectory_model_dirs"].items()},
    trajectory_config_path=None,
    lower_model_dir={str(seed): path for seed, path in payload["lower_model_dirs"].items()},
    lower_model_name="constrained_attention_offload_mappo",
    training_seeds=[int(seed) for seed in payload["training_seeds"]],
    seeds=[int(payload["workload_seed"])],
    episodes_per_seed=int(payload["episodes_per_seed"]),
    steps_per_episode=int(payload["steps_per_episode"]),
    run_root=run_root,
)
summary = {
    "metadata": payload["metadata"],
    "per_policy": {payload["profile"]: result},
    "per_combo": {payload["profile"]: result},
}
summary_path = run_root / "joint_experiment_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary["metadata"], indent=2, ensure_ascii=False))
"""


def make_train_tasks(args: argparse.Namespace, profiles: list[str], run_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    for profile in profiles:
        overrides = PROFILES[profile]
        for seed in args.training_seeds:
            upper_dir = source_upper_dir(args.source_run, args.source_episodes, seed)
            timestamp = tuned_timestamp(args.run_name, profile, seed, args.train_episodes)
            payload = {
                "profile": profile,
                "overrides": overrides,
                "seed": int(seed),
                "num_episodes": int(args.train_episodes),
                "timestamp": timestamp,
                "trajectory_model_dir": str(upper_dir),
            }
            tasks.append(
                Task(
                    name=f"train_{profile}_seed{seed}",
                    cmd=[sys.executable, "-c", train_inline_code(), dumps_json(payload)],
                    done_paths=(tuned_summary_path(args.run_name, profile, seed, args.train_episodes), tuned_lower_dir(args.run_name, profile, seed, args.train_episodes) / "offload_mappo.pth"),
                    log_path=run_dir / "logs" / f"train_{profile}_seed{seed}.log",
                )
            )
    return tasks


def make_eval_tasks(args: argparse.Namespace, profiles: list[str], run_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    for profile in profiles:
        for workload_seed in args.workload_seeds:
            run_root = PROJECT_ROOT / "results" / "dsr_tuning" / args.run_name / f"{profile}_workload{workload_seed}"
            payload = {
                "profile": profile,
                "overrides": PROFILES[profile],
                "training_seeds": [int(seed) for seed in args.training_seeds],
                "workload_seed": int(workload_seed),
                "episodes_per_seed": int(args.episodes_per_seed),
                "steps_per_episode": int(args.steps_per_episode),
                "run_root": str(run_root),
                "trajectory_model_dirs": {
                    str(seed): str(source_upper_dir(args.source_run, args.source_episodes, int(seed)))
                    for seed in args.training_seeds
                },
                "lower_model_dirs": {
                    str(seed): str(tuned_lower_dir(args.run_name, profile, int(seed), args.train_episodes))
                    for seed in args.training_seeds
                },
                "metadata": {
                    "run_name": args.run_name,
                    "profile": profile,
                    "profile_overrides": PROFILES[profile],
                    "source_run": args.source_run,
                    "training_seeds": [int(seed) for seed in args.training_seeds],
                    "workload_seed": int(workload_seed),
                    "episodes_per_seed": int(args.episodes_per_seed),
                    "steps_per_episode": int(args.steps_per_episode),
                },
            }
            tasks.append(
                Task(
                    name=f"eval_{profile}_workload{workload_seed}",
                    cmd=[sys.executable, "-c", eval_inline_code(), dumps_json(payload)],
                    done_paths=(run_root / "joint_experiment_summary.json",),
                    log_path=run_dir / "logs" / f"eval_{profile}_workload{workload_seed}.log",
                )
            )
    return tasks


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        values = [float(item[metric]) for item in metric_dicts if metric in item]
        if not values:
            out[metric] = {"mean": 0.0, "std": 0.0}
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        out[metric] = {"mean": float(mean), "std": float(math.sqrt(variance))}
    return out


def merge_results(args: argparse.Namespace, profiles: list[str]) -> Path:
    merged_per_combo: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        pieces = []
        for workload_seed in args.workload_seeds:
            path = PROJECT_ROOT / "results" / "dsr_tuning" / args.run_name / f"{profile}_workload{workload_seed}" / "joint_experiment_summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
            summary = json.loads(path.read_text(encoding="utf-8"))
            pieces.append(summary["per_combo"][profile])
        per_seed = []
        for piece in pieces:
            per_seed.extend(piece["per_seed"])
        aggregate_units = [entry["per_seed_mean"] for entry in per_seed]
        template = dict(pieces[0])
        template["per_seed"] = per_seed
        template["aggregate"] = aggregate_metric_dicts(aggregate_units)
        merged_per_combo[profile] = template

    out_dir = PROJECT_ROOT / "results" / "dsr_tuning" / args.run_name / "merged"
    ensure_dirs(out_dir)
    summary = {
        "metadata": {
            "run_name": args.run_name,
            "profiles": profiles,
            "profile_overrides": {profile: PROFILES[profile] for profile in profiles},
            "source_run": args.source_run,
            "training_seeds": [int(seed) for seed in args.training_seeds],
            "workload_seeds": [int(seed) for seed in args.workload_seeds],
            "train_episodes": int(args.train_episodes),
            "episodes_per_seed": int(args.episodes_per_seed),
            "steps_per_episode": int(args.steps_per_episode),
        },
        "per_combo": merged_per_combo,
        "per_policy": merged_per_combo,
    }
    summary_path = out_dir / "joint_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    stats_md = out_dir / "statistics.md"
    stats_json = out_dir / "statistics.json"
    subprocess.run(
        [
            sys.executable,
            "analyze_experiment_statistics.py",
            str(summary_path),
            "--reference",
            args.reference_profile,
            "--output_json",
            str(stats_json),
            "--output_md",
            str(stats_md),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    print(f"[merge] {summary_path}")
    print(f"[stats] {stats_md}")
    print_summary_table(summary)
    return summary_path


def print_summary_table(summary: dict[str, Any]) -> None:
    print("\nProfile summary:")
    for profile, details in summary["per_combo"].items():
        agg = details["aggregate"]
        print(
            profile,
            "reward=", round(agg["reward"]["mean"], 3),
            "DSR=", round(agg["deadline_satisfaction_rate"]["mean"], 4),
            "latency=", round(agg["latency"]["mean"], 2),
            "energy=", round(agg["energy"]["mean"], 2),
            "MBS=", round(agg["mbs_load_ratio"]["mean"], 4),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-upper DSR/latency tuning sweeps for lower MAPPO.")
    parser.add_argument("--run_name", default=time.strftime("dsr_tune_%Y%m%d_%H%M%S"))
    parser.add_argument("--profiles", nargs="+", default=["current", "dsr_balanced", "dsr_strong", "latency_first"], choices=sorted(PROFILES))
    parser.add_argument("--source_run", default="formula_fixed_full_20260511", help="Existing run_name that provides trained upper attention_mappo models.")
    parser.add_argument("--source_episodes", type=int, default=200)
    parser.add_argument("--training_seeds", type=int, nargs="+", default=DEFAULT_TRAINING_SEEDS)
    parser.add_argument("--workload_seeds", type=int, nargs="+", default=DEFAULT_WORKLOAD_SEEDS)
    parser.add_argument("--train_episodes", type=int, default=60)
    parser.add_argument("--episodes_per_seed", type=int, default=3)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--max_workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--torch_threads", type=int, default=None)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--reference_profile", default="current", choices=sorted(PROFILES))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--all_workloads", action="store_true", help="Use the full 10 workload seeds instead of the quick tuning set.")
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    workers = max(1, int(args.max_workers))
    torch_threads = args.torch_threads
    if torch_threads is None:
        torch_threads = max(1, (os.cpu_count() or 1) // workers)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(torch_threads),
            "MKL_NUM_THREADS": str(torch_threads),
            "OPENBLAS_NUM_THREADS": str(torch_threads),
            "NUMEXPR_NUM_THREADS": str(torch_threads),
        }
    )
    return env


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    if args.all_workloads:
        args.workload_seeds = ALL_WORKLOAD_SEEDS
    profiles = list(dict.fromkeys(args.profiles))
    if args.reference_profile not in profiles:
        profiles.insert(0, args.reference_profile)

    for seed in args.training_seeds:
        upper_dir = source_upper_dir(args.source_run, args.source_episodes, int(seed))
        if not (upper_dir / "attention_mappo.pth").exists():
            raise FileNotFoundError(f"Upper model missing for seed {seed}: {upper_dir / 'attention_mappo.pth'}")

    run_dir = PROJECT_ROOT / "results" / "dsr_tuning" / args.run_name
    state_dir = run_dir / "state"
    ensure_dirs(run_dir / "logs", state_dir)
    manifest = {
        "run_name": args.run_name,
        "profiles": profiles,
        "profile_overrides": {profile: PROFILES[profile] for profile in profiles},
        "source_run": args.source_run,
        "source_episodes": args.source_episodes,
        "training_seeds": args.training_seeds,
        "workload_seeds": args.workload_seeds,
        "train_episodes": args.train_episodes,
        "episodes_per_seed": args.episodes_per_seed,
        "steps_per_episode": args.steps_per_episode,
        "max_workers": args.max_workers,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    train_tasks = make_train_tasks(args, profiles, run_dir)
    eval_tasks = make_eval_tasks(args, profiles, run_dir)
    if args.dry_run:
        print(f"Run directory: {run_dir}")
        for task in train_tasks + eval_tasks:
            state = "done" if all_done(task.done_paths) else "pending"
            print(f"[{state}] {task.name}")
            print(f"        {shell_join(task.cmd)}")
            print(f"        log: {task.log_path}")
        return

    env = build_env(args)
    print("\n## stage: train tuned lower policies")
    run_tasks(train_tasks, state_dir, int(args.max_workers), int(args.retries), env)
    print("\n## stage: evaluate tuned lower policies")
    run_tasks(eval_tasks, state_dir, int(args.max_workers), int(args.retries), env)
    print("\n## stage: merge")
    merge_results(args, profiles)


if __name__ == "__main__":
    main()
