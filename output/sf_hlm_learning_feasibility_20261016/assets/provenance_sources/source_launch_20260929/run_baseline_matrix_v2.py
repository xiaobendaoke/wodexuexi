#!/usr/bin/env python3
"""
Baseline Matrix v2 experiment runner.

This runner trains each learning method once per training seed and then reuses
that trained model object to evaluate every workload seed in the same Python
process. Non-learning methods are evaluated once per workload seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from environment.env import Env
from marl_models.utils import get_model
from run_baseline_comparison_experiment import (
    train_hierarchical_baseline,
    train_ippo_baseline,
    train_joint_mappo_baseline,
)
from utils.baseline_metrics import (
    aggregate_metric_dicts,
    run_single_episode,
    run_single_episode_joint,
    set_global_seed,
)

DEFAULT_METHODS = ["random", "uniform", "ippo", "vanilla_mappo", "joint_mappo", "proposed"]
DEFAULT_TRAINING_SEEDS = [42, 84, 126]
DEFAULT_WORKLOAD_SEEDS = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420]
DEFAULT_TRAIN_EPISODES = 200
DEFAULT_EVAL_EPISODES = 6
DEFAULT_OUTPUT_DIR = "results/baseline_matrix_v2"

NON_LEARNING_METHODS = {"random", "uniform"}
LEARNING_METHODS = {"ippo", "vanilla_mappo", "joint_mappo", "proposed"}


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def get_git_info() -> dict[str, str]:
    git_info: dict[str, str] = {}
    commands = {
        "commit_hash": ["git", "rev-parse", "HEAD"],
        "commit_message": ["git", "log", "-1", "--pretty=%B"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }
    for key, cmd in commands.items():
        try:
            git_info[key] = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_info[key] = "unknown"
    return git_info


def get_config_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in dir(config):
        if key.isupper() and not key.startswith("__") and not callable(getattr(config, key)):
            snapshot[key] = getattr(config, key)
    return snapshot


def save_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    manifest = {
        "experiment_name": "baseline_matrix_v2",
        "description": "Multi-seed, multi-workload baseline matrix with clarified metric denominators.",
        "created_at": datetime.now().isoformat(),
        "methods": args.methods,
        "training_seeds": args.training_seeds,
        "workload_seeds": args.workload_seeds,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "output_dir": str(output_dir),
        "force_service_admission": bool(args.force_service_admission),
        "git_info": get_git_info(),
        "config_snapshot": get_config_snapshot(),
        "unit_definition": {
            "learning": "one unit = one (training_seed, workload_seed) aggregate over eval_episodes",
            "non_learning": "one unit = one workload_seed aggregate over eval_episodes",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8"
    )


def metrics_path(output_dir: Path, method: str, seed: int | None, workload_seed: int) -> Path:
    if method in LEARNING_METHODS:
        if seed is None:
            raise ValueError("Learning methods require a training seed.")
        return output_dir / method / f"seed_{seed}" / f"workload_{workload_seed}_metrics.json"
    return output_dir / method / f"workload_{workload_seed}_metrics.json"


def training_summary_path(output_dir: Path, method: str, seed: int) -> Path:
    return output_dir / method / f"seed_{seed}" / "training_summary.json"


def save_metrics(
    output_dir: Path,
    method: str,
    seed: int | None,
    workload_seed: int,
    eval_episodes: int,
    episode_metrics: list[dict[str, float]],
    elapsed_seconds: float,
) -> Path:
    path = metrics_path(output_dir, method, seed, workload_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "seed": seed,
        "training_seed": seed if method in LEARNING_METHODS else None,
        "workload_seed": workload_seed,
        "num_episodes": eval_episodes,
        "aggregate": aggregate_metric_dicts(episode_metrics),
        "per_episode": episode_metrics,
        "elapsed_seconds": elapsed_seconds,
        "unit_id": f"seed{seed}_workload{workload_seed}" if method in LEARNING_METHODS else f"workload{workload_seed}",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return path


def evaluate_model(
    method: str,
    trajectory_model: Any,
    offload_model: Any,
    workload_seed: int,
    eval_episodes: int,
) -> list[dict[str, float]]:
    set_global_seed(workload_seed)
    episode_metrics: list[dict[str, float]] = []

    for ep_idx in range(eval_episodes):
        run_seed = int(workload_seed + ep_idx * 1000)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        env = Env()
        if method in NON_LEARNING_METHODS:
            metrics, _ = run_single_episode(
                env,
                trajectory_model,
                offload_model=None,
                exploration=False,
                record_trajectory=False,
                policy_type="non_learning",
            )
        elif method == "joint_mappo":
            metrics, _ = run_single_episode_joint(
                env,
                trajectory_model,
                exploration=False,
                record_trajectory=False,
                policy_type="learned",
            )
        else:
            metrics, _ = run_single_episode(
                env,
                trajectory_model,
                offload_model=offload_model,
                exploration=False,
                record_trajectory=False,
                policy_type="learned",
            )
        metrics["eval_episode_idx"] = float(ep_idx)
        metrics["eval_run_seed"] = float(run_seed)
        episode_metrics.append(metrics)

    return episode_metrics


def train_learning_method(method: str, seed: int, train_episodes: int, output_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    timestamp = datetime.now().strftime(f"baseline_matrix_v2_{method}_seed{seed}_%Y%m%d_%H%M%S")
    set_global_seed(seed)
    start = time.time()

    if method in {"proposed", "vanilla_mappo"}:
        summary = train_hierarchical_baseline(method, train_episodes, seed, timestamp)
        traj_model_name = summary.get("trajectory_model_name", "attention_mappo" if method == "proposed" else "vanilla_mappo")
        trajectory_model = get_model(str(traj_model_name))
        if summary.get("trajectory_model_dir"):
            trajectory_model.load(summary["trajectory_model_dir"])
        offload_model = get_model("constrained_attention_offload_mappo")
        if summary.get("offload_model_dir"):
            offload_model.load(summary["offload_model_dir"])
    elif method == "ippo":
        trajectory_model, offload_model, training_curve = train_ippo_baseline(train_episodes, seed, timestamp)
        summary = {"method": method, "seed": seed, "timestamp": timestamp, "training_curve": training_curve}
    elif method == "joint_mappo":
        summary = train_joint_mappo_baseline(train_episodes, seed, timestamp)
        trajectory_model = get_model("joint_mappo")
        if summary.get("model_dir"):
            trajectory_model.load(summary["model_dir"])
        offload_model = None
    else:
        raise ValueError(f"Unsupported learning method: {method}")

    summary = dict(summary)
    summary.update({"method": method, "seed": seed, "train_episodes": train_episodes, "elapsed_seconds": time.time() - start})
    summary_path = training_summary_path(output_dir, method, seed)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return trajectory_model, offload_model, summary


def run_learning_method(
    method: str,
    training_seeds: list[int],
    workload_seeds: list[int],
    train_episodes: int,
    eval_episodes: int,
    output_dir: Path,
    resume: bool,
    overwrite: bool,
) -> None:
    print(f"\n{'=' * 60}\nLearning method: {method}\n{'=' * 60}")
    for seed in training_seeds:
        pending = [w for w in workload_seeds if overwrite or not metrics_path(output_dir, method, seed, w).exists()]
        if resume and not pending:
            print(f"  seed={seed}: all workload seeds already complete, skipping.")
            continue

        print(f"  seed={seed}: training once, then evaluating {len(pending)} pending workload seeds.")
        trajectory_model, offload_model, _ = train_learning_method(method, seed, train_episodes, output_dir)

        for workload_seed in pending:
            print(f"    workload_seed={workload_seed}: ", end="", flush=True)
            start = time.time()
            episode_metrics = evaluate_model(method, trajectory_model, offload_model, workload_seed, eval_episodes)
            path = save_metrics(output_dir, method, seed, workload_seed, eval_episodes, episode_metrics, time.time() - start)
            print(f"saved {path}")


def run_non_learning_method(
    method: str,
    workload_seeds: list[int],
    eval_episodes: int,
    output_dir: Path,
    resume: bool,
    overwrite: bool,
) -> None:
    print(f"\n{'=' * 60}\nNon-learning method: {method}\n{'=' * 60}")
    model = get_model(f"{method}_baseline")
    for workload_seed in workload_seeds:
        path = metrics_path(output_dir, method, None, workload_seed)
        if resume and not overwrite and path.exists():
            print(f"  workload_seed={workload_seed}: already complete, skipping.")
            continue
        print(f"  workload_seed={workload_seed}: ", end="", flush=True)
        start = time.time()
        episode_metrics = evaluate_model(method, model, None, workload_seed, eval_episodes)
        saved = save_metrics(output_dir, method, None, workload_seed, eval_episodes, episode_metrics, time.time() - start)
        print(f"saved {saved}")


def validate_args(args: argparse.Namespace) -> None:
    methods = set(args.methods)
    unknown = methods.difference(DEFAULT_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Matrix v2 experiment runner")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--training_seeds", nargs="+", type=int, default=DEFAULT_TRAINING_SEEDS)
    parser.add_argument("--workload_seeds", nargs="+", type=int, default=DEFAULT_WORKLOAD_SEEDS)
    parser.add_argument("--train_episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true", help="Skip completed metrics files.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run metrics even when output files exist.")
    parser.add_argument(
        "--force_service_admission",
        action="store_true",
        help="Admit service requests outside all UAV coverage disks through the nearest UAV.",
    )
    args = parser.parse_args()
    validate_args(args)
    config.FORCE_SERVICE_ADMISSION = bool(args.force_service_admission)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(output_dir, args)

    print(f"\n{'=' * 60}")
    print("Baseline Matrix v2 Experiment")
    print(f"Methods: {args.methods}")
    print(f"Training seeds: {args.training_seeds}")
    print(f"Workload seeds: {args.workload_seeds}")
    print(f"Train episodes: {args.train_episodes}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Output dir: {output_dir}")
    print(f"Force service admission: {args.force_service_admission}")
    print(f"Resume: {args.resume}; overwrite: {args.overwrite}")
    print(f"{'=' * 60}\n")

    for method in [m for m in args.methods if m in LEARNING_METHODS]:
        run_learning_method(method, args.training_seeds, args.workload_seeds, args.train_episodes, args.eval_episodes, output_dir, args.resume, args.overwrite)

    for method in [m for m in args.methods if m in NON_LEARNING_METHODS]:
        run_non_learning_method(method, args.workload_seeds, args.eval_episodes, output_dir, args.resume, args.overwrite)

    print(f"\n{'=' * 60}\nAll experiments completed at {datetime.now().isoformat()}\n{'=' * 60}")


if __name__ == "__main__":
    main()
