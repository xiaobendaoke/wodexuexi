#!/usr/bin/env python3
"""Evaluate already-trained force-admission hierarchical variants.

This script does not train.  It loads training summaries produced by
``scripts/run_force_admission_replacement_job.sh`` and evaluates the saved
models on a fixed workload-seed matrix using the same episode runner and metric
schema as ``run_baseline_matrix_v2.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from marl_models.utils import get_model
from utils.baseline_metrics import aggregate_metric_dicts, run_single_episode, set_global_seed


DEFAULT_RUN_ID = "20260628_204054"
DEFAULT_REPORTS_DIR = Path("results/reports")
DEFAULT_OUTPUT_DIR = Path("results/final_force_admission/evaluation_20260630")
DEFAULT_WORKLOAD_SEEDS = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420]
DEFAULT_TRAINING_SEEDS = [42, 84, 126]
DEFAULT_EVAL_EPISODES = 6

SUMMARY_PATTERNS = [
    "hierarchical_mappo_summary_force_admission_main_*_seed{seed}_{run_id}.json",
    "hierarchical_mappo_summary_force_admission_ablation_*_seed{seed}_{run_id}.json",
    "hierarchical_mappo_summary_force_admission_dsr_priority_full_seed{seed}_{run_id}.json",
]

SUMMARY_METRICS = [
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
    "forced_admission_ratio",
    "natural_coverage_service_ratio",
]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_git_info() -> dict[str, str]:
    git_info: dict[str, str] = {}
    commands = {
        "commit_hash": ["git", "rev-parse", "HEAD"],
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


def variant_name(summary: dict[str, Any], summary_path: Path) -> str:
    name = summary_path.name
    mode = str(summary.get("mode", "unknown"))
    ablation = str(summary.get("lower_ablation", "full"))
    if "_dsr_priority_" in name:
        return "dsr_priority_full"
    if "_ablation_" in name:
        return f"ablation_{ablation}"
    if mode == "upper_only":
        return "main_upper_only"
    if mode == "lower_only_fixed_upper":
        return "main_lower_only_fixed_upper"
    if mode == "full_hierarchical":
        return "main_full_hierarchical"
    return f"main_{mode}"


def discover_summaries(reports_dir: Path, run_id: str, training_seeds: list[int]) -> list[tuple[str, int, Path]]:
    found: list[tuple[str, int, Path]] = []
    seen: set[Path] = set()
    for seed in training_seeds:
        for pattern in SUMMARY_PATTERNS:
            for path in sorted(reports_dir.glob(pattern.format(seed=seed, run_id=run_id))):
                if path in seen:
                    continue
                summary = read_json(path)
                variant = variant_name(summary, path)
                found.append((variant, seed, path))
                seen.add(path)
    return sorted(found, key=lambda item: (item[0], item[1], str(item[2])))


def apply_summary_config(summary: dict[str, Any]) -> None:
    config.FORCE_SERVICE_ADMISSION = True
    if "offload_mask_mode" in summary:
        config.OFFLOAD_MASK_MODE = str(summary["offload_mask_mode"])
    if "offload_use_attention" in summary:
        config.OFFLOAD_USE_ATTENTION = bool(summary["offload_use_attention"])
    if "constraint_mode" in summary:
        config.OFFLOAD_CONSTRAINT_MODE = str(summary["constraint_mode"])
    if "offload_dsr_target" in summary:
        config.OFFLOAD_DSR_TARGET = float(summary["offload_dsr_target"])
    if "offload_mbs_load_ceiling" in summary:
        config.OFFLOAD_MBS_LOAD_CEILING = float(summary["offload_mbs_load_ceiling"])


def require_model_dir(path: str | None, *, label: str, required: bool) -> str | None:
    if not path:
        if required:
            raise FileNotFoundError(f"Missing {label} in training summary.")
        return None
    model_dir = Path(path)
    if not model_dir.exists():
        raise FileNotFoundError(f"{label} does not exist: {model_dir}")
    if not any(model_dir.glob("*.pth")):
        raise FileNotFoundError(f"{label} contains no .pth file: {model_dir}")
    return str(model_dir)


def load_models(summary: dict[str, Any], seed: int) -> tuple[Any, Any | None]:
    """Load or reconstruct the trajectory/offload policies for one variant."""
    apply_summary_config(summary)
    set_global_seed(seed)

    trajectory_model_name = str(summary.get("trajectory_model_name", "attention_mappo"))
    trajectory_model = get_model(trajectory_model_name)
    trajectory_model_dir = summary.get("trajectory_model_dir")
    if trajectory_model_dir:
        trajectory_model.load(require_model_dir(str(trajectory_model_dir), label="trajectory_model_dir", required=True))
    else:
        # lower_only_fixed_upper used a seed-initialized, fixed upper policy.
        print(f"      fixed upper policy: initialized {trajectory_model_name} with seed={seed}")

    offload_model = None
    offload_model_dir = summary.get("offload_model_dir")
    if offload_model_dir:
        offload_model_name = str(summary.get("offload_model_name", "constrained_attention_offload_mappo"))
        offload_model = get_model(offload_model_name)
        offload_model.load(require_model_dir(str(offload_model_dir), label="offload_model_dir", required=True))
    return trajectory_model, offload_model


def metrics_path(output_dir: Path, variant: str, seed: int, workload_seed: int) -> Path:
    return output_dir / variant / f"seed_{seed}" / f"workload_{workload_seed}_metrics.json"


def evaluate_model(trajectory_model: Any, offload_model: Any | None, workload_seed: int, eval_episodes: int) -> list[dict[str, float]]:
    set_global_seed(workload_seed)
    episode_metrics: list[dict[str, float]] = []
    for ep_idx in range(eval_episodes):
        run_seed = int(workload_seed + ep_idx * 1000)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        env = Env()
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


def save_metrics(
    output_dir: Path,
    variant: str,
    seed: int,
    workload_seed: int,
    eval_episodes: int,
    episode_metrics: list[dict[str, float]],
    elapsed_seconds: float,
    summary_path: Path,
) -> Path:
    path = metrics_path(output_dir, variant, seed, workload_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": variant,
        "variant": variant,
        "seed": seed,
        "training_seed": seed,
        "workload_seed": workload_seed,
        "num_episodes": eval_episodes,
        "aggregate": aggregate_metric_dicts(episode_metrics),
        "per_episode": episode_metrics,
        "elapsed_seconds": elapsed_seconds,
        "unit_id": f"seed{seed}_workload{workload_seed}",
        "training_summary": str(summary_path),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return path


def unit_metrics(payload: dict[str, Any]) -> dict[str, float]:
    episodes = payload.get("per_episode", [])
    if not isinstance(episodes, list) or not episodes:
        aggregate = payload.get("aggregate", {})
        return {metric: float(aggregate.get(metric, {}).get("mean", 0.0)) for metric in SUMMARY_METRICS}

    sums = {
        "service_requests_generated": sum(float(ep.get("service_requests_generated", 0.0)) for ep in episodes),
        "service_requests_processed": sum(float(ep.get("service_requests_processed", 0.0)) for ep in episodes),
        "deadline_satisfied_service_requests": sum(float(ep.get("deadline_satisfied_service_requests", 0.0)) for ep in episodes),
        "service_offloads_local": sum(float(ep.get("service_offloads_local", 0.0)) for ep in episodes),
        "service_offloads_cooperative": sum(float(ep.get("service_offloads_cooperative", 0.0)) for ep in episodes),
        "service_offloads_mbs": sum(float(ep.get("service_offloads_mbs", 0.0)) for ep in episodes),
        "energy": sum(float(ep.get("energy", 0.0)) for ep in episodes),
        "forced_service_admissions": sum(float(ep.get("forced_service_admissions", 0.0)) for ep in episodes),
    }
    generated = max(sums["service_requests_generated"], float(config.EPSILON))
    processed = max(sums["service_requests_processed"], float(config.EPSILON))
    energy = max(sums["energy"], float(config.EPSILON))
    satisfied = sums["deadline_satisfied_service_requests"]
    out = {
        "reward": float(np.mean([float(ep.get("reward", 0.0)) for ep in episodes])),
        "latency": float(np.mean([float(ep.get("latency", 0.0)) for ep in episodes])),
        "energy": float(np.mean([float(ep.get("energy", 0.0)) for ep in episodes])),
        "energy_efficiency_global": float(satisfied / energy),
        "fairness_final": float(np.mean([float(ep.get("fairness_final", 0.0)) for ep in episodes])),
        "fairness_step_mean": float(np.mean([float(ep.get("fairness_step_mean", 0.0)) for ep in episodes])),
        "offline_rate_final": float(np.mean([float(ep.get("offline_rate_final", 0.0)) for ep in episodes])),
        "offline_rate_step_mean": float(np.mean([float(ep.get("offline_rate_step_mean", 0.0)) for ep in episodes])),
        "dsr_request_weighted": float(satisfied / generated),
        "offloading_ratio_local_processed": float(sums["service_offloads_local"] / processed),
        "offloading_ratio_cooperative_processed": float(sums["service_offloads_cooperative"] / processed),
        "offloading_ratio_mbs_processed": float(sums["service_offloads_mbs"] / processed),
        "mbs_load_ratio_generated": float(sums["service_offloads_mbs"] / generated),
        "processed_request_ratio": float(sums["service_requests_processed"] / generated),
        "deadline_satisfied_per_processed": float(satisfied / processed),
        "forced_admission_ratio": float(sums["forced_service_admissions"] / generated),
        "natural_coverage_service_ratio": float(1.0 - sums["forced_service_admissions"] / generated),
    }
    return out


def write_summary_tables(output_dir: Path) -> None:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in sorted(output_dir.rglob("*_metrics.json")):
        payload = read_json(path)
        variant = str(payload.get("variant") or payload.get("method") or "unknown")
        grouped[variant].append(unit_metrics(payload))

    stats: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "unit_note": "one unit = one (training_seed, workload_seed), aggregated over eval episodes",
        "methods": {},
    }
    lines = ["Variant\tN\t" + "\t".join(SUMMARY_METRICS)]
    for variant in sorted(grouped):
        units = grouped[variant]
        stats["methods"][variant] = {"n": len(units), "metrics": {}}
        row = [variant, str(len(units))]
        for metric in SUMMARY_METRICS:
            vals = np.asarray([float(unit.get(metric, 0.0)) for unit in units], dtype=np.float64)
            mean = float(np.mean(vals)) if vals.size else 0.0
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            stats["methods"][variant]["metrics"][metric] = {"mean": mean, "std": std}
            row.append(f"{mean:.6g} +/- {std:.6g}")
        lines.append("\t".join(row))

    (output_dir / "variant_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8"
    )
    (output_dir / "variant_summary_table.tsv").write_text("\n".join(lines), encoding="utf-8")


def save_manifest(output_dir: Path, args: argparse.Namespace, selected: list[tuple[str, int, Path]]) -> None:
    manifest = {
        "experiment_name": "force_admission_trained_variant_evaluation",
        "created_at": datetime.now().isoformat(),
        "run_id": args.run_id,
        "reports_dir": str(args.reports_dir),
        "output_dir": str(output_dir),
        "training_seeds": args.training_seeds,
        "workload_seeds": args.workload_seeds,
        "eval_episodes": args.eval_episodes,
        "force_service_admission": True,
        "selected_summaries": [
            {"variant": variant, "seed": seed, "path": str(path)}
            for variant, seed, path in selected
        ],
        "git_info": get_git_info(),
        "config_snapshot": get_config_snapshot(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--reports_dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training_seeds", nargs="+", type=int, default=DEFAULT_TRAINING_SEEDS)
    parser.add_argument("--workload_seeds", nargs="+", type=int, default=DEFAULT_WORKLOAD_SEEDS)
    parser.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--variants", nargs="+", default=None, help="Optional variant names to evaluate.")
    parser.add_argument("--resume", action="store_true", help="Skip existing workload metric files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing workload metric files.")
    parser.add_argument("--dry_run", action="store_true", help="Only list selected summaries.")
    parser.add_argument("--copy_summaries", action="store_true", help="Copy training summaries into each variant/seed directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("Use either --resume or --overwrite, not both.")

    selected = discover_summaries(args.reports_dir, args.run_id, args.training_seeds)
    if args.variants:
        allowed = set(args.variants)
        selected = [item for item in selected if item[0] in allowed]
    if not selected:
        raise SystemExit("No matching training summaries found.")

    print("Selected training summaries:")
    for variant, seed, path in selected:
        print(f"  {variant:34s} seed={seed:<3d} {path}")
    if args.dry_run:
        return

    config.FORCE_SERVICE_ADMISSION = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(args.output_dir, args, selected)

    for variant, seed, summary_path in selected:
        summary = read_json(summary_path)
        print(f"\n=== {variant} seed={seed} ===", flush=True)
        trajectory_model, offload_model = load_models(summary, seed)
        if args.copy_summaries:
            target = args.output_dir / variant / f"seed_{seed}" / "training_summary.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary_path, target)

        for workload_seed in args.workload_seeds:
            out_path = metrics_path(args.output_dir, variant, seed, workload_seed)
            if args.resume and not args.overwrite and out_path.exists():
                print(f"  workload={workload_seed}: exists, skip", flush=True)
                continue
            print(f"  workload={workload_seed}: evaluating...", end="", flush=True)
            start = time.time()
            episode_metrics = evaluate_model(trajectory_model, offload_model, workload_seed, args.eval_episodes)
            saved = save_metrics(
                args.output_dir,
                variant,
                seed,
                workload_seed,
                args.eval_episodes,
                episode_metrics,
                time.time() - start,
                summary_path,
            )
            processed = np.mean([m.get("processed_request_ratio", 0.0) for m in episode_metrics])
            dsr = np.mean([m.get("dsr_request_weighted", 0.0) for m in episode_metrics])
            print(f" saved {saved} processed={processed:.4f} dsr={dsr:.4f}", flush=True)

    write_summary_tables(args.output_dir)
    print(f"\nDone. Summary table: {args.output_dir / 'variant_summary_table.tsv'}")


if __name__ == "__main__":
    main()
