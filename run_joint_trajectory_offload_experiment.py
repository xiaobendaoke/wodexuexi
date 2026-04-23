from __future__ import annotations

import argparse
import copy
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from environment.env import Env
from environment.uavs import UAV
from marl_models.base_model import MARLModel
from marl_models.utils import get_model
from paths import results_path
from utils.comparative_plots import compare_algorithms
from utils.logger import Log, Logger, load_configs
from utils.plot_logs import generate_plots


SUMMARY_METRIC_NAMES: tuple[str, ...] = (
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


def configure_fp32_precision() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Please use the new API settings to control TF32 behavior.*",
        category=UserWarning,
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def configure_compile_backend() -> None:
    if os.name == "nt":
        def _no_compile(module, *args, **kwargs):
            return module

        torch.compile = _no_compile  # type: ignore[attr-defined]


def snapshot_config() -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for key in dir(config):
        if key.isupper() and not key.startswith("__"):
            value = getattr(config, key)
            snapshot[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    return snapshot


def restore_config(snapshot: dict[str, object]) -> None:
    for key, value in snapshot.items():
        setattr(config, key, value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))


def _get_episode_runtime_audit(env: Env) -> dict[str, object]:
    audit: dict[str, object] = env.last_runtime_audit or {}
    return {
        "service_learned_decision_count": float(audit.get("episode_service_learned_decision_count", 0.0)),
        "service_heuristic_decision_count": float(audit.get("episode_service_heuristic_decision_count", 0.0)),
        "service_fallback_count": float(audit.get("episode_service_fallback_count", 0.0)),
        "service_predict_exception_fallback_count": float(
            audit.get("episode_service_predict_exception_fallback_count", 0.0)
        ),
        "service_offload_policy_requested": str(audit.get("service_offload_policy_requested", "heuristic")),
        "service_offload_policy_loaded": bool(audit.get("service_offload_policy_loaded", False)),
        "service_offload_policy_checkpoint_path": audit.get("service_offload_policy_checkpoint_path"),
        "service_offload_policy_feature_family": audit.get("service_offload_policy_feature_family"),
    }


def resolve_latest_training_artifacts(trajectory_run_root: str | Path, model_name: str) -> tuple[Path | None, Path]:
    run_root = Path(trajectory_run_root)
    config_candidates: list[Path] = []
    primary_config_dir = run_root / "train_logs" / model_name
    if primary_config_dir.exists():
        config_candidates.extend(primary_config_dir.glob("config_*.json"))

    fallback_config_dir = Path("train_logs") / model_name
    if fallback_config_dir.exists():
        config_candidates.extend(fallback_config_dir.glob("config_*.json"))

    config_candidates = sorted(
        {path.resolve(): path for path in config_candidates}.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    model_candidates: list[Path] = []
    primary_model_root = run_root / "saved_models"
    if primary_model_root.exists():
        model_candidates.extend([path for path in primary_model_root.glob(f"{model_name}_*") if (path / "final").exists()])

    fallback_model_root = Path("saved_models")
    if fallback_model_root.exists():
        model_candidates.extend([path for path in fallback_model_root.glob(f"{model_name}_*") if (path / "final").exists()])

    model_candidates = sorted(
        {path.resolve(): path for path in model_candidates}.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not model_candidates:
        raise FileNotFoundError(
            f"No saved model run with a 'final' directory found under either '{primary_model_root}' "
            f"or '{fallback_model_root}' for model '{model_name}'."
        )

    resolved_config = config_candidates[0] if config_candidates else None
    return resolved_config, model_candidates[0] / "final"


def resolve_offload_checkpoints(offload_experiment_root: str | Path) -> tuple[Path, Path]:
    root = Path(offload_experiment_root)
    surrogate_checkpoint = root / "checkpoints" / "offload_policy_surrogate_runtime.pt"
    rich_checkpoint = root / "checkpoints" / "offload_policy_rich_runtime.pt"
    if not surrogate_checkpoint.exists():
        raise FileNotFoundError(f"Surrogate checkpoint not found: {surrogate_checkpoint}")
    if not rich_checkpoint.exists():
        raise FileNotFoundError(f"Rich reduced checkpoint not found: {rich_checkpoint}")
    return surrogate_checkpoint, rich_checkpoint


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {metric_name: {"mean": 0.0, "std": 0.0} for metric_name in SUMMARY_METRIC_NAMES}
    arrays = {
        metric_name: np.asarray([entry[metric_name] for entry in metric_dicts], dtype=np.float64)
        for metric_name in SUMMARY_METRIC_NAMES
    }
    return {
        metric_name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for metric_name, values in arrays.items()
    }


def set_service_offload_policy(policy_label: str, checkpoint_path: str | None) -> None:
    if policy_label == "heuristic":
        config.SERVICE_OFFLOAD_POLICY = "heuristic"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None
    elif policy_label in {"surrogate", "rich_reduced"}:
        config.SERVICE_OFFLOAD_POLICY = "learned"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = checkpoint_path
    else:
        raise ValueError(f"Unsupported joint-eval policy label: {policy_label}")
    UAV._policy_cache.clear()


def run_single_episode(env: Env, model: MARLModel) -> tuple[dict[str, float], dict[str, object]]:
    obs = env.reset()
    model.reset()

    episode_reward: float = 0.0
    episode_latency: float = 0.0
    episode_energy: float = 0.0
    episode_fairness: float = 0.0
    episode_offline_rate: float = 0.0
    episode_deadline_satisfaction_sum: float = 0.0
    episode_offload_local_sum: float = 0.0
    episode_offload_cooperative_sum: float = 0.0
    episode_offload_mbs_sum: float = 0.0
    episode_mbs_load_sum: float = 0.0

    for step in range(1, config.STEPS_PER_EPISODE + 1):
        obs_arr = np.asarray(obs, dtype=np.float32)
        actions = model.select_actions(obs_arr, exploration=False)
        next_obs, rewards, metrics = env.step(actions)
        obs = next_obs

        episode_reward += float(np.sum(rewards))
        episode_latency += float(metrics["latency"])
        episode_energy += float(metrics["energy"])
        episode_fairness = float(metrics["fairness"])
        episode_offline_rate = float(metrics["offline_rate"])
        episode_deadline_satisfaction_sum += float(metrics["deadline_satisfaction_rate"])
        episode_offload_local_sum += float(metrics["offloading_ratio_local"])
        episode_offload_cooperative_sum += float(metrics["offloading_ratio_cooperative"])
        episode_offload_mbs_sum += float(metrics["offloading_ratio_mbs"])
        episode_mbs_load_sum += float(metrics["mbs_load_ratio"])

        if step >= config.STEPS_PER_EPISODE:
            break

    episode_length = max(float(config.STEPS_PER_EPISODE), 1.0)
    runtime_audit = _get_episode_runtime_audit(env)
    episode_metrics = {
        "reward": float(episode_reward),
        "latency": float(episode_latency),
        "energy": float(episode_energy),
        "fairness": float(episode_fairness),
        "offline_rate": float(episode_offline_rate),
        "deadline_satisfaction_rate": float(episode_deadline_satisfaction_sum / episode_length),
        "offloading_ratio_local": float(episode_offload_local_sum / episode_length),
        "offloading_ratio_cooperative": float(episode_offload_cooperative_sum / episode_length),
        "offloading_ratio_mbs": float(episode_offload_mbs_sum / episode_length),
        "mbs_load_ratio": float(episode_mbs_load_sum / episode_length),
        "service_learned_decision_count": float(runtime_audit["service_learned_decision_count"]),
        "service_heuristic_decision_count": float(runtime_audit["service_heuristic_decision_count"]),
        "service_fallback_count": float(runtime_audit["service_fallback_count"]),
        "service_predict_exception_fallback_count": float(runtime_audit["service_predict_exception_fallback_count"]),
    }
    return episode_metrics, runtime_audit


def evaluate_joint_policy(
    *,
    policy_label: str,
    checkpoint_path: str | None,
    trajectory_model_name: str,
    trajectory_model_dir: str | Path,
    trajectory_config_path: str | Path | None,
    seeds: list[int],
    episodes_per_seed: int,
    steps_per_episode: int | None,
    run_root: Path,
) -> dict[str, object]:
    if trajectory_config_path is not None:
        load_configs(str(trajectory_config_path))
    config.MODEL = trajectory_model_name
    if steps_per_episode is not None:
        config.STEPS_PER_EPISODE = int(steps_per_episode)

    set_service_offload_policy(policy_label, checkpoint_path)

    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{policy_label}")
    log_dir = run_root / "test_logs" / policy_label / timestamp
    plot_dir = run_root / "test_plots" / policy_label / timestamp
    logger = Logger(str(log_dir), timestamp)
    logger.log_configs()
    episode_log = Log()

    start_time = time.time()
    per_seed_runs: list[dict[str, object]] = []
    aggregate_units: list[dict[str, float]] = []
    global_episode_idx = 0

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = Env()
        model = get_model(trajectory_model_name)
        model.load(str(trajectory_model_dir))

        episode_metrics_for_seed: list[dict[str, float]] = []
        for episode_idx in range(episodes_per_seed):
            run_seed = int(seed + episode_idx * 1000)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

            episode_metrics, runtime_audit = run_single_episode(env, model)
            episode_metrics_for_seed.append(episode_metrics)
            aggregate_units.append(episode_metrics)

            episode_log.append(
                episode_metrics["reward"],
                episode_metrics["latency"],
                episode_metrics["energy"],
                episode_metrics["fairness"],
                episode_metrics["offline_rate"],
                deadline_satisfaction_rate=episode_metrics["deadline_satisfaction_rate"],
                offloading_ratio_local=episode_metrics["offloading_ratio_local"],
                offloading_ratio_cooperative=episode_metrics["offloading_ratio_cooperative"],
                offloading_ratio_mbs=episode_metrics["offloading_ratio_mbs"],
                mbs_load_ratio=episode_metrics["mbs_load_ratio"],
                service_learned_decision_count=episode_metrics["service_learned_decision_count"],
                service_heuristic_decision_count=episode_metrics["service_heuristic_decision_count"],
                service_fallback_count=episode_metrics["service_fallback_count"],
                service_predict_exception_fallback_count=episode_metrics["service_predict_exception_fallback_count"],
                service_offload_policy_requested=str(runtime_audit["service_offload_policy_requested"]),
                service_offload_policy_loaded=bool(runtime_audit["service_offload_policy_loaded"]),
                service_offload_policy_checkpoint_path=runtime_audit["service_offload_policy_checkpoint_path"],
                service_offload_policy_feature_family=runtime_audit["service_offload_policy_feature_family"],
            )
            global_episode_idx += 1
            logger.log_metrics(global_episode_idx, episode_log, 1, time.time() - start_time)

        per_seed_mean = {
            metric_name: float(np.mean([entry[metric_name] for entry in episode_metrics_for_seed]))
            for metric_name in SUMMARY_METRIC_NAMES
        }
        per_seed_runs.append(
            {
                "policy": policy_label,
                "seed": int(seed),
                "episodes": episode_metrics_for_seed,
                "per_seed_mean": per_seed_mean,
            }
        )

    generate_plots(str(logger.json_file_path), str(plot_dir), "joint_test", timestamp, smoothing_window=2)
    return {
        "log_dir": str(log_dir),
        "plot_dir": str(plot_dir),
        "log_json_path": str(logger.json_file_path),
        "config_path": str(logger.config_file_path),
        "trajectory_config_source_path": str(trajectory_config_path) if trajectory_config_path is not None else None,
        "aggregate": aggregate_metric_dicts(aggregate_units),
        "per_seed": per_seed_runs,
    }


def build_delta_vs_reference(
    *,
    results_by_policy: dict[str, dict[str, object]],
    reference_policy: str,
) -> dict[str, dict[str, dict[str, float]]]:
    reference_units = [entry["per_seed_mean"] for entry in results_by_policy[reference_policy]["per_seed"]]
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for policy_label, details in results_by_policy.items():
        if policy_label == reference_policy:
            continue
        policy_units = [entry["per_seed_mean"] for entry in details["per_seed"]]
        deltas[policy_label] = {
            metric_name: {
                "mean_delta": float(
                    np.mean(
                        [float(policy_units[idx][metric_name] - reference_units[idx][metric_name]) for idx in range(len(policy_units))]
                    )
                ),
                "std_delta": float(
                    np.std(
                        [float(policy_units[idx][metric_name] - reference_units[idx][metric_name]) for idx in range(len(policy_units))]
                    )
                ),
            }
            for metric_name in SUMMARY_METRIC_NAMES
        }
    return deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run joint trajectory-control + offloading-policy comparison experiments.")
    parser.add_argument("--name", type=str, default=time.strftime("joint_exp_%Y%m%d_%H%M%S"), help="Joint experiment name.")
    parser.add_argument("--trajectory_run_root", type=str, required=True, help="Run root containing the trained trajectory model artifacts.")
    parser.add_argument("--trajectory_model", type=str, default="attention_mappo", help="Trajectory model name to load.")
    parser.add_argument("--trajectory_config", type=str, default=None, help="Optional explicit config_*.json path.")
    parser.add_argument("--trajectory_model_dir", type=str, default=None, help="Optional explicit saved model final directory.")
    parser.add_argument("--offload_experiment_root", type=str, required=True, help="Offload experiment root containing checkpoints/.")
    parser.add_argument("--surrogate_checkpoint", type=str, default=None, help="Optional explicit surrogate checkpoint path.")
    parser.add_argument("--rich_checkpoint", type=str, default=None, help="Optional explicit rich reduced checkpoint path.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["heuristic", "surrogate", "rich_reduced"],
        choices=["heuristic", "surrogate", "rich_reduced"],
        help="Offloading policies to compare.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Evaluation seeds.")
    parser.add_argument("--episodes_per_seed", type=int, default=8, help="Episodes per seed and policy.")
    parser.add_argument("--steps_per_episode", type=int, default=None, help="Optional override for STEPS_PER_EPISODE.")
    parser.add_argument("--comparison_smoothing", type=int, default=3, help="Smoothing window for cross-policy comparison plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fp32_precision()
    configure_compile_backend()

    base_snapshot = snapshot_config()
    run_root = results_path("joint_experiments", args.name)
    run_root.mkdir(parents=True, exist_ok=True)

    if args.trajectory_config is None or args.trajectory_model_dir is None:
        inferred_config_path, inferred_model_dir = resolve_latest_training_artifacts(
            args.trajectory_run_root,
            args.trajectory_model,
        )
        trajectory_config_path = Path(args.trajectory_config) if args.trajectory_config is not None else inferred_config_path
        trajectory_model_dir = Path(args.trajectory_model_dir) if args.trajectory_model_dir is not None else inferred_model_dir
    else:
        trajectory_config_path = Path(args.trajectory_config)
        trajectory_model_dir = Path(args.trajectory_model_dir)

    if trajectory_config_path is None:
        print(
            "[joint] warning: no training config_*.json was found for the trajectory model; "
            "falling back to the current config.py values."
        )

    if args.surrogate_checkpoint is None or args.rich_checkpoint is None:
        inferred_surrogate_checkpoint, inferred_rich_checkpoint = resolve_offload_checkpoints(args.offload_experiment_root)
        surrogate_checkpoint = Path(args.surrogate_checkpoint) if args.surrogate_checkpoint is not None else inferred_surrogate_checkpoint
        rich_checkpoint = Path(args.rich_checkpoint) if args.rich_checkpoint is not None else inferred_rich_checkpoint
    else:
        surrogate_checkpoint = Path(args.surrogate_checkpoint)
        rich_checkpoint = Path(args.rich_checkpoint)

    policy_to_checkpoint = {
        "heuristic": None,
        "surrogate": str(surrogate_checkpoint),
        "rich_reduced": str(rich_checkpoint),
    }

    results_by_policy: dict[str, dict[str, object]] = {}
    try:
        for policy_label in args.policies:
            print(f"[joint] evaluating policy={policy_label}")
            restore_config(base_snapshot)
            results_by_policy[policy_label] = evaluate_joint_policy(
                policy_label=policy_label,
                checkpoint_path=policy_to_checkpoint[policy_label],
                trajectory_model_name=args.trajectory_model,
                trajectory_model_dir=trajectory_model_dir,
                trajectory_config_path=trajectory_config_path,
                seeds=[int(seed) for seed in args.seeds],
                episodes_per_seed=args.episodes_per_seed,
                steps_per_episode=args.steps_per_episode,
                run_root=run_root,
            )

        log_dirs = [results_by_policy[policy_label]["log_dir"] for policy_label in args.policies]
        comparison_dir = run_root / "comparisons"
        compare_algorithms(log_dirs, args.policies, str(comparison_dir), smoothing_window=args.comparison_smoothing)

        summary = {
            "metadata": {
                "experiment_name": args.name,
                "trajectory_run_root": str(Path(args.trajectory_run_root).resolve()),
                "trajectory_model": args.trajectory_model,
                "trajectory_config_path": str(trajectory_config_path.resolve()) if trajectory_config_path is not None else None,
                "trajectory_model_dir": str(trajectory_model_dir.resolve()),
                "offload_experiment_root": str(Path(args.offload_experiment_root).resolve()),
                "surrogate_checkpoint": str(surrogate_checkpoint.resolve()),
                "rich_checkpoint": str(rich_checkpoint.resolve()),
                "policies": args.policies,
                "seeds": [int(seed) for seed in args.seeds],
                "episodes_per_seed": int(args.episodes_per_seed),
                "steps_per_episode": int(args.steps_per_episode) if args.steps_per_episode is not None else None,
                "comparison_dir": str(comparison_dir),
            },
            "per_policy": results_by_policy,
            "delta_vs_heuristic": build_delta_vs_reference(results_by_policy=results_by_policy, reference_policy="heuristic")
            if "heuristic" in results_by_policy
            else {},
        }
        summary_path = run_root / "joint_experiment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        restore_config(base_snapshot)
        UAV._policy_cache.clear()


if __name__ == "__main__":
    main()
