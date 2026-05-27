from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from environment.env import Env
from marl_models.utils import get_model
from utils.logger import Log, Logger

SUMMARY_METRIC_NAMES = [
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
]


def make_joint_obs(trajectory_obs: np.ndarray, offload_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([trajectory_obs, offload_obs], axis=-1).astype(np.float32)


def aggregate_metric_dicts(entries: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([entry[metric] for entry in entries])) if entries else 0.0,
            "std": float(np.std([entry[metric] for entry in entries])) if entries else 0.0,
        }
        for metric in SUMMARY_METRIC_NAMES
    }


def run_single_episode(env: Env, model) -> tuple[dict[str, float], dict[str, object]]:
    obs = env.reset()
    trajectory_obs = np.asarray(obs, dtype=np.float32)
    episode_reward = 0.0
    episode_latency = 0.0
    episode_energy = 0.0
    episode_fairness = 0.0
    episode_offline_rate = 0.0
    episode_deadline = 0.0
    episode_local = 0.0
    episode_coop = 0.0
    episode_mbs = 0.0
    episode_mbs_load = 0.0
    for _ in range(config.STEPS_PER_EPISODE):
        offload_obs, offload_masks = env.get_offloading_obs_and_masks()
        joint_obs = make_joint_obs(trajectory_obs, offload_obs)
        trajectory_actions, offload_actions, _, _ = model.get_action_and_value(joint_obs, masks=offload_masks, exploration=False)
        next_obs, rewards, metrics = env.step(trajectory_actions, offloading_actions=offload_actions)
        trajectory_obs = np.asarray(next_obs, dtype=np.float32)
        episode_reward += float(np.sum(rewards))
        episode_latency += float(metrics["latency"])
        episode_energy += float(metrics["energy"])
        episode_fairness = float(metrics["fairness"])
        episode_offline_rate = float(metrics["offline_rate"])
        episode_deadline += float(metrics["deadline_satisfaction_rate"])
        episode_local += float(metrics["offloading_ratio_local"])
        episode_coop += float(metrics["offloading_ratio_cooperative"])
        episode_mbs += float(metrics["offloading_ratio_mbs"])
        episode_mbs_load += float(metrics["mbs_load_ratio"])
    episode_length = max(float(config.STEPS_PER_EPISODE), 1.0)
    runtime_audit = dict(env.last_runtime_audit)
    return (
        {
            "reward": float(episode_reward),
            "latency": float(episode_latency),
            "energy": float(episode_energy),
            "fairness": float(episode_fairness),
            "offline_rate": float(episode_offline_rate),
            "deadline_satisfaction_rate": float(episode_deadline / episode_length),
            "offloading_ratio_local": float(episode_local / episode_length),
            "offloading_ratio_cooperative": float(episode_coop / episode_length),
            "offloading_ratio_mbs": float(episode_mbs / episode_length),
            "mbs_load_ratio": float(episode_mbs_load / episode_length),
            "service_learned_decision_count": float(runtime_audit.get("episode_service_learned_decision_count", 0.0)),
            "service_heuristic_decision_count": float(runtime_audit.get("episode_service_heuristic_decision_count", 0.0)),
            "service_fallback_count": float(runtime_audit.get("episode_service_fallback_count", 0.0)),
            "service_predict_exception_fallback_count": float(runtime_audit.get("episode_service_predict_exception_fallback_count", 0.0)),
        },
        runtime_audit,
    )


def evaluate(model_dir: str, name: str, seeds: list[int], episodes_per_seed: int, steps_per_episode: int | None = None) -> dict[str, object]:
    config.MODEL = "joint_mappo"
    if steps_per_episode is not None:
        config.STEPS_PER_EPISODE = int(steps_per_episode)
    run_root = Path("results") / "joint_e2e_evaluations" / name
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{name}")
    log_dir = run_root / "test_logs" / timestamp
    logger = Logger(str(log_dir), timestamp)
    logger.log_configs()
    episode_log = Log()
    start_time = time.time()
    model = get_model("joint_mappo")
    model.load(model_dir)
    aggregate_units: list[dict[str, float]] = []
    per_seed_runs: list[dict[str, object]] = []
    global_episode_idx = 0

    for seed in seeds:
        episode_metrics_for_seed: list[dict[str, float]] = []
        for episode_idx in range(episodes_per_seed):
            run_seed = int(seed + episode_idx * 1000)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            env = Env()
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
                service_offload_policy_requested="joint_mappo",
                service_offload_policy_loaded=True,
                service_offload_policy_feature_family="joint_end_to_end_mappo",
            )
            global_episode_idx += 1
            logger.log_metrics(global_episode_idx, episode_log, 1, time.time() - start_time)
        per_seed_mean = {metric: float(np.mean([entry[metric] for entry in episode_metrics_for_seed])) for metric in SUMMARY_METRIC_NAMES}
        per_seed_runs.append(
            {
                "policy": "joint_end_to_end_mappo",
                "workload_seed": int(seed),
                "seed": int(seed),
                "unit_id": f"workload{int(seed)}",
                "episodes": episode_metrics_for_seed,
                "per_seed_mean": per_seed_mean,
                "unit_mean": per_seed_mean,
            }
        )

    summary = {
        "metadata": {
            "experiment_name": name,
            "model_dir": str(Path(model_dir).resolve()),
            "model_name": "joint_mappo",
            "seeds": [int(seed) for seed in seeds],
            "episodes_per_seed": int(episodes_per_seed),
            "steps_per_episode": int(config.STEPS_PER_EPISODE),
            "log_json": logger.json_file_path,
        },
        "policy": "joint_end_to_end_mappo",
        "aggregate": aggregate_metric_dicts(aggregate_units),
        "per_seed": per_seed_runs,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "joint_e2e_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a true end-to-end joint MAPPO baseline.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="joint_e2e_eval")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126, 168])
    parser.add_argument("--episodes_per_seed", type=int, default=8)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args.model_dir, args.name, [int(seed) for seed in args.seeds], int(args.episodes_per_seed), args.steps_per_episode)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
