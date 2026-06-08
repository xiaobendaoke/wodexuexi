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
from marl_models.buffer_and_helpers import JointMAPPOBuffer
from marl_models.utils import get_model
from utils.logger import Log, Logger


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_joint_obs(trajectory_obs: np.ndarray, offload_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([trajectory_obs, offload_obs], axis=-1).astype(np.float32)


def update_joint_model(model, buffer: JointMAPPOBuffer, last_values: np.ndarray) -> dict[str, float]:
    buffer.compute_returns_and_advantages(last_values, config.DISCOUNT_FACTOR, config.PPO_GAE_LAMBDA)
    losses: dict[str, list[float]] = {"actor": [], "critic": [], "entropy": []}
    for _ in range(config.PPO_EPOCHS):
        for batch in buffer.get_batches(config.PPO_BATCH_SIZE):
            loss = model.update(batch)
            for key in losses:
                value = loss.get(key) if loss else None
                if value is not None:
                    losses[key].append(float(value))
    buffer.clear()
    return {key: float(np.mean(values)) if values else 0.0 for key, values in losses.items()}


def train_joint_end_to_end_mappo(num_episodes: int, timestamp: str | None = None, seed: int | None = None) -> dict[str, object]:
    set_global_seed(seed)
    config.MODEL = "joint_mappo"
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_joint_e2e_mappo")
    env = Env()
    model = get_model("joint_mappo")
    buffer = JointMAPPOBuffer(
        num_agents=config.NUM_UAVS,
        joint_obs_dim=config.OBS_DIM_SINGLE + config.OFFLOAD_OBS_DIM_SINGLE,
        trajectory_action_dim=config.ACTION_DIM,
        max_requests=config.MAX_OFFLOAD_REQUESTS_PER_UAV,
        num_offload_actions=config.OFFLOAD_NUM_ACTIONS,
        buffer_size=config.PPO_ROLLOUT_LENGTH,
        device=model.device,
    )
    logger = Logger(log_dir="train_logs/joint_end_to_end_mappo", timestamp=timestamp)
    logger.log_configs()
    episode_log = Log()
    start_time = time.time()

    obs = env.reset()
    trajectory_obs = np.asarray(obs, dtype=np.float32)
    episode = 1
    episode_step = 0
    max_time_steps = num_episodes * config.STEPS_PER_EPISODE
    num_updates = max_time_steps // config.PPO_ROLLOUT_LENGTH
    if num_updates <= 0:
        raise ValueError("num_updates is 0; increase num_episodes or reduce PPO_ROLLOUT_LENGTH.")

    recent_rewards: list[float] = []
    recent_losses = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
    episode_totals = {
        "reward": 0.0,
        "latency": 0.0,
        "energy": 0.0,
        "fairness": 0.0,
        "offline_rate": 0.0,
        "deadline": 0.0,
        "local": 0.0,
        "coop": 0.0,
        "mbs": 0.0,
        "mbs_load": 0.0,
    }

    for _ in range(1, num_updates + 1):
        for _ in range(config.PPO_ROLLOUT_LENGTH):
            offload_obs, offload_masks = env.get_offloading_obs_and_masks()
            joint_obs = make_joint_obs(trajectory_obs, offload_obs)
            trajectory_actions, offload_actions, log_probs, values = model.get_action_and_value(
                joint_obs,
                masks=offload_masks,
                exploration=True,
            )
            next_obs, system_rewards, metrics = env.step(trajectory_actions, offloading_actions=offload_actions)
            episode_step += 1
            done = episode_step >= config.STEPS_PER_EPISODE
            buffer.add(joint_obs, trajectory_actions, offload_actions, offload_masks, log_probs, system_rewards, done, values)
            trajectory_obs = np.asarray(next_obs, dtype=np.float32)

            episode_totals["reward"] += float(np.sum(system_rewards))
            episode_totals["latency"] += float(metrics["latency"])
            episode_totals["energy"] += float(metrics["energy"])
            episode_totals["fairness"] = float(metrics["fairness"])
            episode_totals["offline_rate"] = float(metrics["offline_rate"])
            episode_totals["deadline"] += float(metrics["deadline_satisfaction_rate"])
            episode_totals["local"] += float(metrics["offloading_ratio_local"])
            episode_totals["coop"] += float(metrics["offloading_ratio_cooperative"])
            episode_totals["mbs"] += float(metrics["offloading_ratio_mbs"])
            episode_totals["mbs_load"] += float(metrics["mbs_load_ratio"])

            if done:
                recent_rewards.append(episode_totals["reward"])
                episode_length = max(float(episode_step), 1.0)
                episode_log.append(
                    episode_totals["reward"],
                    episode_totals["latency"],
                    episode_totals["energy"],
                    episode_totals["fairness"],
                    episode_totals["offline_rate"],
                    deadline_satisfaction_rate=episode_totals["deadline"] / episode_length,
                    offloading_ratio_local=episode_totals["local"] / episode_length,
                    offloading_ratio_cooperative=episode_totals["coop"] / episode_length,
                    offloading_ratio_mbs=episode_totals["mbs"] / episode_length,
                    mbs_load_ratio=episode_totals["mbs_load"] / episode_length,
                    service_learned_decision_count=float(env.last_runtime_audit.get("episode_service_learned_decision_count", 0.0)),
                    service_heuristic_decision_count=float(env.last_runtime_audit.get("episode_service_heuristic_decision_count", 0.0)),
                    service_fallback_count=float(env.last_runtime_audit.get("episode_service_fallback_count", 0.0)),
                    service_predict_exception_fallback_count=float(env.last_runtime_audit.get("episode_service_predict_exception_fallback_count", 0.0)),
                    service_offload_policy_requested="joint_mappo",
                    service_offload_policy_loaded=True,
                    actor_loss=recent_losses.get("actor"),
                    critic_loss=recent_losses.get("critic"),
                    entropy_loss=recent_losses.get("entropy"),
                )
                if episode % config.LOG_FREQ == 0:
                    logger.log_metrics(episode, episode_log, config.LOG_FREQ, time.time() - start_time, losses=recent_losses)
                obs = env.reset()
                trajectory_obs = np.asarray(obs, dtype=np.float32)
                episode += 1
                episode_step = 0
                for key in episode_totals:
                    episode_totals[key] = 0.0

        bootstrap_offload_obs, bootstrap_offload_masks = env.get_offloading_obs_and_masks()
        bootstrap_joint_obs = make_joint_obs(trajectory_obs, bootstrap_offload_obs)
        with torch.no_grad():
            _, _, _, last_values = model.get_action_and_value(bootstrap_joint_obs, masks=bootstrap_offload_masks, exploration=False)
        recent_losses = update_joint_model(model, buffer, last_values)

    save_dir = Path("saved_models") / f"joint_mappo_{timestamp}" / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir))
    summary = {
        "timestamp": timestamp,
        "model_name": "joint_mappo",
        "seed": int(seed) if seed is not None else None,
        "num_episodes": int(num_episodes),
        "mean_recent_reward": float(np.mean(recent_rewards[-max(1, int(num_episodes * 0.1)) :])) if recent_rewards else 0.0,
        "model_dir": str(save_dir),
        "log_json": logger.json_file_path,
        "final_losses": {key: float(value) for key, value in recent_losses.items()},
    }
    summary_path = Path("results") / "reports" / f"joint_e2e_mappo_summary_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a true end-to-end joint MAPPO baseline.")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_joint_end_to_end_mappo(args.num_episodes, args.timestamp, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
