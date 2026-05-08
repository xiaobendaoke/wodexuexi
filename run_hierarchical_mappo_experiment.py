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
from marl_models.buffer_and_helpers import AttentionRolloutBuffer, DiscreteOffloadRolloutBuffer
from marl_models.utils import get_model, save_models
from utils.logger import Log, Logger


def lower_rewards_from_metrics(metrics: dict[str, float], system_rewards: list[float]) -> list[float]:
    deadline_penalty = 1.0 - float(metrics["deadline_satisfaction_rate"])
    latency_term = np.log(float(metrics["latency"]) + config.EPSILON)
    energy_term = np.log(float(metrics["energy"]) + config.EPSILON)
    mbs_term = float(metrics["mbs_load_ratio"])
    coop_term = float(metrics["offloading_ratio_cooperative"])
    success_term = float(metrics["deadline_satisfaction_rate"])
    lower_reward = (
        config.OFFLOAD_REWARD_SUCCESS_BONUS * success_term
        + config.OFFLOAD_REWARD_COOP_BONUS * coop_term
        - config.OFFLOAD_REWARD_DEADLINE_WEIGHT * deadline_penalty
        - config.OFFLOAD_REWARD_LATENCY_WEIGHT * latency_term
        - config.OFFLOAD_REWARD_ENERGY_WEIGHT * energy_term
        - config.OFFLOAD_REWARD_MBS_WEIGHT * mbs_term
    )
    lower_reward *= config.OFFLOAD_REWARD_SCALING_FACTOR
    return [float(lower_reward)] * len(system_rewards)


def update_on_policy_model(model, buffer, last_values: np.ndarray) -> dict[str, float]:
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


def train_hierarchical_mappo(num_episodes: int, timestamp: str | None = None) -> dict[str, object]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_hierarchical_mappo")
    env = Env()
    trajectory_model = get_model("attention_mappo")
    offload_model = get_model("offload_mappo")

    trajectory_buffer = AttentionRolloutBuffer(
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        buffer_size=config.PPO_ROLLOUT_LENGTH,
        device=trajectory_model.device,
    )
    offload_buffer = DiscreteOffloadRolloutBuffer(
        num_agents=config.NUM_UAVS,
        obs_dim=config.OFFLOAD_OBS_DIM_SINGLE,
        max_requests=config.MAX_OFFLOAD_REQUESTS_PER_UAV,
        num_actions=config.OFFLOAD_NUM_ACTIONS,
        buffer_size=config.PPO_ROLLOUT_LENGTH,
        device=offload_model.device,
    )

    logger = Logger(log_dir=f"train_logs/hierarchical_mappo", timestamp=timestamp)
    logger.log_configs()
    episode_log = Log()
    recent_losses = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
    start_time = time.time()

    obs = env.reset()
    traj_obs_arr = np.asarray(obs, dtype=np.float32)
    traj_state = np.concatenate(obs, axis=0, dtype=np.float32)
    episode = 1
    episode_step = 0
    episode_reward = 0.0
    episode_latency = 0.0
    episode_energy = 0.0
    episode_fairness = 0.0
    episode_offline_rate = 0.0
    episode_deadline_sum = 0.0
    episode_local_sum = 0.0
    episode_coop_sum = 0.0
    episode_mbs_sum = 0.0
    episode_mbs_load_sum = 0.0
    recent_rewards: list[float] = []

    max_time_steps = num_episodes * config.STEPS_PER_EPISODE
    num_updates = max_time_steps // config.PPO_ROLLOUT_LENGTH
    if num_updates <= 0:
        raise ValueError("num_updates is 0; increase num_episodes or reduce PPO_ROLLOUT_LENGTH.")

    for update in range(1, num_updates + 1):
        last_traj_obs = traj_obs_arr
        last_offload_obs = None
        for _ in range(config.PPO_ROLLOUT_LENGTH):
            offload_obs, offload_masks = env.get_offloading_obs_and_masks()
            traj_actions_raw, traj_log_probs, traj_values = trajectory_model.get_action_and_value(traj_obs_arr, traj_state)
            traj_actions = np.clip(traj_actions_raw, -1.0, 1.0)
            offload_actions, offload_log_probs, offload_values = offload_model.get_action_and_value(
                offload_obs,
                masks=offload_masks,
                exploration=True,
            )

            next_obs, system_rewards, metrics = env.step(traj_actions, offloading_actions=offload_actions)
            lower_rewards = lower_rewards_from_metrics(metrics, system_rewards)
            next_traj_state = np.concatenate(next_obs, axis=0, dtype=np.float32)

            episode_step += 1
            done = episode_step >= config.STEPS_PER_EPISODE
            trajectory_buffer.add(traj_state, traj_obs_arr, traj_actions_raw, traj_log_probs, system_rewards, done, traj_values)
            offload_buffer.add(offload_obs, offload_actions, offload_masks, offload_log_probs, lower_rewards, done, offload_values)

            traj_obs_arr = np.asarray(next_obs, dtype=np.float32)
            traj_state = next_traj_state
            last_traj_obs = traj_obs_arr
            last_offload_obs = offload_obs

            episode_reward += float(np.sum(system_rewards))
            episode_latency += float(metrics["latency"])
            episode_energy += float(metrics["energy"])
            episode_fairness = float(metrics["fairness"])
            episode_offline_rate = float(metrics["offline_rate"])
            episode_deadline_sum += float(metrics["deadline_satisfaction_rate"])
            episode_local_sum += float(metrics["offloading_ratio_local"])
            episode_coop_sum += float(metrics["offloading_ratio_cooperative"])
            episode_mbs_sum += float(metrics["offloading_ratio_mbs"])
            episode_mbs_load_sum += float(metrics["mbs_load_ratio"])

            if done:
                recent_rewards.append(episode_reward)
                episode_length = max(float(episode_step), 1.0)
                episode_log.append(
                    episode_reward,
                    episode_latency,
                    episode_energy,
                    episode_fairness,
                    episode_offline_rate,
                    deadline_satisfaction_rate=episode_deadline_sum / episode_length,
                    offloading_ratio_local=episode_local_sum / episode_length,
                    offloading_ratio_cooperative=episode_coop_sum / episode_length,
                    offloading_ratio_mbs=episode_mbs_sum / episode_length,
                    mbs_load_ratio=episode_mbs_load_sum / episode_length,
                    service_learned_decision_count=float(env.last_runtime_audit.get("episode_service_learned_decision_count", 0.0)),
                    service_heuristic_decision_count=float(env.last_runtime_audit.get("episode_service_heuristic_decision_count", 0.0)),
                    service_fallback_count=float(env.last_runtime_audit.get("episode_service_fallback_count", 0.0)),
                    service_predict_exception_fallback_count=float(env.last_runtime_audit.get("episode_service_predict_exception_fallback_count", 0.0)),
                    service_offload_policy_requested="hierarchical_mappo",
                    service_offload_policy_loaded=True,
                )
                if episode % config.LOG_FREQ == 0:
                    logger.log_metrics(episode, episode_log, config.LOG_FREQ, time.time() - start_time, losses=recent_losses)
                obs = env.reset()
                traj_obs_arr = np.asarray(obs, dtype=np.float32)
                traj_state = np.concatenate(obs, axis=0, dtype=np.float32)
                episode += 1
                episode_step = 0
                episode_reward = 0.0
                episode_latency = 0.0
                episode_energy = 0.0
                episode_fairness = 0.0
                episode_offline_rate = 0.0
                episode_deadline_sum = 0.0
                episode_local_sum = 0.0
                episode_coop_sum = 0.0
                episode_mbs_sum = 0.0
                episode_mbs_load_sum = 0.0

        with torch.no_grad():
            _, _, last_traj_values = trajectory_model.get_action_and_value(last_traj_obs, np.concatenate(last_traj_obs, axis=0, dtype=np.float32))
            next_offload_obs, next_offload_masks = env.get_offloading_obs_and_masks()
            if last_offload_obs is None:
                last_offload_obs = next_offload_obs
            _, _, last_offload_values = offload_model.get_action_and_value(
                next_offload_obs,
                masks=next_offload_masks,
                exploration=False,
            )

        traj_losses = update_on_policy_model(trajectory_model, trajectory_buffer, last_traj_values)
        offload_losses = update_on_policy_model(offload_model, offload_buffer, last_offload_values)
        recent_losses = {
            "actor": float((traj_losses["actor"] + offload_losses["actor"]) / 2.0),
            "critic": float((traj_losses["critic"] + offload_losses["critic"]) / 2.0),
            "entropy": float((traj_losses["entropy"] + offload_losses["entropy"]) / 2.0),
        }

    save_models(trajectory_model, -1, "update", timestamp, final=True)
    offload_save_dir = Path("saved_models") / f"offload_mappo_{timestamp}" / "final"
    offload_save_dir.mkdir(parents=True, exist_ok=True)
    offload_model.save(str(offload_save_dir))

    summary = {
        "timestamp": timestamp,
        "num_episodes": num_episodes,
        "mean_recent_reward": float(np.mean(recent_rewards[-max(1, int(num_episodes * 0.1)) :])) if recent_rewards else 0.0,
        "trajectory_model_dir": f"saved_models/{trajectory_model.model_name}_{timestamp}/final",
        "offload_model_dir": str(offload_save_dir),
        "log_json": logger.json_file_path,
    }
    summary_path = Path("results") / "reports" / f"hierarchical_mappo_summary_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical attention-MAPPO trajectory and offloading policies.")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--timestamp", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_hierarchical_mappo(num_episodes=args.num_episodes, timestamp=args.timestamp)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
