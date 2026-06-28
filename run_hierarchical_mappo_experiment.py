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
from utils.logger import Log, Logger, load_configs


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def apply_lower_ablation(ablation: str) -> str:
    if ablation == "full":
        config.OFFLOAD_MASK_MODE = "quality"
        config.OFFLOAD_CONSTRAINT_MODE = "lagrange"
        config.OFFLOAD_USE_ATTENTION = True
        return "constrained_attention_offload_mappo"
    if ablation == "no_mask":
        config.OFFLOAD_MASK_MODE = "none"
        config.OFFLOAD_CONSTRAINT_MODE = "lagrange"
        config.OFFLOAD_USE_ATTENTION = True
        return "constrained_attention_offload_mappo"
    if ablation == "no_lagrange":
        config.OFFLOAD_MASK_MODE = "quality"
        config.OFFLOAD_CONSTRAINT_MODE = "none"
        config.OFFLOAD_USE_ATTENTION = True
        return "constrained_attention_offload_mappo"
    if ablation == "no_attention":
        config.OFFLOAD_MASK_MODE = "quality"
        config.OFFLOAD_CONSTRAINT_MODE = "lagrange"
        config.OFFLOAD_USE_ATTENTION = False
        return "no_attention_offload_mappo"
    raise ValueError(f"Unknown lower-layer ablation: {ablation}")


def compute_constraint_diagnostics(
    metrics: dict[str, float],
    lambda_dsr: float,
    lambda_mbs: float,
) -> dict[str, float]:
    dsr_violation = max(0.0, float(config.OFFLOAD_DSR_TARGET) - float(metrics["deadline_satisfaction_rate"]))
    mbs_load_violation = max(0.0, float(metrics["mbs_load_ratio"]) - float(config.OFFLOAD_MBS_LOAD_CEILING))
    if getattr(config, "OFFLOAD_CONSTRAINT_MODE", "lagrange") == "lagrange":
        constraint_penalty = lambda_dsr * dsr_violation + lambda_mbs * mbs_load_violation
    else:
        constraint_penalty = 0.0
    return {
        "dsr_violation": float(dsr_violation),
        "mbs_load_violation": float(mbs_load_violation),
        "constraint_penalty": float(constraint_penalty),
    }


def lower_rewards_from_metrics(
    metrics: dict[str, float],
    system_rewards: list[float],
    lambda_dsr: float = 0.0,
    lambda_mbs: float = 0.0,
) -> tuple[list[float], dict[str, float]]:
    deadline_penalty = 1.0 - float(metrics["deadline_satisfaction_rate"])
    latency_term = float(metrics["latency"]) / (config.NUM_UES * config.NON_SERVED_LATENCY_PENALTY + config.EPSILON)
    energy_term = float(metrics["energy"]) / (config.NUM_UAVS * config.OFFLOAD_ENERGY_NORM_REF + config.EPSILON)
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
    diagnostics = compute_constraint_diagnostics(metrics, lambda_dsr, lambda_mbs)
    lower_reward -= diagnostics["constraint_penalty"]
    lower_reward *= config.OFFLOAD_REWARD_SCALING_FACTOR
    diagnostics["constraint_penalty"] *= float(config.OFFLOAD_REWARD_SCALING_FACTOR)
    return [float(lower_reward)] * len(system_rewards), diagnostics


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


def _make_trajectory_buffer(model):
    return AttentionRolloutBuffer(
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        buffer_size=config.PPO_ROLLOUT_LENGTH,
        device=model.device,
    )


def _make_offload_buffer(model):
    return DiscreteOffloadRolloutBuffer(
        num_agents=config.NUM_UAVS,
        obs_dim=config.OFFLOAD_OBS_DIM_SINGLE,
        max_requests=config.MAX_OFFLOAD_REQUESTS_PER_UAV,
        num_actions=config.OFFLOAD_NUM_ACTIONS,
        buffer_size=config.PPO_ROLLOUT_LENGTH,
        device=model.device,
    )


def train_hierarchical_mappo(
    num_episodes: int,
    timestamp: str | None = None,
    mode: str = "full_hierarchical",
    seed: int | None = None,
    trajectory_model_name: str = "attention_mappo",
    offload_model_name: str | None = None,
    trajectory_model_dir: str | None = None,
    trajectory_config_path: str | None = None,
    lower_ablation: str = "full",
    constraint_mode: str | None = None,
    mask_mode: str | None = None,
) -> dict[str, object]:
    if trajectory_config_path is not None:
        load_configs(trajectory_config_path)
    set_global_seed(seed)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_hierarchical_mappo")
    resolved_offload_model_name = offload_model_name or apply_lower_ablation(lower_ablation)
    if constraint_mode is not None:
        config.OFFLOAD_CONSTRAINT_MODE = constraint_mode
    if mask_mode is not None:
        config.OFFLOAD_MASK_MODE = mask_mode
    train_trajectory = mode in {"upper_only", "full_hierarchical"}
    train_offload = mode in {"lower_only_fixed_upper", "full_hierarchical"}
    use_offload_actions = train_offload

    env = Env()
    trajectory_model = get_model(trajectory_model_name)
    if trajectory_model_dir is not None:
        trajectory_model.load(trajectory_model_dir)
    offload_model = get_model(resolved_offload_model_name) if train_offload else None

    trajectory_buffer = _make_trajectory_buffer(trajectory_model) if train_trajectory else None
    offload_buffer = _make_offload_buffer(offload_model) if offload_model is not None else None

    logger = Logger(log_dir="train_logs/hierarchical_mappo", timestamp=timestamp)
    logger.log_configs()
    episode_log = Log()
    recent_losses = {
        "actor": 0.0,
        "critic": 0.0,
        "entropy": 0.0,
        "trajectory_actor": 0.0,
        "trajectory_critic": 0.0,
        "trajectory_entropy": 0.0,
        "lower_actor": 0.0,
        "lower_critic": 0.0,
        "lower_entropy": 0.0,
    }
    start_time = time.time()

    obs = env.reset()
    traj_obs_arr = np.asarray(obs, dtype=np.float32)
    traj_state = np.concatenate(obs, axis=0, dtype=np.float32)
    episode = 1
    episode_step = 0
    recent_rewards: list[float] = []
    training_curve: list[dict[str, float | int]] = []
    lambda_dsr = 0.0
    lambda_mbs = 0.0

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
        "lambda_dsr": 0.0,
        "lambda_mbs": 0.0,
        "dsr_violation": 0.0,
        "mbs_violation": 0.0,
        "constraint_penalty": 0.0,
        "coop_masked": 0.0,
        "mbs_masked": 0.0,
    }

    max_time_steps = num_episodes * config.STEPS_PER_EPISODE
    num_updates = max_time_steps // config.PPO_ROLLOUT_LENGTH
    if num_updates <= 0:
        raise ValueError("num_updates is 0; increase num_episodes or reduce PPO_ROLLOUT_LENGTH.")

    for _ in range(1, num_updates + 1):
        last_traj_obs = traj_obs_arr
        update_dsr_violations: list[float] = []
        update_mbs_violations: list[float] = []
        for _ in range(config.PPO_ROLLOUT_LENGTH):
            offload_obs = None
            offload_masks = None
            offload_actions = None
            offload_log_probs = None
            offload_values = None
            mask_audit: dict[str, object] = {}
            if use_offload_actions and offload_model is not None:
                offload_obs, offload_masks = env.get_offloading_obs_and_masks()
                mask_audit = dict(env.last_runtime_audit)
                offload_actions, offload_log_probs, offload_values = offload_model.get_action_and_value(
                    offload_obs,
                    masks=offload_masks,
                    exploration=True,
                )

            if train_trajectory:
                traj_actions_raw, traj_log_probs, traj_values = trajectory_model.get_action_and_value(traj_obs_arr, traj_state)
                traj_actions = np.clip(traj_actions_raw, -1.0, 1.0)
            else:
                traj_actions = trajectory_model.select_actions(traj_obs_arr, exploration=False)
                traj_actions_raw = traj_actions
                traj_log_probs = np.zeros(config.NUM_UAVS, dtype=np.float32)
                traj_values = np.zeros(config.NUM_UAVS, dtype=np.float32)

            next_obs, system_rewards, metrics = env.step(traj_actions, offloading_actions=offload_actions)
            lower_rewards, constraint_diag = lower_rewards_from_metrics(metrics, system_rewards, lambda_dsr, lambda_mbs)
            update_dsr_violations.append(constraint_diag["dsr_violation"])
            update_mbs_violations.append(constraint_diag["mbs_load_violation"])
            next_traj_state = np.concatenate(next_obs, axis=0, dtype=np.float32)

            episode_step += 1
            done = episode_step >= config.STEPS_PER_EPISODE
            if trajectory_buffer is not None:
                trajectory_buffer.add(traj_state, traj_obs_arr, traj_actions_raw, traj_log_probs, system_rewards, done, traj_values)
            if offload_buffer is not None and offload_obs is not None and offload_masks is not None and offload_actions is not None and offload_log_probs is not None and offload_values is not None:
                offload_buffer.add(offload_obs, offload_actions, offload_masks, offload_log_probs, lower_rewards, done, offload_values)

            traj_obs_arr = np.asarray(next_obs, dtype=np.float32)
            traj_state = next_traj_state
            last_traj_obs = traj_obs_arr

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
            episode_totals["lambda_dsr"] += float(lambda_dsr)
            episode_totals["lambda_mbs"] += float(lambda_mbs)
            episode_totals["dsr_violation"] += float(constraint_diag["dsr_violation"])
            episode_totals["mbs_violation"] += float(constraint_diag["mbs_load_violation"])
            episode_totals["constraint_penalty"] += float(constraint_diag["constraint_penalty"])
            episode_totals["coop_masked"] += float(mask_audit.get("step_coop_masked_count", 0.0))
            episode_totals["mbs_masked"] += float(mask_audit.get("step_mbs_masked_count", 0.0))

            if done:
                recent_rewards.append(episode_totals["reward"])
                episode_length = max(float(episode_step), 1.0)
                episode_curve_entry = {
                    "episode": int(episode),
                    "reward": float(episode_totals["reward"]),
                    "latency": float(episode_totals["latency"]),
                    "energy": float(episode_totals["energy"]),
                    "fairness": float(episode_totals["fairness"]),
                    "offline_rate": float(episode_totals["offline_rate"]),
                    "deadline_satisfaction_rate": float(episode_totals["deadline"] / episode_length),
                    "offloading_ratio_local": float(episode_totals["local"] / episode_length),
                    "offloading_ratio_cooperative": float(episode_totals["coop"] / episode_length),
                    "offloading_ratio_mbs": float(episode_totals["mbs"] / episode_length),
                    "mbs_load_ratio": float(episode_totals["mbs_load"] / episode_length),
                    "lambda_dsr": float(episode_totals["lambda_dsr"] / episode_length),
                    "lambda_mbs": float(episode_totals["lambda_mbs"] / episode_length),
                    "dsr_violation": float(episode_totals["dsr_violation"] / episode_length),
                    "mbs_load_violation": float(episode_totals["mbs_violation"] / episode_length),
                    "constraint_penalty": float(episode_totals["constraint_penalty"] / episode_length),
                    "coop_masked_count": float(episode_totals["coop_masked"]),
                    "mbs_masked_count": float(episode_totals["mbs_masked"]),
                }
                training_curve.append(episode_curve_entry)
                episode_log.append(
                    episode_totals["reward"],
                    episode_totals["latency"],
                    episode_totals["energy"],
                    episode_totals["fairness"],
                    episode_totals["offline_rate"],
                    deadline_satisfaction_rate=episode_curve_entry["deadline_satisfaction_rate"],
                    offloading_ratio_local=episode_curve_entry["offloading_ratio_local"],
                    offloading_ratio_cooperative=episode_curve_entry["offloading_ratio_cooperative"],
                    offloading_ratio_mbs=episode_curve_entry["offloading_ratio_mbs"],
                    mbs_load_ratio=episode_curve_entry["mbs_load_ratio"],
                    service_learned_decision_count=float(env.last_runtime_audit.get("episode_service_learned_decision_count", 0.0)),
                    service_heuristic_decision_count=float(env.last_runtime_audit.get("episode_service_heuristic_decision_count", 0.0)),
                    service_fallback_count=float(env.last_runtime_audit.get("episode_service_fallback_count", 0.0)),
                    service_predict_exception_fallback_count=float(env.last_runtime_audit.get("episode_service_predict_exception_fallback_count", 0.0)),
                    service_offload_policy_requested=resolved_offload_model_name if use_offload_actions else "heuristic",
                    service_offload_policy_loaded=bool(use_offload_actions),
                    lambda_dsr=episode_totals["lambda_dsr"] / episode_length,
                    lambda_mbs=episode_totals["lambda_mbs"] / episode_length,
                    dsr_violation=episode_totals["dsr_violation"] / episode_length,
                    mbs_load_violation=episode_totals["mbs_violation"] / episode_length,
                    constraint_penalty=episode_totals["constraint_penalty"] / episode_length,
                    coop_masked_count=episode_totals["coop_masked"],
                    mbs_masked_count=episode_totals["mbs_masked"],
                    actor_loss=recent_losses.get("actor"),
                    critic_loss=recent_losses.get("critic"),
                    entropy_loss=recent_losses.get("entropy"),
                )
                if episode % config.LOG_FREQ == 0:
                    logger.log_metrics(episode, episode_log, config.LOG_FREQ, time.time() - start_time, losses=recent_losses)
                obs = env.reset()
                traj_obs_arr = np.asarray(obs, dtype=np.float32)
                traj_state = np.concatenate(obs, axis=0, dtype=np.float32)
                episode += 1
                episode_step = 0
                for key in episode_totals:
                    episode_totals[key] = 0.0

        traj_losses = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
        offload_losses = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
        if trajectory_buffer is not None:
            with torch.no_grad():
                _, _, last_traj_values = trajectory_model.get_action_and_value(last_traj_obs, np.concatenate(last_traj_obs, axis=0, dtype=np.float32))
            traj_losses = update_on_policy_model(trajectory_model, trajectory_buffer, last_traj_values)
        if offload_buffer is not None and offload_model is not None:
            with torch.no_grad():
                next_offload_obs, next_offload_masks = env.get_offloading_obs_and_masks()
                _, _, last_offload_values = offload_model.get_action_and_value(
                    next_offload_obs,
                    masks=next_offload_masks,
                    exploration=False,
                )
            offload_losses = update_on_policy_model(offload_model, offload_buffer, last_offload_values)
        if train_offload and getattr(config, "OFFLOAD_CONSTRAINT_MODE", "lagrange") == "lagrange":
            mean_dsr_violation = float(np.mean(update_dsr_violations)) if update_dsr_violations else 0.0
            mean_mbs_violation = float(np.mean(update_mbs_violations)) if update_mbs_violations else 0.0
            lambda_dsr = float(np.clip(lambda_dsr + config.OFFLOAD_LAGRANGE_LR * mean_dsr_violation, 0.0, config.OFFLOAD_LAGRANGE_MAX))
            lambda_mbs = float(np.clip(lambda_mbs + config.OFFLOAD_LAGRANGE_LR * mean_mbs_violation, 0.0, config.OFFLOAD_LAGRANGE_MAX))

        divisor = float(max(int(train_trajectory) + int(train_offload), 1))
        recent_losses = {
            "actor": float((traj_losses["actor"] + offload_losses["actor"]) / divisor),
            "critic": float((traj_losses["critic"] + offload_losses["critic"]) / divisor),
            "entropy": float((traj_losses["entropy"] + offload_losses["entropy"]) / divisor),
            "trajectory_actor": float(traj_losses["actor"]),
            "trajectory_critic": float(traj_losses["critic"]),
            "trajectory_entropy": float(traj_losses["entropy"]),
            "lower_actor": float(offload_losses["actor"]),
            "lower_critic": float(offload_losses["critic"]),
            "lower_entropy": float(offload_losses["entropy"]),
        }

    trajectory_model_dir_out: str | None = trajectory_model_dir
    offload_model_dir_out: str | None = None
    if train_trajectory:
        save_models(trajectory_model, -1, "update", timestamp, final=True)
        trajectory_model_dir_out = f"saved_models/{trajectory_model.model_name}_{timestamp}/final"
    if train_offload and offload_model is not None:
        offload_save_dir = Path("saved_models") / f"offload_mappo_{timestamp}" / "final"
        offload_save_dir.mkdir(parents=True, exist_ok=True)
        offload_model.save(str(offload_save_dir))
        offload_model_dir_out = str(offload_save_dir)

    training_curve_path = Path("train_logs") / "hierarchical_mappo" / f"training_curve_{timestamp}.json"
    training_curve_path.parent.mkdir(parents=True, exist_ok=True)
    with training_curve_path.open("w", encoding="utf-8") as f:
        json.dump(training_curve, f, indent=2)

    summary = {
        "timestamp": timestamp,
        "mode": mode,
        "seed": int(seed) if seed is not None else None,
        "trajectory_model_name": trajectory_model_name,
        "lower_ablation": lower_ablation,
        "num_episodes": num_episodes,
        "mean_recent_reward": float(np.mean(recent_rewards[-max(1, int(num_episodes * 0.1)) :])) if recent_rewards else 0.0,
        "trajectory_model_dir": trajectory_model_dir_out,
        "offload_model_dir": offload_model_dir_out,
        "offload_model_name": resolved_offload_model_name,
        "offload_mask_mode": str(getattr(config, "OFFLOAD_MASK_MODE", "quality")),
        "offload_use_attention": bool(getattr(config, "OFFLOAD_USE_ATTENTION", True)),
        "constraint_mode": str(getattr(config, "OFFLOAD_CONSTRAINT_MODE", "lagrange")),
        "final_lambda_dsr": float(lambda_dsr),
        "final_lambda_mbs": float(lambda_mbs),
        "final_losses": {key: float(value) for key, value in recent_losses.items()},
        "offload_dsr_target": float(config.OFFLOAD_DSR_TARGET),
        "force_service_admission": bool(config.FORCE_SERVICE_ADMISSION),
        "offload_mbs_load_ceiling": float(config.OFFLOAD_MBS_LOAD_CEILING),
        "log_json": logger.json_file_path,
        "training_curve_json": str(training_curve_path),
    }
    summary_path = Path("results") / "reports" / f"hierarchical_mappo_summary_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train upper/lower hierarchical MARL policies for multi-UAV MEC.")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full_hierarchical", choices=["upper_only", "lower_only_fixed_upper", "full_hierarchical"])
    parser.add_argument("--seed", type=int, default=None, help="Training seed for NumPy and PyTorch.")
    parser.add_argument(
        "--trajectory_model",
        type=str,
        default="attention_mappo",
        help="Upper-layer trajectory controller used during hierarchical training.",
    )
    parser.add_argument("--lower_ablation", type=str, default="full", choices=["full", "no_mask", "no_lagrange", "no_attention"])
    parser.add_argument("--offload_model", type=str, default=None)
    parser.add_argument("--trajectory_model_dir", type=str, default=None)
    parser.add_argument("--trajectory_config_path", type=str, default=None)
    parser.add_argument("--constraint_mode", type=str, default=None, choices=["none", "lagrange"])
    parser.add_argument("--mask_mode", type=str, default=None, choices=["quality", "none"])
    parser.add_argument("--dsr_target", type=float, default=None)
    parser.add_argument("--force_service_admission", action="store_true", help="Enable nearest-UAV fallback admission for uncovered service requests")
    parser.add_argument("--mbs_load_ceiling", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.FORCE_SERVICE_ADMISSION = bool(args.force_service_admission)
    if args.constraint_mode is not None:
        config.OFFLOAD_CONSTRAINT_MODE = args.constraint_mode
    if args.mask_mode is not None:
        config.OFFLOAD_MASK_MODE = args.mask_mode
    if args.dsr_target is not None:
        config.OFFLOAD_DSR_TARGET = float(args.dsr_target)
    if args.mbs_load_ceiling is not None:
        config.OFFLOAD_MBS_LOAD_CEILING = float(args.mbs_load_ceiling)
    summary = train_hierarchical_mappo(
        num_episodes=args.num_episodes,
        timestamp=args.timestamp,
        mode=args.mode,
        seed=args.seed,
        trajectory_model_name=args.trajectory_model,
        offload_model_name=args.offload_model,
        trajectory_model_dir=args.trajectory_model_dir,
        trajectory_config_path=args.trajectory_config_path,
        lower_ablation=args.lower_ablation,
        constraint_mode=args.constraint_mode,
        mask_mode=args.mask_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
