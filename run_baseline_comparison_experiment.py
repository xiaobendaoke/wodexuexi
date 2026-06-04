"""
统一 baseline 对比实验入口。

支持的 baseline：
    proposed        - attention_mappo + constrained_attention_offload_mappo（本文方法）
    vanilla_mappo   - vanilla_mappo + constrained_attention_offload_mappo（消融 Attention）
    joint_mappo     - joint_mappo（端到端联合 MAPPO）
    random          - Random Policy（非学习随机策略）
    uniform         - Uniform Policy（非学习负载均衡策略）
    ippo            - IPPO + constrained_attention_offload_mappo（独立 PPO + 共享下层）

用法：
    python run_baseline_comparison_experiment.py --baseline random --eval_episodes 10 --seed 42
    python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 200 --eval_episodes 10 --seed 42
"""

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
from marl_models.buffer_and_helpers import AttentionRolloutBuffer, DiscreteOffloadRolloutBuffer, JointMAPPOBuffer
from marl_models.utils import get_model, save_models
from utils.baseline_metrics import (
    set_global_seed,
    run_single_episode,
    run_single_episode_joint,
    aggregate_metric_dicts,
    save_baseline_result,
    SUMMARY_METRIC_NAMES,
)
from utils.logger import Log, Logger


# ─── 非学习 baseline 列表 ───────────────────────────────────────────────────
NON_LEARNING_BASELINES = {"random", "uniform"}

# ─── 需要双层训练的 baseline（复用 train_hierarchical_mappo） ─────────────────
HIERARCHICAL_BASELINES = {"proposed", "vanilla_mappo"}

# ─── 所有支持的 baseline ─────────────────────────────────────────────────────
ALL_BASELINES = {"proposed", "vanilla_mappo", "joint_mappo", "random", "uniform", "ippo"}


# ─── 训练函数：复用原有脚本 ──────────────────────────────────────────────────


def train_hierarchical_baseline(
    baseline: str,
    num_episodes: int,
    seed: int,
    timestamp: str,
) -> dict:
    """复用 run_hierarchical_mappo_experiment.train_hierarchical_mappo() 训练分层 MAPPO。

    返回 summary dict，包含 trajectory_model_dir 和 offload_model_dir。
    """
    from run_hierarchical_mappo_experiment import train_hierarchical_mappo

    trajectory_model_name = "attention_mappo" if baseline == "proposed" else "vanilla_mappo"

    summary = train_hierarchical_mappo(
        num_episodes=num_episodes,
        timestamp=timestamp,
        mode="full_hierarchical",
        seed=seed,
        trajectory_model_name=trajectory_model_name,
        lower_ablation="full",
    )
    return summary


def train_joint_mappo_baseline(
    num_episodes: int,
    seed: int,
    timestamp: str,
) -> dict:
    """复用 run_joint_end_to_end_mappo_experiment.train_joint_end_to_end_mappo() 训练 Joint MAPPO。"""
    from run_joint_end_to_end_mappo_experiment import train_joint_end_to_end_mappo

    summary = train_joint_end_to_end_mappo(
        num_episodes=num_episodes,
        timestamp=timestamp,
        seed=seed,
    )
    return summary


def train_ippo_baseline(
    num_episodes: int,
    seed: int,
    timestamp: str,
) -> tuple[object, object, list[dict]]:
    """训练 IPPO（上层独立 PPO）+ 共享下层 constrained_attention_offload_mappo。

    返回 (trajectory_model, offload_model, training_curve)。
    关键：IPPO 也配同一个下层卸载模型，保证公平对比。
    """
    set_global_seed(seed)

    env = Env()
    trajectory_model = get_model("ippo_baseline")
    offload_model = get_model("constrained_attention_offload_mappo")

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

    logger = Logger(log_dir="train_logs/baseline_comparison/ippo", timestamp=timestamp)
    logger.log_configs()
    episode_log = Log()
    recent_losses = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
    start_time = time.time()

    obs = env.reset()
    traj_obs_arr = np.asarray(obs, dtype=np.float32)
    traj_state = np.concatenate(obs, axis=0, dtype=np.float32)
    episode = 1
    episode_step = 0
    recent_rewards: list[float] = []
    lambda_dsr = 0.0
    lambda_mbs = 0.0

    episode_totals = {key: 0.0 for key in [
        "reward", "latency", "energy", "fairness", "offline_rate",
        "deadline", "local", "coop", "mbs", "mbs_load",
    ]}

    max_time_steps = num_episodes * config.STEPS_PER_EPISODE
    num_updates = max_time_steps // config.PPO_ROLLOUT_LENGTH
    if num_updates <= 0:
        raise ValueError("num_updates is 0; increase num_episodes or reduce PPO_ROLLOUT_LENGTH.")

    training_curve: list[dict] = []

    def _update_model(model, buffer, last_values):
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

    for _ in range(1, num_updates + 1):
        last_traj_obs = traj_obs_arr
        update_dsr_violations: list[float] = []
        update_mbs_violations: list[float] = []

        for _ in range(config.PPO_ROLLOUT_LENGTH):
            # 下层卸载动作（使用共享的 offload_model）
            offload_obs, offload_masks = env.get_offloading_obs_and_masks()
            offload_actions, offload_log_probs, offload_values = offload_model.get_action_and_value(
                offload_obs, masks=offload_masks, exploration=True,
            )

            # 上层轨迹动作（IPPO：独立 PPO，critic 用局部 obs）
            traj_actions_raw, traj_log_probs, traj_values = trajectory_model.get_action_and_value(traj_obs_arr, traj_state)
            traj_actions = np.clip(traj_actions_raw, -1.0, 1.0)

            next_obs, system_rewards, metrics = env.step(traj_actions, offloading_actions=offload_actions)
            next_traj_state = np.concatenate(next_obs, axis=0, dtype=np.float32)

            # 下层奖励计算
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
            dsr_violation = max(0.0, float(config.OFFLOAD_DSR_TARGET) - float(metrics["deadline_satisfaction_rate"]))
            mbs_load_violation = max(0.0, float(metrics["mbs_load_ratio"]) - float(config.OFFLOAD_MBS_LOAD_CEILING))
            constraint_penalty = lambda_dsr * dsr_violation + lambda_mbs * mbs_load_violation
            lower_reward -= constraint_penalty
            lower_reward *= config.OFFLOAD_REWARD_SCALING_FACTOR
            lower_rewards = [float(lower_reward)] * config.NUM_UAVS

            update_dsr_violations.append(dsr_violation)
            update_mbs_violations.append(mbs_load_violation)

            episode_step += 1
            done = episode_step >= config.STEPS_PER_EPISODE

            trajectory_buffer.add(traj_state, traj_obs_arr, traj_actions_raw, traj_log_probs, system_rewards, done, traj_values)
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
                    actor_loss=recent_losses.get("actor"),
                    critic_loss=recent_losses.get("critic"),
                    entropy_loss=recent_losses.get("entropy"),
                )
                if episode % config.LOG_FREQ == 0:
                    logger.log_metrics(episode, episode_log, config.LOG_FREQ, time.time() - start_time, losses=recent_losses)
                training_curve.append({"episode": episode, "reward": episode_totals["reward"]})
                obs = env.reset()
                traj_obs_arr = np.asarray(obs, dtype=np.float32)
                traj_state = np.concatenate(obs, axis=0, dtype=np.float32)
                episode += 1
                episode_step = 0
                for key in episode_totals:
                    episode_totals[key] = 0.0

        # 上层 IPPO 更新
        with torch.no_grad():
            _, _, last_traj_values = trajectory_model.get_action_and_value(last_traj_obs, np.concatenate(last_traj_obs, axis=0, dtype=np.float32))
        traj_losses = _update_model(trajectory_model, trajectory_buffer, last_traj_values)

        # 下层 offload model 更新
        with torch.no_grad():
            next_offload_obs, next_offload_masks = env.get_offloading_obs_and_masks()
            _, _, last_offload_values = offload_model.get_action_and_value(
                next_offload_obs, masks=next_offload_masks, exploration=False,
            )
        offload_losses = _update_model(offload_model, offload_buffer, last_offload_values)

        # Lagrange 更新
        if getattr(config, "OFFLOAD_CONSTRAINT_MODE", "lagrange") == "lagrange":
            mean_dsr_violation = float(np.mean(update_dsr_violations)) if update_dsr_violations else 0.0
            mean_mbs_violation = float(np.mean(update_mbs_violations)) if update_mbs_violations else 0.0
            lambda_dsr = float(np.clip(lambda_dsr + config.OFFLOAD_LAGRANGE_LR * mean_dsr_violation, 0.0, config.OFFLOAD_LAGRANGE_MAX))
            lambda_mbs = float(np.clip(lambda_mbs + config.OFFLOAD_LAGRANGE_LR * mean_mbs_violation, 0.0, config.OFFLOAD_LAGRANGE_MAX))

        recent_losses = {
            "actor": float((traj_losses["actor"] + offload_losses["actor"]) / 2),
            "critic": float((traj_losses["critic"] + offload_losses["critic"]) / 2),
            "entropy": float((traj_losses["entropy"] + offload_losses["entropy"]) / 2),
        }

    # 保存模型
    save_models(trajectory_model, -1, "update", timestamp, final=True)
    offload_save_dir = Path("saved_models") / f"offload_mappo_{timestamp}" / "final"
    offload_save_dir.mkdir(parents=True, exist_ok=True)
    offload_model.save(str(offload_save_dir))

    return trajectory_model, offload_model, training_curve


# ─── 评估函数 ────────────────────────────────────────────────────────────────


def evaluate_baseline(
    baseline: str,
    trajectory_model,
    offload_model,
    eval_episodes: int,
    seed: int,
    workload_seed: int,
) -> tuple[list[dict[str, float]], list[dict]]:
    """评估 baseline，返回 (episode_metrics_list, trajectory_data)。"""
    set_global_seed(workload_seed)

    episode_metrics_list: list[dict[str, float]] = []
    all_trajectories: list[dict] = []

    for ep_idx in range(eval_episodes):
        run_seed = int(workload_seed + ep_idx * 1000)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        env = Env()

        if baseline in NON_LEARNING_BASELINES:
            # 非学习模型：自带卸载逻辑
            episode_metrics, trajectory = run_single_episode(
                env, trajectory_model, offload_model=None, exploration=False, record_trajectory=True,
                policy_type="non_learning",
            )
        elif baseline == "joint_mappo":
            # Joint MAPPO：单模型同时输出轨迹+卸载
            episode_metrics, trajectory = run_single_episode_joint(
                env, trajectory_model, exploration=False, record_trajectory=True,
                policy_type="learned",
            )
        else:
            # 分层 MAPPO / IPPO：trajectory_model + offload_model
            episode_metrics, trajectory = run_single_episode(
                env, trajectory_model, offload_model=offload_model, exploration=False, record_trajectory=True,
                policy_type="learned",
            )

        episode_metrics_list.append(episode_metrics)
        all_trajectories.append({"episode_idx": ep_idx, "seed": run_seed, "trajectory": trajectory})

    return episode_metrics_list, all_trajectories


# ─── 主入口 ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="统一 baseline 对比实验入口")
    parser.add_argument("--baseline", type=str, required=True, choices=sorted(ALL_BASELINES),
                        help="Baseline 方法名")
    parser.add_argument("--train_episodes", type=int, default=0,
                        help="训练 episode 数（非学习 baseline 忽略）")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="评估 episode 数")
    parser.add_argument("--seed", type=int, default=42,
                        help="训练种子")
    parser.add_argument("--workload_seed", type=int, default=None,
                        help="评估 workload 种子（默认等于 seed）")
    parser.add_argument("--output_dir", type=str, default="results/baseline_comparison",
                        help="输出目录")
    args = parser.parse_args()

    workload_seed = args.workload_seed if args.workload_seed is not None else args.seed
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{args.baseline}")

    print(f"\n{'='*60}")
    print(f"Baseline: {args.baseline}")
    print(f"Train episodes: {args.train_episodes}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Seed: {args.seed}")
    print(f"Workload seed: {workload_seed}")
    print(f"{'='*60}\n")

    # ─── 训练阶段 ────────────────────────────────────────────────────────────
    trajectory_model = None
    offload_model = None
    training_curve = None
    training_summary = None

    if args.baseline in NON_LEARNING_BASELINES:
        # 非学习 baseline：直接创建模型，不训练
        print(f"[{args.baseline}] Non-learning baseline, skipping training.")
        trajectory_model = get_model(f"{args.baseline}_baseline")
        offload_model = None  # 非学习模型自带卸载逻辑

    elif args.baseline in HIERARCHICAL_BASELINES:
        if args.train_episodes > 0:
            print(f"[{args.baseline}] Training hierarchical MAPPO for {args.train_episodes} episodes...")
            training_summary = train_hierarchical_baseline(args.baseline, args.train_episodes, args.seed, timestamp)
            # 从 summary 中获取模型路径，加载训练好的模型
            traj_dir = training_summary.get("trajectory_model_dir")
            offload_dir = training_summary.get("offload_model_dir")
            traj_model_name = training_summary.get("trajectory_model_name", "attention_mappo")
            trajectory_model = get_model(traj_model_name)
            if traj_dir:
                trajectory_model.load(traj_dir)
            offload_model = get_model("constrained_attention_offload_mappo")
            if offload_dir:
                offload_model.load(offload_dir)
            # 从 log_json 提取训练曲线
            log_json_path = training_summary.get("log_json")
            if log_json_path and Path(log_json_path).exists():
                with open(log_json_path, "r") as f:
                    log_data = json.load(f)
                training_curve = [{"episode": entry.get("episode", i), "reward": entry.get("reward", 0.0)} for i, entry in enumerate(log_data)]
        else:
            print(f"[{args.baseline}] No training requested, creating fresh model...")
            traj_model_name = "attention_mappo" if args.baseline == "proposed" else "vanilla_mappo"
            trajectory_model = get_model(traj_model_name)
            offload_model = get_model("constrained_attention_offload_mappo")

    elif args.baseline == "joint_mappo":
        if args.train_episodes > 0:
            print(f"[{args.baseline}] Training Joint MAPPO for {args.train_episodes} episodes...")
            training_summary = train_joint_mappo_baseline(args.train_episodes, args.seed, timestamp)
            # Joint MAPPO 只有一个模型
            model_dir = training_summary.get("model_dir")
            trajectory_model = get_model("joint_mappo")
            if model_dir:
                trajectory_model.load(model_dir)
            offload_model = None  # Joint MAPPO 自带卸载
        else:
            print(f"[{args.baseline}] No training requested, creating fresh model...")
            trajectory_model = get_model("joint_mappo")
            offload_model = None

    elif args.baseline == "ippo":
        if args.train_episodes > 0:
            print(f"[{args.baseline}] Training IPPO + shared offload model for {args.train_episodes} episodes...")
            trajectory_model, offload_model, training_curve = train_ippo_baseline(
                args.train_episodes, args.seed, timestamp,
            )
            training_summary = {"ippo_timestamp": timestamp}
        else:
            print(f"[{args.baseline}] No training requested, creating fresh model...")
            trajectory_model = get_model("ippo_baseline")
            offload_model = get_model("constrained_attention_offload_mappo")

    # ─── 评估阶段 ────────────────────────────────────────────────────────────
    print(f"\n[{args.baseline}] Evaluating for {args.eval_episodes} episodes...")
    episode_metrics_list, trajectory_data = evaluate_baseline(
        args.baseline, trajectory_model, offload_model,
        args.eval_episodes, args.seed, workload_seed,
    )

    # ─── 保存结果 ────────────────────────────────────────────────────────────
    result_path = save_baseline_result(
        method=args.baseline,
        seed=args.seed,
        workload_seed=workload_seed,
        episode_metrics_list=episode_metrics_list,
        output_dir=args.output_dir,
        training_curve=training_curve,
        trajectory_data=trajectory_data,
    )

    # 保存 training summary（如果有）
    if training_summary is not None:
        summary_path = Path(args.output_dir) / args.baseline / f"training_summary_seed_{args.seed}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(training_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 打印摘要
    aggregate = aggregate_metric_dicts(episode_metrics_list)
    print(f"\n{'='*60}")
    print(f"Results for {args.baseline}:")
    print(f"{'='*60}")
    for metric, values in aggregate.items():
        print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
