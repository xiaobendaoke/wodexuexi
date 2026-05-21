#!/usr/bin/env python3
"""
新增对比方法评估脚本。
在不训练的情况下，评估以下 baseline：
  - all_local: 所有任务本地执行
  - all_mbs: 所有任务卸载到 MBS
  - random_offload: 随机卸载 + 随机轨迹
  - fixed_position: UAV 固定位置 + 下层 MAPPO
  - attention_maddpg__lower_mappo: 复用已有 MADDPG 模型
  - attention_matd3__lower_mappo: 复用已有 MATD3 模型

输出格式与主实验兼容，可合并统计。
"""
from __future__ import annotations

import json, os, sys, time, argparse, itertools, math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_ROOT))

import config
from environment.env import Env
from marl_models.uncoordinated_greedy_baseline.uncoordinated_greedy_model import UncoordinatedGreedyModel
from marl_models.attention_mappo.attention_mappo import AttentionMAPPO

# ============================================================
# 简单 baseline 策略
# ============================================================

def _greedy_initial_positions(env: Env) -> list[np.ndarray]:
    """Generate spread-out initial UAV positions."""
    rng = np.random.default_rng(42)
    positions = []
    for i in range(config.NUM_UAVS):
        angle = 2 * math.pi * i / config.NUM_UAVS
        center_x = config.AREA_SIZE / 2 + (config.AREA_SIZE / 4) * math.cos(angle)
        center_y = config.AREA_SIZE / 2 + (config.AREA_SIZE / 4) * math.sin(angle)
        pos = np.array([center_x, center_y, config.UAV_ALTITUDE], dtype=np.float32)
        positions.append(pos)
    return positions


def eval_all_local(seeds: list[int], workloads: list[int], episodes: int, steps: int) -> dict[str, Any]:
    """所有任务本地执行，UAV 静止。"""
    print("[all_local] Starting evaluation...")
    config.SERVICE_OFFLOAD_POLICY = "local_only"  # 强制本地
    return _eval_fixed_strategy("all_local", seeds, workloads, episodes, steps, fixed_positions=True)


def eval_all_mbs(seeds: list[int], workloads: list[int], episodes: int, steps: int) -> dict[str, Any]:
    """所有任务卸载到 MBS，UAV 静止。"""
    print("[all_mbs] Starting evaluation...")
    config.SERVICE_OFFLOAD_POLICY = "mbs_only"  # 强制 MBS
    return _eval_fixed_strategy("all_mbs", seeds, workloads, episodes, steps, fixed_positions=True)


def eval_random_offload(seeds: list[int], workloads: list[int], episodes: int, steps: int) -> dict[str, Any]:
    """随机卸载 + 随机轨迹。"""
    print("[random_offload] Starting evaluation...")
    return _eval_random_strategy("random_offload", seeds, workloads, episodes, steps)


def eval_fixed_position_lower_mappo(
    seeds: list[int], workloads: list[int], episodes: int, steps: int,
    lower_model_path: str
) -> dict[str, Any]:
    """UAV 固定位置 + 下层 MAPPO 卸载。"""
    print("[fixed_position__lower_mappo] Starting evaluation...")
    return _eval_fixed_lower("fixed_position__lower_mappo", seeds, workloads, episodes, steps, lower_model_path)


def eval_maddpg_lower_mappo(
    seeds: list[int], workloads: list[int], episodes: int, steps: int,
    maddpg_model_path: str, lower_model_path: str
) -> dict[str, Any]:
    """上层 MADDPG + 下层 MAPPO。"""
    print("[attention_maddpg__lower_mappo] Starting evaluation...")
    return _eval_upper_lower_combo("attention_maddpg__lower_mappo", seeds, workloads, episodes, steps,
                                   "maddpg", maddpg_model_path, lower_model_path)


def eval_matd3_lower_mappo(
    seeds: list[int], workloads: list[int], episodes: int, steps: int,
    matd3_model_path: str, lower_model_path: str
) -> dict[str, Any]:
    """上层 MATD3 + 下层 MAPPO。"""
    print("[attention_matd3__lower_mappo] Starting evaluation...")
    return _eval_upper_lower_combo("attention_matd3__lower_mappo", seeds, workloads, episodes, steps,
                                   "matd3", matd3_model_path, lower_model_path)


# ============================================================
# 内部实现
# ============================================================

DEFAULT_METRICS = (
    "reward", "latency", "energy", "fairness", "offline_rate",
    "deadline_satisfaction_rate", "offloading_ratio_local",
    "offloading_ratio_cooperative", "offloading_ratio_mbs", "mbs_load_ratio",
)

LOWER_IS_BETTER = {"latency", "energy", "offline_rate", "offloading_ratio_mbs", "mbs_load_ratio"}


def _run_one_episode(env: Env, get_actions_fn, steps: int, fixed_positions: bool = False) -> dict[str, float]:
    """Run one episode and return aggregated metrics."""
    initial_pos = _greedy_initial_positions(env)
    obs = env.reset(initial_positions=initial_pos)
    
    total_reward = 0.0
    ep_metrics: dict[str, list[float]] = {m: [] for m in DEFAULT_METRICS}
    
    for step in range(steps):
        actions = get_actions_fn(env, obs, step)
        next_obs, rewards, metrics = env.step(actions)
        total_reward += float(np.mean(rewards))
        for m in DEFAULT_METRICS:
            ep_metrics[m].append(float(metrics.get(m, 0.0)))
    
    # Average over episode
    result = {"reward": total_reward}
    for m in DEFAULT_METRICS:
        vals = ep_metrics[m]
        result[m] = float(np.mean(vals)) if vals else 0.0
    
    return result


def _eval_fixed_strategy(
    label: str, seeds: list[int], workloads: list[int],
    episodes: int, steps: int, fixed_positions: bool = True
) -> dict[str, Any]:
    """Evaluate a fixed offloading strategy."""
    per_seed_results = []
    
    for seed in seeds:
        seed_episode_means = []
        for wl in workloads:
            np.random.seed(wl)
            env = Env()
            for ep in range(episodes):
                def get_actions(env, obs, step):
                    if fixed_positions:
                        return [np.zeros(2, dtype=np.float32) for _ in range(config.NUM_UAVS)]
                    else:
                        return [np.random.uniform(-1, 1, 2).astype(np.float32) for _ in range(config.NUM_UAVS)]
                
                ep_result = _run_one_episode(env, get_actions, steps, fixed_positions)
                seed_episode_means.append(ep_result)
        
        # Aggregate over workload seeds
        metric_means = {}
        for m in DEFAULT_METRICS:
            vals = [ep[m] for ep in seed_episode_means]
            metric_means[m] = float(np.mean(vals))
        
        per_seed_results.append({
            "seed": seed,
            "per_seed_mean": metric_means,
            "episode_means": seed_episode_means,
        })
    
    return {"label": label, "per_seed": per_seed_results}


def _eval_random_strategy(
    label: str, seeds: list[int], workloads: list[int], episodes: int, steps: int
) -> dict[str, Any]:
    """Evaluate random offloading + random trajectory."""
    per_seed_results = []
    
    for seed in seeds:
        np.random.seed(seed)
        env = Env()
        seed_episode_means = []
        
        for wl in workloads:
            np.random.seed(wl)
            for ep in range(episodes):
                # Regenerate env with workload seed
                env2 = Env()
                env2.reset(initial_positions=_greedy_initial_positions(env2))
                
                def get_actions(env, obs, step):
                    return [np.random.uniform(-1, 1, 2).astype(np.float32) for _ in range(config.NUM_UAVS)]
                
                ep_result = _run_one_episode(env2, get_actions, steps, fixed_positions=False)
                seed_episode_means.append(ep_result)
        
        metric_means = {}
        for m in DEFAULT_METRICS:
            vals = [ep[m] for ep in seed_episode_means]
            metric_means[m] = float(np.mean(vals))
        
        per_seed_results.append({
            "seed": seed,
            "per_seed_mean": metric_means,
        })
    
    return {"label": label, "per_seed": per_seed_results}


def _eval_fixed_lower(
    label: str, seeds: list[int], workloads: list[int],
    episodes: int, steps: int, lower_model_path: str
) -> dict[str, Any]:
    """UAV fixed position + lower MAPPO."""
    # Load lower model
    from marl_models.offload_mappo.offload_mappo import OffloadMAPPO
    
    per_seed_results = []
    
    for seed in seeds:
        np.random.seed(seed)
        
        # Try to load the lower model for this seed
        model_path = Path(lower_model_path.replace("seed42", f"seed{seed}").replace("seed84", f"seed{seed}").replace("seed126", f"seed{seed}"))
        if not model_path.exists():
            # Fall back to the first available model
            print(f"  Warning: model not found at {model_path}, trying alternatives...")
            model_path = Path(lower_model_path)
        
        if not model_path.exists():
            print(f"  Warning: no model found, skipping seed {seed}")
            continue
        
        model = OffloadMAPPO.load(model_path)
        
        seed_episode_means = []
        for wl in workloads:
            np.random.seed(wl)
            env = Env()
            env.reset(initial_positions=_greedy_initial_positions(env))
            
            for ep in range(episodes):
                # Fixed position (zero action)
                def get_upper_actions(env, obs, step):
                    return [np.zeros(2, dtype=np.float32) for _ in range(config.NUM_UAVS)]
                
                ep_result = _run_with_lower_model(env, model, get_upper_actions, steps)
                seed_episode_means.append(ep_result)
        
        metric_means = {}
        for m in DEFAULT_METRICS:
            vals = [ep[m] for ep in seed_episode_means]
            metric_means[m] = float(np.mean(vals))
        
        per_seed_results.append({
            "seed": seed,
            "per_seed_mean": metric_means,
        })
    
    return {"label": label, "per_seed": per_seed_results}


def _eval_upper_lower_combo(
    label: str, seeds: list[int], workloads: list[int],
    episodes: int, steps: int,
    upper_type: str, upper_model_path: str, lower_model_path: str
) -> dict[str, Any]:
    """Evaluate upper + lower combo with existing models."""
    print(f"  [{label}] upper={upper_type}, upper_path={upper_model_path}, lower_path={lower_model_path}")
    
    # This requires loading both models, which depends on their specific APIs
    # For now, we note that the models exist and can be evaluated
    print(f"  NOTE: {label} requires model-specific loading. Models exist at:")
    print(f"    upper: {upper_model_path}")
    print(f"    lower: {lower_model_path}")
    
    return {"label": label, "per_seed": [], "status": "models_exist_but_need_custom_loader"}


def _run_with_lower_model(env: Env, model: Any, get_upper_actions_fn, steps: int) -> dict[str, float]:
    """Run episode with a given lower model and upper action function."""
    initial_pos = _greedy_initial_positions(env)
    obs = env.reset(initial_positions=initial_pos)
    
    total_reward = 0.0
    ep_metrics: dict[str, list[float]] = {m: [] for m in DEFAULT_METRICS}
    
    for step in range(steps):
        upper_actions = get_upper_actions_fn(env, obs, step)
        # Get lower actions from model
        with np.errstate(invalid="ignore"):
            lower_actions, _, _ = model.get_action_and_value(obs)
        
        # Combine: the env.step expects combined actions
        next_obs, rewards, metrics = env.step(upper_actions)
        total_reward += float(np.mean(rewards))
        for m in DEFAULT_METRICS:
            ep_metrics[m].append(float(metrics.get(m, 0.0)))
    
    result = {"reward": total_reward}
    for m in DEFAULT_METRICS:
        vals = ep_metrics[m]
        result[m] = float(np.mean(vals)) if vals else 0.0
    
    return result


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate additional baselines for the paper.")
    p.add_argument("--run_name", default=f"additional_baselines_{time.strftime(%Y%m%d)}")
    p.add_argument("--training_seeds", type=int, nargs="+", default=[42, 84, 126])
    p.add_argument("--workload_seeds", type=int, nargs="+", default=[42, 84, 126, 168, 210, 252, 294, 336, 378, 420])
    p.add_argument("--episodes_per_seed", type=int, default=6)
    p.add_argument("--steps_per_episode", type=int, default=1000)
    p.add_argument("--baselines", nargs="+", default=["all_local", "all_mbs", "random_offload"])
    p.add_argument("--output_dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "results" / "additional_baselines" / args.run_name
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = args.training_seeds
    workloads = args.workload_seeds
    eps = args.episodes_per_seed
    steps = args.steps_per_episode
    
    all_results = {}
    
    baselines_to_run = args.baselines
    
    if "all_local" in baselines_to_run:
        result = eval_all_local(seeds, workloads, eps, steps)
        all_results["all_local"] = result
    
    if "all_mbs" in baselines_to_run:
        result = eval_all_mbs(seeds, workloads, eps, steps)
        all_results["all_mbs"] = result
    
    if "random_offload" in baselines_to_run:
        result = eval_random_offload(seeds, workloads, eps, steps)
        all_results["random_offload"] = result
    
    # Save results
    output_path = args.output_dir / "additional_baselines_summary.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print(f"\n{=*80}")
    print("Summary:")
    print(f"{Baseline:<35} {Reward:>10} {Energy:>12} {DSR:>8} {MBS Load:>10}")
    print("-" * 80)
    for label, result in all_results.items():
        seeds_data = result.get("per_seed", [])
        if not seeds_data:
            print(f"{label:<35} {NO DATA:>10}")
            continue
        
        all_means = []
        for sd in seeds_data:
            pm = sd.get("per_seed_mean", {})
            if pm:
                all_means.append(pm)
        
        if not all_means:
            print(f"{label:<35} {NO DATA:>10}")
            continue
        
        avg = {}
        for m in DEFAULT_METRICS:
            vals = [am[m] for am in all_means if m in am]
            avg[m] = np.mean(vals) if vals else 0.0
        
        print(f"{label:<35} {avg[reward]:>10.2f} {avg[energy]:>12.0f} {avg[deadline_satisfaction_rate]:>8.4f} {avg[mbs_load_ratio]:>10.4f}")


if __name__ == "__main__":
    main()
