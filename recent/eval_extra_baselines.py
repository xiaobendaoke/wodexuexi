#!/usr/bin/env python3
"""Evaluate additional baselines for the paper."""
import sys, os, json, time, math, argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_ROOT))

import config
from environment.env import Env

METRICS = [
    "reward", "latency", "energy", "fairness", "offline_rate",
    "deadline_satisfaction_rate", "offloading_ratio_local",
    "offloading_ratio_cooperative", "offloading_ratio_mbs", "mbs_load_ratio",
]

def spread_positions():
    positions = []
    for i in range(config.NUM_UAVS):
        angle = 2 * math.pi * i / config.NUM_UAVS
        cx = config.AREA_WIDTH / 2 + (config.AREA_WIDTH / 4) * math.cos(angle)
        cy = config.AREA_WIDTH / 2 + (config.AREA_WIDTH / 4) * math.sin(angle)
        positions.append(np.array([cx, cy], dtype=np.float32))
    return positions

def run_episode(env, get_actions_fn, steps):
    obs = env.reset(initial_positions=spread_positions())
    total_r = 0.0
    ep_metrics = {m: [] for m in METRICS}
    for step in range(steps):
        actions = get_actions_fn(env, obs, step)
        next_obs, rewards, metrics = env.step(actions)
        total_r += float(np.mean(rewards))
        for m in METRICS:
            ep_metrics[m].append(float(metrics.get(m, 0.0)))
    result = {"reward": total_r}
    for m in METRICS:
        vals = [v for v in ep_metrics[m] if np.isfinite(v)]
        result[m] = float(np.mean(vals)) if vals else 0.0
    return result

def zero_actions(env, obs, step):
    return [np.zeros(2, dtype=np.float32) for _ in range(config.NUM_UAVS)]

def random_actions(env, obs, step):
    rng = np.random.default_rng(step)
    return [rng.uniform(-1, 1, 2).astype(np.float32) for _ in range(config.NUM_UAVS)]

def eval_all_local(training_seeds, workload_seeds, episodes, steps):
    print("[all_local] evaluating...")
    results = []
    for seed in training_seeds:
        seed_means = []
        for wl in workload_seeds:
            np.random.seed(wl)
            for ep in range(episodes):
                env = Env()
                r = run_episode(env, zero_actions, steps)
                seed_means.append(r)
        results.append({"seed": seed, "per_seed_mean": {m: float(np.mean([x[m] for x in seed_means])) for m in METRICS}})
        print("  seed=%d done, reward=%.1f" % (seed, results[-1]["per_seed_mean"]["reward"]))
    return results

def eval_all_mbs(training_seeds, workload_seeds, episodes, steps):
    print("[all_mbs] evaluating...")
    import environment.uavs as uavs_mod
    _orig = uavs_mod.UAV._process_service_request

    def _force_mbs(self, ue, ue_uav_rate, target_idx, target_uav):
        return _orig(self, ue, ue_uav_rate, 2, None)

    uavs_mod.UAV._process_service_request = _force_mbs

    results = []
    for seed in training_seeds:
        seed_means = []
        for wl in workload_seeds:
            np.random.seed(wl)
            for ep in range(episodes):
                env = Env()
                r = run_episode(env, zero_actions, steps)
                seed_means.append(r)
        results.append({"seed": seed, "per_seed_mean": {m: float(np.mean([x[m] for x in seed_means])) for m in METRICS}})
        print("  seed=%d done, reward=%.1f" % (seed, results[-1]["per_seed_mean"]["reward"]))

    uavs_mod.UAV._process_service_request = _orig
    return results

def eval_random_offload(training_seeds, workload_seeds, episodes, steps):
    print("[random_offload] evaluating...")
    results = []
    for seed in training_seeds:
        seed_means = []
        for wl in workload_seeds:
            np.random.seed(wl)
            for ep in range(episodes):
                env = Env()
                r = run_episode(env, random_actions, steps)
                seed_means.append(r)
        results.append({"seed": seed, "per_seed_mean": {m: float(np.mean([x[m] for x in seed_means])) for m in METRICS}})
        print("  seed=%d done, reward=%.1f" % (seed, results[-1]["per_seed_mean"]["reward"]))
    return results

def eval_fixed_position(training_seeds, workload_seeds, episodes, steps):
    print("[fixed_position] evaluating...")
    results = []
    for seed in training_seeds:
        seed_means = []
        for wl in workload_seeds:
            np.random.seed(wl)
            for ep in range(episodes):
                env = Env()
                r = run_episode(env, zero_actions, steps)
                seed_means.append(r)
        results.append({"seed": seed, "per_seed_mean": {m: float(np.mean([x[m] for x in seed_means])) for m in METRICS}})
        print("  seed=%d done, reward=%.1f" % (seed, results[-1]["per_seed_mean"]["reward"]))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="extra_baselines_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--training_seeds", type=int, nargs="+", default=[42, 84, 126])
    parser.add_argument("--workload_seeds", type=int, nargs="+", default=[42, 84, 126, 168, 210, 252, 294, 336, 378, 420])
    parser.add_argument("--episodes_per_seed", type=int, default=6)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--baselines", nargs="+", default=["all_local", "all_mbs", "random_offload", "fixed_position"])
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "results" / "extra_baselines" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.training_seeds
    wls = args.workload_seeds
    eps = args.episodes_per_seed
    steps = args.steps_per_episode

    all_data = {"metadata": {"run_name": args.run_name, "training_seeds": seeds, "workload_seeds": wls}}

    if "all_local" in args.baselines:
        all_data["all_local"] = {"per_seed": eval_all_local(seeds, wls, eps, steps)}
    if "all_mbs" in args.baselines:
        all_data["all_mbs"] = {"per_seed": eval_all_mbs(seeds, wls, eps, steps)}
    if "random_offload" in args.baselines:
        all_data["random_offload"] = {"per_seed": eval_random_offload(seeds, wls, eps, steps)}
    if "fixed_position" in args.baselines:
        all_data["fixed_position__heuristic"] = {"per_seed": eval_fixed_position(seeds, wls, eps, steps)}

    out_path = out_dir / "extra_baselines_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)

    print("\n" + "=" * 90)
    print("Summary saved to: %s" % out_path)
    header = "%-30s %10s %12s %8s %10s %8s" % ("Baseline", "Reward", "Energy", "DSR", "MBS Load", "Offline")
    print(header)
    print("-" * 90)
    for label in sorted(all_data.keys()):
        if label == "metadata":
            continue
        data = all_data[label]
        all_means = [sd["per_seed_mean"] for sd in data["per_seed"]]
        avg = {}
        for m in METRICS:
            avg[m] = np.mean([am[m] for am in all_means])
        row = "%-30s %10.2f %12.0f %8.4f %10.4f %8.4f" % (
            label, avg["reward"], avg["energy"],
            avg["deadline_satisfaction_rate"], avg["mbs_load_ratio"], avg["offline_rate"]
        )
        print(row)

if __name__ == "__main__":
    main()
