from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from environment.env import Env
from environment.uavs import UAV
from marl_models.static_baseline.static_model import StaticModel


METRIC_NAMES: tuple[str, ...] = (
    "latency",
    "energy",
    "deadline_satisfaction_rate",
    "offloading_ratio_local",
    "offloading_ratio_cooperative",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
)


@dataclass(frozen=True, slots=True)
class RuntimeScenario:
    name: str
    description: str


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


def _scaled_int_array(base_values: np.ndarray, scale: float, minimum: int) -> np.ndarray:
    scaled = np.maximum(np.round(base_values.astype(np.float64) * scale).astype(np.int64), minimum)
    return scaled.astype(np.int64)


def build_runtime_scenarios() -> list[RuntimeScenario]:
    return [
        RuntimeScenario(
            name="default_uniform",
            description="Repository default configuration with uniformly distributed UEs.",
        ),
        RuntimeScenario(
            name="hotspot_deadline_stress",
            description="Hotspot-heavy traffic with tighter deadlines, heavier compute, and weaker backhaul.",
        ),
        RuntimeScenario(
            name="cooperative_friendly",
            description="Backhaul-limited cooperative regime with heterogeneous UAV compute and denser neighbor availability.",
        ),
        RuntimeScenario(
            name="mbs_heavy_jobs",
            description="Large service jobs and strong backhaul that make MBS routing more attractive.",
        ),
    ]


def apply_runtime_scenario(scenario: RuntimeScenario, base_snapshot: dict[str, object]) -> None:
    restore_config(base_snapshot)

    if scenario.name == "default_uniform":
        config.USE_HOTSPOTS = False
    elif scenario.name == "hotspot_deadline_stress":
        config.USE_HOTSPOTS = True
        config.HOTSPOT_UE_PROB = 0.9
        config.SERVICE_DEADLINE_MIN = 0.35 * config.TIME_SLOT_DURATION
        config.SERVICE_DEADLINE_MAX = 1.10 * config.TIME_SLOT_DURATION
        config.BANDWIDTH_BACKHAUL = 6 * 10**6
        config.CPU_CYCLES_PER_BYTE = _scaled_int_array(np.asarray(base_snapshot["CPU_CYCLES_PER_BYTE"]), scale=1.25, minimum=200)
        config.FILE_SIZES = _scaled_int_array(np.asarray(base_snapshot["FILE_SIZES"]), scale=1.10, minimum=1)
    elif scenario.name == "cooperative_friendly":
        config.USE_HOTSPOTS = True
        config.HOTSPOT_UE_PROB = 0.85
        config.SERVICE_DEADLINE_MIN = 0.55 * config.TIME_SLOT_DURATION
        config.SERVICE_DEADLINE_MAX = 1.75 * config.TIME_SLOT_DURATION
        config.BANDWIDTH_BACKHAUL = 3 * 10**6
        config.UAV_SENSING_RANGE = 360.0
        base_compute = np.asarray(base_snapshot["UAV_COMPUTING_CAPACITY"], dtype=np.int64)
        if base_compute.size >= 5:
            cooperative_compute = np.array([8, 30, 26, 16, 12], dtype=np.int64)[: base_compute.size] * 10**9
            config.UAV_COMPUTING_CAPACITY = cooperative_compute.astype(np.int64)
        else:
            boosted = base_compute.copy()
            if boosted.size > 1:
                boosted[1:] = _scaled_int_array(boosted[1:], scale=1.6, minimum=5 * 10**9)
            boosted[0] = int(max(boosted[0] * 0.8, 5 * 10**9))
            config.UAV_COMPUTING_CAPACITY = boosted.astype(np.int64)
        config.CPU_CYCLES_PER_BYTE = _scaled_int_array(np.asarray(base_snapshot["CPU_CYCLES_PER_BYTE"]), scale=1.10, minimum=200)
    elif scenario.name == "mbs_heavy_jobs":
        config.USE_HOTSPOTS = False
        config.SERVICE_DEADLINE_MIN = 0.45 * config.TIME_SLOT_DURATION
        config.SERVICE_DEADLINE_MAX = 1.25 * config.TIME_SLOT_DURATION
        config.BANDWIDTH_BACKHAUL = 18 * 10**6
        config.MIN_INPUT_SIZE = int(1.5 * 10**6)
        config.MAX_INPUT_SIZE = int(6.5 * 10**6)
        config.CPU_CYCLES_PER_BYTE = _scaled_int_array(np.asarray(base_snapshot["CPU_CYCLES_PER_BYTE"]), scale=1.40, minimum=200)
        config.FILE_SIZES = _scaled_int_array(np.asarray(base_snapshot["FILE_SIZES"]), scale=1.20, minimum=1)
    else:
        raise ValueError(f"Unsupported runtime scenario: {scenario.name}")

    config.AVG_FILE_SIZE = float(np.mean(config.FILE_SIZES))


def set_runtime_policy(policy_mode: str, checkpoint_path: str | None) -> None:
    if policy_mode == "heuristic":
        config.SERVICE_OFFLOAD_POLICY = "heuristic"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None
    elif policy_mode == "learned":
        config.SERVICE_OFFLOAD_POLICY = "learned"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = checkpoint_path
    else:
        raise ValueError(f"Unsupported runtime policy mode: {policy_mode}")
    UAV._policy_cache.clear()


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {metric_name: {"mean": 0.0, "std": 0.0} for metric_name in METRIC_NAMES}
    arrays = {metric_name: np.asarray([entry[metric_name] for entry in metric_dicts], dtype=np.float64) for metric_name in METRIC_NAMES}
    return {
        metric_name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for metric_name, values in arrays.items()
    }


def run_policy_for_seed(
    *,
    scenario: RuntimeScenario,
    policy_label: str,
    checkpoint_path: str | None,
    base_snapshot: dict[str, object],
    seed: int,
    episodes_per_seed: int,
    steps_per_episode: int,
) -> dict[str, object]:
    seed_episode_metrics: list[dict[str, float]] = []

    for episode_idx in range(episodes_per_seed):
        apply_runtime_scenario(scenario, base_snapshot)
        set_runtime_policy("heuristic" if checkpoint_path is None else "learned", checkpoint_path)

        run_seed = int(seed + episode_idx * 1000)
        np.random.seed(run_seed)
        static_model = StaticModel("static", config.NUM_UAVS, config.OBS_DIM_SINGLE, config.ACTION_DIM, "cpu")
        env = Env()
        env.reset(initial_positions=static_model.static_positions)

        episode_totals: dict[str, float] = {metric_name: 0.0 for metric_name in METRIC_NAMES}
        for _ in range(steps_per_episode):
            actions = np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
            _, _, metrics = env.step(actions)
            for metric_name in METRIC_NAMES:
                episode_totals[metric_name] += float(metrics[metric_name])

        seed_episode_metrics.append(
            {metric_name: total_value / float(steps_per_episode) for metric_name, total_value in episode_totals.items()}
        )

    per_seed_mean = {
        metric_name: float(np.mean([episode_metrics[metric_name] for episode_metrics in seed_episode_metrics]))
        for metric_name in METRIC_NAMES
    }
    return {
        "policy": policy_label,
        "seed": seed,
        "scenario": scenario.name,
        "episode_metrics": seed_episode_metrics,
        "per_seed_mean": per_seed_mean,
    }


def compare_runtime_offload_policies(
    *,
    surrogate_checkpoint: str | Path,
    rich_reduced_checkpoint: str | Path,
    seeds: list[int],
    episodes_per_seed: int,
    steps_per_episode: int,
    output_path: str | Path,
) -> dict[str, object]:
    base_snapshot = snapshot_config()
    scenarios = build_runtime_scenarios()
    policy_specs = {
        "heuristic_offloading": None,
        "surrogate_baseline": str(surrogate_checkpoint),
        "rich_reduced_runtime_policy": str(rich_reduced_checkpoint),
    }

    scenario_results: dict[str, dict[str, object]] = {}
    overall_seed_summaries: dict[str, list[dict[str, float]]] = {policy_name: [] for policy_name in policy_specs}

    try:
        for scenario in scenarios:
            scenario_policy_results: dict[str, object] = {}
            for policy_name, checkpoint_path in policy_specs.items():
                per_seed_runs: list[dict[str, object]] = []
                per_seed_means: list[dict[str, float]] = []
                for seed in seeds:
                    run_result = run_policy_for_seed(
                        scenario=scenario,
                        policy_label=policy_name,
                        checkpoint_path=checkpoint_path,
                        base_snapshot=base_snapshot,
                        seed=seed,
                        episodes_per_seed=episodes_per_seed,
                        steps_per_episode=steps_per_episode,
                    )
                    per_seed_runs.append(run_result)
                    per_seed_means.append(run_result["per_seed_mean"])
                    overall_seed_summaries[policy_name].append(run_result["per_seed_mean"])

                scenario_policy_results[policy_name] = {
                    "description": scenario.description,
                    "per_seed": per_seed_runs,
                    "aggregate": aggregate_metric_dicts(per_seed_means),
                }
            scenario_results[scenario.name] = scenario_policy_results
    finally:
        restore_config(base_snapshot)
        UAV._policy_cache.clear()

    overall_results = {
        policy_name: aggregate_metric_dicts(per_seed_means) for policy_name, per_seed_means in overall_seed_summaries.items()
    }

    delta_vs_heuristic: dict[str, dict[str, object]] = {}
    heuristic_seed_units = overall_seed_summaries["heuristic_offloading"]
    for policy_name in ("surrogate_baseline", "rich_reduced_runtime_policy"):
        policy_seed_units = overall_seed_summaries[policy_name]
        delta_vs_heuristic[policy_name] = {
            metric_name: {
                "mean_delta": float(
                    np.mean(
                        [
                            float(policy_seed_units[idx][metric_name] - heuristic_seed_units[idx][metric_name])
                            for idx in range(len(policy_seed_units))
                        ]
                    )
                ),
                "std_delta": float(
                    np.std(
                        [
                            float(policy_seed_units[idx][metric_name] - heuristic_seed_units[idx][metric_name])
                            for idx in range(len(policy_seed_units))
                        ]
                    )
                ),
            }
            for metric_name in METRIC_NAMES
        }

    scenario_wins: dict[str, dict[str, int]] = {}
    for policy_name in ("surrogate_baseline", "rich_reduced_runtime_policy"):
        latency_wins = 0
        energy_wins = 0
        deadline_wins = 0
        for scenario in scenarios:
            scenario_name = scenario.name
            heuristic_metrics = scenario_results[scenario_name]["heuristic_offloading"]["aggregate"]
            policy_metrics = scenario_results[scenario_name][policy_name]["aggregate"]
            if policy_metrics["latency"]["mean"] < heuristic_metrics["latency"]["mean"]:
                latency_wins += 1
            if policy_metrics["energy"]["mean"] < heuristic_metrics["energy"]["mean"]:
                energy_wins += 1
            if policy_metrics["deadline_satisfaction_rate"]["mean"] > heuristic_metrics["deadline_satisfaction_rate"]["mean"]:
                deadline_wins += 1
        scenario_wins[policy_name] = {
            "latency_wins": latency_wins,
            "energy_wins": energy_wins,
            "deadline_wins": deadline_wins,
            "num_scenarios": len(scenarios),
        }

    report: dict[str, object] = {
        "metadata": {
            "surrogate_checkpoint": str(surrogate_checkpoint),
            "rich_reduced_checkpoint": str(rich_reduced_checkpoint),
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "steps_per_episode": steps_per_episode,
            "trajectory_policy": "static baseline with zero motion actions",
            "metric_semantics": "Per-episode means of per-step metrics, then aggregated across seeds.",
        },
        "scenarios": [{ "name": scenario.name, "description": scenario.description } for scenario in scenarios],
        "per_scenario": scenario_results,
        "overall": overall_results,
        "delta_vs_heuristic": delta_vs_heuristic,
        "scenario_win_counts_vs_heuristic": scenario_wins,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heuristic, surrogate, and rich reduced runtime offloading policies.")
    parser.add_argument("--surrogate_checkpoint", type=str, required=True, help="Full-feature surrogate checkpoint.")
    parser.add_argument("--rich_checkpoint", type=str, required=True, help="Rich reduced runtime checkpoint.")
    parser.add_argument("--output", type=str, default="saved_offload_policies/runtime_offload_policy_comparison.json", help="Output JSON report.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Seeds used for the multi-seed comparison.")
    parser.add_argument("--episodes_per_seed", type=int, default=4, help="Episodes per seed and scenario.")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="Steps per episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_runtime_offload_policies(
        surrogate_checkpoint=args.surrogate_checkpoint,
        rich_reduced_checkpoint=args.rich_checkpoint,
        seeds=[int(seed) for seed in args.seeds],
        episodes_per_seed=args.episodes_per_seed,
        steps_per_episode=args.steps_per_episode,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
