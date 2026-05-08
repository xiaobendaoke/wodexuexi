"""Run one-factor scale generalization tests for runtime offloading policies.

The script varies UAV count, UE count, hotspot count, and backhaul bandwidth
while keeping the lower-layer policy checkpoint fixed. It is designed to show
whether the request-level offloading mechanism generalizes across environment
sizes and traffic regimes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from compare_runtime_offload_policies import (
    METRIC_NAMES,
    aggregate_metric_dicts,
    restore_config,
    set_runtime_policy,
    snapshot_config,
)
from environment.env import Env
from environment.uavs import UAV
from marl_models.static_baseline.static_model import StaticModel
from paths import results_path


@dataclass(frozen=True)
class ScaleCase:
    name: str
    num_uavs: int
    num_ues: int
    num_hotspots: int
    backhaul_bandwidth: int
    description: str


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    policy_mode: str
    checkpoint_path: str | None
    description: str


def _resize_array(base_values: object, length: int, *, fallback: int) -> np.ndarray:
    values = np.asarray(base_values, dtype=np.int64).reshape(-1)
    if values.size == 0:
        values = np.asarray([fallback], dtype=np.int64)
    if values.size >= length:
        return values[:length].copy()
    repeat_count = int(np.ceil(length / values.size))
    return np.tile(values, repeat_count)[:length].astype(np.int64)


def recompute_derived_config() -> None:
    config.NUM_FILES = int(config.NUM_SERVICES + config.NUM_CONTENTS)
    config.MAX_UAV_NEIGHBORS = max(0, int(config.NUM_UAVS) - 1)
    config.MAX_ASSOCIATED_UES = min(30, max(1, int(config.NUM_UES) // max(int(config.NUM_UAVS), 1) + 10))
    config.AVG_FILE_SIZE = float(np.mean(config.FILE_SIZES))

    config.SELF_OBS_DIM = 2 + int(config.NUM_FILES)
    config.REQUEST_OBS_DIM = 5
    config.UE_OBS_DIM = 2 + int(config.REQUEST_OBS_DIM) + 1
    config.NEIGHBOR_OBS_DIM = 2
    config.OBS_DIM_SINGLE = (
        int(config.SELF_OBS_DIM)
        + int(config.MAX_UAV_NEIGHBORS) * int(config.NEIGHBOR_OBS_DIM)
        + int(config.MAX_ASSOCIATED_UES) * int(config.UE_OBS_DIM)
    )


def apply_scale_case(case: ScaleCase, base_snapshot: dict[str, object]) -> None:
    restore_config(base_snapshot)

    config.NUM_UAVS = int(case.num_uavs)
    config.NUM_UES = int(case.num_ues)
    config.NUM_HOTSPOTS = int(case.num_hotspots)
    config.USE_HOTSPOTS = int(case.num_hotspots) > 0
    config.BANDWIDTH_BACKHAUL = int(case.backhaul_bandwidth)

    config.UAV_STORAGE_CAPACITY = _resize_array(
        base_snapshot["UAV_STORAGE_CAPACITY"],
        config.NUM_UAVS,
        fallback=150 * 10**6,
    )
    config.UAV_COMPUTING_CAPACITY = _resize_array(
        base_snapshot["UAV_COMPUTING_CAPACITY"],
        config.NUM_UAVS,
        fallback=60 * 10**9,
    )

    # Four hotspots do not fit cleanly with the default 400 m separation in a
    # 700 m x 700 m map. Keep non-overlap while making the stress case feasible.
    if config.NUM_HOTSPOTS >= 3:
        config.HOTSPOT_SEPARATION = max(2.0 * float(config.HOTSPOT_RADIUS), 200.0)

    recompute_derived_config()
    UAV._policy_cache.clear()


def build_scale_cases(
    *,
    baseline_uavs: int,
    baseline_ues: int,
    baseline_hotspots: int,
    baseline_backhaul: int,
    uav_values: list[int],
    ue_values: list[int],
    hotspot_values: list[int],
    backhaul_values: list[int],
) -> list[ScaleCase]:
    raw_cases: list[ScaleCase] = [
        ScaleCase(
            name=f"baseline_u{baseline_uavs}_ue{baseline_ues}_h{baseline_hotspots}_bh{baseline_backhaul}",
            num_uavs=baseline_uavs,
            num_ues=baseline_ues,
            num_hotspots=baseline_hotspots,
            backhaul_bandwidth=baseline_backhaul,
            description="Baseline scale used as the anchor for one-factor generalization.",
        )
    ]

    raw_cases.extend(
        ScaleCase(
            name=f"uav_{value}",
            num_uavs=int(value),
            num_ues=baseline_ues,
            num_hotspots=baseline_hotspots,
            backhaul_bandwidth=baseline_backhaul,
            description=f"Vary UAV count to {value}; keep other scale factors fixed.",
        )
        for value in uav_values
    )
    raw_cases.extend(
        ScaleCase(
            name=f"ue_{value}",
            num_uavs=baseline_uavs,
            num_ues=int(value),
            num_hotspots=baseline_hotspots,
            backhaul_bandwidth=baseline_backhaul,
            description=f"Vary UE count to {value}; keep other scale factors fixed.",
        )
        for value in ue_values
    )
    raw_cases.extend(
        ScaleCase(
            name=f"hotspots_{value}",
            num_uavs=baseline_uavs,
            num_ues=baseline_ues,
            num_hotspots=int(value),
            backhaul_bandwidth=baseline_backhaul,
            description=f"Vary hotspot count to {value}; keep other scale factors fixed.",
        )
        for value in hotspot_values
    )
    raw_cases.extend(
        ScaleCase(
            name=f"backhaul_{value}",
            num_uavs=baseline_uavs,
            num_ues=baseline_ues,
            num_hotspots=baseline_hotspots,
            backhaul_bandwidth=int(value),
            description=f"Vary backhaul bandwidth to {value} Hz; keep other scale factors fixed.",
        )
        for value in backhaul_values
    )

    seen: set[tuple[int, int, int, int]] = set()
    cases: list[ScaleCase] = []
    for case in raw_cases:
        key = (case.num_uavs, case.num_ues, case.num_hotspots, case.backhaul_bandwidth)
        if key in seen:
            continue
        seen.add(key)
        cases.append(case)
    return cases


def build_policy_variants(surrogate_checkpoint: str) -> list[PolicyVariant]:
    return [
        PolicyVariant(
            name="heuristic_offloading",
            policy_mode="heuristic",
            checkpoint_path=None,
            description="Original heuristic service-request offloading baseline.",
        ),
        PolicyVariant(
            name="classifier_only",
            policy_mode="learned",
            checkpoint_path=surrogate_checkpoint,
            description="Oracle-guided surrogate classifier without SC-OGO reranking.",
        ),
        PolicyVariant(
            name="sc_ogo_offloading",
            policy_mode="sc_ogo",
            checkpoint_path=surrogate_checkpoint,
            description="SC-OGO policy with safety-constrained reranking.",
        ),
    ]


def run_policy_for_scale_seed(
    *,
    case: ScaleCase,
    variant: PolicyVariant,
    base_snapshot: dict[str, object],
    seed: int,
    episodes_per_seed: int,
    steps_per_episode: int,
) -> dict[str, object]:
    seed_episode_metrics: list[dict[str, float]] = []

    for episode_idx in range(episodes_per_seed):
        apply_scale_case(case, base_snapshot)
        set_runtime_policy(variant.policy_mode, variant.checkpoint_path)

        run_seed = int(seed + episode_idx * 1000)
        np.random.seed(run_seed)
        static_model = StaticModel("static", config.NUM_UAVS, config.OBS_DIM_SINGLE, config.ACTION_DIM, "cpu")
        env = Env()
        env.reset(initial_positions=static_model.static_positions)

        episode_totals = {metric_name: 0.0 for metric_name in METRIC_NAMES}
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
        "policy": variant.name,
        "seed": seed,
        "scale_case": case.name,
        "episode_metrics": seed_episode_metrics,
        "per_seed_mean": per_seed_mean,
    }


def paired_delta_vs_heuristic(case_policy_means: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    reference_name = "heuristic_offloading"
    reference_units = case_policy_means[reference_name]
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for policy_name, policy_units in case_policy_means.items():
        if policy_name == reference_name:
            continue
        deltas[policy_name] = {}
        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [
                    float(policy_units[idx][metric_name] - reference_units[idx][metric_name])
                    for idx in range(len(reference_units))
                ],
                dtype=np.float64,
            )
            deltas[policy_name][metric_name] = {
                "mean_delta": float(np.mean(values)),
                "std_delta": float(np.std(values)),
            }
    return deltas


def run_scale_generalization(
    *,
    surrogate_checkpoint: str | Path,
    seeds: list[int],
    episodes_per_seed: int,
    steps_per_episode: int,
    output_path: str | Path,
    uav_values: list[int],
    ue_values: list[int],
    hotspot_values: list[int],
    backhaul_values: list[int],
) -> dict[str, object]:
    base_snapshot = snapshot_config()
    cases = build_scale_cases(
        baseline_uavs=int(base_snapshot["NUM_UAVS"]),
        baseline_ues=int(base_snapshot["NUM_UES"]),
        baseline_hotspots=int(base_snapshot["NUM_HOTSPOTS"]),
        baseline_backhaul=int(base_snapshot["BANDWIDTH_BACKHAUL"]),
        uav_values=uav_values,
        ue_values=ue_values,
        hotspot_values=hotspot_values,
        backhaul_values=backhaul_values,
    )
    variants = build_policy_variants(str(surrogate_checkpoint))

    case_results: dict[str, dict[str, object]] = {}
    try:
        for case in cases:
            case_policy_results: dict[str, object] = {}
            case_policy_means: dict[str, list[dict[str, float]]] = {variant.name: [] for variant in variants}
            for variant in variants:
                per_seed_runs: list[dict[str, object]] = []
                per_seed_means: list[dict[str, float]] = []
                for seed in seeds:
                    run_result = run_policy_for_scale_seed(
                        case=case,
                        variant=variant,
                        base_snapshot=base_snapshot,
                        seed=int(seed),
                        episodes_per_seed=episodes_per_seed,
                        steps_per_episode=steps_per_episode,
                    )
                    per_seed_runs.append(run_result)
                    per_seed_means.append(run_result["per_seed_mean"])
                    case_policy_means[variant.name].append(run_result["per_seed_mean"])

                case_policy_results[variant.name] = {
                    "description": variant.description,
                    "per_seed": per_seed_runs,
                    "aggregate": aggregate_metric_dicts(per_seed_means),
                }

            case_results[case.name] = {
                "config": {
                    "num_uavs": case.num_uavs,
                    "num_ues": case.num_ues,
                    "num_hotspots": case.num_hotspots,
                    "backhaul_bandwidth": case.backhaul_bandwidth,
                },
                "description": case.description,
                "policies": case_policy_results,
                "delta_vs_heuristic": paired_delta_vs_heuristic(case_policy_means),
            }
    finally:
        restore_config(base_snapshot)
        UAV._policy_cache.clear()

    report: dict[str, object] = {
        "metadata": {
            "surrogate_checkpoint": str(surrogate_checkpoint),
            "seeds": [int(seed) for seed in seeds],
            "episodes_per_seed": int(episodes_per_seed),
            "steps_per_episode": int(steps_per_episode),
            "trajectory_policy": "static baseline with zero motion actions",
            "scale_design": "One-factor-at-a-time scale sweep around the current config.py baseline.",
            "note": "Changing NUM_UAVS for a trained upper-layer MARL checkpoint requires retraining or a compatible architecture; this script isolates lower-layer runtime generalization.",
        },
        "policies": [
            {
                "name": variant.name,
                "policy_mode": variant.policy_mode,
                "description": variant.description,
            }
            for variant in variants
        ],
        "scale_cases": [
            {
                "name": case.name,
                "num_uavs": case.num_uavs,
                "num_ues": case.num_ues,
                "num_hotspots": case.num_hotspots,
                "backhaul_bandwidth": case.backhaul_bandwidth,
                "description": case.description,
            }
            for case in cases
        ],
        "per_case": case_results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_int_list(values: list[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(token) for token in value.replace(",", " ").split() if token.strip())
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scale generalization tests for runtime offloading policies.")
    parser.add_argument("--surrogate_checkpoint", type=str, required=True, help="Surrogate checkpoint used by classifier and SC-OGO.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(results_path("reports", "scale_generalization_runtime.json")),
        help="Output JSON report.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Workload seeds.")
    parser.add_argument("--episodes_per_seed", type=int, default=3, help="Episodes per seed and scale case.")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="Steps per episode.")
    parser.add_argument("--uav_values", type=str, nargs="+", default=["3", "5", "7"], help="UAV-count sweep values.")
    parser.add_argument("--ue_values", type=str, nargs="+", default=["50", "100", "150"], help="UE-count sweep values.")
    parser.add_argument("--hotspot_values", type=str, nargs="+", default=["1", "2", "3", "4"], help="Hotspot-count sweep values.")
    parser.add_argument(
        "--backhaul_values",
        type=str,
        nargs="+",
        default=["120000", "350000", "750000", "1500000", "5000000"],
        help="Backhaul-bandwidth sweep values in Hz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_scale_generalization(
        surrogate_checkpoint=args.surrogate_checkpoint,
        seeds=[int(seed) for seed in args.seeds],
        episodes_per_seed=int(args.episodes_per_seed),
        steps_per_episode=int(args.steps_per_episode),
        output_path=args.output,
        uav_values=parse_int_list(args.uav_values),
        ue_values=parse_int_list(args.ue_values),
        hotspot_values=parse_int_list(args.hotspot_values),
        backhaul_values=parse_int_list(args.backhaul_values),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
