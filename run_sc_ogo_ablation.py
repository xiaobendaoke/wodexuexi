"""Run runtime component ablations for the SC-OGO offloading policy.

This script keeps the trained surrogate classifier fixed and changes only the
online SC-OGO reranking terms. It is intended for mechanism-level paper
evidence: which term explains MBS-load reduction, DSR safety, queue awareness,
and cooperative-path adjustment.
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
    RuntimeScenario,
    aggregate_metric_dicts,
    apply_runtime_scenario,
    build_runtime_scenarios,
    restore_config,
    set_runtime_policy,
    snapshot_config,
)
from environment.env import Env
from environment.uavs import UAV
from marl_models.static_baseline.static_model import StaticModel
from paths import results_path


@dataclass(frozen=True)
class AblationVariant:
    name: str
    policy_mode: str
    checkpoint_path: str | None
    config_overrides: dict[str, float]
    description: str


def build_ablation_variants(surrogate_checkpoint: str) -> list[AblationVariant]:
    return [
        AblationVariant(
            name="heuristic_offloading",
            policy_mode="heuristic",
            checkpoint_path=None,
            config_overrides={},
            description="Original heuristic service-request offloading baseline.",
        ),
        AblationVariant(
            name="classifier_only",
            policy_mode="learned",
            checkpoint_path=surrogate_checkpoint,
            config_overrides={},
            description="Oracle-guided surrogate classifier without SC-OGO reranking.",
        ),
        AblationVariant(
            name="sc_ogo_full",
            policy_mode="sc_ogo",
            checkpoint_path=surrogate_checkpoint,
            config_overrides={},
            description="Full SC-OGO reranking with default MBS, deadline-margin, queue, and cooperative terms.",
        ),
        AblationVariant(
            name="sc_ogo_no_mbs_penalty",
            policy_mode="sc_ogo",
            checkpoint_path=surrogate_checkpoint,
            config_overrides={"SC_OGO_MBS_WEIGHT": 0.0},
            description="Remove the online MBS fallback penalty to test where MBS compression comes from.",
        ),
        AblationVariant(
            name="sc_ogo_no_deadline_margin",
            policy_mode="sc_ogo",
            checkpoint_path=surrogate_checkpoint,
            config_overrides={
                "SC_OGO_MARGIN_WEIGHT": 0.0,
                "SC_OGO_DEADLINE_MARGIN": 1.0,
                "SC_OGO_HARD_DEADLINE_RATIO": 10.0,
            },
            description="Remove the pre-deadline safety margin and hard safety switch while keeping violation cost.",
        ),
        AblationVariant(
            name="sc_ogo_no_queue_pressure",
            policy_mode="sc_ogo",
            checkpoint_path=surrogate_checkpoint,
            config_overrides={"SC_OGO_QUEUE_WEIGHT": 0.0},
            description="Remove online queue-pressure cost to test whether queue awareness matters.",
        ),
        AblationVariant(
            name="sc_ogo_no_coop_term",
            policy_mode="sc_ogo",
            checkpoint_path=surrogate_checkpoint,
            config_overrides={"SC_OGO_COOP_WEIGHT": 0.0},
            description="Remove the online cooperative-path term to test cooperative routing adjustment.",
        ),
    ]


def select_scenarios(requested_names: list[str] | None) -> list[RuntimeScenario]:
    scenarios = build_runtime_scenarios()
    if not requested_names:
        return scenarios
    by_name = {scenario.name: scenario for scenario in scenarios}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown scenario(s): {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in requested_names]


def apply_variant_overrides(variant: AblationVariant) -> None:
    for key, value in variant.config_overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"config has no attribute {key!r}")
        setattr(config, key, value)


def run_variant_for_seed(
    *,
    scenario: RuntimeScenario,
    variant: AblationVariant,
    base_snapshot: dict[str, object],
    seed: int,
    episodes_per_seed: int,
    steps_per_episode: int,
) -> dict[str, object]:
    seed_episode_metrics: list[dict[str, float]] = []

    for episode_idx in range(episodes_per_seed):
        apply_runtime_scenario(scenario, base_snapshot)
        apply_variant_overrides(variant)
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
        "scenario": scenario.name,
        "episode_metrics": seed_episode_metrics,
        "per_seed_mean": per_seed_mean,
    }


def paired_deltas(
    seed_summaries: dict[str, list[dict[str, float]]],
    *,
    reference_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if reference_name not in seed_summaries:
        return {}
    reference_units = seed_summaries[reference_name]
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for policy_name, policy_units in seed_summaries.items():
        if policy_name == reference_name:
            continue
        if len(policy_units) != len(reference_units):
            raise ValueError(f"Cannot pair {policy_name} with {reference_name}: different unit counts.")
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


def run_sc_ogo_ablation(
    *,
    surrogate_checkpoint: str | Path,
    seeds: list[int],
    episodes_per_seed: int,
    steps_per_episode: int,
    output_path: str | Path,
    scenario_names: list[str] | None = None,
) -> dict[str, object]:
    base_snapshot = snapshot_config()
    scenarios = select_scenarios(scenario_names)
    variants = build_ablation_variants(str(surrogate_checkpoint))

    scenario_results: dict[str, dict[str, object]] = {}
    overall_seed_summaries: dict[str, list[dict[str, float]]] = {variant.name: [] for variant in variants}

    try:
        for scenario in scenarios:
            scenario_variant_results: dict[str, object] = {}
            for variant in variants:
                per_seed_runs: list[dict[str, object]] = []
                per_seed_means: list[dict[str, float]] = []
                for seed in seeds:
                    run_result = run_variant_for_seed(
                        scenario=scenario,
                        variant=variant,
                        base_snapshot=base_snapshot,
                        seed=int(seed),
                        episodes_per_seed=episodes_per_seed,
                        steps_per_episode=steps_per_episode,
                    )
                    per_seed_runs.append(run_result)
                    per_seed_means.append(run_result["per_seed_mean"])
                    overall_seed_summaries[variant.name].append(run_result["per_seed_mean"])

                scenario_variant_results[variant.name] = {
                    "description": variant.description,
                    "config_overrides": variant.config_overrides,
                    "per_seed": per_seed_runs,
                    "aggregate": aggregate_metric_dicts(per_seed_means),
                }
            scenario_results[scenario.name] = scenario_variant_results
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
            "metric_semantics": "Per-episode means of per-step metrics, then aggregated across seeds and scenarios.",
        },
        "variants": [
            {
                "name": variant.name,
                "policy_mode": variant.policy_mode,
                "description": variant.description,
                "config_overrides": variant.config_overrides,
            }
            for variant in variants
        ],
        "scenarios": [{"name": scenario.name, "description": scenario.description} for scenario in scenarios],
        "per_scenario": scenario_results,
        "overall": {
            policy_name: aggregate_metric_dicts(per_seed_means)
            for policy_name, per_seed_means in overall_seed_summaries.items()
        },
        "delta_vs_heuristic": paired_deltas(overall_seed_summaries, reference_name="heuristic_offloading"),
        "delta_vs_classifier_only": paired_deltas(overall_seed_summaries, reference_name="classifier_only"),
        "delta_vs_sc_ogo_full": paired_deltas(overall_seed_summaries, reference_name="sc_ogo_full"),
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SC-OGO runtime component ablations.")
    parser.add_argument("--surrogate_checkpoint", type=str, required=True, help="Surrogate checkpoint used by classifier and SC-OGO.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(results_path("reports", "sc_ogo_component_ablation.json")),
        help="Output JSON report.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Workload seeds.")
    parser.add_argument("--episodes_per_seed", type=int, default=4, help="Episodes per seed and scenario.")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="Steps per episode.")
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of runtime scenario names. Defaults to all runtime scenarios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_sc_ogo_ablation(
        surrogate_checkpoint=args.surrogate_checkpoint,
        seeds=[int(seed) for seed in args.seeds],
        episodes_per_seed=int(args.episodes_per_seed),
        steps_per_episode=int(args.steps_per_episode),
        output_path=args.output,
        scenario_names=args.scenarios,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
