"""
Run constrained CQL-DQN offloading sensitivity experiments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import config
from collect_offload_dataset import collect_cql_transition_dataset
from compare_runtime_offload_policies import compare_runtime_offload_policies
from paths import results_path
from train_cql_offload_policy import train_cql_offload_policy
from train_offload_policy import parse_hidden_dims


@dataclass(frozen=True)
class CQLVariant:
    name: str
    deadline_weight: float
    mbs_weight: float
    queue_weight: float


def parse_variant(value: str) -> CQLVariant:
    tokens = [token.strip() for token in value.split(",")]
    if len(tokens) != 4:
        raise ValueError("Variant must use NAME,DEADLINE_WEIGHT,MBS_WEIGHT,QUEUE_WEIGHT")
    return CQLVariant(
        name=tokens[0],
        deadline_weight=float(tokens[1]),
        mbs_weight=float(tokens[2]),
        queue_weight=float(tokens[3]),
    )


def default_variants() -> list[CQLVariant]:
    return [
        CQLVariant("dsr_guard_dw5_mbs010", 5.0, 0.10, 0.05),
        CQLVariant("dsr_guard_dw6_mbs008", 6.0, 0.08, 0.05),
        CQLVariant("balanced_dw4_mbs012", 4.0, 0.12, 0.05),
    ]


def metric_mean(report: dict[str, object], policy: str, metric: str) -> float:
    return float(report["overall"][policy][metric]["mean"])  # type: ignore[index]


def metric_delta(report: dict[str, object], policy: str, metric: str) -> float:
    return float(report["delta_vs_heuristic"][policy][metric]["mean_delta"])  # type: ignore[index]


def run_variant(
    variant: CQLVariant,
    *,
    output_root: Path,
    surrogate_checkpoint: str,
    rich_checkpoint: str,
    per_class_target: int,
    max_attempts: int,
    procedural_train_per_class: int,
    dataset_seed: int,
    train_seed: int,
    epsilon_random: float,
    transition_horizon: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: tuple[int, ...],
    device: str,
    cql_alpha: float,
    gamma: float,
    runtime_seeds: list[int],
    episodes_per_seed: int,
    steps_per_episode: int,
) -> dict[str, object]:
    variant_root = output_root / variant.name
    dataset_path = variant_root / "datasets" / "offload_dataset_cql_transition.npz"
    checkpoint_path = variant_root / "checkpoints" / "offload_policy_cql.pt"
    report_path = variant_root / "reports" / "runtime_offload_policy_comparison.json"
    variant_root.mkdir(parents=True, exist_ok=True)

    print(f"[cql-sensitivity] collecting {variant.name} -> {dataset_path}")
    dataset_metadata = collect_cql_transition_dataset(
        output_path=dataset_path,
        template_per_class_target=per_class_target,
        template_max_attempts=max_attempts,
        procedural_train_per_class=procedural_train_per_class,
        seed=dataset_seed,
        label_mode="enhanced_oracle",
        epsilon_random=epsilon_random,
        transition_horizon=transition_horizon,
        deadline_weight=variant.deadline_weight,
        mbs_weight=variant.mbs_weight,
        queue_weight=variant.queue_weight,
    )

    print(f"[cql-sensitivity] training {variant.name} -> {checkpoint_path}")
    training_summary = train_cql_offload_policy(
        dataset_path=dataset_path,
        output_path=checkpoint_path,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=train_seed,
        device=device,
        gamma=gamma,
        cql_alpha=cql_alpha,
        target_update_freq=config.CQL_OFFLOAD_TARGET_UPDATE_FREQ,
    )

    print(f"[cql-sensitivity] comparing {variant.name} -> {report_path}")
    runtime_report = compare_runtime_offload_policies(
        surrogate_checkpoint=surrogate_checkpoint,
        rich_reduced_checkpoint=rich_checkpoint,
        cql_checkpoint=checkpoint_path,
        seeds=runtime_seeds,
        episodes_per_seed=episodes_per_seed,
        steps_per_episode=steps_per_episode,
        output_path=report_path,
    )

    summary = {
        "variant": variant.name,
        "weights": {
            "deadline": variant.deadline_weight,
            "mbs": variant.mbs_weight,
            "queue": variant.queue_weight,
        },
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "runtime_report_path": str(report_path),
        "dataset": {
            "transitions": dataset_metadata.get("collected_transitions"),
            "action_counts": dataset_metadata.get("action_counts"),
            "reward_mean": dataset_metadata.get("reward_mean"),
            "reward_std": dataset_metadata.get("reward_std"),
        },
        "training": {
            "best_epoch": training_summary.get("best_epoch"),
            "best_eval": training_summary.get("best_eval"),
        },
        "runtime": {
            "cql": {
                "dsr": metric_mean(runtime_report, "cql_dqn_offloading", "deadline_satisfaction_rate"),
                "mbs_load_ratio": metric_mean(runtime_report, "cql_dqn_offloading", "mbs_load_ratio"),
                "mbs_ratio": metric_mean(runtime_report, "cql_dqn_offloading", "offloading_ratio_mbs"),
                "energy": metric_mean(runtime_report, "cql_dqn_offloading", "energy"),
                "latency": metric_mean(runtime_report, "cql_dqn_offloading", "latency"),
            },
            "delta_vs_heuristic": {
                "dsr": metric_delta(runtime_report, "cql_dqn_offloading", "deadline_satisfaction_rate"),
                "mbs_load_ratio": metric_delta(runtime_report, "cql_dqn_offloading", "mbs_load_ratio"),
                "mbs_ratio": metric_delta(runtime_report, "cql_dqn_offloading", "offloading_ratio_mbs"),
                "energy": metric_delta(runtime_report, "cql_dqn_offloading", "energy"),
                "latency": metric_delta(runtime_report, "cql_dqn_offloading", "latency"),
            },
        },
    }
    (variant_root / "variant_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CQL-DQN offloading reward-weight sensitivity experiments.")
    parser.add_argument("--output_root", type=str, default=str(results_path("full_offload_experiments", "cql_sensitivity")), help="Output root for variant artifacts.")
    parser.add_argument("--surrogate_checkpoint", type=str, default="saved_offload_policies/offload_policy_surrogate_runtime.pt")
    parser.add_argument("--rich_checkpoint", type=str, default="saved_offload_policies/offload_policy_rich_runtime.pt")
    parser.add_argument("--variant", action="append", default=None, help="Variant as NAME,DEADLINE_WEIGHT,MBS_WEIGHT,QUEUE_WEIGHT. Repeatable.")
    parser.add_argument("--per_class_target", type=int, default=2000)
    parser.add_argument("--max_attempts", type=int, default=60000)
    parser.add_argument("--procedural_train_per_class", type=int, default=1800)
    parser.add_argument("--dataset_seed", type=int, default=config.SEED)
    parser.add_argument("--train_seed", type=int, default=config.SEED)
    parser.add_argument("--epsilon_random", type=float, default=config.CQL_OFFLOAD_EPSILON_RANDOM)
    parser.add_argument("--transition_horizon", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dims", type=str, default="128,128")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gamma", type=float, default=config.CQL_OFFLOAD_GAMMA)
    parser.add_argument("--cql_alpha", type=float, default=config.CQL_OFFLOAD_ALPHA)
    parser.add_argument("--runtime_seeds", type=int, nargs="+", default=[42, 84, 126, 168])
    parser.add_argument("--episodes_per_seed", type=int, default=4)
    parser.add_argument("--steps_per_episode", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [parse_variant(value) for value in args.variant] if args.variant else default_variants()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_variant(
            variant,
            output_root=output_root,
            surrogate_checkpoint=args.surrogate_checkpoint,
            rich_checkpoint=args.rich_checkpoint,
            per_class_target=args.per_class_target,
            max_attempts=args.max_attempts,
            procedural_train_per_class=args.procedural_train_per_class,
            dataset_seed=args.dataset_seed,
            train_seed=args.train_seed,
            epsilon_random=args.epsilon_random,
            transition_horizon=args.transition_horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dims=parse_hidden_dims(args.hidden_dims),
            device=args.device,
            cql_alpha=args.cql_alpha,
            gamma=args.gamma,
            runtime_seeds=[int(seed) for seed in args.runtime_seeds],
            episodes_per_seed=args.episodes_per_seed,
            steps_per_episode=args.steps_per_episode,
        )
        for variant in variants
    ]
    summary_path = output_root / "cql_sensitivity_summary.json"
    summary_path.write_text(json.dumps({"variants": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "variants": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
