"""
Run RADCC-Offload sensitivity experiments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import config
from collect_offload_dataset import collect_radcc_cost_dataset
from compare_runtime_offload_policies import compare_runtime_offload_policies
from paths import results_path
from train_offload_policy import parse_hidden_dims
from train_radcc_offload_policy import train_radcc_offload_policy


@dataclass(frozen=True, slots=True)
class RADCCVariant:
    name: str
    deadline_weight: float
    mbs_weight: float
    coop_weight: float
    queue_weight: float
    risk_beta: float
    cvar_alpha: float


def parse_variant(value: str) -> RADCCVariant:
    tokens = [token.strip() for token in value.split(",")]
    if len(tokens) != 7:
        raise ValueError("Variant must use NAME,DEADLINE_WEIGHT,MBS_WEIGHT,COOP_WEIGHT,QUEUE_WEIGHT,RISK_BETA,CVAR_ALPHA")
    return RADCCVariant(
        name=tokens[0],
        deadline_weight=float(tokens[1]),
        mbs_weight=float(tokens[2]),
        coop_weight=float(tokens[3]),
        queue_weight=float(tokens[4]),
        risk_beta=float(tokens[5]),
        cvar_alpha=float(tokens[6]),
    )


def default_variants() -> list[RADCCVariant]:
    return [
        RADCCVariant("radcc_dw6_mbs005_coop005_r0", 6.0, 0.05, 0.05, 0.05, 0.0, 0.80),
        RADCCVariant("radcc_dw8_mbs003_coop008_r0", 8.0, 0.03, 0.08, 0.05, 0.0, 0.80),
        RADCCVariant("radcc_dw8_mbs003_coop010_r015", 8.0, 0.03, 0.10, 0.05, 0.15, 0.80),
    ]


def metric_mean(report: dict[str, object], policy: str, metric: str) -> float:
    return float(report["overall"][policy][metric]["mean"])  # type: ignore[index]


def metric_delta(report: dict[str, object], policy: str, metric: str) -> float:
    return float(report["delta_vs_heuristic"][policy][metric]["mean_delta"])  # type: ignore[index]


def run_variant(
    variant: RADCCVariant,
    *,
    output_root: Path,
    surrogate_checkpoint: str,
    rich_checkpoint: str,
    cql_checkpoint: str | None,
    per_class_target: int,
    max_attempts: int,
    procedural_train_per_class: int,
    dataset_seed: int,
    train_seed: int,
    num_quantiles: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: tuple[int, ...],
    device: str,
    runtime_seeds: list[int],
    episodes_per_seed: int,
    steps_per_episode: int,
) -> dict[str, object]:
    variant_root = output_root / variant.name
    dataset_path = variant_root / "datasets" / "offload_dataset_radcc_cost.npz"
    checkpoint_path = variant_root / "checkpoints" / "offload_policy_radcc.pt"
    report_path = variant_root / "reports" / "runtime_offload_policy_comparison.json"
    variant_root.mkdir(parents=True, exist_ok=True)

    print(f"[radcc-sensitivity] collecting {variant.name} -> {dataset_path}")
    dataset_metadata = collect_radcc_cost_dataset(
        output_path=dataset_path,
        template_per_class_target=per_class_target,
        template_max_attempts=max_attempts,
        procedural_train_per_class=procedural_train_per_class,
        seed=dataset_seed,
        label_mode="enhanced_oracle",
        num_quantiles=num_quantiles,
        deadline_weight=variant.deadline_weight,
        mbs_weight=variant.mbs_weight,
        coop_weight=variant.coop_weight,
        queue_weight=variant.queue_weight,
    )

    print(f"[radcc-sensitivity] training {variant.name} -> {checkpoint_path}")
    training_summary = train_radcc_offload_policy(
        dataset_path=dataset_path,
        output_path=checkpoint_path,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=train_seed,
        device=device,
        risk_beta=variant.risk_beta,
        cvar_alpha=variant.cvar_alpha,
    )

    print(f"[radcc-sensitivity] comparing {variant.name} -> {report_path}")
    runtime_report = compare_runtime_offload_policies(
        surrogate_checkpoint=surrogate_checkpoint,
        rich_reduced_checkpoint=rich_checkpoint,
        cql_checkpoint=cql_checkpoint,
        radcc_checkpoint=checkpoint_path,
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
            "cooperative": variant.coop_weight,
            "queue": variant.queue_weight,
            "risk_beta": variant.risk_beta,
            "cvar_alpha": variant.cvar_alpha,
        },
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "runtime_report_path": str(report_path),
        "dataset": {
            "samples": dataset_metadata.get("collected_samples"),
            "mean_valid_action_costs": dataset_metadata.get("mean_valid_action_costs"),
        },
        "training": {
            "best_epoch": training_summary.get("best_epoch"),
            "best_eval": training_summary.get("best_eval"),
        },
        "runtime": {
            "radcc": {
                "dsr": metric_mean(runtime_report, "radcc_offloading", "deadline_satisfaction_rate"),
                "mbs_load_ratio": metric_mean(runtime_report, "radcc_offloading", "mbs_load_ratio"),
                "mbs_ratio": metric_mean(runtime_report, "radcc_offloading", "offloading_ratio_mbs"),
                "local_ratio": metric_mean(runtime_report, "radcc_offloading", "offloading_ratio_local"),
                "coop_ratio": metric_mean(runtime_report, "radcc_offloading", "offloading_ratio_cooperative"),
                "energy": metric_mean(runtime_report, "radcc_offloading", "energy"),
                "latency": metric_mean(runtime_report, "radcc_offloading", "latency"),
            },
            "delta_vs_heuristic": {
                "dsr": metric_delta(runtime_report, "radcc_offloading", "deadline_satisfaction_rate"),
                "mbs_load_ratio": metric_delta(runtime_report, "radcc_offloading", "mbs_load_ratio"),
                "mbs_ratio": metric_delta(runtime_report, "radcc_offloading", "offloading_ratio_mbs"),
                "local_ratio": metric_delta(runtime_report, "radcc_offloading", "offloading_ratio_local"),
                "coop_ratio": metric_delta(runtime_report, "radcc_offloading", "offloading_ratio_cooperative"),
                "energy": metric_delta(runtime_report, "radcc_offloading", "energy"),
                "latency": metric_delta(runtime_report, "radcc_offloading", "latency"),
            },
        },
    }
    (variant_root / "variant_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RADCC-Offload sensitivity experiments.")
    parser.add_argument("--output_root", type=str, default=str(results_path("full_offload_experiments", "radcc_sensitivity")))
    parser.add_argument("--surrogate_checkpoint", type=str, default="saved_offload_policies/offload_policy_surrogate_runtime.pt")
    parser.add_argument("--rich_checkpoint", type=str, default="saved_offload_policies/offload_policy_rich_runtime.pt")
    parser.add_argument("--cql_checkpoint", type=str, default=None)
    parser.add_argument("--variant", action="append", default=None, help="NAME,DEADLINE_WEIGHT,MBS_WEIGHT,COOP_WEIGHT,QUEUE_WEIGHT,RISK_BETA,CVAR_ALPHA")
    parser.add_argument("--per_class_target", type=int, default=2000)
    parser.add_argument("--max_attempts", type=int, default=60000)
    parser.add_argument("--procedural_train_per_class", type=int, default=1800)
    parser.add_argument("--dataset_seed", type=int, default=config.SEED)
    parser.add_argument("--train_seed", type=int, default=config.SEED)
    parser.add_argument("--num_quantiles", type=int, default=config.RADCC_OFFLOAD_NUM_QUANTILES)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dims", type=str, default="128,128")
    parser.add_argument("--device", type=str, default="auto")
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
            cql_checkpoint=args.cql_checkpoint,
            per_class_target=args.per_class_target,
            max_attempts=args.max_attempts,
            procedural_train_per_class=args.procedural_train_per_class,
            dataset_seed=args.dataset_seed,
            train_seed=args.train_seed,
            num_quantiles=args.num_quantiles,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dims=parse_hidden_dims(args.hidden_dims),
            device=args.device,
            runtime_seeds=[int(seed) for seed in args.runtime_seeds],
            episodes_per_seed=args.episodes_per_seed,
            steps_per_episode=args.steps_per_episode,
        )
        for variant in variants
    ]
    summary_path = output_root / "radcc_sensitivity_summary.json"
    summary_path.write_text(json.dumps({"variants": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "variants": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
