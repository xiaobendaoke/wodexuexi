from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from collect_offload_dataset import collect_offload_dataset
from compare_runtime_offload_policies import compare_runtime_offload_policies
from marl_models.offload_policy import OFFLOAD_FEATURE_FAMILY_FULL, OFFLOAD_FEATURE_FAMILY_RICH_REDUCED
from paths import results_path
from profile_offload_dataset import profile_offload_dataset
from train_offload_policy import parse_hidden_dims, train_offload_policy


def _default_experiment_name() -> str:
    return time.strftime("offload_full_%Y%m%d_%H%M%S")


def run_full_offload_experiment(
    *,
    experiment_name: str,
    per_class_target: int,
    procedural_train_per_class: int,
    max_attempts: int,
    dataset_seed: int,
    train_seed: int,
    runtime_seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_ratio: float,
    hidden_dims: tuple[int, ...],
    device: str,
    sampler_mode: str,
    episodes_per_seed: int,
    steps_per_episode: int,
) -> dict[str, object]:
    """Run the current repository's end-to-end offloading experiment pipeline.

    This script stays within the current project scope:
    1. collect a rich runtime candidate dataset
    2. profile the dataset
    3. train the full-feature surrogate baseline
    4. train the rich reduced runtime policy
    5. compare heuristic vs surrogate vs rich runtime policy
    """

    output_root = results_path("full_offload_experiments", experiment_name)
    dataset_dir = output_root / "datasets"
    checkpoint_dir = output_root / "checkpoints"
    report_dir = output_root / "reports"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / "offload_dataset_runtime_candidate.npz"
    dataset_profile_path = dataset_dir / "offload_dataset_runtime_candidate.profile.json"
    surrogate_checkpoint = checkpoint_dir / "offload_policy_surrogate_runtime.pt"
    rich_checkpoint = checkpoint_dir / "offload_policy_rich_runtime.pt"
    comparison_report_path = report_dir / "runtime_offload_policy_comparison.json"
    experiment_summary_path = output_root / "experiment_summary.json"

    print(f"[full-exp] collecting dataset -> {dataset_path}")
    dataset_metadata = collect_offload_dataset(
        output_path=dataset_path,
        mode="rich_candidate_mixed",
        per_class_target=per_class_target,
        max_attempts=max_attempts,
        procedural_train_per_class=procedural_train_per_class,
        seed=dataset_seed,
    )

    print(f"[full-exp] profiling dataset -> {dataset_profile_path}")
    dataset_profile = profile_offload_dataset(dataset_path)
    dataset_profile_path.write_text(json.dumps(dataset_profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[full-exp] training surrogate baseline -> {surrogate_checkpoint}")
    surrogate_summary = train_offload_policy(
        dataset_path=dataset_path,
        output_path=surrogate_checkpoint,
        feature_family=OFFLOAD_FEATURE_FAMILY_FULL,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        seed=train_seed,
        device=device,
        sampler_mode=sampler_mode,
    )

    print(f"[full-exp] training rich reduced runtime policy -> {rich_checkpoint}")
    rich_summary = train_offload_policy(
        dataset_path=dataset_path,
        output_path=rich_checkpoint,
        feature_family=OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        seed=train_seed,
        device=device,
        sampler_mode=sampler_mode,
    )

    print(f"[full-exp] comparing runtime policies -> {comparison_report_path}")
    comparison_report = compare_runtime_offload_policies(
        surrogate_checkpoint=surrogate_checkpoint,
        rich_reduced_checkpoint=rich_checkpoint,
        seeds=runtime_seeds,
        episodes_per_seed=episodes_per_seed,
        steps_per_episode=steps_per_episode,
        output_path=comparison_report_path,
    )

    summary: dict[str, object] = {
        "experiment_name": experiment_name,
        "dataset": {
            "path": str(dataset_path),
            "metadata": dataset_metadata,
            "profile_path": str(dataset_profile_path),
            "profile": dataset_profile,
        },
        "surrogate_training": surrogate_summary,
        "rich_reduced_training": rich_summary,
        "runtime_comparison_path": str(comparison_report_path),
        "runtime_comparison": comparison_report,
    }
    experiment_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repository's full request-level offloading experiment pipeline.")
    parser.add_argument("--name", type=str, default=_default_experiment_name(), help="Experiment output folder name.")
    parser.add_argument("--per_class_target", type=int, default=2000, help="Template-scenario samples per class.")
    parser.add_argument("--procedural_train_per_class", type=int, default=1800, help="Procedural rich samples per class.")
    parser.add_argument("--max_attempts", type=int, default=60000, help="Maximum attempts for template scenario sampling.")
    parser.add_argument("--dataset_seed", type=int, default=42, help="Seed for dataset collection.")
    parser.add_argument("--train_seed", type=int, default=42, help="Seed for classifier training.")
    parser.add_argument("--runtime_seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Seeds for runtime comparison.")
    parser.add_argument("--epochs", type=int, default=25, help="Classifier training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Classifier training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Classifier learning rate.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--hidden_dims", type=str, default="64,64", help="Comma-separated classifier hidden dims.")
    parser.add_argument("--device", type=str, default="auto", help="Training device: auto, cpu, cuda, or mps.")
    parser.add_argument("--sampler_mode", type=str, default="auto", choices=["auto", "weighted", "none"], help="Sampler mode for classifier training.")
    parser.add_argument("--episodes_per_seed", type=int, default=4, help="Episodes per seed in runtime comparison.")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="Steps per episode in runtime comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_full_offload_experiment(
        experiment_name=args.name,
        per_class_target=args.per_class_target,
        procedural_train_per_class=args.procedural_train_per_class,
        max_attempts=args.max_attempts,
        dataset_seed=args.dataset_seed,
        train_seed=args.train_seed,
        runtime_seeds=[int(seed) for seed in args.runtime_seeds],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        device=args.device,
        sampler_mode=args.sampler_mode,
        episodes_per_seed=args.episodes_per_seed,
        steps_per_episode=args.steps_per_episode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
