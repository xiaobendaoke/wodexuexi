from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
import verify_offload_policy_validity as advanced_offload_validity
from environment.env import Env
from evaluate_offload_validity import run_validity_evaluation
from marl_models.base_model import MARLModel
from marl_models.utils import get_model
from paths import results_path
from run_full_offload_experiment import run_full_offload_experiment
from test import test_model
from train import train_baselines, train_off_policy, train_on_policy
from utils.comparative_plots import compare_algorithms
from utils.logger import Logger
from utils.plot_logs import generate_plots


ALL_MODELS: tuple[str, ...] = (
    "maddpg",
    "matd3",
    "mappo",
    "masac",
    "attention_maddpg",
    "attention_matd3",
    "attention_mappo",
    "attention_masac",
    "random",
    "static",
    "nearest_greedy",
    "uncoordinated_greedy",
)


def configure_fp32_precision() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Please use the new API settings to control TF32 behavior.*",
        category=UserWarning,
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def configure_compile_backend() -> None:
    if os.name == "nt":
        def _no_compile(module, *args, **kwargs):
            return module

        torch.compile = _no_compile  # type: ignore[attr-defined]


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


def is_off_policy(model_name: str) -> bool:
    return model_name in {"maddpg", "attention_maddpg", "matd3", "attention_matd3", "masac", "attention_masac"}


def is_on_policy(model_name: str) -> bool:
    return model_name in {"mappo", "attention_mappo"}


def copy_saved_model_run(model_name: str, timestamp: str, run_root: Path) -> Path | None:
    source_dir = Path("saved_models") / f"{model_name}_{timestamp}"
    if not source_dir.exists():
        return None

    destination_dir = run_root / "saved_models" / source_dir.name
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    return destination_dir


def run_single_rl_experiment(
    *,
    model_name: str,
    train_episodes: int,
    test_episodes: int,
    run_name: str,
) -> dict[str, object]:
    run_root = results_path("full_runs", run_name)
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{model_name}")

    config.MODEL = model_name
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    env = Env()
    model: MARLModel = get_model(model_name)

    train_log_dir = run_root / "train_logs" / model_name
    train_logger = Logger(str(train_log_dir), timestamp)
    train_logger.log_configs()

    train_start = time.time()
    if is_off_policy(model_name):
        train_score = train_off_policy(env, model, train_logger, train_episodes, total_step_count=0)
    elif is_on_policy(model_name):
        train_score = train_on_policy(env, model, train_logger, train_episodes)
    else:
        train_score = train_baselines(env, model, train_logger, train_episodes)
    train_seconds = float(time.time() - train_start)

    train_json_path = train_log_dir / f"log_data_{timestamp}.json"
    train_plot_dir = run_root / "train_plots" / model_name
    generate_plots(str(train_json_path), str(train_plot_dir), "train", timestamp)

    copied_model_dir = copy_saved_model_run(model_name, timestamp, run_root)
    final_model_dir = copied_model_dir / "final" if copied_model_dir is not None else None

    test_summary: dict[str, object] | None = None
    if test_episodes > 0:
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        test_env = Env()
        test_model_instance: MARLModel = get_model(model_name)
        if final_model_dir is not None and final_model_dir.exists():
            test_model_instance.load(str(final_model_dir))
        else:
            test_model_instance = model

        test_log_dir = run_root / "test_logs" / model_name
        test_logger = Logger(str(test_log_dir), timestamp)
        test_start = time.time()
        test_model(test_env, test_model_instance, test_logger, test_episodes)
        test_seconds = float(time.time() - test_start)

        test_json_path = test_log_dir / f"log_data_{timestamp}.json"
        test_plot_dir = run_root / "test_plots" / model_name
        generate_plots(str(test_json_path), str(test_plot_dir), "test", timestamp, smoothing_window=2)

        test_summary = {
            "episodes": int(test_episodes),
            "seconds": test_seconds,
            "log_dir": str(test_log_dir),
            "plot_dir": str(test_plot_dir),
            "log_json_path": str(test_json_path),
        }

    return {
        "model": model_name,
        "timestamp": timestamp,
        "train": {
            "episodes": int(train_episodes),
            "score": float(train_score),
            "seconds": train_seconds,
            "log_dir": str(train_log_dir),
            "plot_dir": str(train_plot_dir),
            "log_json_path": str(train_json_path),
        },
        "saved_model_dir": str(copied_model_dir) if copied_model_dir is not None else None,
        "saved_model_final_dir": str(final_model_dir) if final_model_dir is not None else None,
        "test": test_summary,
    }


def run_rl_suite(
    *,
    models: list[str],
    train_episodes: int,
    test_episodes: int,
    run_name: str,
    smoothing_window: int,
) -> dict[str, object]:
    run_root = results_path("full_runs", run_name)
    rl_results: list[dict[str, object]] = []

    for model_name in models:
        print(f"[rl] training/testing model={model_name}")
        rl_results.append(
            run_single_rl_experiment(
                model_name=model_name,
                train_episodes=train_episodes,
                test_episodes=test_episodes,
                run_name=run_name,
            )
        )

    train_log_dirs = [entry["train"]["log_dir"] for entry in rl_results]
    comparison_dir = run_root / "comparisons" / "training"
    compare_algorithms(train_log_dirs, models, str(comparison_dir), smoothing_window=smoothing_window)

    return {
        "models": models,
        "train_episodes": int(train_episodes),
        "test_episodes": int(test_episodes),
        "comparison_dir": str(comparison_dir),
        "runs": rl_results,
    }


def run_offload_suite(
    *,
    run_name: str,
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
    validity_dataset: str | None,
    validity_epochs: int,
    run_advanced_validity: bool,
) -> dict[str, object]:
    print("[offload] running full pipeline")
    full_pipeline_summary = run_full_offload_experiment(
        experiment_name=f"{run_name}_offload",
        per_class_target=per_class_target,
        procedural_train_per_class=procedural_train_per_class,
        max_attempts=max_attempts,
        dataset_seed=dataset_seed,
        train_seed=train_seed,
        runtime_seeds=runtime_seeds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        hidden_dims=hidden_dims,
        device=device,
        sampler_mode=sampler_mode,
        episodes_per_seed=episodes_per_seed,
        steps_per_episode=steps_per_episode,
    )

    validity_report_path: str | None = None
    validity_report: dict[str, object] | None = None
    if validity_dataset is not None:
        print(f"[offload] running paper-validity evaluation on {validity_dataset}")
        validity_report_path = str(results_path("full_runs", run_name, "reports", "offload_policy_validity_report.json"))
        validity_report = run_validity_evaluation(
            dataset_path=validity_dataset,
            output_path=validity_report_path,
            seed=train_seed,
            epochs=validity_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            sampler_mode=sampler_mode,
        )

    advanced_validity_outputs: dict[str, str] | None = None
    if run_advanced_validity:
        print("[offload] running advanced repository validator")
        advanced_offload_validity.main()
        advanced_validity_outputs = {
            "json_report": str(results_path("reports", "offload_policy_paper_validity.json")),
            "markdown_summary": str(results_path("reports", "offload_policy_paper_validity_summary.md")),
        }

    return {
        "full_pipeline": full_pipeline_summary,
        "validity_dataset": validity_dataset,
        "validity_report_path": validity_report_path,
        "validity_report": validity_report,
        "advanced_validity_outputs": advanced_validity_outputs,
    }


def parse_hidden_dims(hidden_dims_arg: str) -> tuple[int, ...]:
    dims = tuple(int(token.strip()) for token in hidden_dims_arg.split(",") if token.strip())
    if not dims:
        raise ValueError("At least one hidden dimension must be provided.")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repository's full RL and offloading experiment suite.")
    parser.add_argument("--name", type=str, default=time.strftime("all_exp_%Y%m%d_%H%M%S"), help="Experiment run name.")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), help="RL models to include in the RL suite.")
    parser.add_argument("--train_episodes", type=int, default=100, help="Training episodes per RL model.")
    parser.add_argument("--test_episodes", type=int, default=20, help="Testing episodes per RL model.")
    parser.add_argument("--comparison_smoothing", type=int, default=5, help="Smoothing window for RL comparison plots.")
    parser.add_argument("--skip_rl", action="store_true", help="Skip the RL algorithm suite.")
    parser.add_argument("--skip_offload", action="store_true", help="Skip the offloading suite.")
    parser.add_argument("--skip_advanced_validity", action="store_true", help="Skip verify_offload_policy_validity.py.")
    parser.add_argument("--per_class_target", type=int, default=2000, help="Template samples per class for offload dataset collection.")
    parser.add_argument("--procedural_train_per_class", type=int, default=1800, help="Procedural rich samples per class.")
    parser.add_argument("--max_attempts", type=int, default=60000, help="Maximum attempts during offload dataset collection.")
    parser.add_argument("--dataset_seed", type=int, default=42, help="Dataset collection seed.")
    parser.add_argument("--train_seed", type=int, default=42, help="Training seed for offload classifier stages.")
    parser.add_argument("--runtime_seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Runtime comparison seeds.")
    parser.add_argument("--offload_epochs", type=int, default=25, help="Training epochs for offload classifiers.")
    parser.add_argument("--validity_epochs", type=int, default=20, help="Epochs for evaluate_offload_validity.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for offload classifiers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for offload classifiers.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio for offload classifier training.")
    parser.add_argument("--hidden_dims", type=str, default="64,64", help="Comma-separated hidden dims for offload classifiers.")
    parser.add_argument("--device", type=str, default="auto", help="Offload classifier device: auto, cpu, cuda, or mps.")
    parser.add_argument("--sampler_mode", type=str, default="auto", choices=["auto", "weighted", "none"], help="Offload classifier sampler mode.")
    parser.add_argument("--episodes_per_seed", type=int, default=4, help="Episodes per seed for runtime offload comparison.")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="Steps per episode for runtime offload comparison.")
    parser.add_argument(
        "--validity_dataset",
        type=str,
        default="offload_datasets/offload_dataset_balanced.npz",
        help="Dataset for evaluate_offload_validity. Set empty string to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fp32_precision()
    configure_compile_backend()

    base_snapshot = snapshot_config()
    run_root = results_path("full_runs", args.name)
    run_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "run_name": args.name,
        "run_root": str(run_root),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "rl_suite": None,
        "offload_suite": None,
    }

    try:
        if not args.skip_rl:
            restore_config(base_snapshot)
            summary["rl_suite"] = run_rl_suite(
                models=args.models,
                train_episodes=args.train_episodes,
                test_episodes=args.test_episodes,
                run_name=args.name,
                smoothing_window=args.comparison_smoothing,
            )

        if not args.skip_offload:
            restore_config(base_snapshot)
            validity_dataset = args.validity_dataset if args.validity_dataset.strip() else None
            summary["offload_suite"] = run_offload_suite(
                run_name=args.name,
                per_class_target=args.per_class_target,
                procedural_train_per_class=args.procedural_train_per_class,
                max_attempts=args.max_attempts,
                dataset_seed=args.dataset_seed,
                train_seed=args.train_seed,
                runtime_seeds=[int(seed) for seed in args.runtime_seeds],
                epochs=args.offload_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                val_ratio=args.val_ratio,
                hidden_dims=parse_hidden_dims(args.hidden_dims),
                device=args.device,
                sampler_mode=args.sampler_mode,
                episodes_per_seed=args.episodes_per_seed,
                steps_per_episode=args.steps_per_episode,
                validity_dataset=validity_dataset,
                validity_epochs=args.validity_epochs,
                run_advanced_validity=not args.skip_advanced_validity,
            )
    finally:
        restore_config(base_snapshot)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path = run_root / "experiment_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
