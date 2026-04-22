from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marl_models.offload_policy import OFFLOAD_NUM_CLASSES, OFFLOAD_TARGET_MBS, OFFLOAD_TARGET_NAMES, get_offload_feature_specs


def _feature_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _collect_feature_summary(features: np.ndarray, labels: np.ndarray) -> tuple[dict[str, dict[str, object]], list[str]]:
    feature_specs = get_offload_feature_specs()
    feature_summary: dict[str, dict[str, object]] = {}
    low_signal_features: list[str] = []

    for feature_idx, spec in enumerate(feature_specs):
        feature_name = spec["name"]
        values = features[:, feature_idx]
        unique_values, unique_counts = np.unique(np.round(values, decimals=6), return_counts=True)
        dominant_ratio: float = float(np.max(unique_counts) / max(values.size, 1))
        global_stats = _feature_stats(values)
        per_class: dict[str, dict[str, float]] = {}
        for class_idx in range(OFFLOAD_NUM_CLASSES):
            class_values = values[labels == class_idx]
            per_class[OFFLOAD_TARGET_NAMES[class_idx]] = _feature_stats(class_values)

        feature_summary[feature_name] = {
            "description": spec["description"],
            "normalization": spec["normalization"],
            "global": global_stats,
            "per_class": per_class,
            "unique_value_count_rounded_6dp": int(unique_values.size),
            "dominant_value_ratio_rounded_6dp": dominant_ratio,
        }

        value_range: float = float(global_stats["max"] - global_stats["min"])
        if global_stats["std"] < 1e-4 or value_range < 1e-4 or dominant_ratio > 0.98:
            low_signal_features.append(feature_name)

    return feature_summary, low_signal_features


def _diagnose_dataset(features: np.ndarray, labels: np.ndarray) -> list[str]:
    diagnostics: list[str] = []
    total_samples: int = int(labels.size)
    if total_samples == 0:
        diagnostics.append("Dataset is empty.")
        return diagnostics

    class_counts = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES)
    class_ratios = class_counts / float(total_samples)
    missing_classes = [OFFLOAD_TARGET_NAMES[idx] for idx in range(OFFLOAD_NUM_CLASSES) if class_counts[idx] == 0]
    if missing_classes:
        diagnostics.append(f"Label collapse detected: missing classes = {missing_classes}.")
    elif np.max(class_ratios) > 0.8:
        dominant_class = OFFLOAD_TARGET_NAMES[int(np.argmax(class_counts))]
        diagnostics.append(
            f"Strong class imbalance detected: dominant class '{dominant_class}' accounts for {float(np.max(class_ratios)):.3f} of samples."
        )

    latency_features = features[:, :3]
    latency_argmin = np.argmin(latency_features, axis=1)
    latency_argmin_counts = np.bincount(latency_argmin, minlength=OFFLOAD_NUM_CLASSES)
    dominant_latency_class = int(np.argmax(latency_argmin_counts))
    if latency_argmin_counts[dominant_latency_class] / max(total_samples, 1) > 0.9:
        diagnostics.append(
            f"The normalized latency proxy for '{OFFLOAD_TARGET_NAMES[dominant_latency_class]}' is the smallest in "
            f"{float(latency_argmin_counts[dominant_latency_class] / total_samples):.3f} of samples."
        )

    dominant_latency_ratio = float(latency_argmin_counts[dominant_latency_class] / total_samples)
    if dominant_latency_class == OFFLOAD_TARGET_MBS and dominant_latency_ratio > 0.6:
        diagnostics.append(
            "MBS latency proxy is consistently smaller than local/cooperative latency proxies, "
            "which strongly pushes the heuristic toward MBS."
        )

    cooperative_feature = features[:, 6]
    if float(np.std(cooperative_feature)) < 1e-6:
        diagnostics.append("Feature 'cooperative_available' is effectively constant in this dataset.")

    queue_feature = features[:, 7]
    if float(np.mean(queue_feature)) < 0.15:
        diagnostics.append(
            "Local queue load stays low in most samples, so queue-relief driven cooperative decisions are rarely activated."
        )

    return diagnostics


def profile_offload_dataset(dataset_path: str | Path) -> dict[str, object]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    dataset = np.load(dataset_file)
    features = np.asarray(dataset["features"], dtype=np.float32)
    labels = np.asarray(dataset["labels"], dtype=np.int64)
    scenario_ids = np.asarray(dataset["scenario_ids"], dtype=np.int64) if "scenario_ids" in dataset.files else None
    scenario_names = dataset["scenario_names"].tolist() if "scenario_names" in dataset.files else []
    scenario_names = [str(name) for name in scenario_names]

    total_samples = int(labels.size)
    class_counts = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES)
    class_ratios = class_counts / max(float(total_samples), 1.0)
    feature_summary, low_signal_features = _collect_feature_summary(features, labels)
    diagnostics = _diagnose_dataset(features, labels)

    latency_argmin = np.argmin(features[:, :3], axis=1) if total_samples > 0 else np.zeros((0,), dtype=np.int64)
    latency_argmin_counts = np.bincount(latency_argmin, minlength=OFFLOAD_NUM_CLASSES)

    scenario_summary: dict[str, object] = {}
    if scenario_ids is not None and scenario_names:
        scenario_counts = np.bincount(scenario_ids, minlength=len(scenario_names))
        scenario_summary = {
            scenario_names[idx]: int(scenario_counts[idx]) for idx in range(len(scenario_names)) if scenario_counts[idx] > 0
        }

    profile: dict[str, object] = {
        "dataset_path": str(dataset_file),
        "num_samples": total_samples,
        "class_counts": {OFFLOAD_TARGET_NAMES[idx]: int(class_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "class_ratios": {OFFLOAD_TARGET_NAMES[idx]: float(class_ratios[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "latency_argmin_counts": {OFFLOAD_TARGET_NAMES[idx]: int(latency_argmin_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "feature_statistics": feature_summary,
        "low_signal_features": low_signal_features,
        "diagnostics": diagnostics,
    }
    if scenario_summary:
        profile["scenario_counts"] = scenario_summary
    return profile


def add_reference_comparison(profile: dict[str, object], reference_profile: dict[str, object]) -> dict[str, object]:
    comparison = {
        "reference_dataset_path": reference_profile["dataset_path"],
        "target_dataset_path": profile["dataset_path"],
        "class_count_delta": {},
        "class_ratio_delta": {},
    }
    for class_name in OFFLOAD_TARGET_NAMES:
        comparison["class_count_delta"][class_name] = int(profile["class_counts"][class_name]) - int(reference_profile["class_counts"][class_name])
        comparison["class_ratio_delta"][class_name] = float(profile["class_ratios"][class_name]) - float(reference_profile["class_ratios"][class_name])
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile an offload dataset and save a short JSON report.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset .npz file.")
    parser.add_argument("--output", type=str, default=None, help="Optional output profile path. Defaults next to the dataset.")
    parser.add_argument("--reference", type=str, default=None, help="Optional reference dataset for comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = profile_offload_dataset(args.dataset)
    if args.reference is not None:
        reference_profile = profile_offload_dataset(args.reference)
        profile["comparison_to_reference"] = add_reference_comparison(profile, reference_profile)

    output_path = Path(args.output) if args.output is not None else Path(args.dataset).with_suffix(".profile.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
