from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl_models.offload_policy import (
    OFFLOAD_FEATURE_FAMILY_FULL,
    OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
    OFFLOAD_NUM_CLASSES,
    OFFLOAD_TARGET_NAMES,
    load_offload_policy_checkpoint,
)
from train_offload_policy import stratified_train_val_split


def select_features(dataset: np.lib.npyio.NpzFile, feature_family: str) -> np.ndarray:
    if feature_family == OFFLOAD_FEATURE_FAMILY_FULL:
        key = "features_full" if "features_full" in dataset else "features"
        return np.asarray(dataset[key], dtype=np.float32)
    if feature_family == OFFLOAD_FEATURE_FAMILY_RICH_REDUCED:
        if "features_rich_reduced" not in dataset:
            raise KeyError("Dataset does not contain features_rich_reduced.")
        return np.asarray(dataset["features_rich_reduced"], dtype=np.float32)
    raise ValueError(f"Unsupported feature family: {feature_family}")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((OFFLOAD_NUM_CLASSES, OFFLOAD_NUM_CLASSES), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred, strict=False):
        matrix[int(truth), int(pred)] += 1
    return matrix


def macro_f1_from_confusion(matrix: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    per_class: list[dict[str, float]] = []
    for class_idx, class_name in enumerate(OFFLOAD_TARGET_NAMES):
        tp = float(matrix[class_idx, class_idx])
        fp = float(np.sum(matrix[:, class_idx]) - tp)
        fn = float(np.sum(matrix[class_idx, :]) - tp)
        support = float(np.sum(matrix[class_idx, :]))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append(
            {
                "class": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )
    return float(np.mean([entry["f1"] for entry in per_class])), per_class


def calibration_bins(confidence: np.ndarray, correct: np.ndarray, num_bins: int) -> tuple[list[dict[str, float]], float]:
    bins: list[dict[str, float]] = []
    ece = 0.0
    total = max(int(confidence.size), 1)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    for idx in range(num_bins):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        if idx == num_bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        count = int(np.sum(mask))
        if count == 0:
            bins.append({"bin_low": low, "bin_high": high, "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        bin_conf = float(np.mean(confidence[mask]))
        bin_acc = float(np.mean(correct[mask]))
        ece += (count / total) * abs(bin_acc - bin_conf)
        bins.append({"bin_low": low, "bin_high": high, "count": count, "confidence": bin_conf, "accuracy": bin_acc})
    return bins, float(ece)


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(OFFLOAD_NUM_CLASSES, dtype=np.float64)[labels.astype(np.int64)]
    return float(np.mean(np.sum((probabilities.astype(np.float64) - one_hot) ** 2, axis=1)))


def analyze_classifier_quality(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_json: str | Path | None = None,
    split: str = "validation",
    val_ratio: float = 0.2,
    seed: int | None = None,
    num_bins: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    checkpoint_file = Path(checkpoint_path)
    model, metadata = load_offload_policy_checkpoint(checkpoint_file, device=device)
    feature_family = str(metadata.get("feature_family", OFFLOAD_FEATURE_FAMILY_FULL))
    metrics = metadata.get("metrics", {})
    resolved_seed = int(seed if seed is not None else metrics.get("seed", 42))

    dataset = np.load(dataset_file, allow_pickle=True)
    features = select_features(dataset, feature_family)
    labels = np.asarray(dataset["labels"], dtype=np.int64)

    if split == "validation":
        _, indices = stratified_train_val_split(labels, val_ratio=val_ratio, seed=resolved_seed)
    elif split == "all":
        indices = np.arange(labels.size, dtype=np.int64)
    else:
        raise ValueError("split must be 'validation' or 'all'.")

    selected_features = features[indices].astype(np.float32)
    selected_labels = labels[indices].astype(np.int64)
    scaler_mean = metadata.get("scaler_mean")
    scaler_std = metadata.get("scaler_std")
    if scaler_mean is not None and scaler_std is not None:
        mean = np.asarray(scaler_mean, dtype=np.float32)
        std = np.asarray(scaler_std, dtype=np.float32)
        selected_features = ((selected_features - mean) / np.where(std < 1e-6, 1.0, std)).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(selected_features).to(torch.device(device)))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    matrix = confusion_matrix(selected_labels, predictions)
    macro_f1, per_class = macro_f1_from_confusion(matrix)
    confidence = np.max(probabilities, axis=1)
    correct = (predictions == selected_labels).astype(np.float64)
    bins, ece = calibration_bins(confidence, correct, num_bins=num_bins)
    brier = multiclass_brier_score(probabilities, selected_labels)

    summary: dict[str, Any] = {
        "dataset_path": str(dataset_file),
        "checkpoint_path": str(checkpoint_file),
        "feature_family": feature_family,
        "split": split,
        "seed": resolved_seed,
        "val_ratio": float(val_ratio),
        "num_samples": int(selected_labels.size),
        "accuracy": float(np.mean(correct)),
        "macro_f1": macro_f1,
        "ece": ece,
        "brier": brier,
        "confusion_matrix": matrix.tolist(),
        "per_class_metrics": per_class,
        "calibration_bins": bins,
        "class_names": list(OFFLOAD_TARGET_NAMES),
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze offload classifier confusion and calibration quality.")
    parser.add_argument("--dataset", required=True, help="Path to offload dataset .npz.")
    parser.add_argument("--checkpoint", required=True, help="Path to offload classifier checkpoint.")
    parser.add_argument("--output_json", default=None, help="Optional output JSON path.")
    parser.add_argument("--split", choices=["validation", "all"], default="validation")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_classifier_quality(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_json=args.output_json,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_bins=args.num_bins,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
