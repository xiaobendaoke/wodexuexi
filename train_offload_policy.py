from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import config
from marl_models.offload_policy import (
    OFFLOAD_FEATURE_FAMILY_FULL,
    OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
    OFFLOAD_NUM_CLASSES,
    OFFLOAD_TARGET_NAMES,
    OffloadMLP,
    get_offload_feature_dim,
    get_offload_feature_specs,
    save_offload_policy_checkpoint,
)


def parse_hidden_dims(hidden_dims_arg: str) -> tuple[int, ...]:
    dims = tuple(int(token.strip()) for token in hidden_dims_arg.split(",") if token.strip())
    if not dims:
        raise ValueError("At least one hidden layer size must be provided.")
    return dims


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_feature_family_matrix(dataset: dict[str, np.ndarray], feature_family: str) -> np.ndarray:
    """Load the requested feature family while preserving backward compatibility with old datasets."""

    if feature_family == OFFLOAD_FEATURE_FAMILY_FULL:
        if "features_full" in dataset:
            return np.asarray(dataset["features_full"], dtype=np.float32)
        return np.asarray(dataset["features"], dtype=np.float32)
    if feature_family == OFFLOAD_FEATURE_FAMILY_RICH_REDUCED:
        if "features_rich_reduced" not in dataset:
            raise KeyError(
                "Dataset does not contain 'features_rich_reduced'. Re-collect it with the updated dataset pipeline."
            )
        return np.asarray(dataset["features_rich_reduced"], dtype=np.float32)
    raise ValueError(f"Unsupported feature family: {feature_family}")


def standardize_feature_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using the training split statistics only."""

    mean = np.mean(x_train, axis=0, keepdims=True).astype(np.float32)
    std = np.std(x_train, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_std = ((x_train - mean) / std).astype(np.float32)
    x_val_std = ((x_val - mean) / std).astype(np.float32)
    return x_train_std, x_val_std, mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


def stratified_train_val_split(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    for class_idx in range(OFFLOAD_NUM_CLASSES):
        class_indices = np.where(labels == class_idx)[0]
        if class_indices.size == 0:
            continue
        shuffled = rng.permutation(class_indices)
        if class_indices.size == 1:
            train_indices.append(shuffled)
            continue

        val_count: int = int(round(class_indices.size * val_ratio))
        val_count = max(1, min(val_count, class_indices.size - 1))
        val_indices.append(shuffled[:val_count])
        train_indices.append(shuffled[val_count:])

    train_idx = np.concatenate(train_indices) if train_indices else np.zeros((0,), dtype=np.int64)
    val_idx = np.concatenate(val_indices) if val_indices else np.zeros((0,), dtype=np.int64)
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    if val_idx.size == 0:
        raise ValueError("Validation split is empty. Collect more data or increase val_ratio.")
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((OFFLOAD_NUM_CLASSES, OFFLOAD_NUM_CLASSES), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred, strict=False):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def compute_per_class_metrics(confusion_matrix: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for class_idx, class_name in enumerate(OFFLOAD_TARGET_NAMES):
        true_positive = int(confusion_matrix[class_idx, class_idx])
        false_positive = int(np.sum(confusion_matrix[:, class_idx]) - true_positive)
        false_negative = int(np.sum(confusion_matrix[class_idx, :]) - true_positive)
        support = int(np.sum(confusion_matrix[class_idx, :]))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return metrics


def compute_macro_precision_recall_f1(per_class_metrics: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    precisions = [metrics["precision"] for metrics in per_class_metrics.values()]
    recalls = [metrics["recall"] for metrics in per_class_metrics.values()]
    f1_scores = [metrics["f1"] for metrics in per_class_metrics.values()]
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores))


def class_distribution(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES)
    total = max(int(labels.size), 1)
    return {OFFLOAD_TARGET_NAMES[idx]: float(counts[idx] / total) for idx in range(OFFLOAD_NUM_CLASSES)}


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES)
    return {OFFLOAD_TARGET_NAMES[idx]: int(counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)}


def should_use_weighted_sampler(train_labels: np.ndarray, sampler_mode: str) -> bool:
    if sampler_mode == "weighted":
        return True
    if sampler_mode == "none":
        return False
    counts = np.bincount(train_labels, minlength=OFFLOAD_NUM_CLASSES).astype(np.float32)
    nonzero_counts = counts[counts > 0]
    if nonzero_counts.size <= 1:
        return False
    imbalance_ratio = float(np.max(nonzero_counts) / np.min(nonzero_counts))
    return imbalance_ratio > 1.5


def build_train_loader(x_train: np.ndarray, y_train: np.ndarray, batch_size: int, sampler_mode: str) -> tuple[DataLoader, str]:
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    if should_use_weighted_sampler(y_train, sampler_mode):
        label_counts = np.bincount(y_train, minlength=OFFLOAD_NUM_CLASSES).astype(np.float32)
        sample_weights = np.array([1.0 / max(label_counts[label], 1.0) for label in y_train], dtype=np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler), "weighted"
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), "shuffle"


def evaluate_classifier(model: OffloadMLP, features: np.ndarray, labels: np.ndarray, device: torch.device) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = compute_accuracy(labels, predictions)
    confusion_matrix = compute_confusion_matrix(labels, predictions)
    per_class_metrics = compute_per_class_metrics(confusion_matrix)
    macro_precision, macro_recall, macro_f1 = compute_macro_precision_recall_f1(per_class_metrics)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "predictions": predictions,
    }


def train_offload_policy(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL,
    hidden_dims: tuple[int, ...] = (64, 64),
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = config.SEED,
    device: str = "auto",
    sampler_mode: str = "auto",
) -> dict[str, object]:
    """Train a simple request-level classifier from heuristic imitation data."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    dataset = np.load(dataset_file)
    features = load_feature_family_matrix(dataset, feature_family)
    labels = np.asarray(dataset["labels"], dtype=np.int64)
    expected_dim: int = get_offload_feature_dim(feature_family)
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected feature shape {features.shape}; expected (*, {expected_dim}) for feature family '{feature_family}'."
        )
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("Labels must be a 1D array aligned with the feature matrix.")
    missing_classes = [OFFLOAD_TARGET_NAMES[idx] for idx in range(OFFLOAD_NUM_CLASSES) if np.sum(labels == idx) == 0]

    train_idx, val_idx = stratified_train_val_split(labels, val_ratio=val_ratio, seed=seed)
    x_train = features[train_idx]
    y_train = labels[train_idx]
    x_val = features[val_idx]
    y_val = labels[val_idx]
    x_train, x_val, scaler_mean, scaler_std = standardize_feature_splits(x_train, x_val)

    train_loader, effective_sampler_mode = build_train_loader(x_train, y_train, batch_size, sampler_mode)

    device_obj = select_device(device)
    model = OffloadMLP(input_dim=expected_dim, hidden_dims=hidden_dims, num_classes=OFFLOAD_NUM_CLASSES).to(device_obj)

    train_label_counts = np.bincount(y_train, minlength=OFFLOAD_NUM_CLASSES).astype(np.float32)
    class_weights = np.zeros_like(train_label_counts)
    nonzero_mask = train_label_counts > 0
    if np.any(nonzero_mask):
        class_weights[nonzero_mask] = float(y_train.size) / (float(np.sum(nonzero_mask)) * train_label_counts[nonzero_mask])
    else:
        class_weights[:] = 1.0

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device_obj))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, object] | None = None
    best_epoch: int = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss: float = 0.0
        total_batches: int = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device_obj)
            batch_labels = batch_labels.to(device_obj)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            total_batches += 1

        evaluation = evaluate_classifier(model, x_val, y_val, device_obj)
        if best_eval is None or float(evaluation["macro_f1"]) > float(best_eval["macro_f1"]):
            best_state_dict = copy.deepcopy(model.state_dict())
            best_eval = evaluation
            best_epoch = epoch

        mean_epoch_loss = epoch_loss / max(total_batches, 1)
        print(
            f"epoch={epoch:03d} loss={mean_epoch_loss:.6f} "
            f"val_acc={float(evaluation['accuracy']):.4f} val_macro_f1={float(evaluation['macro_f1']):.4f}"
        )

    if best_state_dict is None or best_eval is None:
        raise RuntimeError("Classifier training did not produce a valid checkpoint.")

    model.load_state_dict(best_state_dict)

    summary: dict[str, object] = {
        "dataset_path": str(dataset_file),
        "checkpoint_path": str(output_path),
        "device": str(device_obj),
        "feature_family": feature_family,
        "feature_dim": int(expected_dim),
        "feature_names": [spec["name"] for spec in get_offload_feature_specs(feature_family)],
        "seed": seed,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "hidden_dims": list(hidden_dims),
        "best_epoch": int(best_epoch),
        "val_accuracy": float(best_eval["accuracy"]),
        "val_macro_precision": float(best_eval["macro_precision"]),
        "val_macro_recall": float(best_eval["macro_recall"]),
        "val_macro_f1": float(best_eval["macro_f1"]),
        "val_per_class_metrics": best_eval["per_class_metrics"],
        "val_confusion_matrix": best_eval["confusion_matrix"],
        "train_samples": int(y_train.size),
        "val_samples": int(y_val.size),
        "train_class_counts": class_counts(y_train),
        "val_class_counts": class_counts(y_val),
        "train_class_distribution": class_distribution(y_train),
        "val_class_distribution": class_distribution(y_val),
        "overall_class_distribution": class_distribution(labels),
        "overall_class_counts": class_counts(labels),
        "missing_classes": missing_classes,
        "sampler_mode_requested": sampler_mode,
        "sampler_mode_used": effective_sampler_mode,
        "loss_class_weights": {OFFLOAD_TARGET_NAMES[idx]: float(class_weights[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "todo": "This remains an imitation-learning baseline; formal thesis experiments still need multi-seed evaluation and scenario generalization tests.",
    }
    if missing_classes:
        summary["warning"] = (
            "At least one offloading class is absent from the dataset. "
            "This checkpoint is still usable for runtime integration and fallback tests, but it is not a balanced experimental model."
        )

    saved_path = save_offload_policy_checkpoint(
        model,
        output_path,
        feature_family=feature_family,
        hidden_dims=hidden_dims,
        scaler_mean=scaler_mean.tolist(),
        scaler_std=scaler_std.tolist(),
        metrics=summary,
        extra_metadata={"dataset_seed": seed, "dataset_rows": int(labels.size), "feature_family": feature_family},
    )
    summary["checkpoint_path"] = saved_path

    summary_path = Path(saved_path).with_suffix(".metrics.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a request-level offloading classifier from heuristic labels.")
    parser.add_argument("--dataset", type=str, default="offload_datasets/offload_dataset_minimal.npz", help="Path to the collected dataset (.npz).")
    parser.add_argument("--output", type=str, default="saved_offload_policies/offload_policy_minimal.pt", help="Checkpoint output path.")
    parser.add_argument(
        "--feature_family",
        type=str,
        default=OFFLOAD_FEATURE_FAMILY_FULL,
        choices=[OFFLOAD_FEATURE_FAMILY_FULL, OFFLOAD_FEATURE_FAMILY_RICH_REDUCED],
        help="Feature family to train on.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--hidden_dims", type=str, default="64,64", help="Comma-separated hidden layer sizes, e.g. '64,64'.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps.")
    parser.add_argument("--sampler_mode", type=str, default="auto", choices=["auto", "weighted", "none"], help="Training sampler mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_offload_policy(
        dataset_path=args.dataset,
        output_path=args.output,
        feature_family=args.feature_family,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        sampler_mode=args.sampler_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
