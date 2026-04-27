"""
中文注释说明：train_offload_policy.py

文件作用：
    训练任务卸载策略分类器，读取 collect_offload_dataset.py 生成的离线 .npz 数据集，
    按指定特征族构造 MLP 分类器，并把标准化参数、特征定义、类别映射和模型权重一起保存成
    可由 LearnedClassifierOffloadPolicy 在运行时加载的检查点文件。

整体流程：
    1. 加载离线数据集，并根据 feature_family 选择 full_features 或 rich_reduced_features。
    2. 按类别分层切分训练集和验证集，减少类别不均衡对验证结果的影响。
    3. 仅使用训练集统计量对训练/验证特征做标准化，避免验证集信息泄漏。
    4. 根据类别分布决定是否启用 WeightedRandomSampler，以缓解卸载标签类别不平衡。
    5. 训练 OffloadMLP，记录最佳验证集准确率对应的模型参数。
    6. 保存检查点和 summary JSON，供运行时策略、有效性验证和论文图表复用。

关键变量与对象：
    - dataset_path: 输入的 .npz 数据集路径。
    - output_path: 输出的 .pt 检查点路径。
    - feature_family: 选择训练使用的特征族，决定输入维度和特征名称。
    - hidden_dims: MLP 隐藏层结构。
    - val_ratio: 验证集比例，按类别分层抽样。
    - sampler_mode: weighted/none/auto，控制是否对少数类样本加权采样。
    - mean / std: 训练集特征标准化参数，必须随检查点保存以保证推理一致。
    - best_state: 验证集表现最好的模型参数快照。
    - confusion_matrix: 验证集混淆矩阵，用于分析各卸载类别的错误来源。
    - train_offload_policy(): 主训练函数，返回包含训练指标和检查点路径的 summary。

主要依赖：
    argparse, copy, json, pathlib, numpy, torch, config, marl_models

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

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


# 函数 parse_hidden_dims：把命令行隐藏层配置字符串转换成 MLP 可直接使用的整数元组。
def parse_hidden_dims(hidden_dims_arg: str) -> tuple[int, ...]:
    dims = tuple(int(token.strip()) for token in hidden_dims_arg.split(",") if token.strip())
    # 没有任何隐藏层尺寸时无法构造分类器，提前报错便于定位参数问题。
    if not dims:
        # 主动报错：提示用户至少提供一个隐藏层宽度。
        raise ValueError("At least one hidden layer size must be provided.")
    # 返回例如 (64, 64) 的结构，后续传入 OffloadMLP。
    return dims


# 函数 select_device：根据命令行参数选择训练设备，auto 模式优先使用 CUDA，其次 MPS，最后 CPU。
def select_device(device_arg: str) -> torch.device:
    # auto 模式让同一脚本可以在不同机器上直接运行，无需手工改代码。
    if device_arg == "auto":
        # NVIDIA GPU 可用时优先走 CUDA。
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon 环境可用时使用 MPS 后端。
        if torch.backends.mps.is_available():
            return torch.device("mps")
        # 没有硬件加速时退回 CPU。
        return torch.device("cpu")
    # 用户显式指定 cpu/cuda/mps 时尊重该设置。
    return torch.device(device_arg)


# 函数 load_feature_family_matrix：从数据集字典中读取指定特征族矩阵，并兼容旧版数据集字段。
def load_feature_family_matrix(dataset: dict[str, np.ndarray], feature_family: str) -> np.ndarray:
    """Load the requested feature family while preserving backward compatibility with old datasets."""

    # full_features 是完整特征族；旧数据集可能只有 features 字段，所以这里提供回退。
    if feature_family == OFFLOAD_FEATURE_FAMILY_FULL:
        # 新数据集显式保存 features_full。
        if "features_full" in dataset:
            return np.asarray(dataset["features_full"], dtype=np.float32)
        # 兼容旧数据集：features 等价于完整特征。
        return np.asarray(dataset["features"], dtype=np.float32)
    # rich_reduced_features 是运行时更容易获得的精简特征族，必须由新版采集脚本生成。
    if feature_family == OFFLOAD_FEATURE_FAMILY_RICH_REDUCED:
        # 缺少字段说明数据集版本不匹配，需要重新采集。
        if "features_rich_reduced" not in dataset:
            raise KeyError(
                "Dataset does not contain 'features_rich_reduced'. Re-collect it with the updated dataset pipeline."
            )
        return np.asarray(dataset["features_rich_reduced"], dtype=np.float32)
    # 未知特征族直接报错，避免悄悄训练出输入维度不匹配的模型。
    raise ValueError(f"Unsupported feature family: {feature_family}")


# 函数 standardize_feature_splits：用训练集均值和标准差标准化训练/验证特征，防止验证集信息泄漏。
def standardize_feature_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using the training split statistics only."""

    mean = np.mean(x_train, axis=0, keepdims=True).astype(np.float32)
    std = np.std(x_train, axis=0, keepdims=True).astype(np.float32)
    # 标准差过小的维度视为常量特征，置为 1 避免除零或数值爆炸。
    std = np.where(std < 1e-6, 1.0, std)
    x_train_std = ((x_train - mean) / std).astype(np.float32)
    x_val_std = ((x_val - mean) / std).astype(np.float32)
    # 同时返回标准化后的特征和一维 mean/std，后者会保存进检查点用于推理。
    return x_train_std, x_val_std, mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


# 函数 stratified_train_val_split：按卸载类别分层划分训练集和验证集，尽量保持类别比例一致。
def stratified_train_val_split(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    # 分别处理 local/cooperative/mbs 三个类别，避免少数类被随机划分完全抽走。
    for class_idx in range(OFFLOAD_NUM_CLASSES):
        class_indices = np.where(labels == class_idx)[0]
        # 当前类别没有样本时跳过，后续 class distribution 会反映这个问题。
        if class_indices.size == 0:
            continue
        shuffled = rng.permutation(class_indices)
        # 单样本类别不能再切验证集，否则训练集中会缺失该类别。
        if class_indices.size == 1:
            train_indices.append(shuffled)
            continue

        # 每个非空类别至少留 1 个验证样本，同时至少保留 1 个训练样本。
        val_count: int = int(round(class_indices.size * val_ratio))
        val_count = max(1, min(val_count, class_indices.size - 1))
        val_indices.append(shuffled[:val_count])
        train_indices.append(shuffled[val_count:])

    train_idx = np.concatenate(train_indices) if train_indices else np.zeros((0,), dtype=np.int64)
    val_idx = np.concatenate(val_indices) if val_indices else np.zeros((0,), dtype=np.int64)
    # 再次打乱合并后的索引，避免 DataLoader 看到按类别分块的数据顺序。
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    # 如果验证集为空，说明数据量太少或 val_ratio 设置不合理。
    if val_idx.size == 0:
        raise ValueError("Validation split is empty. Collect more data or increase val_ratio.")
    # 返回训练和验证样本索引。
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


# 函数 compute_accuracy：关键函数，承载本模块的一段可复用实验逻辑，主要参数：y_true, y_pred。
def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if y_true.size == 0:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return 0.0
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.mean(y_true == y_pred))


# 函数 compute_confusion_matrix：关键函数，承载本模块的一段可复用实验逻辑，主要参数：y_true, y_pred。
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((OFFLOAD_NUM_CLASSES, OFFLOAD_NUM_CLASSES), dtype=np.int64)
    # 循环处理：遍历 (true_label, predicted_label) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for true_label, predicted_label in zip(y_true, y_pred, strict=False):
        matrix[int(true_label), int(predicted_label)] += 1
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return matrix


# 函数 compute_per_class_metrics：关键函数，承载本模块的一段可复用实验逻辑，主要参数：confusion_matrix。
def compute_per_class_metrics(confusion_matrix: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    # 循环处理：遍历 (class_idx, class_name) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return metrics


# 函数 compute_macro_precision_recall_f1：关键函数，承载本模块的一段可复用实验逻辑，主要参数：per_class_metrics。
def compute_macro_precision_recall_f1(per_class_metrics: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    precisions = [metrics["precision"] for metrics in per_class_metrics.values()]
    recalls = [metrics["recall"] for metrics in per_class_metrics.values()]
    f1_scores = [metrics["f1"] for metrics in per_class_metrics.values()]
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores))


# 函数 class_distribution：关键函数，承载本模块的一段可复用实验逻辑，主要参数：labels。
def class_distribution(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES)
    total = max(int(labels.size), 1)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {OFFLOAD_TARGET_NAMES[idx]: float(counts[idx] / total) for idx in range(OFFLOAD_NUM_CLASSES)}


# 函数 class_counts：关键函数，承载本模块的一段可复用实验逻辑，主要参数：labels。
def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {OFFLOAD_TARGET_NAMES[idx]: int(counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)}


# 函数 should_use_weighted_sampler：关键函数，承载本模块的一段可复用实验逻辑，主要参数：train_labels, sampler_mode。
def should_use_weighted_sampler(train_labels: np.ndarray, sampler_mode: str) -> bool:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if sampler_mode == "weighted":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return True
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if sampler_mode == "none":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return False
    counts = np.bincount(train_labels, minlength=OFFLOAD_NUM_CLASSES).astype(np.float32)
    nonzero_counts = counts[counts > 0]
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if nonzero_counts.size <= 1:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return False
    imbalance_ratio = float(np.max(nonzero_counts) / np.min(nonzero_counts))
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return imbalance_ratio > 1.5


# 函数 build_train_loader：关键函数，承载本模块的一段可复用实验逻辑，主要参数：x_train, y_train, batch_size, sampler_mode。
def build_train_loader(x_train: np.ndarray, y_train: np.ndarray, batch_size: int, sampler_mode: str) -> tuple[DataLoader, str]:
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if should_use_weighted_sampler(y_train, sampler_mode):
        label_counts = np.bincount(y_train, minlength=OFFLOAD_NUM_CLASSES).astype(np.float32)
        sample_weights = np.array([1.0 / max(label_counts[label], 1.0) for label in y_train], dtype=np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler), "weighted"
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), "shuffle"


# 函数 evaluate_classifier：关键函数，承载本模块的一段可复用实验逻辑，主要参数：model, features, labels, device。
def evaluate_classifier(model: OffloadMLP, features: np.ndarray, labels: np.ndarray, device: torch.device) -> dict[str, object]:
    model.eval()
    # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = compute_accuracy(labels, predictions)
    confusion_matrix = compute_confusion_matrix(labels, predictions)
    per_class_metrics = compute_per_class_metrics(confusion_matrix)
    macro_precision, macro_recall, macro_f1 = compute_macro_precision_recall_f1(per_class_metrics)

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "predictions": predictions,
    }


# 函数 train_offload_policy：执行模型训练流程。
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
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not dataset_file.exists():
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    dataset = np.load(dataset_file)
    features = load_feature_family_matrix(dataset, feature_family)
    labels = np.asarray(dataset["labels"], dtype=np.int64)
    expected_dim: int = get_offload_feature_dim(feature_family)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if features.ndim != 2 or features.shape[1] != expected_dim:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(
            f"Unexpected feature shape {features.shape}; expected (*, {expected_dim}) for feature family '{feature_family}'."
        )
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
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
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if np.any(nonzero_mask):
        class_weights[nonzero_mask] = float(y_train.size) / (float(np.sum(nonzero_mask)) * train_label_counts[nonzero_mask])
    else:
        class_weights[:] = 1.0

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device_obj))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, object] | None = None
    best_epoch: int = 0

    # 循环处理：遍历 epoch 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss: float = 0.0
        total_batches: int = 0

        # 循环处理：遍历 (batch_features, batch_labels) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
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
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if best_eval is None or float(evaluation["macro_f1"]) > float(best_eval["macro_f1"]):
            best_state_dict = copy.deepcopy(model.state_dict())
            best_eval = evaluation
            best_epoch = epoch

        mean_epoch_loss = epoch_loss / max(total_batches, 1)
        print(
            f"epoch={epoch:03d} loss={mean_epoch_loss:.6f} "
            f"val_acc={float(evaluation['accuracy']):.4f} val_macro_f1={float(evaluation['macro_f1']):.4f}"
        )

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if best_state_dict is None or best_eval is None:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
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
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return summary


# 函数 parse_args：解析命令行参数，并为实验脚本提供可覆盖的默认配置。
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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return parser.parse_args()


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
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


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
