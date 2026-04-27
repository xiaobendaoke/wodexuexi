"""
中文注释说明：evaluate_offload_validity.py

文件作用：
    评估任务卸载策略标签和分类器的论文有效性，输出精度、混淆矩阵和关键统计报告。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - FEATURE_VARIANTS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - load_dataset(): 任务卸载监督学习数据集或样本集合。
    - build_feature_matrix(): 关键函数，承载本模块的一段可复用实验逻辑。
    - standardize_features(): 输入分类器或神经网络的特征矩阵。
    - compute_class_weights(): 关键函数，承载本模块的一段可复用实验逻辑。
    - train_and_evaluate_split(): 执行模型训练流程。
    - random_split_indices(): 关键函数，承载本模块的一段可复用实验逻辑。
    - paired_held_out_scenario_split(): 关键函数，承载本模块的一段可复用实验逻辑。
    - simplify_metrics(): 关键函数，承载本模块的一段可复用实验逻辑。
    - build_textual_conclusion(): 关键函数，承载本模块的一段可复用实验逻辑。
    - run_validity_evaluation(): 关键函数，承载本模块的一段可复用实验逻辑。
    - parse_args(): 解析命令行参数，并为实验脚本提供可覆盖的默认配置。
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    argparse, json, pathlib, numpy, torch, config, paths, marl_models, train_offload_policy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from paths import results_path
from marl_models.offload_policy import OFFLOAD_NUM_CLASSES, OFFLOAD_TARGET_NAMES, OffloadMLP
from train_offload_policy import build_train_loader, evaluate_classifier, select_device, stratified_train_val_split


# 关键变量 FEATURE_VARIANTS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
FEATURE_VARIANTS: dict[str, dict[str, object]] = {
    "full_features": {
        "description": "All normalized features, including the three latency proxy features.",
    },
    "no_latency_features": {
        "description": "Normalized features after removing local/cooperative/mbs latency proxy features.",
    },
    "raw_state_only": {
        "description": "Raw request/state features only: deadline, priority, local_cache_hit, cooperative_available, local_queue_length.",
    },
}


# 函数 load_dataset：任务卸载监督学习数据集或样本集合，主要参数：dataset_path。
def load_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    dataset_file = Path(dataset_path)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not dataset_file.exists():
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    loaded = np.load(dataset_file)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {key: loaded[key] for key in loaded.files}


# 函数 build_feature_matrix：关键函数，承载本模块的一段可复用实验逻辑，主要参数：dataset, variant_name。
def build_feature_matrix(dataset: dict[str, np.ndarray], variant_name: str) -> np.ndarray:
    features = np.asarray(dataset["features"], dtype=np.float32)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if variant_name == "full_features":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return features
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if variant_name == "no_latency_features":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return np.asarray(features[:, 3:], dtype=np.float32)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if variant_name == "raw_state_only":
        raw_state = np.stack(
            [
                np.asarray(dataset["deadlines"], dtype=np.float32),
                np.asarray(dataset["priorities"], dtype=np.float32),
                np.asarray(dataset["local_cache_hits"], dtype=np.float32),
                np.asarray(dataset["cooperative_available"], dtype=np.float32),
                np.asarray(dataset["local_queue_lengths"], dtype=np.float32),
            ],
            axis=1,
        )
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return raw_state.astype(np.float32)
    # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
    raise ValueError(f"Unknown feature variant: {variant_name}")


# 函数 standardize_features：输入分类器或神经网络的特征矩阵，主要参数：x_train, x_val, x_test。
def standardize_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    standardized_train = ((x_train - mean) / std).astype(np.float32)
    standardized_val = ((x_val - mean) / std).astype(np.float32)
    standardized_test = ((x_test - mean) / std).astype(np.float32)
    scaler = {
        "mean": mean.squeeze(0).astype(np.float32).tolist(),
        "std": std.squeeze(0).astype(np.float32).tolist(),
    }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return standardized_train, standardized_val, standardized_test, scaler


# 函数 compute_class_weights：关键函数，承载本模块的一段可复用实验逻辑，主要参数：labels。
def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    label_counts = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES).astype(np.float32)
    class_weights = np.zeros_like(label_counts)
    nonzero_mask = label_counts > 0
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if np.any(nonzero_mask):
        class_weights[nonzero_mask] = float(labels.size) / (float(np.sum(nonzero_mask)) * label_counts[nonzero_mask])
    else:
        class_weights[:] = 1.0
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return class_weights


# 函数 train_and_evaluate_split：执行模型训练流程，主要参数：features, labels。
def train_and_evaluate_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    device: str,
    hidden_dims: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sampler_mode: str,
    split_name: str,
) -> dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_train_full = np.asarray(features[train_indices], dtype=np.float32)
    y_train_full = np.asarray(labels[train_indices], dtype=np.int64)
    x_test = np.asarray(features[test_indices], dtype=np.float32)
    y_test = np.asarray(labels[test_indices], dtype=np.int64)

    relative_train_idx, relative_val_idx = stratified_train_val_split(y_train_full, val_ratio=0.15, seed=seed)
    x_train = x_train_full[relative_train_idx]
    y_train = y_train_full[relative_train_idx]
    x_val = x_train_full[relative_val_idx]
    y_val = y_train_full[relative_val_idx]

    x_train_std, x_val_std, x_test_std, scaler = standardize_features(x_train, x_val, x_test)
    train_loader, effective_sampler_mode = build_train_loader(x_train_std, y_train, batch_size, sampler_mode)

    device_obj = select_device(device)
    model = OffloadMLP(input_dim=x_train_std.shape[1], hidden_dims=hidden_dims, num_classes=OFFLOAD_NUM_CLASSES).to(device_obj)
    class_weights = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device_obj))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_eval: dict[str, object] | None = None
    best_epoch: int = 0

    # 循环处理：遍历 epoch 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for epoch in range(1, epochs + 1):
        model.train()
        # 循环处理：遍历 (batch_features, batch_labels) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device_obj)
            batch_labels = batch_labels.to(device_obj)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        val_eval = evaluate_classifier(model, x_val_std, y_val, device_obj)
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if best_val_eval is None or float(val_eval["macro_f1"]) > float(best_val_eval["macro_f1"]):
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_eval = val_eval
            best_epoch = epoch

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if best_state_dict is None or best_val_eval is None:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise RuntimeError(f"Training failed for split '{split_name}'.")

    model.load_state_dict(best_state_dict)
    model.to(device_obj)
    test_eval = evaluate_classifier(model, x_test_std, y_test, device_obj)
    test_eval.pop("predictions", None)
    best_val_eval = dict(best_val_eval)
    best_val_eval.pop("predictions", None)

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "split_name": split_name,
        "device": str(device_obj),
        "train_samples": int(y_train.size),
        "val_samples": int(y_val.size),
        "test_samples": int(y_test.size),
        "best_epoch": int(best_epoch),
        "sampler_mode_used": effective_sampler_mode,
        "class_weights": {OFFLOAD_TARGET_NAMES[idx]: float(class_weights[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "scaler": scaler,
        "validation": best_val_eval,
        "test": test_eval,
    }


# 函数 random_split_indices：关键函数，承载本模块的一段可复用实验逻辑，主要参数：labels, seed, test_ratio。
def random_split_indices(labels: np.ndarray, seed: int, test_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    # 循环处理：遍历 class_idx 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for class_idx in range(OFFLOAD_NUM_CLASSES):
        class_indices = np.where(labels == class_idx)[0]
        shuffled = rng.permutation(class_indices)
        test_count = int(round(class_indices.size * test_ratio))
        test_count = max(1, min(test_count, class_indices.size - 1))
        test_indices.append(shuffled[:test_count])
        train_indices.append(shuffled[test_count:])

    train_idx = rng.permutation(np.concatenate(train_indices)).astype(np.int64)
    test_idx = rng.permutation(np.concatenate(test_indices)).astype(np.int64)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return train_idx, test_idx


# 函数 paired_held_out_scenario_split：关键函数，承载本模块的一段可复用实验逻辑，主要参数：scenario_ids, scenario_names, labels。
def paired_held_out_scenario_split(
    scenario_ids: np.ndarray,
    scenario_names: list[str],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, str]]:
    scenario_majority_label: dict[int, int] = {}
    # 循环处理：遍历 scenario_id 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for scenario_id in range(len(scenario_names)):
        scenario_labels = labels[scenario_ids == scenario_id]
        majority_label = int(np.argmax(np.bincount(scenario_labels, minlength=OFFLOAD_NUM_CLASSES)))
        scenario_majority_label[scenario_id] = majority_label

    scenarios_by_label: dict[int, list[int]] = {label: [] for label in range(OFFLOAD_NUM_CLASSES)}
    # 循环处理：遍历 (scenario_id, label) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for scenario_id, label in scenario_majority_label.items():
        scenarios_by_label[label].append(scenario_id)

    train_scenario_ids: list[int] = []
    test_scenario_ids: list[int] = []
    pairing: dict[str, str] = {}
    # 循环处理：遍历 label 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for label in range(OFFLOAD_NUM_CLASSES):
        candidate_ids = sorted(scenarios_by_label[label])
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if len(candidate_ids) < 2:
            # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
            raise ValueError(f"Need at least two scenarios for class '{OFFLOAD_TARGET_NAMES[label]}' to run held-out evaluation.")
        train_id = candidate_ids[0]
        test_id = candidate_ids[1]
        train_scenario_ids.append(train_id)
        test_scenario_ids.append(test_id)
        pairing[scenario_names[train_id]] = scenario_names[test_id]

    train_mask = np.isin(scenario_ids, np.asarray(train_scenario_ids, dtype=np.int64))
    test_mask = np.isin(scenario_ids, np.asarray(test_scenario_ids, dtype=np.int64))
    train_indices = np.where(train_mask)[0].astype(np.int64)
    test_indices = np.where(test_mask)[0].astype(np.int64)

    train_scenarios = [scenario_names[idx] for idx in train_scenario_ids]
    test_scenarios = [scenario_names[idx] for idx in test_scenario_ids]
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return train_indices, test_indices, train_scenarios, test_scenarios, pairing


# 函数 simplify_metrics：关键函数，承载本模块的一段可复用实验逻辑，主要参数：result。
def simplify_metrics(result: dict[str, object]) -> dict[str, object]:
    test_metrics = result["test"]
    per_class_recall = {
        class_name: float(test_metrics["per_class_metrics"][class_name]["recall"]) for class_name in OFFLOAD_TARGET_NAMES
    }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "per_class_recall": per_class_recall,
        "confusion_matrix": test_metrics["confusion_matrix"],
    }


# 函数 build_textual_conclusion：关键函数，承载本模块的一段可复用实验逻辑，主要参数：report。
def build_textual_conclusion(report: dict[str, object]) -> dict[str, str]:
    iid_full = report["feature_ablation"]["full_features"]["iid_random_split"]["test"]
    held_full = report["feature_ablation"]["full_features"]["cross_scenario_split"]["test"]
    iid_no_latency = report["feature_ablation"]["no_latency_features"]["iid_random_split"]["test"]
    held_no_latency = report["feature_ablation"]["no_latency_features"]["cross_scenario_split"]["test"]
    held_raw_state = report["feature_ablation"]["raw_state_only"]["cross_scenario_split"]["test"]

    iid_full_f1 = float(iid_full["macro_f1"])
    held_full_f1 = float(held_full["macro_f1"])
    iid_no_latency_f1 = float(iid_no_latency["macro_f1"])
    held_no_latency_f1 = float(held_no_latency["macro_f1"])
    held_raw_state_f1 = float(held_raw_state["macro_f1"])

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if held_full_f1 >= 0.95 and (iid_full_f1 - held_full_f1) <= 0.05 and held_no_latency_f1 >= 0.8:
        classifier_type = "heuristic surrogate / distillation baseline"
        rationale = (
            "Cross-scenario performance remains high, but the labels still come from the heuristic and the features are "
            "engineered around the heuristic decision state. This supports describing the model as a heuristic-surrogate baseline."
        )
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif held_full_f1 >= 0.95 and held_no_latency_f1 < held_full_f1 - 0.2:
        classifier_type = "heuristic surrogate / distillation baseline"
        rationale = (
            "The full-feature model generalizes better than the ablated versions, but the advantage is strongly tied to the latency "
            "proxy features that mirror the heuristic rule. This is closer to distillation than to an independently learned policy."
        )
    else:
        classifier_type = "heuristic surrogate / distillation baseline"
        rationale = (
            "The model should still be presented as a heuristic-surrogate baseline because both the labels and the feature set are derived "
            "from the heuristic decision rule and curated scenarios."
        )

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if iid_full_f1 - held_full_f1 > 0.1:
        scenario_gap = (
            "There is a visible IID-to-held-out drop, which suggests the current classifier is sensitive to scenario composition and may "
            "retain some scenario-specific shortcuts."
        )
    else:
        scenario_gap = (
            "IID and held-out metrics are close on the current paired split, so the main validity caveat is not immediate collapse under "
            "held-out scenarios, but the fact that the dataset is curated from heuristic-driven synthetic scenarios."
        )

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if iid_no_latency_f1 < iid_full_f1 - 0.2:
        shortcut_text = (
            "The latency proxy features contribute materially to the near-perfect result, which means the current model is learning a close "
            "surrogate of the heuristic scoring rule."
        )
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif held_raw_state_f1 >= 0.8:
        shortcut_text = (
            "Removing latency proxies does not fully break performance, so the dataset also contains strong non-latency shortcuts from the "
            "scenario construction itself, especially cache/queue/cooperative-availability patterns."
        )
    else:
        shortcut_text = (
            "Both latency proxies and scenario-specific state patterns contribute to the result, so the current 1.0 should not be claimed "
            "as evidence of broad learned generalization."
        )

    next_steps = (
        "1. Evaluate on procedurally generated held-out scenarios that are not one of the six collection templates.\n"
        "2. Redesign the dataset so local/cooperative/MBS each appear under overlapping state regions instead of nearly template-specific patterns."
    )

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "recommended_thesis_label": classifier_type,
        "rationale": rationale,
        "scenario_generalization_comment": scenario_gap,
        "shortcut_comment": shortcut_text,
        "next_steps": next_steps,
    }


# 函数 run_validity_evaluation：关键函数，承载本模块的一段可复用实验逻辑。
def run_validity_evaluation(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    seed: int = config.SEED,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "auto",
    sampler_mode: str = "auto",
) -> dict[str, object]:
    dataset = load_dataset(dataset_path)
    labels = np.asarray(dataset["labels"], dtype=np.int64)
    scenario_ids = np.asarray(dataset["scenario_ids"], dtype=np.int64)
    scenario_names = [str(name) for name in np.asarray(dataset["scenario_names"]).tolist()]

    iid_train_idx, iid_test_idx = random_split_indices(labels, seed=seed, test_ratio=0.2)
    held_train_idx, held_test_idx, train_scenarios, test_scenarios, pairing = paired_held_out_scenario_split(
        scenario_ids,
        scenario_names,
        labels,
    )

    report: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "splits": {
            "iid_random_split": {
                "description": "Stratified random split over all samples.",
            },
            "cross_scenario_split": {
                "description": "Train on one scenario per class and test on a held-out counterpart scenario per class.",
                "train_scenarios": train_scenarios,
                "test_scenarios": test_scenarios,
                "scenario_pairing": pairing,
            },
        },
        "feature_ablation": {},
    }

    # 循环处理：遍历 (variant_name, variant_meta) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for variant_name, variant_meta in FEATURE_VARIANTS.items():
        feature_matrix = build_feature_matrix(dataset, variant_name)
        iid_result = train_and_evaluate_split(
            feature_matrix,
            labels,
            train_indices=iid_train_idx,
            test_indices=iid_test_idx,
            seed=seed,
            device=device,
            hidden_dims=(64, 64),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            sampler_mode=sampler_mode,
            split_name=f"{variant_name}_iid_random_split",
        )
        held_result = train_and_evaluate_split(
            feature_matrix,
            labels,
            train_indices=held_train_idx,
            test_indices=held_test_idx,
            seed=seed,
            device=device,
            hidden_dims=(64, 64),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            sampler_mode=sampler_mode,
            split_name=f"{variant_name}_cross_scenario_split",
        )

        report["feature_ablation"][variant_name] = {
            "description": str(variant_meta["description"]),
            "feature_dim": int(feature_matrix.shape[1]),
            "iid_random_split": iid_result,
            "cross_scenario_split": held_result,
            "summary": {
                "iid_random_split": simplify_metrics(iid_result),
                "cross_scenario_split": simplify_metrics(held_result),
            },
        }

    full_iid = report["feature_ablation"]["full_features"]["summary"]["iid_random_split"]
    full_held = report["feature_ablation"]["full_features"]["summary"]["cross_scenario_split"]
    report["same_vs_cross_scenario_gap"] = {
        "full_features_accuracy_gap": float(full_iid["accuracy"] - full_held["accuracy"]),
        "full_features_macro_f1_gap": float(full_iid["macro_f1"] - full_held["macro_f1"]),
    }
    report["conclusion"] = build_textual_conclusion(report)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return report


# 函数 parse_args：解析命令行参数，并为实验脚本提供可覆盖的默认配置。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-validity checks for the request-level offloading classifier.")
    parser.add_argument("--dataset", type=str, default="offload_datasets/offload_dataset_balanced.npz", help="Dataset path.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(results_path("reports", "offload_policy_validity_report.json")),
        help="Output report path.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per evaluation.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps.")
    parser.add_argument("--sampler_mode", type=str, default="auto", choices=["auto", "weighted", "none"], help="Sampler mode.")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return parser.parse_args()


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main() -> None:
    args = parse_args()
    report = run_validity_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        sampler_mode=args.sampler_mode,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
