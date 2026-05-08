"""
中文注释说明：verify_offload_policy_validity.py

文件作用：
    从更完整的实验角度验证任务卸载策略是否符合论文设定，生成 JSON 与 Markdown 形式的验证结论。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - REPO_ROOT: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - RESULTS_DIR: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - RESULTS_PATH: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - REPORT_PATH: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - FULL_FEATURE_NAMES: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - RICH_REDUCED_FEATURE_NAMES: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - TrainedClassifier: 核心类，封装本模块中的主要状态和行为。
    - TemplateSample: 核心类，封装本模块中的主要状态和行为。
    - snapshot_config(): 全局配置模块，保存环境参数和训练超参数。
    - restore_config(): 全局配置模块，保存环境参数和训练超参数。
    - build_feature_family_metadata(): 关键函数，承载本模块的一段可复用实验逻辑。
    - normalize_deadline(): 关键函数，承载本模块的一段可复用实验逻辑。
    - normalize_priority(): 关键函数，承载本模块的一段可复用实验逻辑。
    - safe_log10(): 关键函数，承载本模块的一段可复用实验逻辑。
    - extract_rich_runtime_features(): 输入分类器或神经网络的特征矩阵。
    - extract_variant_features_from_context(): 输入分类器或神经网络的特征矩阵。
    - standardize_features(): 输入分类器或神经网络的特征矩阵。
    - compute_class_weights(): 关键函数，承载本模块的一段可复用实验逻辑。
    - train_classifier_from_arrays(): 执行模型训练流程。
    - simplify_metrics(): 关键函数，承载本模块的一段可复用实验逻辑。
    - build_feature_matrices_from_template_samples(): 关键函数，承载本模块的一段可复用实验逻辑。
    - collect_template_rich_dataset(): 任务卸载监督学习数据集或样本集合。

主要依赖：
    copy, json, collections, contextlib, dataclasses, pathlib, typing, numpy, torch, collect_offload_dataset

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

import collect_offload_dataset as dataset_module
import config
import environment.comm_model as comms
import environment.uavs as uav_module
from paths import results_path
from environment.env import Env
from environment.request_types import Request
from marl_models.offload_policy import (
    OFFLOAD_NUM_CLASSES,
    OFFLOAD_TARGET_COOPERATIVE,
    OFFLOAD_TARGET_LOCAL,
    OFFLOAD_TARGET_MBS,
    OFFLOAD_TARGET_NAMES,
    OffloadMLP,
    ServiceOffloadContext,
    context_to_feature_vector,
)
from marl_models.static_baseline.static_model import StaticModel
from train_offload_policy import (
    build_train_loader,
    evaluate_classifier,
    select_device,
    stratified_train_val_split,
)


# 关键变量 REPO_ROOT：全局常量或配置项，会影响环境规模、训练过程或实验输出。
REPO_ROOT = Path(__file__).resolve().parent
# 关键变量 RESULTS_DIR：全局常量或配置项，会影响环境规模、训练过程或实验输出。
RESULTS_DIR = results_path("reports")
# 关键变量 RESULTS_PATH：全局常量或配置项，会影响环境规模、训练过程或实验输出。
RESULTS_PATH = RESULTS_DIR / "offload_policy_paper_validity.json"
# 关键变量 REPORT_PATH：全局常量或配置项，会影响环境规模、训练过程或实验输出。
REPORT_PATH = RESULTS_DIR / "offload_policy_paper_validity_summary.md"

# 关键变量 FULL_FEATURE_NAMES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
FULL_FEATURE_NAMES: tuple[str, ...] = (
    "local_latency_vs_deadline",
    "cooperative_latency_vs_deadline",
    "mbs_latency_vs_deadline",
    "deadline_normalized",
    "priority_normalized",
    "local_cache_hit",
    "cooperative_available",
    "local_queue_fraction",
)
# 关键变量 RICH_REDUCED_FEATURE_NAMES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
RICH_REDUCED_FEATURE_NAMES: tuple[str, ...] = (
    "request_size_normalized",
    "service_file_size_normalized",
    "cpu_density_normalized",
    "deadline_normalized",
    "priority_normalized",
    "local_cache_hit",
    "cooperative_available",
    "local_queue_fraction",
    "neighbor_count_fraction",
    "ue_uav_rate_log10",
    "uav_mbs_rate_log10",
    "best_uav_uav_rate_log10",
    "best_neighbor_mbs_rate_log10",
    "local_compute_share_normalized",
    "best_neighbor_compute_share_normalized",
    "best_neighbor_cache_belief",
)


# 类 TrainedClassifier：核心类，封装本模块中的主要状态和行为。
@dataclass(frozen=True)
class TrainedClassifier:
    name: str
    model: OffloadMLP
    device: torch.device
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    source_description: str
    variant_group: str

    # 函数 predict_from_array：关键函数，承载本模块的一段可复用实验逻辑，主要参数：features。
    def predict_from_array(self, features: np.ndarray) -> int:
        standardized = ((np.asarray(features, dtype=np.float32) - self.mean) / self.std).astype(np.float32)
        feature_tensor = torch.from_numpy(standardized).unsqueeze(0).to(self.device)
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with torch.no_grad():
            logits = self.model(feature_tensor)
            prediction = torch.argmax(logits, dim=1)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return int(prediction.item())

    # 函数 predict_from_context：关键函数，承载本模块的一段可复用实验逻辑，主要参数：context。
    def predict_from_context(self, context: ServiceOffloadContext) -> int:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.predict_from_array(extract_variant_features_from_context(context, self.variant_group))


# 类 TemplateSample：核心类，封装本模块中的主要状态和行为。
@dataclass
class TemplateSample:
    full_features: np.ndarray
    latency_only_features: np.ndarray
    no_latency_features: np.ndarray
    rich_reduced_features: np.ndarray
    label: int
    scenario_name: str


# 函数 snapshot_config：全局配置模块，保存环境参数和训练超参数。
def snapshot_config() -> dict[str, object]:
    snapshot: dict[str, object] = {}
    # 循环处理：遍历 key 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for key in dir(config):
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if key.isupper() and not key.startswith("__"):
            value = getattr(config, key)
            snapshot[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return snapshot


# 函数 restore_config：全局配置模块，保存环境参数和训练超参数，主要参数：snapshot。
def restore_config(snapshot: dict[str, object]) -> None:
    # 循环处理：遍历 (key, value) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for key, value in snapshot.items():
        setattr(config, key, value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))


# 函数 build_feature_family_metadata：关键函数，承载本模块的一段可复用实验逻辑。
def build_feature_family_metadata() -> dict[str, dict[str, object]]:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "full_features": {
            "description": "Distillation surrogate with all 8 normalized features, including the 3 latency proxies.",
            "feature_names": list(FULL_FEATURE_NAMES),
            "uses_latency_proxies": True,
        },
        "latency_only": {
            "description": "Diagnostic shortcut model with only the 3 latency proxy features.",
            "feature_names": list(FULL_FEATURE_NAMES[:3]),
            "uses_latency_proxies": True,
        },
        "no_latency_features": {
            "description": "Current reduced-leakage baseline using only the 5 non-latency fields already present in the existing request-level context.",
            "feature_names": list(FULL_FEATURE_NAMES[3:]),
            "uses_latency_proxies": False,
        },
        "rich_reduced_features": {
            "description": "Reduced-leakage candidate using richer raw runtime state: request size, compute/cache/queue state, rates, and neighbor state, without the 3 latency proxies.",
            "feature_names": list(RICH_REDUCED_FEATURE_NAMES),
            "uses_latency_proxies": False,
        },
    }


# 函数 normalize_deadline：关键函数，承载本模块的一段可复用实验逻辑，主要参数：deadline。
def normalize_deadline(deadline: float) -> float:
    span = float(config.SERVICE_DEADLINE_MAX - config.SERVICE_DEADLINE_MIN)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if span <= float(config.EPSILON):
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return 0.0
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip((float(deadline) - float(config.SERVICE_DEADLINE_MIN)) / span, 0.0, 1.0))


# 函数 normalize_priority：关键函数，承载本模块的一段可复用实验逻辑，主要参数：priority。
def normalize_priority(priority: int) -> float:
    span = int(config.SERVICE_PRIORITY_MAX - config.SERVICE_PRIORITY_MIN)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if span <= 0:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return 0.0
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip((float(priority) - float(config.SERVICE_PRIORITY_MIN)) / float(span), 0.0, 1.0))


# 函数 safe_log10：关键函数，承载本模块的一段可复用实验逻辑，主要参数：value。
def safe_log10(value: float) -> float:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.log10(max(float(value), float(config.EPSILON))))


# 函数 extract_rich_runtime_features：输入分类器或神经网络的特征矩阵，主要参数：source_uav, request, ue_uav_rate, context, cooperative_uav。
def extract_rich_runtime_features(
    source_uav: "uav_module.UAV",
    request: Request,
    ue_uav_rate: float,
    context: ServiceOffloadContext,
    cooperative_uav: "uav_module.UAV | None",
) -> np.ndarray:
    max_file_size = max(float(np.max(config.FILE_SIZES)), 1.0)
    cpu_min = float(np.min(config.CPU_CYCLES_PER_BYTE))
    cpu_max = float(np.max(config.CPU_CYCLES_PER_BYTE))
    cpu_span = max(cpu_max - cpu_min, float(config.EPSILON))
    max_compute_capacity = max(float(np.max(config.UAV_COMPUTING_CAPACITY)), 1.0)
    request_size_span = max(float(config.MAX_INPUT_SIZE - config.MIN_INPUT_SIZE), 1.0)
    service_load = max(int(source_uav._current_service_request_count), 1)

    request_size_normalized = np.clip(
        (float(request.req_size) - float(config.MIN_INPUT_SIZE)) / request_size_span,
        0.0,
        1.0,
    )
    file_size_normalized = float(config.FILE_SIZES[request.req_id]) / max_file_size
    cpu_density_normalized = (float(config.CPU_CYCLES_PER_BYTE[request.req_id]) - cpu_min) / cpu_span
    local_queue_fraction = float(np.clip(source_uav._current_service_request_count / max(float(config.MAX_ASSOCIATED_UES), 1.0), 0.0, 1.0))
    neighbor_count_fraction = float(np.clip(len(source_uav.neighbors) / max(float(config.NUM_UAVS - 1), 1.0), 0.0, 1.0))
    local_compute_share = float(config.UAV_COMPUTING_CAPACITY[source_uav.id]) / float(service_load)
    local_compute_share_normalized = float(np.clip(local_compute_share / max_compute_capacity, 0.0, 4.0))

    best_uav_uav_rate = 0.0
    best_neighbor_mbs_rate = 0.0
    best_neighbor_compute_share_normalized = 0.0
    best_neighbor_cache_belief = 0.0
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if cooperative_uav is not None:
        best_uav_uav_rate = float(
            comms.calculate_uav_uav_rate(comms.calculate_channel_gain(source_uav.pos, cooperative_uav.pos))
        )
        best_neighbor_mbs_rate = float(
            comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(cooperative_uav.pos, config.MBS_POS))
        )
        neighbor_load = max(int(cooperative_uav._current_service_request_count) + 1, 1)
        best_neighbor_compute_share = float(config.UAV_COMPUTING_CAPACITY[cooperative_uav.id]) / float(neighbor_load)
        best_neighbor_compute_share_normalized = float(np.clip(best_neighbor_compute_share / max_compute_capacity, 0.0, 4.0))
        best_neighbor_cache_belief = float(uav_module._get_belief_probability(request.req_id, cooperative_uav.id))

    features = np.array(
        [
            request_size_normalized,
            float(np.clip(file_size_normalized, 0.0, 1.0)),
            float(np.clip(cpu_density_normalized, 0.0, 1.0)),
            normalize_deadline(float(request.deadline)),
            normalize_priority(int(request.priority)),
            float(context.local_cache_hit),
            float(context.cooperative_available),
            local_queue_fraction,
            neighbor_count_fraction,
            safe_log10(float(ue_uav_rate)),
            safe_log10(float(source_uav._uav_mbs_rate)),
            safe_log10(best_uav_uav_rate),
            safe_log10(best_neighbor_mbs_rate),
            local_compute_share_normalized,
            best_neighbor_compute_share_normalized,
            float(np.clip(best_neighbor_cache_belief, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    assert features.shape == (len(RICH_REDUCED_FEATURE_NAMES),)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return features


# 函数 extract_variant_features_from_context：输入分类器或神经网络的特征矩阵，主要参数：context, variant_name。
def extract_variant_features_from_context(context: ServiceOffloadContext, variant_name: str) -> np.ndarray:
    full_features = context_to_feature_vector(context)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if variant_name == "full_features":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return np.asarray(full_features, dtype=np.float32)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if variant_name == "latency_only":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return np.asarray(full_features[:3], dtype=np.float32)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if variant_name == "no_latency_features":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return np.asarray(full_features[3:], dtype=np.float32)
    # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
    raise ValueError(f"Variant '{variant_name}' needs raw runtime state, not a bare ServiceOffloadContext.")


# 函数 standardize_features：输入分类器或神经网络的特征矩阵，主要参数：x_train, x_val, x_test。
def standardize_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0, keepdims=True).astype(np.float32)
    std = np.std(x_train, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        mean.squeeze(0).astype(np.float32),
        std.squeeze(0).astype(np.float32),
    )


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


# 函数 train_classifier_from_arrays：执行模型训练流程。
def train_classifier_from_arrays(
    *,
    name: str,
    feature_family: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    sampler_mode: str,
    source_description: str,
) -> tuple[TrainedClassifier, dict[str, object]]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    relative_train_idx, relative_val_idx = stratified_train_val_split(train_labels, val_ratio=0.15, seed=seed)
    x_train = np.asarray(train_features[relative_train_idx], dtype=np.float32)
    y_train = np.asarray(train_labels[relative_train_idx], dtype=np.int64)
    x_val = np.asarray(train_features[relative_val_idx], dtype=np.float32)
    y_val = np.asarray(train_labels[relative_val_idx], dtype=np.int64)
    x_test = np.asarray(test_features, dtype=np.float32)
    y_test = np.asarray(test_labels, dtype=np.int64)

    x_train_std, x_val_std, x_test_std, mean, std = standardize_features(x_train, x_val, x_test)
    train_loader, sampler_used = build_train_loader(x_train_std, y_train, batch_size, sampler_mode)

    device_obj = select_device(device)
    model = OffloadMLP(input_dim=x_train_std.shape[1], hidden_dims=(64, 64), num_classes=OFFLOAD_NUM_CLASSES).to(device_obj)
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
            best_val_eval = dict(val_eval)
            best_epoch = epoch

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if best_state_dict is None or best_val_eval is None:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise RuntimeError(f"Training failed for variant '{name}'.")

    model.load_state_dict(best_state_dict)
    model.to(device_obj)
    model.eval()
    test_eval = evaluate_classifier(model, x_test_std, y_test, device_obj)
    best_val_eval.pop("predictions", None)
    test_eval.pop("predictions", None)

    trained_policy = TrainedClassifier(
        name=name,
        model=model,
        device=device_obj,
        feature_names=tuple(build_feature_family_metadata()[feature_family]["feature_names"]),
        mean=mean,
        std=std,
        source_description=source_description,
        variant_group=feature_family,
    )
    metrics = {
        "train_samples": int(y_train.size),
        "val_samples": int(y_val.size),
        "test_samples": int(y_test.size),
        "best_epoch": int(best_epoch),
        "sampler_mode_used": sampler_used,
        "class_weights": {OFFLOAD_TARGET_NAMES[idx]: float(class_weights[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "validation": best_val_eval,
        "test": test_eval,
    }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return trained_policy, metrics


# 函数 simplify_metrics：关键函数，承载本模块的一段可复用实验逻辑，主要参数：metrics。
def simplify_metrics(metrics: dict[str, object]) -> dict[str, object]:
    per_class_metrics = metrics["per_class_metrics"]
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_precision": {class_name: float(per_class_metrics[class_name]["precision"]) for class_name in OFFLOAD_TARGET_NAMES},
        "per_class_recall": {class_name: float(per_class_metrics[class_name]["recall"]) for class_name in OFFLOAD_TARGET_NAMES},
        "per_class_f1": {class_name: float(per_class_metrics[class_name]["f1"]) for class_name in OFFLOAD_TARGET_NAMES},
        "confusion_matrix": metrics["confusion_matrix"],
    }


# 函数 build_feature_matrices_from_template_samples：关键函数，承载本模块的一段可复用实验逻辑，主要参数：samples。
def build_feature_matrices_from_template_samples(samples: list[TemplateSample]) -> dict[str, np.ndarray]:
    scenario_name_to_id: dict[str, int] = {}
    scenario_ids: list[int] = []
    labels: list[int] = []
    full_features: list[np.ndarray] = []
    latency_only: list[np.ndarray] = []
    no_latency: list[np.ndarray] = []
    rich_reduced: list[np.ndarray] = []

    # 循环处理：遍历 sample 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for sample in samples:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if sample.scenario_name not in scenario_name_to_id:
            scenario_name_to_id[sample.scenario_name] = len(scenario_name_to_id)
        scenario_ids.append(scenario_name_to_id[sample.scenario_name])
        labels.append(int(sample.label))
        full_features.append(np.asarray(sample.full_features, dtype=np.float32))
        latency_only.append(np.asarray(sample.latency_only_features, dtype=np.float32))
        no_latency.append(np.asarray(sample.no_latency_features, dtype=np.float32))
        rich_reduced.append(np.asarray(sample.rich_reduced_features, dtype=np.float32))

    scenario_names = [name for name, _ in sorted(scenario_name_to_id.items(), key=lambda item: item[1])]
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "scenario_ids": np.asarray(scenario_ids, dtype=np.int64),
        "scenario_names": np.asarray(scenario_names, dtype="<U64"),
        "full_features": np.stack(full_features, axis=0).astype(np.float32),
        "latency_only": np.stack(latency_only, axis=0).astype(np.float32),
        "no_latency_features": np.stack(no_latency, axis=0).astype(np.float32),
        "rich_reduced_features": np.stack(rich_reduced, axis=0).astype(np.float32),
    }


# 函数 collect_template_rich_dataset：任务卸载监督学习数据集或样本集合。
def collect_template_rich_dataset(*, per_class_target: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_snapshot = snapshot_config()
    env = Env()
    env.reset()
    scenario_entries = dataset_module._build_balanced_scenarios()
    scenarios_by_label: dict[int, list[tuple[Any, Callable[..., tuple[ServiceOffloadContext, int]]]]] = {
        label: [entry for entry in scenario_entries if entry[0].intended_label == label] for label in range(OFFLOAD_NUM_CLASSES)
    }
    scenario_indices: dict[int, int] = {label: 0 for label in range(OFFLOAD_NUM_CLASSES)}
    accepted_counts = np.zeros(OFFLOAD_NUM_CLASSES, dtype=np.int64)
    samples: list[TemplateSample] = []

    original_compute_service_sample = dataset_module._compute_service_sample
    pending_payload: dict[str, Any] = {}

    # 函数 wrapped_compute_service_sample：关键函数，承载本模块的一段可复用实验逻辑，主要参数：local_env, request。
    def wrapped_compute_service_sample(local_env: Env, request: Request, *, covered_ues_count: int):
        source_uav = local_env.uavs[0]
        source_uav._uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(source_uav.pos, config.MBS_POS))
        ue_offset = np.array([1.0, 1.0], dtype=np.float32)
        ue_pos = np.array([source_uav.pos[0] + ue_offset[0], source_uav.pos[1] + ue_offset[1], 0.0], dtype=np.float32)
        ue_uav_rate = comms.calculate_ue_uav_rate(
            comms.calculate_channel_gain(ue_pos, source_uav.pos),
            max(int(covered_ues_count), 1),
        )
        context, cooperative_uav = source_uav._build_service_offload_context(request, ue_uav_rate)
        label, _ = source_uav._select_service_target_from_context(context, cooperative_uav)
        full_features = context_to_feature_vector(context)
        pending_payload.clear()
        pending_payload.update(
            {
                "full_features": full_features,
                "latency_only_features": full_features[:3],
                "no_latency_features": full_features[3:],
                "rich_reduced_features": extract_rich_runtime_features(source_uav, request, ue_uav_rate, context, cooperative_uav),
                "label": int(label),
            }
        )
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return context, int(label)

    dataset_module._compute_service_sample = wrapped_compute_service_sample
    # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
    try:
        # 循环控制：在条件满足期间持续推进采样、训练或搜索流程。
        while int(np.sum(np.maximum(per_class_target - accepted_counts, 0))) > 0:
            deficits = np.maximum(per_class_target - accepted_counts, 0)
            target_label = int(np.argmax(deficits))
            candidate_scenarios = scenarios_by_label[target_label]
            scenario_position = scenario_indices[target_label] % len(candidate_scenarios)
            scenario_indices[target_label] += 1
            scenario_spec, scenario_sampler = candidate_scenarios[scenario_position]

            restore_config(base_snapshot)
            scenario_sampler(env, rng)
            label = int(pending_payload["label"])
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if accepted_counts[label] >= per_class_target:
                continue

            samples.append(
                TemplateSample(
                    full_features=np.asarray(pending_payload["full_features"], dtype=np.float32),
                    latency_only_features=np.asarray(pending_payload["latency_only_features"], dtype=np.float32),
                    no_latency_features=np.asarray(pending_payload["no_latency_features"], dtype=np.float32),
                    rich_reduced_features=np.asarray(pending_payload["rich_reduced_features"], dtype=np.float32),
                    label=label,
                    scenario_name=str(scenario_spec.name),
                )
            )
            accepted_counts[label] += 1
    finally:
        dataset_module._compute_service_sample = original_compute_service_sample
        restore_config(base_snapshot)

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return build_feature_matrices_from_template_samples(samples)


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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return (
        train_indices,
        test_indices,
        [scenario_names[idx] for idx in train_scenario_ids],
        [scenario_names[idx] for idx in test_scenario_ids],
        pairing,
    )


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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return (
        rng.permutation(np.concatenate(train_indices)).astype(np.int64),
        rng.permutation(np.concatenate(test_indices)).astype(np.int64),
    )


# 函数 evaluate_feature_variants_on_template：关键函数，承载本模块的一段可复用实验逻辑，主要参数：template_dataset。
def evaluate_feature_variants_on_template(
    template_dataset: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    sampler_mode: str,
) -> tuple[dict[str, object], dict[str, TrainedClassifier]]:
    labels = np.asarray(template_dataset["labels"], dtype=np.int64)
    scenario_ids = np.asarray(template_dataset["scenario_ids"], dtype=np.int64)
    scenario_names = [str(name) for name in np.asarray(template_dataset["scenario_names"]).tolist()]
    iid_train_idx, iid_test_idx = random_split_indices(labels, seed=seed, test_ratio=0.2)
    held_train_idx, held_test_idx, train_scenarios, test_scenarios, pairing = paired_held_out_scenario_split(
        scenario_ids,
        scenario_names,
        labels,
    )

    report: dict[str, object] = {
        "template_dataset_size": int(labels.size),
        "train_scenarios": train_scenarios,
        "test_scenarios": test_scenarios,
        "scenario_pairing": pairing,
        "variants": {},
    }
    trained_policies: dict[str, TrainedClassifier] = {}
    feature_meta = build_feature_family_metadata()

    # 循环处理：遍历 variant_name 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for variant_name in ("full_features", "latency_only", "no_latency_features", "rich_reduced_features"):
        variant_features = np.asarray(template_dataset[variant_name], dtype=np.float32)
        iid_policy, iid_metrics = train_classifier_from_arrays(
            name=f"{variant_name}_iid",
            feature_family=variant_name,
            train_features=variant_features[iid_train_idx],
            train_labels=labels[iid_train_idx],
            test_features=variant_features[iid_test_idx],
            test_labels=labels[iid_test_idx],
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            sampler_mode=sampler_mode,
            source_description=f"Template balanced dataset ({variant_name}) with IID random split.",
        )
        held_policy, held_metrics = train_classifier_from_arrays(
            name=f"{variant_name}_heldout",
            feature_family=variant_name,
            train_features=variant_features[held_train_idx],
            train_labels=labels[held_train_idx],
            test_features=variant_features[held_test_idx],
            test_labels=labels[held_test_idx],
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            sampler_mode=sampler_mode,
            source_description=f"Template balanced dataset ({variant_name}) with paired held-out scenario split.",
        )
        trained_policies[variant_name] = held_policy
        report["variants"][variant_name] = {
            "description": feature_meta[variant_name]["description"],
            "feature_names": feature_meta[variant_name]["feature_names"],
            "iid_random_split": {
                "train": simplify_metrics(iid_metrics["validation"]),
                "test": simplify_metrics(iid_metrics["test"]),
            },
            "held_out_template_split": {
                "train_scenarios": train_scenarios,
                "test_scenarios": test_scenarios,
                "test": simplify_metrics(held_metrics["test"]),
            },
        }

    report["same_vs_cross_scenario_gap"] = {
        variant_name: {
            "accuracy_gap": float(
                report["variants"][variant_name]["iid_random_split"]["test"]["accuracy"]
                - report["variants"][variant_name]["held_out_template_split"]["test"]["accuracy"]
            ),
            "macro_f1_gap": float(
                report["variants"][variant_name]["iid_random_split"]["test"]["macro_f1"]
                - report["variants"][variant_name]["held_out_template_split"]["test"]["macro_f1"]
            ),
        }
        for variant_name in report["variants"]
    }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return report, trained_policies


# 函数 compute_heuristic_label：关键函数，承载本模块的一段可复用实验逻辑，主要参数：context。
def compute_heuristic_label(context: ServiceOffloadContext) -> int:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not context.cooperative_available:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return OFFLOAD_TARGET_MBS if context.mbs_latency < context.local_latency else OFFLOAD_TARGET_LOCAL
    latencies = np.array([context.local_latency, context.cooperative_latency, context.mbs_latency], dtype=np.float32)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return int(np.argmin(latencies))


# 函数 generate_procedural_contexts：通信链路速率。
def generate_procedural_contexts(
    *,
    split: str,
    samples_per_class: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    full_features: list[np.ndarray] = []
    latency_only: list[np.ndarray] = []
    no_latency: list[np.ndarray] = []
    rich_features: list[np.ndarray] = []
    labels: list[int] = []

    max_file_size = max(float(np.max(config.FILE_SIZES)), 1.0)
    cpu_min = float(np.min(config.CPU_CYCLES_PER_BYTE))
    cpu_max = float(np.max(config.CPU_CYCLES_PER_BYTE))
    cpu_span = max(cpu_max - cpu_min, float(config.EPSILON))
    max_compute = max(float(np.max(config.UAV_COMPUTING_CAPACITY)), 1.0)
    request_size_span = max(float(config.MAX_INPUT_SIZE - config.MIN_INPUT_SIZE), 1.0)

    collected = np.zeros(OFFLOAD_NUM_CLASSES, dtype=np.int64)
    # 循环控制：在条件满足期间持续推进采样、训练或搜索流程。
    while np.any(collected < samples_per_class):
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if split == "train":
            deadline_norm = float(rng.uniform(0.12, 0.92))
            request_size_norm = float(rng.uniform(0.06, 0.85))
            file_size_norm = float(rng.uniform(0.18, 0.84))
            cpu_density_norm = float(rng.uniform(0.18, 0.82))
            local_queue_fraction = float(rng.uniform(0.02, 0.78))
            neighbor_count_fraction = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
            cooperative_available = bool(rng.random() < 0.72)
            local_cache_hit = bool(rng.random() < 0.42)
            priority = int(rng.integers(config.SERVICE_PRIORITY_MIN, config.SERVICE_PRIORITY_MAX + 1))
            ue_uav_rate_log10 = float(rng.uniform(6.9, 8.1))
            uav_mbs_rate_log10 = float(rng.uniform(5.9, 7.7))
            best_uav_uav_rate_log10 = float(rng.uniform(6.3, 8.0))
            best_neighbor_mbs_rate_log10 = float(rng.uniform(5.8, 7.6))
            local_compute_share_norm = float(rng.uniform(0.20, 1.60))
            best_neighbor_compute_share_norm = float(rng.uniform(0.18, 1.85))
            best_neighbor_cache_belief = float(rng.uniform(0.18, 0.86))
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        elif split == "test":
            deadline_norm = float(rng.uniform(0.02, 1.00))
            request_size_norm = float(rng.uniform(0.01, 1.00))
            file_size_norm = float(rng.uniform(0.08, 1.00))
            cpu_density_norm = float(rng.uniform(0.05, 0.98))
            local_queue_fraction = float(rng.uniform(0.00, 1.00))
            neighbor_count_fraction = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
            cooperative_available = bool(rng.random() < 0.78)
            local_cache_hit = bool(rng.random() < 0.34)
            priority = int(rng.integers(config.SERVICE_PRIORITY_MIN, config.SERVICE_PRIORITY_MAX + 1))
            ue_uav_rate_log10 = float(rng.uniform(6.5, 8.4))
            uav_mbs_rate_log10 = float(rng.uniform(5.5, 8.2))
            best_uav_uav_rate_log10 = float(rng.uniform(6.0, 8.3))
            best_neighbor_mbs_rate_log10 = float(rng.uniform(5.5, 8.1))
            local_compute_share_norm = float(rng.uniform(0.12, 1.90))
            best_neighbor_compute_share_norm = float(rng.uniform(0.10, 2.10))
            best_neighbor_cache_belief = float(rng.uniform(0.05, 0.95))
        else:
            # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
            raise ValueError(f"Unsupported split: {split}")

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if not cooperative_available:
            neighbor_count_fraction = 0.0
            best_uav_uav_rate_log10 = 0.0
            best_neighbor_mbs_rate_log10 = 0.0
            best_neighbor_compute_share_norm = 0.0
            best_neighbor_cache_belief = 0.0

        request_size = float(config.MIN_INPUT_SIZE) + request_size_norm * request_size_span
        file_size = max(1.0, file_size_norm * max_file_size)
        cpu_density = cpu_min + cpu_density_norm * cpu_span
        cpu_cycles = request_size * cpu_density
        deadline = float(config.SERVICE_DEADLINE_MIN) + deadline_norm * float(config.SERVICE_DEADLINE_MAX - config.SERVICE_DEADLINE_MIN)
        ue_uav_rate = 10.0 ** ue_uav_rate_log10
        uav_mbs_rate = 10.0 ** uav_mbs_rate_log10
        best_uav_uav_rate = 10.0 ** best_uav_uav_rate_log10 if cooperative_available else float("inf")
        best_neighbor_mbs_rate = 10.0 ** best_neighbor_mbs_rate_log10 if cooperative_available else float("inf")
        local_compute_share = max(float(config.EPSILON), local_compute_share_norm * max_compute)
        best_neighbor_compute_share = max(float(config.EPSILON), best_neighbor_compute_share_norm * max_compute) if cooperative_available else float("inf")

        local_latency = (request_size / ue_uav_rate) + ((1.0 - float(local_cache_hit)) * file_size / uav_mbs_rate) + (cpu_cycles / local_compute_share)
        mbs_latency = (request_size / ue_uav_rate) + (request_size / uav_mbs_rate)
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if cooperative_available:
            cooperative_latency = (
                (request_size / ue_uav_rate)
                + (request_size / best_uav_uav_rate)
                + ((1.0 - best_neighbor_cache_belief) * file_size / best_neighbor_mbs_rate)
                + (cpu_cycles / best_neighbor_compute_share)
            )
        else:
            cooperative_latency = float("inf")

        context = ServiceOffloadContext(
            local_latency=float(local_latency),
            cooperative_latency=float(cooperative_latency),
            mbs_latency=float(mbs_latency),
            deadline=float(deadline),
            priority=int(priority),
            local_cache_hit=bool(local_cache_hit),
            cooperative_available=bool(cooperative_available),
            local_queue_length=int(round(local_queue_fraction * float(config.MAX_ASSOCIATED_UES))),
        )
        label = compute_heuristic_label(context)
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if collected[label] >= samples_per_class:
            continue

        full_vector = context_to_feature_vector(context)
        full_features.append(full_vector)
        latency_only.append(full_vector[:3])
        no_latency.append(full_vector[3:])
        rich_features.append(
            np.array(
                [
                    request_size_norm,
                    file_size_norm,
                    cpu_density_norm,
                    deadline_norm,
                    normalize_priority(priority),
                    float(local_cache_hit),
                    float(cooperative_available),
                    local_queue_fraction,
                    neighbor_count_fraction,
                    ue_uav_rate_log10,
                    uav_mbs_rate_log10,
                    best_uav_uav_rate_log10,
                    best_neighbor_mbs_rate_log10,
                    local_compute_share_norm,
                    best_neighbor_compute_share_norm,
                    best_neighbor_cache_belief,
                ],
                dtype=np.float32,
            )
        )
        labels.append(int(label))
        collected[label] += 1

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "full_features": np.stack(full_features, axis=0).astype(np.float32),
        "latency_only": np.stack(latency_only, axis=0).astype(np.float32),
        "no_latency_features": np.stack(no_latency, axis=0).astype(np.float32),
        "rich_reduced_features": np.stack(rich_features, axis=0).astype(np.float32),
    }


# 函数 evaluate_policy_on_arrays：关键函数，承载本模块的一段可复用实验逻辑，主要参数：policy, features, labels。
def evaluate_policy_on_arrays(
    policy: TrainedClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    predictions = np.asarray([policy.predict_from_array(row) for row in np.asarray(features, dtype=np.float32)], dtype=np.int64)
    confusion = np.zeros((OFFLOAD_NUM_CLASSES, OFFLOAD_NUM_CLASSES), dtype=np.int64)
    # 循环处理：遍历 (true_label, predicted_label) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for true_label, predicted_label in zip(labels, predictions, strict=False):
        confusion[int(true_label), int(predicted_label)] += 1

    per_class_metrics: dict[str, dict[str, float]] = {}
    # 循环处理：遍历 (class_idx, class_name) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for class_idx, class_name in enumerate(OFFLOAD_TARGET_NAMES):
        true_positive = int(confusion[class_idx, class_idx])
        false_positive = int(np.sum(confusion[:, class_idx]) - true_positive)
        false_negative = int(np.sum(confusion[class_idx, :]) - true_positive)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(confusion[class_idx, :])),
        }

    macro_precision = float(np.mean([item["precision"] for item in per_class_metrics.values()]))
    macro_recall = float(np.mean([item["recall"] for item in per_class_metrics.values()]))
    macro_f1 = float(np.mean([item["f1"] for item in per_class_metrics.values()]))
    accuracy = float(np.mean(predictions == labels))
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
    }


# 函数 evaluate_non_template_generalization：关键函数，承载本模块的一段可复用实验逻辑。
def evaluate_non_template_generalization(
    *,
    template_dataset: dict[str, np.ndarray],
    template_policies: dict[str, TrainedClassifier],
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    sampler_mode: str,
) -> tuple[dict[str, object], TrainedClassifier]:
    procedural_train = generate_procedural_contexts(split="train", samples_per_class=1800, seed=seed + 101)
    procedural_test = generate_procedural_contexts(split="test", samples_per_class=1000, seed=seed + 202)

    rich_candidate_train_features = np.concatenate(
        [
            np.asarray(template_dataset["rich_reduced_features"], dtype=np.float32),
            np.asarray(procedural_train["rich_reduced_features"], dtype=np.float32),
        ],
        axis=0,
    )
    rich_candidate_train_labels = np.concatenate(
        [
            np.asarray(template_dataset["labels"], dtype=np.int64),
            np.asarray(procedural_train["labels"], dtype=np.int64),
        ],
        axis=0,
    )
    reduced_candidate_policy, reduced_candidate_metrics = train_classifier_from_arrays(
        name="rich_reduced_candidate",
        feature_family="rich_reduced_features",
        train_features=rich_candidate_train_features,
        train_labels=rich_candidate_train_labels,
        test_features=np.asarray(procedural_test["rich_reduced_features"], dtype=np.float32),
        test_labels=np.asarray(procedural_test["labels"], dtype=np.int64),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        sampler_mode=sampler_mode,
        source_description="Reduced-leakage candidate trained on template rich features plus procedurally generated non-template rich features.",
    )

    report = {
        "generator": {
            "description": "Programmatic random-scenario evaluator with train/test parameter ranges intentionally shifted rather than template-reused.",
            "train_samples_per_class": 1800,
            "test_samples_per_class": 1000,
            "train_ranges": {
                "deadline_normalized": [0.12, 0.92],
                "request_size_normalized": [0.06, 0.85],
                "uav_mbs_rate_log10": [5.9, 7.7],
                "local_queue_fraction": [0.02, 0.78],
            },
            "test_ranges": {
                "deadline_normalized": [0.02, 1.0],
                "request_size_normalized": [0.01, 1.0],
                "uav_mbs_rate_log10": [5.5, 8.2],
                "local_queue_fraction": [0.0, 1.0],
            },
        },
        "surrogate_full": evaluate_policy_on_arrays(
            template_policies["full_features"],
            np.asarray(procedural_test["full_features"], dtype=np.float32),
            np.asarray(procedural_test["labels"], dtype=np.int64),
        ),
        "latency_only": evaluate_policy_on_arrays(
            template_policies["latency_only"],
            np.asarray(procedural_test["latency_only"], dtype=np.float32),
            np.asarray(procedural_test["labels"], dtype=np.int64),
        ),
        "no_latency_features": evaluate_policy_on_arrays(
            template_policies["no_latency_features"],
            np.asarray(procedural_test["no_latency_features"], dtype=np.float32),
            np.asarray(procedural_test["labels"], dtype=np.int64),
        ),
        "rich_reduced_template_only": evaluate_policy_on_arrays(
            template_policies["rich_reduced_features"],
            np.asarray(procedural_test["rich_reduced_features"], dtype=np.float32),
            np.asarray(procedural_test["labels"], dtype=np.int64),
        ),
        "rich_reduced_candidate": {
            "validation": simplify_metrics(reduced_candidate_metrics["validation"]),
            "test": simplify_metrics(reduced_candidate_metrics["test"]),
        },
    }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return report, reduced_candidate_policy


# 函数 resolve_policy_choice：关键函数，承载本模块的一段可复用实验逻辑，主要参数：target_idx, heuristic_target_idx, heuristic_target_uav, cooperative_uav。
def resolve_policy_choice(
    target_idx: int,
    heuristic_target_idx: int,
    heuristic_target_uav: "uav_module.UAV | None",
    cooperative_uav: "uav_module.UAV | None",
) -> tuple[int, "uav_module.UAV | None"]:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if target_idx == OFFLOAD_TARGET_LOCAL:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return OFFLOAD_TARGET_LOCAL, None
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if target_idx == OFFLOAD_TARGET_COOPERATIVE and cooperative_uav is not None:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return OFFLOAD_TARGET_COOPERATIVE, cooperative_uav
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if target_idx == OFFLOAD_TARGET_MBS:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return OFFLOAD_TARGET_MBS, None
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return heuristic_target_idx, heuristic_target_uav


# 函数 runtime_policy_mode：关键函数，承载本模块的一段可复用实验逻辑，主要参数：policy_mode。
@contextmanager
def runtime_policy_mode(
    policy_mode: str,
    *,
    surrogate_policy: TrainedClassifier | None = None,
    reduced_policy: TrainedClassifier | None = None,
):
    original_method = uav_module.UAV._select_service_offloading_target
    original_policy_name = config.SERVICE_OFFLOAD_POLICY
    original_checkpoint = getattr(config, "SERVICE_OFFLOAD_POLICY_CHECKPOINT", None)

    # 函数 patched_select_service_offloading_target：关键函数，承载本模块的一段可复用实验逻辑，主要参数：current_req, ue_uav_rate。
    def patched_select_service_offloading_target(
        self: "uav_module.UAV",
        current_req: Request,
        ue_uav_rate: float,
        *,
        context: ServiceOffloadContext | None = None,
        cooperative_uav: "uav_module.UAV | None" = None,
        heuristic_target_idx: int | None = None,
        heuristic_target_uav: "uav_module.UAV | None" = None,
    ) -> tuple[int, "uav_module.UAV | None"]:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if context is None or heuristic_target_idx is None:
            context, cooperative_uav = self._build_service_offload_context(current_req, ue_uav_rate)
            heuristic_target_idx, heuristic_target_uav = self._select_service_target_from_context(context, cooperative_uav)

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if policy_mode == "heuristic":
            self._service_heuristic_decision_count += 1
            # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
            return heuristic_target_idx, heuristic_target_uav

        # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
        try:
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if policy_mode == "surrogate":
                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if surrogate_policy is None:
                    # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
                    raise RuntimeError("Surrogate policy is unavailable.")
                target_idx = surrogate_policy.predict_from_context(context)
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            elif policy_mode == "reduced":
                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if reduced_policy is None:
                    # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
                    raise RuntimeError("Reduced policy is unavailable.")
                runtime_features = extract_rich_runtime_features(self, current_req, ue_uav_rate, context, cooperative_uav)
                target_idx = reduced_policy.predict_from_array(runtime_features)
            else:
                # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
                raise ValueError(f"Unsupported policy mode: {policy_mode}")
        except Exception:
            self._service_heuristic_decision_count += 1
            self._service_fallback_count += 1
            self._service_predict_exception_fallback_count += 1
            # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
            return heuristic_target_idx, heuristic_target_uav

        resolved_target_idx, resolved_target_uav = resolve_policy_choice(
            int(target_idx),
            int(heuristic_target_idx),
            heuristic_target_uav,
            cooperative_uav,
        )
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if (
            resolved_target_idx == int(target_idx)
            and (
                resolved_target_idx != OFFLOAD_TARGET_COOPERATIVE
                or cooperative_uav is not None
            )
        ):
            self._service_learned_decision_count += 1
        else:
            self._service_heuristic_decision_count += 1
            self._service_fallback_count += 1
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return resolved_target_idx, resolved_target_uav

    # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
    try:
        config.SERVICE_OFFLOAD_POLICY = "heuristic"
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None
        uav_module.UAV._select_service_offloading_target = patched_select_service_offloading_target  # type: ignore[assignment]
        yield
    finally:
        config.SERVICE_OFFLOAD_POLICY = original_policy_name
        config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = original_checkpoint
        uav_module.UAV._select_service_offloading_target = original_method  # type: ignore[assignment]


# 函数 run_system_comparison：关键函数，承载本模块的一段可复用实验逻辑。
def run_system_comparison(
    *,
    surrogate_policy: TrainedClassifier,
    reduced_policy: TrainedClassifier,
    seed: int,
    num_episodes: int,
    steps_per_episode: int,
) -> dict[str, dict[str, float]]:
    snapshot = snapshot_config()
    config.STEPS_PER_EPISODE = steps_per_episode
    static_model = StaticModel("static", config.NUM_UAVS, config.OBS_DIM_SINGLE, config.ACTION_DIM, "cpu")

    # 函数 evaluate_mode：关键函数，承载本模块的一段可复用实验逻辑，主要参数：policy_mode。
    def evaluate_mode(policy_mode: str) -> dict[str, float]:
        aggregated: dict[str, list[float]] = defaultdict(list)
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with runtime_policy_mode(
            policy_mode,
            surrogate_policy=surrogate_policy if policy_mode == "surrogate" else None,
            reduced_policy=reduced_policy if policy_mode == "reduced" else None,
        ):
            # 循环处理：遍历 episode 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for episode in range(num_episodes):
                np.random.seed(seed + episode)
                env = Env()
                env.reset(initial_positions=static_model.static_positions)
                per_episode: dict[str, float] = defaultdict(float)

                # 循环处理：遍历 _ 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
                for _ in range(steps_per_episode):
                    actions = np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
                    _, _, metrics = env.step(actions)
                    # 循环处理：遍历 metric_name 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
                    for metric_name in (
                        "latency",
                        "energy",
                        "deadline_satisfaction_rate",
                        "offloading_ratio_local",
                        "offloading_ratio_cooperative",
                        "offloading_ratio_mbs",
                        "mbs_load_ratio",
                    ):
                        per_episode[metric_name] += float(metrics[metric_name])

                # 循环处理：遍历 (metric_name, total_value) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
                for metric_name, total_value in per_episode.items():
                    aggregated[metric_name].append(total_value / float(steps_per_episode))

        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {metric_name: float(np.mean(values)) for metric_name, values in aggregated.items()}

    # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
    try:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {
            "heuristic_offloading": evaluate_mode("heuristic"),
            "surrogate_classifier": evaluate_mode("surrogate"),
            "reduced_leakage_policy": evaluate_mode("reduced"),
        }
    finally:
        restore_config(snapshot)


# 函数 compute_system_deltas：关键函数，承载本模块的一段可复用实验逻辑，主要参数：system_results。
def compute_system_deltas(system_results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    heuristic = system_results["heuristic_offloading"]
    deltas: dict[str, dict[str, float]] = {}
    # 循环处理：遍历 (policy_name, metrics) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for policy_name, metrics in system_results.items():
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if policy_name == "heuristic_offloading":
            continue
        deltas[policy_name] = {metric_name: float(metrics[metric_name] - heuristic[metric_name]) for metric_name in heuristic}
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return deltas


# 函数 classify_policy_version：关键函数，承载本模块的一段可复用实验逻辑。
def classify_policy_version(
    *,
    non_template_macro_f1: float,
    system_latency_delta: float,
    system_deadline_delta: float,
    uses_latency_proxies: bool,
) -> str:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if uses_latency_proxies:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return "heuristic surrogate baseline"
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if (
        non_template_macro_f1 >= 0.75
        and (abs(system_latency_delta) > 1e-6 or abs(system_deadline_delta) > 1e-6)
        and system_deadline_delta >= -0.01
        and system_latency_delta <= 1.0
    ):
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return "partially generalizable learned policy"
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return "not yet sufficient for paper main claim"


# 函数 build_markdown_summary：实验汇总信息，最终写入报告或 manifest 文件，主要参数：results。
def build_markdown_summary(results: dict[str, object]) -> str:
    template_eval = results["template_eval"]
    non_template_eval = results["non_template_eval"]
    system_results = results["system_comparison"]
    system_deltas = results["system_delta_vs_heuristic"]
    final_judgment = results["final_judgment"]

    # 函数 fmt：关键函数，承载本模块的一段可复用实验逻辑，主要参数：value。
    def fmt(value: float) -> str:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return f"{value:.4f}"

    lines: list[str] = []
    lines.append("# Offloading Policy Validity Check")
    lines.append("")
    lines.append("## Core Answers")
    lines.append(f"1. {results['answer_q1']}")
    lines.append(f"2. {results['answer_q2']}")
    lines.append(f"3. {results['answer_q3']}")
    lines.append("")
    lines.append("## Template Held-Out Evaluation")
    lines.append(f"- Train scenarios: {', '.join(template_eval['train_scenarios'])}")
    lines.append(f"- Test scenarios: {', '.join(template_eval['test_scenarios'])}")
    # 循环处理：遍历 (variant_name, variant_result) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for variant_name, variant_result in template_eval["variants"].items():
        iid_metrics = variant_result["iid_random_split"]["test"]
        held_metrics = variant_result["held_out_template_split"]["test"]
        lines.append(
            f"- {variant_name}: IID acc={fmt(iid_metrics['accuracy'])}, IID macro_F1={fmt(iid_metrics['macro_f1'])}, "
            f"held-out acc={fmt(held_metrics['accuracy'])}, held-out macro_F1={fmt(held_metrics['macro_f1'])}"
        )
    lines.append("")
    lines.append("## Non-Template Evaluation")
    # 循环处理：遍历 variant_name 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for variant_name in ("surrogate_full", "latency_only", "no_latency_features", "rich_reduced_template_only"):
        metrics = non_template_eval[variant_name]
        lines.append(f"- {variant_name}: acc={fmt(metrics['accuracy'])}, macro_F1={fmt(metrics['macro_f1'])}")
    reduced_candidate_metrics = non_template_eval["rich_reduced_candidate"]["test"]
    lines.append(
        f"- rich_reduced_candidate: acc={fmt(reduced_candidate_metrics['accuracy'])}, "
        f"macro_F1={fmt(reduced_candidate_metrics['macro_f1'])}"
    )
    lines.append("")
    lines.append("## Fixed-Trajectory System Comparison")
    lines.append("| Policy | Latency | Energy | Deadline Sat | Offload Local | Offload Coop | Offload MBS | MBS Load |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    # 循环处理：遍历 (policy_name, metrics) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for policy_name, metrics in system_results.items():
        lines.append(
            f"| {policy_name} | {fmt(metrics['latency'])} | {fmt(metrics['energy'])} | "
            f"{fmt(metrics['deadline_satisfaction_rate'])} | {fmt(metrics['offloading_ratio_local'])} | "
            f"{fmt(metrics['offloading_ratio_cooperative'])} | {fmt(metrics['offloading_ratio_mbs'])} | "
            f"{fmt(metrics['mbs_load_ratio'])} |"
        )
    lines.append("")
    lines.append("## Delta Vs Heuristic")
    # 循环处理：遍历 (policy_name, metrics) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for policy_name, metrics in system_deltas.items():
        lines.append(
            f"- {policy_name}: latency={fmt(metrics['latency'])}, energy={fmt(metrics['energy'])}, "
            f"deadline_satisfaction_rate={fmt(metrics['deadline_satisfaction_rate'])}, "
            f"offloading_ratio_local={fmt(metrics['offloading_ratio_local'])}, "
            f"offloading_ratio_cooperative={fmt(metrics['offloading_ratio_cooperative'])}, "
            f"offloading_ratio_mbs={fmt(metrics['offloading_ratio_mbs'])}, "
            f"mbs_load_ratio={fmt(metrics['mbs_load_ratio'])}"
        )
    lines.append("")
    lines.append("## Final Judgment")
    lines.append(f"- Surrogate baseline: {final_judgment['surrogate_baseline']}")
    lines.append(f"- Reduced-leakage candidate: {final_judgment['reduced_candidate']}")
    lines.append(f"- Recommended paper baseline: {final_judgment['recommended_baseline']}")
    lines.append(f"- Recommended paper candidate: {final_judgment['recommended_candidate']}")
    lines.append(f"- Remaining key issues: {final_judgment['remaining_key_issues']}")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return "\n".join(lines)


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main() -> None:
    seed = 20260423
    template_epochs = 16
    reduced_candidate_epochs = 18
    batch_size = 256
    learning_rate = 1e-3
    device = "auto"
    sampler_mode = "auto"

    template_dataset = collect_template_rich_dataset(per_class_target=1500, seed=seed)
    template_eval, template_policies = evaluate_feature_variants_on_template(
        template_dataset,
        seed=seed,
        epochs=template_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        sampler_mode=sampler_mode,
    )
    non_template_eval, reduced_candidate_policy = evaluate_non_template_generalization(
        template_dataset=template_dataset,
        template_policies=template_policies,
        seed=seed,
        epochs=reduced_candidate_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        sampler_mode=sampler_mode,
    )
    system_results = run_system_comparison(
        surrogate_policy=template_policies["full_features"],
        reduced_policy=reduced_candidate_policy,
        seed=seed,
        num_episodes=6,
        steps_per_episode=100,
    )
    system_deltas = compute_system_deltas(system_results)

    surrogate_non_template_f1 = float(non_template_eval["surrogate_full"]["macro_f1"])
    latency_only_non_template_f1 = float(non_template_eval["latency_only"]["macro_f1"])
    no_latency_non_template_f1 = float(non_template_eval["no_latency_features"]["macro_f1"])
    rich_reduced_candidate_non_template_f1 = float(non_template_eval["rich_reduced_candidate"]["test"]["macro_f1"])

    answer_q1 = (
        "Current top-line accuracy is still heavily driven by heuristic copying. "
        f"The full surrogate reaches non-template macro_F1={surrogate_non_template_f1:.3f}, and a latency-only diagnostic remains at "
        f"macro_F1={latency_only_non_template_f1:.3f}, while the current 5-field no-latency baseline drops to "
        f"macro_F1={no_latency_non_template_f1:.3f}. "
        "This means a large fraction of the performance comes from features that sit very close to the heuristic scoring rule itself."
    )
    answer_q2 = (
        "The current classifier family does not collapse completely on programmatically generated non-template scenarios, but generalization quality depends on the feature set. "
        f"The reduced-leakage rich-state candidate reaches macro_F1={rich_reduced_candidate_non_template_f1:.3f} on the held-out non-template split, "
        "which is meaningfully stronger than the current no-latency baseline and therefore shows some independent offline generalization value."
    )
    answer_q3 = (
        "Under a fixed static trajectory policy, the surrogate classifier changes the offloading mix but hurts latency, energy, and deadline satisfaction relative to heuristic. "
        "The reduced-leakage candidate does not currently create a measurable end-to-end separation from heuristic in the reference environment, so the current evidence is still stronger for offline generalization than for system-level superiority."
    )

    surrogate_label = classify_policy_version(
        non_template_macro_f1=surrogate_non_template_f1,
        system_latency_delta=float(system_deltas["surrogate_classifier"]["latency"]),
        system_deadline_delta=float(system_deltas["surrogate_classifier"]["deadline_satisfaction_rate"]),
        uses_latency_proxies=True,
    )
    reduced_label = classify_policy_version(
        non_template_macro_f1=rich_reduced_candidate_non_template_f1,
        system_latency_delta=float(system_deltas["reduced_leakage_policy"]["latency"]),
        system_deadline_delta=float(system_deltas["reduced_leakage_policy"]["deadline_satisfaction_rate"]),
        uses_latency_proxies=False,
    )

    final_judgment = {
        "surrogate_baseline": surrogate_label,
        "reduced_candidate": reduced_label,
        "recommended_baseline": "full-feature classifier explicitly labeled as a heuristic surrogate / distillation baseline",
        "recommended_candidate": "rich reduced-leakage policy trained on template plus non-template procedural raw-state data",
        "remaining_key_issues": (
            "1. The reduced-leakage policy still lacks a measurable end-to-end system win over heuristic under fixed trajectories, and in the current runtime evaluation it effectively collapses back to heuristic-like behavior. "
            "2. There is still a deployment gap between the richer external validation features and the raw state officially exposed by the repository offloading module, so a paper main claim would benefit from promoting richer raw features into the official module rather than leaving them only in an external validator."
        ),
    }

    results: dict[str, object] = {
        "metadata": {
            "seed": seed,
            "template_epochs": template_epochs,
            "reduced_candidate_epochs": reduced_candidate_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "feature_families": build_feature_family_metadata(),
            "note": "System metrics are reported as per-step ratios averaged over each episode, then averaged across episodes, matching the current train/test metric semantics in the reference repository.",
        },
        "template_eval": template_eval,
        "non_template_eval": non_template_eval,
        "system_comparison": system_results,
        "system_delta_vs_heuristic": system_deltas,
        "answer_q1": answer_q1,
        "answer_q2": answer_q2,
        "answer_q3": answer_q3,
        "final_judgment": final_judgment,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text(build_markdown_summary(results), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
