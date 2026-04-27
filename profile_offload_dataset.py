"""
中文注释说明：profile_offload_dataset.py

文件作用：
    统计任务卸载数据集的类别分布、特征范围和样本质量，帮助判断数据集是否适合训练。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - _feature_stats(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _collect_feature_summary(): 实验汇总信息，最终写入报告或 manifest 文件。
    - _diagnose_dataset(): 任务卸载监督学习数据集或样本集合。
    - profile_offload_dataset(): 任务卸载监督学习数据集或样本集合。
    - add_reference_comparison(): 关键函数，承载本模块的一段可复用实验逻辑。
    - parse_args(): 解析命令行参数，并为实验脚本提供可覆盖的默认配置。
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    argparse, json, pathlib, numpy, marl_models

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marl_models.offload_policy import OFFLOAD_NUM_CLASSES, OFFLOAD_TARGET_MBS, OFFLOAD_TARGET_NAMES, get_offload_feature_specs


# 函数 _feature_stats：关键函数，承载本模块的一段可复用实验逻辑，主要参数：values。
def _feature_stats(values: np.ndarray) -> dict[str, float]:
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if values.size == 0:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
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


# 函数 _collect_feature_summary：实验汇总信息，最终写入报告或 manifest 文件，主要参数：features, labels。
def _collect_feature_summary(features: np.ndarray, labels: np.ndarray) -> tuple[dict[str, dict[str, object]], list[str]]:
    feature_specs = get_offload_feature_specs()
    feature_summary: dict[str, dict[str, object]] = {}
    low_signal_features: list[str] = []

    # 循环处理：遍历 (feature_idx, spec) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for feature_idx, spec in enumerate(feature_specs):
        feature_name = spec["name"]
        values = features[:, feature_idx]
        unique_values, unique_counts = np.unique(np.round(values, decimals=6), return_counts=True)
        dominant_ratio: float = float(np.max(unique_counts) / max(values.size, 1))
        global_stats = _feature_stats(values)
        per_class: dict[str, dict[str, float]] = {}
        # 循环处理：遍历 class_idx 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
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
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if global_stats["std"] < 1e-4 or value_range < 1e-4 or dominant_ratio > 0.98:
            low_signal_features.append(feature_name)

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return feature_summary, low_signal_features


# 函数 _diagnose_dataset：任务卸载监督学习数据集或样本集合，主要参数：features, labels。
def _diagnose_dataset(features: np.ndarray, labels: np.ndarray) -> list[str]:
    diagnostics: list[str] = []
    total_samples: int = int(labels.size)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if total_samples == 0:
        diagnostics.append("Dataset is empty.")
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return diagnostics

    class_counts = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES)
    class_ratios = class_counts / float(total_samples)
    missing_classes = [OFFLOAD_TARGET_NAMES[idx] for idx in range(OFFLOAD_NUM_CLASSES) if class_counts[idx] == 0]
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if missing_classes:
        diagnostics.append(f"Label collapse detected: missing classes = {missing_classes}.")
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif np.max(class_ratios) > 0.8:
        dominant_class = OFFLOAD_TARGET_NAMES[int(np.argmax(class_counts))]
        diagnostics.append(
            f"Strong class imbalance detected: dominant class '{dominant_class}' accounts for {float(np.max(class_ratios)):.3f} of samples."
        )

    latency_features = features[:, :3]
    latency_argmin = np.argmin(latency_features, axis=1)
    latency_argmin_counts = np.bincount(latency_argmin, minlength=OFFLOAD_NUM_CLASSES)
    dominant_latency_class = int(np.argmax(latency_argmin_counts))
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if latency_argmin_counts[dominant_latency_class] / max(total_samples, 1) > 0.9:
        diagnostics.append(
            f"The normalized latency proxy for '{OFFLOAD_TARGET_NAMES[dominant_latency_class]}' is the smallest in "
            f"{float(latency_argmin_counts[dominant_latency_class] / total_samples):.3f} of samples."
        )

    dominant_latency_ratio = float(latency_argmin_counts[dominant_latency_class] / total_samples)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if dominant_latency_class == OFFLOAD_TARGET_MBS and dominant_latency_ratio > 0.6:
        diagnostics.append(
            "MBS latency proxy is consistently smaller than local/cooperative latency proxies, "
            "which strongly pushes the heuristic toward MBS."
        )

    cooperative_feature = features[:, 6]
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if float(np.std(cooperative_feature)) < 1e-6:
        diagnostics.append("Feature 'cooperative_available' is effectively constant in this dataset.")

    queue_feature = features[:, 7]
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if float(np.mean(queue_feature)) < 0.15:
        diagnostics.append(
            "Local queue load stays low in most samples, so queue-relief driven cooperative decisions are rarely activated."
        )

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return diagnostics


# 函数 profile_offload_dataset：任务卸载监督学习数据集或样本集合，主要参数：dataset_path。
def profile_offload_dataset(dataset_path: str | Path) -> dict[str, object]:
    dataset_file = Path(dataset_path)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not dataset_file.exists():
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
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
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
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
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if scenario_summary:
        profile["scenario_counts"] = scenario_summary
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return profile


# 函数 add_reference_comparison：关键函数，承载本模块的一段可复用实验逻辑，主要参数：profile, reference_profile。
def add_reference_comparison(profile: dict[str, object], reference_profile: dict[str, object]) -> dict[str, object]:
    comparison = {
        "reference_dataset_path": reference_profile["dataset_path"],
        "target_dataset_path": profile["dataset_path"],
        "class_count_delta": {},
        "class_ratio_delta": {},
    }
    # 循环处理：遍历 class_name 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for class_name in OFFLOAD_TARGET_NAMES:
        comparison["class_count_delta"][class_name] = int(profile["class_counts"][class_name]) - int(reference_profile["class_counts"][class_name])
        comparison["class_ratio_delta"][class_name] = float(profile["class_ratios"][class_name]) - float(reference_profile["class_ratios"][class_name])
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return comparison


# 函数 parse_args：解析命令行参数，并为实验脚本提供可覆盖的默认配置。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile an offload dataset and save a short JSON report.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset .npz file.")
    parser.add_argument("--output", type=str, default=None, help="Optional output profile path. Defaults next to the dataset.")
    parser.add_argument("--reference", type=str, default=None, help="Optional reference dataset for comparison.")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return parser.parse_args()


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main() -> None:
    args = parse_args()
    profile = profile_offload_dataset(args.dataset)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if args.reference is not None:
        reference_profile = profile_offload_dataset(args.reference)
        profile["comparison_to_reference"] = add_reference_comparison(profile, reference_profile)

    output_path = Path(args.output) if args.output is not None else Path(args.dataset).with_suffix(".profile.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
