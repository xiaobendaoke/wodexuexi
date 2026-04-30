"""
中文注释说明：marl_models/offload_policy.py

文件作用：
    实现任务卸载策略模型和特征处理逻辑，为运行时卸载决策提供监督学习分类器。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - OFFLOAD_TARGET_LOCAL: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_TARGET_COOPERATIVE: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_TARGET_MBS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_NUM_CLASSES: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_TARGET_NAMES: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_LATENCY_RATIO_CLIP: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_COMPUTE_SHARE_CLIP: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_FEATURE_FAMILY_FULL: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_FEATURE_FAMILY_RICH_REDUCED: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - FULL_OFFLOAD_FEATURE_SPECS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - RICH_REDUCED_FEATURE_SPECS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_FEATURE_SPECS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_FEATURE_FAMILIES: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_POLICY_INPUT_DIM: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - OFFLOAD_POLICY_RICH_INPUT_DIM: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - ServiceOffloadContext: 核心类，封装本模块中的主要状态和行为。
    - OffloadFeatureSpec: 核心类，封装本模块中的主要状态和行为。
    - OffloadMLP: 核心类，封装本模块中的主要状态和行为。
    - LearnedClassifierOffloadPolicy: 核心类，封装本模块中的主要状态和行为。
    - get_offload_feature_specs(): 关键函数，承载本模块的一段可复用实验逻辑。
    - get_offload_feature_names(): 关键函数，承载本模块的一段可复用实验逻辑。
    - get_offload_feature_dim(): 关键函数，承载本模块的一段可复用实验逻辑。
    - get_offload_label_mapping(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _normalize_ratio(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _normalize_deadline(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _normalize_priority(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _normalize_request_size(): 用户设备产生的任务请求。
    - _normalize_file_size(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _normalize_cpu_density(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _normalize_local_queue_fraction(): 用户设备对象，产生任务请求并等待服务。
    - _normalize_neighbor_count(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _normalize_compute_share(): 关键函数，承载本模块的一段可复用实验逻辑。
    - _safe_log10(): 关键函数，承载本模块的一段可复用实验逻辑。

主要依赖：
    dataclasses, pathlib, typing, numpy, torch, config

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import config


# 关键变量 OFFLOAD_TARGET_LOCAL：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_TARGET_LOCAL: int = 0
# 关键变量 OFFLOAD_TARGET_COOPERATIVE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_TARGET_COOPERATIVE: int = 1
# 关键变量 OFFLOAD_TARGET_MBS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_TARGET_MBS: int = 2
# 关键变量 OFFLOAD_NUM_CLASSES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_NUM_CLASSES: int = 3
# 关键变量 OFFLOAD_TARGET_NAMES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_TARGET_NAMES: tuple[str, ...] = ("local", "cooperative", "mbs")
# 关键变量 OFFLOAD_LATENCY_RATIO_CLIP：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_LATENCY_RATIO_CLIP: float = 10.0
# 关键变量 OFFLOAD_COMPUTE_SHARE_CLIP：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_COMPUTE_SHARE_CLIP: float = 4.0
# 关键变量 OFFLOAD_FEATURE_FAMILY_FULL：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_FEATURE_FAMILY_FULL: str = "full_features"
# 关键变量 OFFLOAD_FEATURE_FAMILY_RICH_REDUCED：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_FEATURE_FAMILY_RICH_REDUCED: str = "rich_reduced_features"


# 类 ServiceOffloadContext：核心类，封装本模块中的主要状态和行为。
@dataclass(slots=True)
class ServiceOffloadContext:
    """Compact service-request context passed to the pluggable offload policy."""

    local_latency: float
    cooperative_latency: float
    mbs_latency: float
    deadline: float
    priority: int
    local_cache_hit: bool
    cooperative_available: bool
    local_queue_length: int
    request_size: int = 0
    service_file_size: int = 0
    cpu_cycles_per_byte: float = 0.0
    neighbor_count: int = 0
    ue_uav_rate: float = 0.0
    uav_mbs_rate: float = 0.0
    best_uav_uav_rate: float = 0.0
    best_neighbor_mbs_rate: float = 0.0
    local_compute_share: float = 0.0
    best_neighbor_compute_share: float = 0.0
    best_neighbor_cache_belief: float = 0.0


# 类 OffloadFeatureSpec：核心类，封装本模块中的主要状态和行为。
@dataclass(frozen=True, slots=True)
class OffloadFeatureSpec:
    name: str
    description: str
    normalization: str


# 关键变量 FULL_OFFLOAD_FEATURE_SPECS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
FULL_OFFLOAD_FEATURE_SPECS: tuple[OffloadFeatureSpec, ...] = (
    OffloadFeatureSpec(
        name="local_latency_vs_deadline",
        description="Estimated local service latency relative to the request deadline.",
        normalization=f"min(local_latency / deadline, {OFFLOAD_LATENCY_RATIO_CLIP}) / {OFFLOAD_LATENCY_RATIO_CLIP}",
    ),
    OffloadFeatureSpec(
        name="cooperative_latency_vs_deadline",
        description="Estimated best cooperative-UAV service latency relative to the request deadline.",
        normalization=(
            f"min(cooperative_latency / deadline, {OFFLOAD_LATENCY_RATIO_CLIP}) / {OFFLOAD_LATENCY_RATIO_CLIP}; "
            "uses 1.0 when no cooperative UAV is available"
        ),
    ),
    OffloadFeatureSpec(
        name="mbs_latency_vs_deadline",
        description="Estimated MBS offloading latency relative to the request deadline.",
        normalization=f"min(mbs_latency / deadline, {OFFLOAD_LATENCY_RATIO_CLIP}) / {OFFLOAD_LATENCY_RATIO_CLIP}",
    ),
    OffloadFeatureSpec(
        name="deadline_normalized",
        description="Service request deadline tightness.",
        normalization="(deadline - SERVICE_DEADLINE_MIN) / (SERVICE_DEADLINE_MAX - SERVICE_DEADLINE_MIN)",
    ),
    OffloadFeatureSpec(
        name="priority_normalized",
        description="Service request priority level.",
        normalization="(priority - SERVICE_PRIORITY_MIN) / (SERVICE_PRIORITY_MAX - SERVICE_PRIORITY_MIN)",
    ),
    OffloadFeatureSpec(
        name="local_cache_hit",
        description="Whether the local UAV already caches the requested service file.",
        normalization="binary in {0, 1}",
    ),
    OffloadFeatureSpec(
        name="cooperative_available",
        description="Whether at least one cooperative UAV candidate is currently available.",
        normalization="binary in {0, 1}",
    ),
    OffloadFeatureSpec(
        name="local_queue_fraction",
        description="Current local service queue length before this decision.",
        normalization="min(local_queue_length / MAX_ASSOCIATED_UES, 1.0)",
    ),
)
# 关键变量 RICH_REDUCED_FEATURE_SPECS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
RICH_REDUCED_FEATURE_SPECS: tuple[OffloadFeatureSpec, ...] = (
    OffloadFeatureSpec(
        name="request_size_normalized",
        description="Normalized service request input size.",
        normalization="(request_size - MIN_INPUT_SIZE) / (MAX_INPUT_SIZE - MIN_INPUT_SIZE)",
    ),
    OffloadFeatureSpec(
        name="service_file_size_normalized",
        description="Requested service file size relative to the largest file in the catalog.",
        normalization="service_file_size / max(FILE_SIZES)",
    ),
    OffloadFeatureSpec(
        name="cpu_density_normalized",
        description="Requested service compute density relative to the configured service range.",
        normalization="(cpu_cycles_per_byte - min(CPU_CYCLES_PER_BYTE)) / (max(CPU_CYCLES_PER_BYTE) - min(CPU_CYCLES_PER_BYTE))",
    ),
    OffloadFeatureSpec(
        name="deadline_normalized",
        description="Service request deadline tightness.",
        normalization="(deadline - SERVICE_DEADLINE_MIN) / (SERVICE_DEADLINE_MAX - SERVICE_DEADLINE_MIN)",
    ),
    OffloadFeatureSpec(
        name="priority_normalized",
        description="Service request priority level.",
        normalization="(priority - SERVICE_PRIORITY_MIN) / (SERVICE_PRIORITY_MAX - SERVICE_PRIORITY_MIN)",
    ),
    OffloadFeatureSpec(
        name="local_cache_hit",
        description="Whether the local UAV already caches the requested service file.",
        normalization="binary in {0, 1}",
    ),
    OffloadFeatureSpec(
        name="cooperative_available",
        description="Whether at least one cooperative UAV candidate is currently available.",
        normalization="binary in {0, 1}",
    ),
    OffloadFeatureSpec(
        name="local_queue_fraction",
        description="Current local service queue length before this decision.",
        normalization="min(local_queue_length / MAX_ASSOCIATED_UES, 1.0)",
    ),
    OffloadFeatureSpec(
        name="neighbor_count_fraction",
        description="Fraction of currently reachable neighbor UAVs.",
        normalization="neighbor_count / max(NUM_UAVS - 1, 1)",
    ),
    OffloadFeatureSpec(
        name="ue_uav_rate_log10",
        description="Log-scale uplink rate from UE to the serving UAV.",
        normalization="log10(max(ue_uav_rate, EPSILON))",
    ),
    OffloadFeatureSpec(
        name="uav_mbs_rate_log10",
        description="Log-scale backhaul rate from the source UAV to the MBS.",
        normalization="log10(max(uav_mbs_rate, EPSILON))",
    ),
    OffloadFeatureSpec(
        name="best_uav_uav_rate_log10",
        description="Log-scale inter-UAV rate to the best cooperative neighbor.",
        normalization="log10(max(best_uav_uav_rate, EPSILON)); 0 when no cooperative UAV is available",
    ),
    OffloadFeatureSpec(
        name="best_neighbor_mbs_rate_log10",
        description="Log-scale backhaul rate from the best cooperative neighbor to the MBS.",
        normalization="log10(max(best_neighbor_mbs_rate, EPSILON)); 0 when no cooperative UAV is available",
    ),
    OffloadFeatureSpec(
        name="local_compute_share_normalized",
        description="Current local compute share available to this request.",
        normalization=f"clip(local_compute_share / max(UAV_COMPUTING_CAPACITY), 0, {OFFLOAD_COMPUTE_SHARE_CLIP})",
    ),
    OffloadFeatureSpec(
        name="best_neighbor_compute_share_normalized",
        description="Current compute share available at the best cooperative neighbor.",
        normalization=f"clip(best_neighbor_compute_share / max(UAV_COMPUTING_CAPACITY), 0, {OFFLOAD_COMPUTE_SHARE_CLIP}); 0 when unavailable",
    ),
    OffloadFeatureSpec(
        name="best_neighbor_cache_belief",
        description="Belief probability that the best cooperative neighbor already caches the requested service.",
        normalization="binary-probability in [0, 1]",
    ),
)
# 关键变量 OFFLOAD_FEATURE_SPECS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_FEATURE_SPECS: tuple[OffloadFeatureSpec, ...] = FULL_OFFLOAD_FEATURE_SPECS
# 关键变量 OFFLOAD_FEATURE_FAMILIES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_FEATURE_FAMILIES: dict[str, tuple[OffloadFeatureSpec, ...]] = {
    OFFLOAD_FEATURE_FAMILY_FULL: FULL_OFFLOAD_FEATURE_SPECS,
    OFFLOAD_FEATURE_FAMILY_RICH_REDUCED: RICH_REDUCED_FEATURE_SPECS,
}
# 关键变量 OFFLOAD_POLICY_INPUT_DIM：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_POLICY_INPUT_DIM: int = len(FULL_OFFLOAD_FEATURE_SPECS)
# 关键变量 OFFLOAD_POLICY_RICH_INPUT_DIM：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_POLICY_RICH_INPUT_DIM: int = len(RICH_REDUCED_FEATURE_SPECS)


# 函数 get_offload_feature_specs：关键函数，承载本模块的一段可复用实验逻辑，主要参数：feature_family。
def get_offload_feature_specs(feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL) -> list[dict[str, str]]:
    """Return JSON-serializable feature metadata for dataset artifacts."""

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return [asdict(spec) for spec in OFFLOAD_FEATURE_FAMILIES[feature_family]]


# 函数 get_offload_feature_names：关键函数，承载本模块的一段可复用实验逻辑，主要参数：feature_family。
def get_offload_feature_names(feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL) -> list[str]:
    """Return the canonical ordered feature-name schema for a feature family."""

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return [spec.name for spec in OFFLOAD_FEATURE_FAMILIES[feature_family]]


# 函数 get_offload_feature_dim：关键函数，承载本模块的一段可复用实验逻辑，主要参数：feature_family。
def get_offload_feature_dim(feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL) -> int:
    """Return the input dimension for a given feature family."""

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return len(OFFLOAD_FEATURE_FAMILIES[feature_family])


# 函数 get_offload_label_mapping：关键函数，承载本模块的一段可复用实验逻辑。
def get_offload_label_mapping() -> dict[int, str]:
    """Return the canonical offloading label mapping."""

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {label: OFFLOAD_TARGET_NAMES[label] for label in range(OFFLOAD_NUM_CLASSES)}


# 函数 _normalize_ratio：关键函数，承载本模块的一段可复用实验逻辑，主要参数：numerator, denominator。
def _normalize_ratio(numerator: float, denominator: float) -> float:
    safe_denominator: float = max(float(denominator), float(config.EPSILON))
    raw_ratio: float = float(numerator) / safe_denominator
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not np.isfinite(raw_ratio):
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return 1.0
    clipped_ratio: float = float(np.clip(raw_ratio, 0.0, OFFLOAD_LATENCY_RATIO_CLIP))
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return clipped_ratio / OFFLOAD_LATENCY_RATIO_CLIP


# 函数 _normalize_deadline：关键函数，承载本模块的一段可复用实验逻辑，主要参数：deadline。
def _normalize_deadline(deadline: float) -> float:
    span: float = float(config.SERVICE_DEADLINE_MAX - config.SERVICE_DEADLINE_MIN)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if span <= float(config.EPSILON):
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return 0.0
    normalized: float = (float(deadline) - float(config.SERVICE_DEADLINE_MIN)) / span
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip(normalized, 0.0, 1.0))


# 函数 _normalize_priority：关键函数，承载本模块的一段可复用实验逻辑，主要参数：priority。
def _normalize_priority(priority: int) -> float:
    span: int = int(config.SERVICE_PRIORITY_MAX - config.SERVICE_PRIORITY_MIN)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if span <= 0:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return 0.0
    normalized: float = (float(priority) - float(config.SERVICE_PRIORITY_MIN)) / float(span)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip(normalized, 0.0, 1.0))


# 函数 _normalize_request_size：用户设备产生的任务请求，主要参数：request_size。
def _normalize_request_size(request_size: int) -> float:
    span: float = float(config.MAX_INPUT_SIZE - config.MIN_INPUT_SIZE)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if span <= float(config.EPSILON):
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return 0.0
    normalized: float = (float(request_size) - float(config.MIN_INPUT_SIZE)) / span
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip(normalized, 0.0, 1.0))


# 函数 _normalize_file_size：关键函数，承载本模块的一段可复用实验逻辑，主要参数：file_size。
def _normalize_file_size(file_size: int) -> float:
    max_file_size: float = max(float(np.max(config.FILE_SIZES)), 1.0)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip(float(file_size) / max_file_size, 0.0, 1.0))


# 函数 _normalize_cpu_density：关键函数，承载本模块的一段可复用实验逻辑，主要参数：cpu_cycles_per_byte。
def _normalize_cpu_density(cpu_cycles_per_byte: float) -> float:
    cpu_min: float = float(np.min(config.CPU_CYCLES_PER_BYTE))
    cpu_max: float = float(np.max(config.CPU_CYCLES_PER_BYTE))
    span: float = max(cpu_max - cpu_min, float(config.EPSILON))
    normalized: float = (float(cpu_cycles_per_byte) - cpu_min) / span
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip(normalized, 0.0, 1.0))


# 函数 _normalize_local_queue_fraction：用户设备对象，产生任务请求并等待服务，主要参数：local_queue_length。
def _normalize_local_queue_fraction(local_queue_length: int) -> float:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip(local_queue_length / max(float(config.MAX_ASSOCIATED_UES), 1.0), 0.0, 1.0))


# 函数 _normalize_neighbor_count：关键函数，承载本模块的一段可复用实验逻辑，主要参数：neighbor_count。
def _normalize_neighbor_count(neighbor_count: int) -> float:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip(neighbor_count / max(float(config.NUM_UAVS - 1), 1.0), 0.0, 1.0))


# 函数 _normalize_compute_share：关键函数，承载本模块的一段可复用实验逻辑，主要参数：compute_share。
def _normalize_compute_share(compute_share: float) -> float:
    max_capacity: float = max(float(np.max(config.UAV_COMPUTING_CAPACITY)), 1.0)
    normalized: float = float(compute_share) / max_capacity
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.clip(normalized, 0.0, OFFLOAD_COMPUTE_SHARE_CLIP))


# 函数 _safe_log10：关键函数，承载本模块的一段可复用实验逻辑，主要参数：value。
def _safe_log10(value: float) -> float:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.log10(max(float(value), float(config.EPSILON))))


# 函数 context_to_feature_vector：关键函数，承载本模块的一段可复用实验逻辑，主要参数：context, feature_family。
def context_to_feature_vector(
    context: ServiceOffloadContext,
    feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL,
) -> np.ndarray:
    """Convert a service offloading context into a normalized feature vector."""

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_family == OFFLOAD_FEATURE_FAMILY_FULL:
        cooperative_latency_feature: float = 1.0
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if context.cooperative_available:
            cooperative_latency_feature = _normalize_ratio(context.cooperative_latency, context.deadline)

        feature_vector = np.array(
            [
                _normalize_ratio(context.local_latency, context.deadline),
                cooperative_latency_feature,
                _normalize_ratio(context.mbs_latency, context.deadline),
                _normalize_deadline(context.deadline),
                _normalize_priority(context.priority),
                float(context.local_cache_hit),
                float(context.cooperative_available),
                _normalize_local_queue_fraction(context.local_queue_length),
            ],
            dtype=np.float32,
        )
        assert feature_vector.shape == (OFFLOAD_POLICY_INPUT_DIM,)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return feature_vector

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_family == OFFLOAD_FEATURE_FAMILY_RICH_REDUCED:
        best_uav_uav_rate: float = context.best_uav_uav_rate if context.cooperative_available else 0.0
        best_neighbor_mbs_rate: float = context.best_neighbor_mbs_rate if context.cooperative_available else 0.0
        best_neighbor_compute_share: float = context.best_neighbor_compute_share if context.cooperative_available else 0.0
        best_neighbor_cache_belief: float = context.best_neighbor_cache_belief if context.cooperative_available else 0.0

        feature_vector = np.array(
            [
                _normalize_request_size(context.request_size),
                _normalize_file_size(context.service_file_size),
                _normalize_cpu_density(context.cpu_cycles_per_byte),
                _normalize_deadline(context.deadline),
                _normalize_priority(context.priority),
                float(context.local_cache_hit),
                float(context.cooperative_available),
                _normalize_local_queue_fraction(context.local_queue_length),
                _normalize_neighbor_count(context.neighbor_count),
                _safe_log10(context.ue_uav_rate),
                _safe_log10(context.uav_mbs_rate),
                _safe_log10(best_uav_uav_rate),
                _safe_log10(best_neighbor_mbs_rate),
                _normalize_compute_share(context.local_compute_share),
                _normalize_compute_share(best_neighbor_compute_share),
                float(np.clip(best_neighbor_cache_belief, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        assert feature_vector.shape == (OFFLOAD_POLICY_RICH_INPUT_DIM,)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return feature_vector

    # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
    raise ValueError(f"Unsupported offload feature family: {feature_family}")


# 函数 context_to_rich_reduced_feature_vector：关键函数，承载本模块的一段可复用实验逻辑，主要参数：context。
def context_to_rich_reduced_feature_vector(context: ServiceOffloadContext) -> np.ndarray:
    """Shortcut for the reduced-leakage raw-state feature family."""

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return context_to_feature_vector(context, feature_family=OFFLOAD_FEATURE_FAMILY_RICH_REDUCED)


# 类 OffloadMLP，继承自 ：核心类，封装本模块中的主要状态和行为。
class OffloadMLP(nn.Module):
    """Small request-level classifier used independently from the MARL trajectory policy."""

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：input_dim, hidden_dims, num_classes。
    def __init__(self, input_dim: int = OFFLOAD_POLICY_INPUT_DIM, hidden_dims: tuple[int, ...] = (64, 64), num_classes: int = OFFLOAD_NUM_CLASSES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim: int = input_dim
        # 循环处理：遍历 hidden_dim 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(last_dim, num_classes)

    # 函数 forward：定义神经网络前向传播计算，主要参数：x。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.backbone(x)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.classifier(embedding)


class OffloadQNetwork(nn.Module):
    """Request-level Q network used by constrained CQL-DQN offloading."""

    def __init__(self, input_dim: int = OFFLOAD_POLICY_RICH_INPUT_DIM, hidden_dims: tuple[int, ...] = (128, 128), num_actions: int = OFFLOAD_NUM_CLASSES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim: int = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(last_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.backbone(x)
        return self.q_head(embedding)


class DistributionalCostNetwork(nn.Module):
    """Distributional action-cost critic for RADCC-Offload."""

    def __init__(
        self,
        input_dim: int = OFFLOAD_POLICY_RICH_INPUT_DIM,
        hidden_dims: tuple[int, ...] = (128, 128),
        num_actions: int = OFFLOAD_NUM_CLASSES,
        num_quantiles: int = 16,
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_quantiles = int(num_quantiles)
        layers: list[nn.Module] = []
        last_dim: int = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.quantile_head = nn.Linear(last_dim, self.num_actions * self.num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.backbone(x)
        raw_quantiles = self.quantile_head(embedding)
        quantiles = raw_quantiles.view(-1, self.num_actions, self.num_quantiles)
        return torch.sort(quantiles, dim=-1).values


def action_mask_from_context(context: ServiceOffloadContext) -> np.ndarray:
    """Return legal request-level offloading actions for the current context."""

    mask = np.ones((OFFLOAD_NUM_CLASSES,), dtype=np.float32)
    if not context.cooperative_available:
        mask[OFFLOAD_TARGET_COOPERATIVE] = 0.0
    return mask


# 类 LearnedClassifierOffloadPolicy：核心类，封装本模块中的主要状态和行为。
class LearnedClassifierOffloadPolicy:
    """Inference wrapper for a trained request-level offloading classifier."""

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：checkpoint_path, device。
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path: str = str(checkpoint_path)
        self.device = torch.device(device)
        self.model, self.metadata = load_offload_policy_checkpoint(self.checkpoint_path, device=device)
        self.feature_family: str = str(self.metadata.get("feature_family", OFFLOAD_FEATURE_FAMILY_FULL))
        scaler_mean = self.metadata.get("scaler_mean")
        scaler_std = self.metadata.get("scaler_std")
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if scaler_mean is not None and scaler_std is not None:
            self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
            self.scaler_std = np.asarray(scaler_std, dtype=np.float32)

    # 函数 predict：关键函数，承载本模块的一段可复用实验逻辑，主要参数：context。
    def predict(self, context: ServiceOffloadContext) -> int:
        features: np.ndarray = context_to_feature_vector(context, feature_family=self.feature_family)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.predict_from_features(features)

    # 函数 predict_from_features：输入分类器或神经网络的特征矩阵，主要参数：features。
    def predict_from_features(self, features: np.ndarray) -> int:
        normalized_features = np.asarray(features, dtype=np.float32)
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if self.scaler_mean is not None and self.scaler_std is not None:
            safe_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)
            normalized_features = ((normalized_features - self.scaler_mean) / safe_std).astype(np.float32)

        feature_tensor: torch.Tensor = torch.from_numpy(normalized_features).unsqueeze(0).to(self.device)
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with torch.no_grad():
            logits: torch.Tensor = self.model(feature_tensor)
            prediction: torch.Tensor = torch.argmax(logits, dim=1)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return int(prediction.item())


class RADCCOffloadPolicy:
    """Risk-aware distributional cost critic offloading policy."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path: str = str(checkpoint_path)
        self.device = torch.device(device)
        self.model, self.metadata = load_radcc_offload_policy_checkpoint(self.checkpoint_path, device=device)
        self.feature_family: str = str(self.metadata.get("feature_family", OFFLOAD_FEATURE_FAMILY_RICH_REDUCED))
        self.risk_beta: float = float(self.metadata.get("risk_beta", getattr(config, "RADCC_OFFLOAD_RISK_BETA", 0.35)))
        self.cvar_alpha: float = float(self.metadata.get("cvar_alpha", getattr(config, "RADCC_OFFLOAD_CVAR_ALPHA", 0.80)))
        scaler_mean = self.metadata.get("scaler_mean")
        scaler_std = self.metadata.get("scaler_std")
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        if scaler_mean is not None and scaler_std is not None:
            self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
            self.scaler_std = np.asarray(scaler_std, dtype=np.float32)

    def predict(self, context: ServiceOffloadContext) -> int:
        features = context_to_feature_vector(context, feature_family=self.feature_family)
        mask = action_mask_from_context(context)
        return self.predict_from_features(features, mask)

    def predict_from_features(self, features: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        normalized_features = np.asarray(features, dtype=np.float32)
        if self.scaler_mean is not None and self.scaler_std is not None:
            safe_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)
            normalized_features = ((normalized_features - self.scaler_mean) / safe_std).astype(np.float32)

        mask = np.ones((OFFLOAD_NUM_CLASSES,), dtype=np.float32) if action_mask is None else np.asarray(action_mask, dtype=np.float32)
        if mask.shape != (OFFLOAD_NUM_CLASSES,):
            raise ValueError(f"Invalid RADCC action mask shape: {mask.shape}")
        if not np.any(mask > 0.0):
            raise ValueError("RADCC action mask has no legal actions.")

        feature_tensor = torch.from_numpy(normalized_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            quantiles = self.model(feature_tensor).squeeze(0)
            mean_cost = torch.mean(quantiles, dim=-1)
            cutoff = max(1, int(np.ceil(float(self.cvar_alpha) * quantiles.shape[-1])))
            tail_cost = torch.mean(quantiles[:, cutoff - 1 :], dim=-1)
            scores = mean_cost + float(self.risk_beta) * tail_cost
            mask_tensor = torch.from_numpy(mask).to(self.device)
            scores = scores.masked_fill(mask_tensor <= 0.0, 1.0e9)
            prediction = torch.argmin(scores, dim=0)
        return int(prediction.item())


class CQLDQNOffloadPolicy:
    """Inference wrapper for a trained constrained CQL-DQN offloading policy."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path: str = str(checkpoint_path)
        self.device = torch.device(device)
        self.model, self.metadata = load_cql_offload_policy_checkpoint(self.checkpoint_path, device=device)
        self.feature_family: str = str(self.metadata.get("feature_family", OFFLOAD_FEATURE_FAMILY_RICH_REDUCED))
        scaler_mean = self.metadata.get("scaler_mean")
        scaler_std = self.metadata.get("scaler_std")
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        if scaler_mean is not None and scaler_std is not None:
            self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
            self.scaler_std = np.asarray(scaler_std, dtype=np.float32)

    def predict(self, context: ServiceOffloadContext) -> int:
        features: np.ndarray = context_to_feature_vector(context, feature_family=self.feature_family)
        mask: np.ndarray = action_mask_from_context(context)
        return self.predict_from_features(features, mask)

    def predict_from_features(self, features: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        normalized_features = np.asarray(features, dtype=np.float32)
        if self.scaler_mean is not None and self.scaler_std is not None:
            safe_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)
            normalized_features = ((normalized_features - self.scaler_mean) / safe_std).astype(np.float32)

        mask = np.ones((OFFLOAD_NUM_CLASSES,), dtype=np.float32) if action_mask is None else np.asarray(action_mask, dtype=np.float32)
        if mask.shape != (OFFLOAD_NUM_CLASSES,):
            raise ValueError(f"Invalid CQL action mask shape: {mask.shape}")
        if not np.any(mask > 0.0):
            raise ValueError("CQL action mask has no legal actions.")

        feature_tensor: torch.Tensor = torch.from_numpy(normalized_features).unsqueeze(0).to(self.device)
        mask_tensor: torch.Tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values: torch.Tensor = self.model(feature_tensor)
            masked_q_values = q_values.masked_fill(mask_tensor <= 0.0, -1.0e9)
            prediction: torch.Tensor = torch.argmax(masked_q_values, dim=1)
        return int(prediction.item())


# 函数 save_offload_policy_checkpoint：保存模型参数或实验结果，主要参数：model, output_path。
def save_offload_policy_checkpoint(
    model: OffloadMLP,
    output_path: str | Path,
    *,
    feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL,
    hidden_dims: tuple[int, ...],
    scaler_mean: list[float] | None = None,
    scaler_std: list[float] | None = None,
    metrics: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Persist a trained classifier checkpoint for later runtime inference."""

    checkpoint_path = Path(output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    input_dim: int = int(model.backbone[0].in_features) if len(model.backbone) > 0 else int(model.classifier.in_features)
    checkpoint: dict[str, Any] = {
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "num_classes": OFFLOAD_NUM_CLASSES,
        "label_mapping": get_offload_label_mapping(),
        "feature_family": feature_family,
        "feature_names": get_offload_feature_names(feature_family),
        "feature_specs": get_offload_feature_specs(feature_family),
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
        "extra_metadata": extra_metadata or {},
    }
    torch.save(checkpoint, checkpoint_path)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return str(checkpoint_path)


def save_cql_offload_policy_checkpoint(
    model: OffloadQNetwork,
    output_path: str | Path,
    *,
    feature_family: str = OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
    hidden_dims: tuple[int, ...],
    scaler_mean: list[float] | None = None,
    scaler_std: list[float] | None = None,
    metrics: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Persist a trained CQL-DQN policy checkpoint for runtime inference."""

    checkpoint_path = Path(output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    input_dim: int = int(model.backbone[0].in_features) if len(model.backbone) > 0 else int(model.q_head.in_features)
    checkpoint: dict[str, Any] = {
        "policy_type": "cql_dqn",
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "num_actions": OFFLOAD_NUM_CLASSES,
        "label_mapping": get_offload_label_mapping(),
        "feature_family": feature_family,
        "feature_names": get_offload_feature_names(feature_family),
        "feature_specs": get_offload_feature_specs(feature_family),
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
        "extra_metadata": extra_metadata or {},
    }
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def save_radcc_offload_policy_checkpoint(
    model: DistributionalCostNetwork,
    output_path: str | Path,
    *,
    feature_family: str = OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
    hidden_dims: tuple[int, ...],
    num_quantiles: int,
    risk_beta: float,
    cvar_alpha: float,
    scaler_mean: list[float] | None = None,
    scaler_std: list[float] | None = None,
    metrics: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    checkpoint_path = Path(output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    input_dim: int = int(model.backbone[0].in_features) if len(model.backbone) > 0 else int(model.quantile_head.in_features)
    checkpoint: dict[str, Any] = {
        "policy_type": "radcc_offload",
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "num_actions": OFFLOAD_NUM_CLASSES,
        "num_quantiles": int(num_quantiles),
        "risk_beta": float(risk_beta),
        "cvar_alpha": float(cvar_alpha),
        "label_mapping": get_offload_label_mapping(),
        "feature_family": feature_family,
        "feature_names": get_offload_feature_names(feature_family),
        "feature_specs": get_offload_feature_specs(feature_family),
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
        "extra_metadata": extra_metadata or {},
    }
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


# 函数 _extract_checkpoint_feature_names：关键函数，承载本模块的一段可复用实验逻辑，主要参数：checkpoint。
def _extract_checkpoint_feature_names(checkpoint: dict[str, Any]) -> list[str]:
    """Read the ordered feature-name schema stored inside a checkpoint."""

    feature_names = checkpoint.get("feature_names")
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_names is not None:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if not isinstance(feature_names, (list, tuple)):
            # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
            raise ValueError("Invalid offload policy checkpoint: 'feature_names' must be a list or tuple.")
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return [str(name) for name in feature_names]

    feature_specs = checkpoint.get("feature_specs")
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_specs is None:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(
            "Invalid offload policy checkpoint: missing feature schema. "
            "Refusing to load because strong schema validation requires 'feature_names' or 'feature_specs'."
        )
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not isinstance(feature_specs, (list, tuple)):
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError("Invalid offload policy checkpoint: 'feature_specs' must be a list or tuple.")

    extracted_names: list[str] = []
    # 循环处理：遍历 spec 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for spec in feature_specs:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if not isinstance(spec, dict) or "name" not in spec:
            # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
            raise ValueError("Invalid offload policy checkpoint: every feature spec must be a dict with a 'name' field.")
        extracted_names.append(str(spec["name"]))
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return extracted_names


# 函数 load_offload_policy_checkpoint：加载模型参数或实验数据，主要参数：checkpoint_path, device。
def load_offload_policy_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[OffloadMLP, dict[str, Any]]:
    """Load a previously trained classifier checkpoint."""

    checkpoint_file = Path(checkpoint_path)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not checkpoint_file.exists():
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise FileNotFoundError(f"Offload policy checkpoint not found: {checkpoint_file}")

    checkpoint: dict[str, Any] = torch.load(checkpoint_file, map_location=device, weights_only=True)
    feature_family: str = str(checkpoint.get("feature_family", OFFLOAD_FEATURE_FAMILY_FULL))
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported offload policy feature family in checkpoint: {feature_family}")
    expected_feature_names: list[str] = get_offload_feature_names(feature_family)
    default_input_dim: int = len(expected_feature_names)
    input_dim: int = int(checkpoint.get("input_dim", default_input_dim))
    hidden_dims: tuple[int, ...] = tuple(int(v) for v in checkpoint.get("hidden_dims", [64, 64]))
    num_classes: int = int(checkpoint.get("num_classes", OFFLOAD_NUM_CLASSES))
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if num_classes != OFFLOAD_NUM_CLASSES:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unexpected offload policy class count: {num_classes} != {OFFLOAD_NUM_CLASSES}")
    expected_dim: int = len(expected_feature_names)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if input_dim != expected_dim:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unexpected offload policy input dim: {input_dim} != {expected_dim} for {feature_family}")

    checkpoint_feature_names: list[str] = _extract_checkpoint_feature_names(checkpoint)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if len(checkpoint_feature_names) != input_dim:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(
            "Invalid offload policy checkpoint: feature schema length does not match input_dim "
            f"({len(checkpoint_feature_names)} != {input_dim})."
        )
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if checkpoint_feature_names != expected_feature_names:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(
            "Offload policy feature schema mismatch. "
            f"Expected ordered features {expected_feature_names}, got {checkpoint_feature_names}."
        )

    scaler_mean = checkpoint.get("scaler_mean")
    scaler_std = checkpoint.get("scaler_std")
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if scaler_mean is not None and len(scaler_mean) != input_dim:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Invalid offload policy checkpoint: scaler_mean length {len(scaler_mean)} != input_dim {input_dim}.")
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if scaler_std is not None and len(scaler_std) != input_dim:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Invalid offload policy checkpoint: scaler_std length {len(scaler_std)} != input_dim {input_dim}.")

    model = OffloadMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return model, checkpoint


def load_cql_offload_policy_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[OffloadQNetwork, dict[str, Any]]:
    """Load a trained constrained CQL-DQN request-level offloading policy."""

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"CQL offload policy checkpoint not found: {checkpoint_file}")

    checkpoint: dict[str, Any] = torch.load(checkpoint_file, map_location=device, weights_only=True)
    policy_type = str(checkpoint.get("policy_type", ""))
    if policy_type and policy_type != "cql_dqn":
        raise ValueError(f"Unexpected CQL offload checkpoint policy_type: {policy_type}")

    feature_family: str = str(checkpoint.get("feature_family", OFFLOAD_FEATURE_FAMILY_RICH_REDUCED))
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported CQL offload policy feature family in checkpoint: {feature_family}")

    expected_feature_names: list[str] = get_offload_feature_names(feature_family)
    input_dim: int = int(checkpoint.get("input_dim", len(expected_feature_names)))
    hidden_dims: tuple[int, ...] = tuple(int(v) for v in checkpoint.get("hidden_dims", [128, 128]))
    num_actions: int = int(checkpoint.get("num_actions", OFFLOAD_NUM_CLASSES))
    if num_actions != OFFLOAD_NUM_CLASSES:
        raise ValueError(f"Unexpected CQL offload action count: {num_actions} != {OFFLOAD_NUM_CLASSES}")
    if input_dim != len(expected_feature_names):
        raise ValueError(f"Unexpected CQL offload input dim: {input_dim} != {len(expected_feature_names)} for {feature_family}")

    checkpoint_feature_names: list[str] = _extract_checkpoint_feature_names(checkpoint)
    if checkpoint_feature_names != expected_feature_names:
        raise ValueError(
            "CQL offload policy feature schema mismatch. "
            f"Expected ordered features {expected_feature_names}, got {checkpoint_feature_names}."
        )

    scaler_mean = checkpoint.get("scaler_mean")
    scaler_std = checkpoint.get("scaler_std")
    if scaler_mean is not None and len(scaler_mean) != input_dim:
        raise ValueError(f"Invalid CQL offload checkpoint: scaler_mean length {len(scaler_mean)} != input_dim {input_dim}.")
    if scaler_std is not None and len(scaler_std) != input_dim:
        raise ValueError(f"Invalid CQL offload checkpoint: scaler_std length {len(scaler_std)} != input_dim {input_dim}.")

    model = OffloadQNetwork(input_dim=input_dim, hidden_dims=hidden_dims, num_actions=num_actions)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, checkpoint


def load_radcc_offload_policy_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[DistributionalCostNetwork, dict[str, Any]]:
    """Load a trained RADCC request-level offloading policy."""

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"RADCC offload policy checkpoint not found: {checkpoint_file}")

    checkpoint: dict[str, Any] = torch.load(checkpoint_file, map_location=device, weights_only=True)
    policy_type = str(checkpoint.get("policy_type", ""))
    if policy_type and policy_type != "radcc_offload":
        raise ValueError(f"Unexpected RADCC checkpoint policy_type: {policy_type}")

    feature_family: str = str(checkpoint.get("feature_family", OFFLOAD_FEATURE_FAMILY_RICH_REDUCED))
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported RADCC feature family in checkpoint: {feature_family}")
    expected_feature_names = get_offload_feature_names(feature_family)
    input_dim = int(checkpoint.get("input_dim", len(expected_feature_names)))
    hidden_dims: tuple[int, ...] = tuple(int(v) for v in checkpoint.get("hidden_dims", [128, 128]))
    num_actions = int(checkpoint.get("num_actions", OFFLOAD_NUM_CLASSES))
    num_quantiles = int(checkpoint.get("num_quantiles", getattr(config, "RADCC_OFFLOAD_NUM_QUANTILES", 16)))
    if num_actions != OFFLOAD_NUM_CLASSES:
        raise ValueError(f"Unexpected RADCC action count: {num_actions} != {OFFLOAD_NUM_CLASSES}")
    if input_dim != len(expected_feature_names):
        raise ValueError(f"Unexpected RADCC input dim: {input_dim} != {len(expected_feature_names)} for {feature_family}")
    checkpoint_feature_names = _extract_checkpoint_feature_names(checkpoint)
    if checkpoint_feature_names != expected_feature_names:
        raise ValueError(
            "RADCC feature schema mismatch. "
            f"Expected ordered features {expected_feature_names}, got {checkpoint_feature_names}."
        )
    scaler_mean = checkpoint.get("scaler_mean")
    scaler_std = checkpoint.get("scaler_std")
    if scaler_mean is not None and len(scaler_mean) != input_dim:
        raise ValueError(f"Invalid RADCC checkpoint: scaler_mean length {len(scaler_mean)} != input_dim {input_dim}.")
    if scaler_std is not None and len(scaler_std) != input_dim:
        raise ValueError(f"Invalid RADCC checkpoint: scaler_std length {len(scaler_std)} != input_dim {input_dim}.")

    model = DistributionalCostNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_actions=num_actions,
        num_quantiles=num_quantiles,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, checkpoint


# 函数 build_offload_policy：关键函数，承载本模块的一段可复用实验逻辑，主要参数：policy_name, checkpoint_path, device。
def build_offload_policy(policy_name: str, checkpoint_path: str | None = None, device: str = "cpu"):
    """Factory for runtime service-request offloading policies."""

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if policy_name not in {"learned", "cql", "radcc"}:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unsupported service offload policy: {policy_name}")

    resolved_checkpoint: str | None = checkpoint_path or getattr(config, "SERVICE_OFFLOAD_POLICY_CHECKPOINT", None)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not resolved_checkpoint:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"SERVICE_OFFLOAD_POLICY_CHECKPOINT must be set when SERVICE_OFFLOAD_POLICY='{policy_name}'.")

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    if policy_name == "cql":
        return CQLDQNOffloadPolicy(resolved_checkpoint, device=device)
    if policy_name == "radcc":
        return RADCCOffloadPolicy(resolved_checkpoint, device=device)
    return LearnedClassifierOffloadPolicy(resolved_checkpoint, device=device)
