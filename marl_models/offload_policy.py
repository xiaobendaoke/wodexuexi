from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import config


OFFLOAD_TARGET_LOCAL: int = 0
OFFLOAD_TARGET_COOPERATIVE: int = 1
OFFLOAD_TARGET_MBS: int = 2
OFFLOAD_NUM_CLASSES: int = 3
OFFLOAD_TARGET_NAMES: tuple[str, ...] = ("local", "cooperative", "mbs")
OFFLOAD_LATENCY_RATIO_CLIP: float = 10.0
OFFLOAD_COMPUTE_SHARE_CLIP: float = 4.0
OFFLOAD_FEATURE_FAMILY_FULL: str = "full_features"
OFFLOAD_FEATURE_FAMILY_RICH_REDUCED: str = "rich_reduced_features"


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


@dataclass(frozen=True, slots=True)
class OffloadFeatureSpec:
    name: str
    description: str
    normalization: str


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
OFFLOAD_FEATURE_SPECS: tuple[OffloadFeatureSpec, ...] = FULL_OFFLOAD_FEATURE_SPECS
OFFLOAD_FEATURE_FAMILIES: dict[str, tuple[OffloadFeatureSpec, ...]] = {
    OFFLOAD_FEATURE_FAMILY_FULL: FULL_OFFLOAD_FEATURE_SPECS,
    OFFLOAD_FEATURE_FAMILY_RICH_REDUCED: RICH_REDUCED_FEATURE_SPECS,
}
OFFLOAD_POLICY_INPUT_DIM: int = len(FULL_OFFLOAD_FEATURE_SPECS)
OFFLOAD_POLICY_RICH_INPUT_DIM: int = len(RICH_REDUCED_FEATURE_SPECS)


def get_offload_feature_specs(feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL) -> list[dict[str, str]]:
    """Return JSON-serializable feature metadata for dataset artifacts."""

    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    return [asdict(spec) for spec in OFFLOAD_FEATURE_FAMILIES[feature_family]]


def get_offload_feature_names(feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL) -> list[str]:
    """Return the canonical ordered feature-name schema for a feature family."""

    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    return [spec.name for spec in OFFLOAD_FEATURE_FAMILIES[feature_family]]


def get_offload_feature_dim(feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL) -> int:
    """Return the input dimension for a given feature family."""

    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported offload feature family: {feature_family}")
    return len(OFFLOAD_FEATURE_FAMILIES[feature_family])


def get_offload_label_mapping() -> dict[int, str]:
    """Return the canonical offloading label mapping."""

    return {label: OFFLOAD_TARGET_NAMES[label] for label in range(OFFLOAD_NUM_CLASSES)}


def _normalize_ratio(numerator: float, denominator: float) -> float:
    safe_denominator: float = max(float(denominator), float(config.EPSILON))
    raw_ratio: float = float(numerator) / safe_denominator
    if not np.isfinite(raw_ratio):
        return 1.0
    clipped_ratio: float = float(np.clip(raw_ratio, 0.0, OFFLOAD_LATENCY_RATIO_CLIP))
    return clipped_ratio / OFFLOAD_LATENCY_RATIO_CLIP


def _normalize_deadline(deadline: float) -> float:
    span: float = float(config.SERVICE_DEADLINE_MAX - config.SERVICE_DEADLINE_MIN)
    if span <= float(config.EPSILON):
        return 0.0
    normalized: float = (float(deadline) - float(config.SERVICE_DEADLINE_MIN)) / span
    return float(np.clip(normalized, 0.0, 1.0))


def _normalize_priority(priority: int) -> float:
    span: int = int(config.SERVICE_PRIORITY_MAX - config.SERVICE_PRIORITY_MIN)
    if span <= 0:
        return 0.0
    normalized: float = (float(priority) - float(config.SERVICE_PRIORITY_MIN)) / float(span)
    return float(np.clip(normalized, 0.0, 1.0))


def _normalize_request_size(request_size: int) -> float:
    span: float = float(config.MAX_INPUT_SIZE - config.MIN_INPUT_SIZE)
    if span <= float(config.EPSILON):
        return 0.0
    normalized: float = (float(request_size) - float(config.MIN_INPUT_SIZE)) / span
    return float(np.clip(normalized, 0.0, 1.0))


def _normalize_file_size(file_size: int) -> float:
    max_file_size: float = max(float(np.max(config.FILE_SIZES)), 1.0)
    return float(np.clip(float(file_size) / max_file_size, 0.0, 1.0))


def _normalize_cpu_density(cpu_cycles_per_byte: float) -> float:
    cpu_min: float = float(np.min(config.CPU_CYCLES_PER_BYTE))
    cpu_max: float = float(np.max(config.CPU_CYCLES_PER_BYTE))
    span: float = max(cpu_max - cpu_min, float(config.EPSILON))
    normalized: float = (float(cpu_cycles_per_byte) - cpu_min) / span
    return float(np.clip(normalized, 0.0, 1.0))


def _normalize_local_queue_fraction(local_queue_length: int) -> float:
    return float(np.clip(local_queue_length / max(float(config.MAX_ASSOCIATED_UES), 1.0), 0.0, 1.0))


def _normalize_neighbor_count(neighbor_count: int) -> float:
    return float(np.clip(neighbor_count / max(float(config.NUM_UAVS - 1), 1.0), 0.0, 1.0))


def _normalize_compute_share(compute_share: float) -> float:
    max_capacity: float = max(float(np.max(config.UAV_COMPUTING_CAPACITY)), 1.0)
    normalized: float = float(compute_share) / max_capacity
    return float(np.clip(normalized, 0.0, OFFLOAD_COMPUTE_SHARE_CLIP))


def _safe_log10(value: float) -> float:
    return float(np.log10(max(float(value), float(config.EPSILON))))


def context_to_feature_vector(
    context: ServiceOffloadContext,
    feature_family: str = OFFLOAD_FEATURE_FAMILY_FULL,
) -> np.ndarray:
    """Convert a service offloading context into a normalized feature vector."""

    if feature_family == OFFLOAD_FEATURE_FAMILY_FULL:
        cooperative_latency_feature: float = 1.0
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
        return feature_vector

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
        return feature_vector

    raise ValueError(f"Unsupported offload feature family: {feature_family}")


def context_to_rich_reduced_feature_vector(context: ServiceOffloadContext) -> np.ndarray:
    """Shortcut for the reduced-leakage raw-state feature family."""

    return context_to_feature_vector(context, feature_family=OFFLOAD_FEATURE_FAMILY_RICH_REDUCED)


class OffloadMLP(nn.Module):
    """Small request-level classifier used independently from the MARL trajectory policy."""

    def __init__(self, input_dim: int = OFFLOAD_POLICY_INPUT_DIM, hidden_dims: tuple[int, ...] = (64, 64), num_classes: int = OFFLOAD_NUM_CLASSES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim: int = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(last_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.backbone(x)
        return self.classifier(embedding)


class LearnedClassifierOffloadPolicy:
    """Inference wrapper for a trained request-level offloading classifier."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path: str = str(checkpoint_path)
        self.device = torch.device(device)
        self.model, self.metadata = load_offload_policy_checkpoint(self.checkpoint_path, device=device)
        self.feature_family: str = str(self.metadata.get("feature_family", OFFLOAD_FEATURE_FAMILY_FULL))
        scaler_mean = self.metadata.get("scaler_mean")
        scaler_std = self.metadata.get("scaler_std")
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        if scaler_mean is not None and scaler_std is not None:
            self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
            self.scaler_std = np.asarray(scaler_std, dtype=np.float32)

    def predict(self, context: ServiceOffloadContext) -> int:
        features: np.ndarray = context_to_feature_vector(context, feature_family=self.feature_family)
        return self.predict_from_features(features)

    def predict_from_features(self, features: np.ndarray) -> int:
        normalized_features = np.asarray(features, dtype=np.float32)
        if self.scaler_mean is not None and self.scaler_std is not None:
            safe_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)
            normalized_features = ((normalized_features - self.scaler_mean) / safe_std).astype(np.float32)

        feature_tensor: torch.Tensor = torch.from_numpy(normalized_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits: torch.Tensor = self.model(feature_tensor)
            prediction: torch.Tensor = torch.argmax(logits, dim=1)
        return int(prediction.item())


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
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
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
    return str(checkpoint_path)


def _extract_checkpoint_feature_names(checkpoint: dict[str, Any]) -> list[str]:
    """Read the ordered feature-name schema stored inside a checkpoint."""

    feature_names = checkpoint.get("feature_names")
    if feature_names is not None:
        if not isinstance(feature_names, (list, tuple)):
            raise ValueError("Invalid offload policy checkpoint: 'feature_names' must be a list or tuple.")
        return [str(name) for name in feature_names]

    feature_specs = checkpoint.get("feature_specs")
    if feature_specs is None:
        raise ValueError(
            "Invalid offload policy checkpoint: missing feature schema. "
            "Refusing to load because strong schema validation requires 'feature_names' or 'feature_specs'."
        )
    if not isinstance(feature_specs, (list, tuple)):
        raise ValueError("Invalid offload policy checkpoint: 'feature_specs' must be a list or tuple.")

    extracted_names: list[str] = []
    for spec in feature_specs:
        if not isinstance(spec, dict) or "name" not in spec:
            raise ValueError("Invalid offload policy checkpoint: every feature spec must be a dict with a 'name' field.")
        extracted_names.append(str(spec["name"]))
    return extracted_names


def load_offload_policy_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[OffloadMLP, dict[str, Any]]:
    """Load a previously trained classifier checkpoint."""

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Offload policy checkpoint not found: {checkpoint_file}")

    checkpoint: dict[str, Any] = torch.load(checkpoint_file, map_location=device, weights_only=True)
    feature_family: str = str(checkpoint.get("feature_family", OFFLOAD_FEATURE_FAMILY_FULL))
    if feature_family not in OFFLOAD_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported offload policy feature family in checkpoint: {feature_family}")
    expected_feature_names: list[str] = get_offload_feature_names(feature_family)
    default_input_dim: int = len(expected_feature_names)
    input_dim: int = int(checkpoint.get("input_dim", default_input_dim))
    hidden_dims: tuple[int, ...] = tuple(int(v) for v in checkpoint.get("hidden_dims", [64, 64]))
    num_classes: int = int(checkpoint.get("num_classes", OFFLOAD_NUM_CLASSES))
    if num_classes != OFFLOAD_NUM_CLASSES:
        raise ValueError(f"Unexpected offload policy class count: {num_classes} != {OFFLOAD_NUM_CLASSES}")
    expected_dim: int = len(expected_feature_names)
    if input_dim != expected_dim:
        raise ValueError(f"Unexpected offload policy input dim: {input_dim} != {expected_dim} for {feature_family}")

    checkpoint_feature_names: list[str] = _extract_checkpoint_feature_names(checkpoint)
    if len(checkpoint_feature_names) != input_dim:
        raise ValueError(
            "Invalid offload policy checkpoint: feature schema length does not match input_dim "
            f"({len(checkpoint_feature_names)} != {input_dim})."
        )
    if checkpoint_feature_names != expected_feature_names:
        raise ValueError(
            "Offload policy feature schema mismatch. "
            f"Expected ordered features {expected_feature_names}, got {checkpoint_feature_names}."
        )

    scaler_mean = checkpoint.get("scaler_mean")
    scaler_std = checkpoint.get("scaler_std")
    if scaler_mean is not None and len(scaler_mean) != input_dim:
        raise ValueError(f"Invalid offload policy checkpoint: scaler_mean length {len(scaler_mean)} != input_dim {input_dim}.")
    if scaler_std is not None and len(scaler_std) != input_dim:
        raise ValueError(f"Invalid offload policy checkpoint: scaler_std length {len(scaler_std)} != input_dim {input_dim}.")

    model = OffloadMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, checkpoint


def build_offload_policy(policy_name: str, checkpoint_path: str | None = None, device: str = "cpu"):
    """Factory for runtime service-request offloading policies."""

    if policy_name != "learned":
        raise ValueError(f"Unsupported service offload policy: {policy_name}")

    resolved_checkpoint: str | None = checkpoint_path or getattr(config, "SERVICE_OFFLOAD_POLICY_CHECKPOINT", None)
    if not resolved_checkpoint:
        raise ValueError("SERVICE_OFFLOAD_POLICY_CHECKPOINT must be set when SERVICE_OFFLOAD_POLICY='learned'.")

    return LearnedClassifierOffloadPolicy(resolved_checkpoint, device=device)
