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


@dataclass(frozen=True, slots=True)
class OffloadFeatureSpec:
    name: str
    description: str
    normalization: str


OFFLOAD_FEATURE_SPECS: tuple[OffloadFeatureSpec, ...] = (
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
OFFLOAD_POLICY_INPUT_DIM: int = len(OFFLOAD_FEATURE_SPECS)


def get_offload_feature_specs() -> list[dict[str, str]]:
    """Return JSON-serializable feature metadata for dataset artifacts."""

    return [asdict(spec) for spec in OFFLOAD_FEATURE_SPECS]


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


def context_to_feature_vector(context: ServiceOffloadContext) -> np.ndarray:
    """Convert a service offloading context into a normalized feature vector."""

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
            float(np.clip(context.local_queue_length / max(float(config.MAX_ASSOCIATED_UES), 1.0), 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    assert feature_vector.shape == (OFFLOAD_POLICY_INPUT_DIM,)
    return feature_vector


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

    def predict(self, context: ServiceOffloadContext) -> int:
        features: np.ndarray = context_to_feature_vector(context)
        return self.predict_from_features(features)

    def predict_from_features(self, features: np.ndarray) -> int:
        feature_tensor: torch.Tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits: torch.Tensor = self.model(feature_tensor)
            prediction: torch.Tensor = torch.argmax(logits, dim=1)
        return int(prediction.item())


def save_offload_policy_checkpoint(
    model: OffloadMLP,
    output_path: str | Path,
    *,
    hidden_dims: tuple[int, ...],
    metrics: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Persist a trained classifier checkpoint for later runtime inference."""

    checkpoint_path = Path(output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "input_dim": OFFLOAD_POLICY_INPUT_DIM,
        "hidden_dims": list(hidden_dims),
        "num_classes": OFFLOAD_NUM_CLASSES,
        "label_mapping": get_offload_label_mapping(),
        "feature_specs": get_offload_feature_specs(),
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
        "extra_metadata": extra_metadata or {},
    }
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def load_offload_policy_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[OffloadMLP, dict[str, Any]]:
    """Load a previously trained classifier checkpoint."""

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Offload policy checkpoint not found: {checkpoint_file}")

    checkpoint: dict[str, Any] = torch.load(checkpoint_file, map_location=device, weights_only=True)
    input_dim: int = int(checkpoint.get("input_dim", OFFLOAD_POLICY_INPUT_DIM))
    hidden_dims: tuple[int, ...] = tuple(int(v) for v in checkpoint.get("hidden_dims", [64, 64]))
    num_classes: int = int(checkpoint.get("num_classes", OFFLOAD_NUM_CLASSES))
    if input_dim != OFFLOAD_POLICY_INPUT_DIM:
        raise ValueError(f"Unexpected offload policy input dim: {input_dim} != {OFFLOAD_POLICY_INPUT_DIM}")
    if num_classes != OFFLOAD_NUM_CLASSES:
        raise ValueError(f"Unexpected offload policy class count: {num_classes} != {OFFLOAD_NUM_CLASSES}")

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
