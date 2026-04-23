from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from environment import comm_model as comms
from environment.env import Env
from environment.request_types import Request
from environment.uavs import _get_belief_probability
from marl_models.offload_policy import (
    OFFLOAD_FEATURE_FAMILY_FULL,
    OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
    OFFLOAD_NUM_CLASSES,
    OFFLOAD_TARGET_COOPERATIVE,
    OFFLOAD_TARGET_LOCAL,
    OFFLOAD_TARGET_MBS,
    OFFLOAD_TARGET_NAMES,
    ServiceOffloadContext,
    context_to_feature_vector,
    get_offload_feature_dim,
    get_offload_feature_specs,
)


def snapshot_config() -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for key in dir(config):
        if key.isupper() and not key.startswith("__"):
            value = getattr(config, key)
            snapshot[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    return snapshot


def restore_config(snapshot: dict[str, object]) -> None:
    for key, value in snapshot.items():
        setattr(config, key, value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))


class OffloadDatasetRecorder:
    """Collect feature vectors, labels, and scenario metadata for later profiling."""

    def __init__(self, default_scenario_name: str = "random_rollout") -> None:
        self.default_scenario_name: str = default_scenario_name
        self.features_full: list[np.ndarray] = []
        self.features_rich_reduced: list[np.ndarray] = []
        self.labels: list[int] = []
        self.scenario_ids: list[int] = []
        self.raw_latencies: list[np.ndarray] = []
        self.deadlines: list[float] = []
        self.priorities: list[int] = []
        self.local_cache_hits: list[int] = []
        self.cooperative_available: list[int] = []
        self.local_queue_lengths: list[int] = []
        self._scenario_name_to_id: dict[str, int] = {}

    def _ensure_scenario_id(self, scenario_name: str) -> int:
        if scenario_name not in self._scenario_name_to_id:
            self._scenario_name_to_id[scenario_name] = len(self._scenario_name_to_id)
        return self._scenario_name_to_id[scenario_name]

    def record(self, context: ServiceOffloadContext, heuristic_label: int, scenario_name: str | None = None) -> None:
        actual_scenario_name: str = scenario_name or self.default_scenario_name
        self.features_full.append(context_to_feature_vector(context, feature_family=OFFLOAD_FEATURE_FAMILY_FULL))
        self.features_rich_reduced.append(context_to_feature_vector(context, feature_family=OFFLOAD_FEATURE_FAMILY_RICH_REDUCED))
        self.labels.append(int(heuristic_label))
        self.scenario_ids.append(self._ensure_scenario_id(actual_scenario_name))
        self.raw_latencies.append(
            np.array([context.local_latency, context.cooperative_latency, context.mbs_latency], dtype=np.float32)
        )
        self.deadlines.append(float(context.deadline))
        self.priorities.append(int(context.priority))
        self.local_cache_hits.append(int(context.local_cache_hit))
        self.cooperative_available.append(int(context.cooperative_available))
        self.local_queue_lengths.append(int(context.local_queue_length))

    def __call__(self, context: ServiceOffloadContext, heuristic_label: int) -> None:
        self.record(context, heuristic_label, scenario_name=self.default_scenario_name)

    def export(self) -> dict[str, np.ndarray]:
        full_feature_dim = get_offload_feature_dim(OFFLOAD_FEATURE_FAMILY_FULL)
        rich_feature_dim = get_offload_feature_dim(OFFLOAD_FEATURE_FAMILY_RICH_REDUCED)
        if not self.features_full:
            scenario_names = sorted(self._scenario_name_to_id, key=self._scenario_name_to_id.get)
            return {
                "features": np.zeros((0, full_feature_dim), dtype=np.float32),
                "features_full": np.zeros((0, full_feature_dim), dtype=np.float32),
                "features_rich_reduced": np.zeros((0, rich_feature_dim), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "scenario_ids": np.zeros((0,), dtype=np.int64),
                "scenario_names": np.asarray(scenario_names, dtype="<U64"),
                "raw_latencies": np.zeros((0, 3), dtype=np.float32),
                "deadlines": np.zeros((0,), dtype=np.float32),
                "priorities": np.zeros((0,), dtype=np.int64),
                "local_cache_hits": np.zeros((0,), dtype=np.int64),
                "cooperative_available": np.zeros((0,), dtype=np.int64),
                "local_queue_lengths": np.zeros((0,), dtype=np.int64),
            }

        scenario_names = [name for name, _ in sorted(self._scenario_name_to_id.items(), key=lambda item: item[1])]
        return {
            "features": np.asarray(self.features_full, dtype=np.float32),
            "features_full": np.asarray(self.features_full, dtype=np.float32),
            "features_rich_reduced": np.asarray(self.features_rich_reduced, dtype=np.float32),
            "labels": np.asarray(self.labels, dtype=np.int64),
            "scenario_ids": np.asarray(self.scenario_ids, dtype=np.int64),
            "scenario_names": np.asarray(scenario_names, dtype="<U64"),
            "raw_latencies": np.asarray(self.raw_latencies, dtype=np.float32),
            "deadlines": np.asarray(self.deadlines, dtype=np.float32),
            "priorities": np.asarray(self.priorities, dtype=np.int64),
            "local_cache_hits": np.asarray(self.local_cache_hits, dtype=np.int64),
            "cooperative_available": np.asarray(self.cooperative_available, dtype=np.int64),
            "local_queue_lengths": np.asarray(self.local_queue_lengths, dtype=np.int64),
        }


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    name: str
    intended_label: int
    description: str


def _save_dataset(output_path: str | Path, recorder: OffloadDatasetRecorder, metadata: dict[str, object]) -> dict[str, object]:
    arrays = recorder.export()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **arrays)

    labels = arrays["labels"]
    label_counts: np.ndarray = np.bincount(labels, minlength=OFFLOAD_NUM_CLASSES)
    total_samples: int = int(labels.size)
    scenario_names = arrays["scenario_names"].tolist()
    scenario_counts: dict[str, int] = {}
    if arrays["scenario_ids"].size > 0 and scenario_names:
        counts = np.bincount(arrays["scenario_ids"], minlength=len(scenario_names))
        scenario_counts = {scenario_names[idx]: int(counts[idx]) for idx in range(len(scenario_names))}

    final_metadata: dict[str, object] = {
        "dataset_path": str(output_file),
        "collected_samples": total_samples,
        "feature_dim": int(arrays["features"].shape[1]) if arrays["features"].ndim == 2 else 0,
        "feature_specs": get_offload_feature_specs(),
        "feature_dims": {
            OFFLOAD_FEATURE_FAMILY_FULL: int(arrays["features_full"].shape[1]) if arrays["features_full"].ndim == 2 else 0,
            OFFLOAD_FEATURE_FAMILY_RICH_REDUCED: int(arrays["features_rich_reduced"].shape[1]) if arrays["features_rich_reduced"].ndim == 2 else 0,
        },
        "feature_families": {
            OFFLOAD_FEATURE_FAMILY_FULL: get_offload_feature_specs(OFFLOAD_FEATURE_FAMILY_FULL),
            OFFLOAD_FEATURE_FAMILY_RICH_REDUCED: get_offload_feature_specs(OFFLOAD_FEATURE_FAMILY_RICH_REDUCED),
        },
        "label_mapping": {str(idx): name for idx, name in enumerate(OFFLOAD_TARGET_NAMES)},
        "class_counts": {OFFLOAD_TARGET_NAMES[idx]: int(label_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "class_ratios": {
            OFFLOAD_TARGET_NAMES[idx]: float(label_counts[idx] / max(total_samples, 1)) for idx in range(OFFLOAD_NUM_CLASSES)
        },
        "scenario_counts": scenario_counts,
    }
    final_metadata.update(metadata)

    metadata_path = output_file.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(final_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return final_metadata


def _clear_synthetic_state(env: Env) -> None:
    for uav in env.uavs:
        uav.cache.fill(False)
        uav._working_cache.fill(False)
        uav._current_service_request_count = 0
        uav._current_covered_ues.clear()
        uav._neighbors = []
        uav._freq_counts.fill(0.0)
        uav._uav_mbs_rate = 0.0
    for ue in env.ues:
        ue.assigned = False
        ue.latency_current_request = 0.0
        ue.current_request = Request.content(req_id=config.NUM_SERVICES)


def _set_source_and_neighbor_positions(env: Env, rng: np.random.Generator, source_xy: np.ndarray, neighbor_xy: np.ndarray | None) -> None:
    env.uavs[0].pos = np.array([source_xy[0], source_xy[1], config.UAV_ALTITUDE], dtype=np.float32)
    if neighbor_xy is not None and len(env.uavs) > 1:
        env.uavs[1].pos = np.array([neighbor_xy[0], neighbor_xy[1], config.UAV_ALTITUDE], dtype=np.float32)

    for idx, uav in enumerate(env.uavs[2:] if neighbor_xy is not None else env.uavs[1:], start=2 if neighbor_xy is not None else 1):
        jitter = rng.uniform(-20.0, 20.0, size=2)
        far_xy = np.array([config.AREA_WIDTH - 40.0, 40.0 + 35.0 * idx], dtype=np.float32) + jitter.astype(np.float32)
        far_xy = np.clip(far_xy, [30.0, 30.0], [config.AREA_WIDTH - 30.0, config.AREA_HEIGHT - 30.0])
        uav.pos = np.array([far_xy[0], far_xy[1], config.UAV_ALTITUDE], dtype=np.float32)

    for uav in env.uavs:
        uav.set_neighbors(env.uavs)


def _make_request(
    *,
    rng: np.random.Generator,
    req_id: int,
    size_range: tuple[int, int],
    deadline_range: tuple[float, float],
) -> Request:
    req_size = int(rng.integers(size_range[0], size_range[1] + 1))
    deadline = float(rng.uniform(deadline_range[0], deadline_range[1]))
    priority = int(rng.integers(config.SERVICE_PRIORITY_MIN, config.SERVICE_PRIORITY_MAX + 1))
    return Request.service(req_size=req_size, req_id=req_id, deadline=deadline, priority=priority)


def _compute_service_sample(env: Env, request: Request, *, covered_ues_count: int) -> tuple[ServiceOffloadContext, int]:
    source_uav = env.uavs[0]
    source_uav._uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(source_uav.pos, config.MBS_POS))
    ue_offset = np.array([1.0, 1.0], dtype=np.float32)
    ue_pos = np.array([source_uav.pos[0] + ue_offset[0], source_uav.pos[1] + ue_offset[1], 0.0], dtype=np.float32)
    ue_uav_rate = comms.calculate_ue_uav_rate(
        comms.calculate_channel_gain(ue_pos, source_uav.pos),
        max(int(covered_ues_count), 1),
    )
    context, cooperative_uav = source_uav._build_service_offload_context(request, ue_uav_rate)
    label, _ = source_uav._select_service_target_from_context(context, cooperative_uav)
    return context, int(label)


def _sample_top_belief_service_id(neighbor_id: int, rng: np.random.Generator, top_k: int = 5) -> int:
    service_scores = np.array([_get_belief_probability(service_id, neighbor_id) for service_id in range(config.NUM_SERVICES)], dtype=np.float32)
    ranked_ids = np.argsort(-service_scores)
    k = min(int(top_k), ranked_ids.size)
    return int(rng.choice(ranked_ids[:k]))


def _scenario_local_cached_backhaul_bottleneck(env: Env, rng: np.random.Generator) -> tuple[ServiceOffloadContext, int]:
    _clear_synthetic_state(env)
    config.BANDWIDTH_BACKHAUL = int(rng.integers(50_000, 180_001))
    config.UAV_COMPUTING_CAPACITY[0] = int(rng.integers(120, 260)) * 10**9

    source_xy = rng.uniform(40.0, 120.0, size=2).astype(np.float32)
    _set_source_and_neighbor_positions(env, rng, source_xy, neighbor_xy=None)

    req_id = int(rng.integers(0, config.NUM_SERVICES))
    config.CPU_CYCLES_PER_BYTE[req_id] = int(rng.integers(300, 900))
    env.uavs[0].cache[req_id] = True
    env.uavs[0]._current_service_request_count = int(rng.integers(1, 3))

    request = _make_request(
        rng=rng,
        req_id=req_id,
        size_range=(150_000, 600_000),
        deadline_range=(0.8 * config.TIME_SLOT_DURATION, 1.8 * config.TIME_SLOT_DURATION),
    )
    return _compute_service_sample(env, request, covered_ues_count=max(env.uavs[0]._current_service_request_count, 1))


def _scenario_local_compute_advantage(env: Env, rng: np.random.Generator) -> tuple[ServiceOffloadContext, int]:
    _clear_synthetic_state(env)
    config.BANDWIDTH_BACKHAUL = int(rng.integers(120_000, 350_001))
    config.UAV_COMPUTING_CAPACITY[0] = int(rng.integers(90, 180)) * 10**9
    if len(env.uavs) > 1:
        config.UAV_COMPUTING_CAPACITY[1] = int(rng.integers(5, 15)) * 10**9

    source_xy = rng.uniform(60.0, 160.0, size=2).astype(np.float32)
    neighbor_xy = np.clip(source_xy + np.array([260.0, 10.0], dtype=np.float32), [30.0, 30.0], [config.AREA_WIDTH - 30.0, config.AREA_HEIGHT - 30.0])
    _set_source_and_neighbor_positions(env, rng, source_xy, neighbor_xy if len(env.uavs) > 1 else None)

    req_id = int(rng.integers(0, config.NUM_SERVICES))
    config.CPU_CYCLES_PER_BYTE[req_id] = int(rng.integers(250, 700))
    env.uavs[0].cache[req_id] = True
    env.uavs[0]._current_service_request_count = int(rng.integers(1, 4))
    if len(env.uavs) > 1:
        env.uavs[1]._current_service_request_count = int(rng.integers(8, 18))

    request = _make_request(
        rng=rng,
        req_id=req_id,
        size_range=(120_000, 450_000),
        deadline_range=(0.7 * config.TIME_SLOT_DURATION, 1.6 * config.TIME_SLOT_DURATION),
    )
    covered_ues = max(env.uavs[0]._current_service_request_count + int(rng.integers(0, 2)), 1)
    return _compute_service_sample(env, request, covered_ues_count=covered_ues)


def _scenario_cooperative_neighbor_cached(env: Env, rng: np.random.Generator) -> tuple[ServiceOffloadContext, int]:
    _clear_synthetic_state(env)
    config.BANDWIDTH_BACKHAUL = int(rng.integers(60_000, 220_001))
    config.UAV_COMPUTING_CAPACITY[0] = int(rng.integers(8, 25)) * 10**9
    if len(env.uavs) > 1:
        config.UAV_COMPUTING_CAPACITY[1] = int(rng.integers(140, 260)) * 10**9
        config.UAV_STORAGE_CAPACITY[1] = int(rng.integers(90, 140)) * 10**6

    source_xy = rng.uniform(60.0, 130.0, size=2).astype(np.float32)
    neighbor_xy = np.clip(source_xy + np.array([rng.uniform(10.0, 25.0), rng.uniform(-8.0, 8.0)], dtype=np.float32), [30.0, 30.0], [config.AREA_WIDTH - 30.0, config.AREA_HEIGHT - 30.0])
    _set_source_and_neighbor_positions(env, rng, source_xy, neighbor_xy if len(env.uavs) > 1 else None)

    req_id = _sample_top_belief_service_id(1 if len(env.uavs) > 1 else 0, rng)
    config.CPU_CYCLES_PER_BYTE[req_id] = int(rng.integers(350, 900))
    if len(env.uavs) > 1:
        env.uavs[1].cache[req_id] = True
        env.uavs[1]._current_service_request_count = int(rng.integers(0, 2))
    env.uavs[0]._current_service_request_count = int(rng.integers(12, 25))

    request = _make_request(
        rng=rng,
        req_id=req_id,
        size_range=(180_000, 900_000),
        deadline_range=(0.8 * config.TIME_SLOT_DURATION, 1.8 * config.TIME_SLOT_DURATION),
    )
    return _compute_service_sample(env, request, covered_ues_count=max(env.uavs[0]._current_service_request_count, 1))


def _scenario_cooperative_queue_relief(env: Env, rng: np.random.Generator) -> tuple[ServiceOffloadContext, int]:
    _clear_synthetic_state(env)
    config.BANDWIDTH_BACKHAUL = int(rng.integers(80_000, 260_001))
    config.UAV_COMPUTING_CAPACITY[0] = int(rng.integers(6, 18)) * 10**9
    if len(env.uavs) > 1:
        config.UAV_COMPUTING_CAPACITY[1] = int(rng.integers(110, 220)) * 10**9
        config.UAV_STORAGE_CAPACITY[1] = int(rng.integers(80, 130)) * 10**6

    source_xy = rng.uniform(80.0, 180.0, size=2).astype(np.float32)
    neighbor_xy = np.clip(source_xy + np.array([rng.uniform(8.0, 20.0), rng.uniform(-6.0, 6.0)], dtype=np.float32), [30.0, 30.0], [config.AREA_WIDTH - 30.0, config.AREA_HEIGHT - 30.0])
    _set_source_and_neighbor_positions(env, rng, source_xy, neighbor_xy if len(env.uavs) > 1 else None)

    req_id = _sample_top_belief_service_id(1 if len(env.uavs) > 1 else 0, rng)
    config.CPU_CYCLES_PER_BYTE[req_id] = int(rng.integers(450, 1200))
    env.uavs[0]._current_service_request_count = int(rng.integers(16, 30))
    if len(env.uavs) > 1:
        env.uavs[1]._current_service_request_count = int(rng.integers(0, 3))
        if rng.random() < 0.6:
            env.uavs[1].cache[req_id] = True

    request = _make_request(
        rng=rng,
        req_id=req_id,
        size_range=(250_000, 1_000_000),
        deadline_range=(0.9 * config.TIME_SLOT_DURATION, 2.0 * config.TIME_SLOT_DURATION),
    )
    return _compute_service_sample(env, request, covered_ues_count=max(env.uavs[0]._current_service_request_count, 1))


def _scenario_mbs_backhaul_advantage(env: Env, rng: np.random.Generator) -> tuple[ServiceOffloadContext, int]:
    _clear_synthetic_state(env)
    config.BANDWIDTH_BACKHAUL = int(rng.integers(8_000_000, 20_000_001))
    config.UAV_COMPUTING_CAPACITY[0] = int(rng.integers(5, 12)) * 10**9

    source_xy = np.clip(config.MBS_POS[:2] + rng.uniform(-25.0, 25.0, size=2), [30.0, 30.0], [config.AREA_WIDTH - 30.0, config.AREA_HEIGHT - 30.0]).astype(np.float32)
    _set_source_and_neighbor_positions(env, rng, source_xy, neighbor_xy=None)

    req_id = int(rng.integers(0, config.NUM_SERVICES))
    config.CPU_CYCLES_PER_BYTE[req_id] = int(rng.integers(2800, 5200))
    env.uavs[0]._current_service_request_count = int(rng.integers(8, 24))

    request = _make_request(
        rng=rng,
        req_id=req_id,
        size_range=(1_000_000, 5_000_000),
        deadline_range=(0.5 * config.TIME_SLOT_DURATION, 1.4 * config.TIME_SLOT_DURATION),
    )
    covered_ues = max(env.uavs[0]._current_service_request_count + int(rng.integers(0, 4)), 1)
    return _compute_service_sample(env, request, covered_ues_count=covered_ues)


def _scenario_mbs_large_job(env: Env, rng: np.random.Generator) -> tuple[ServiceOffloadContext, int]:
    _clear_synthetic_state(env)
    config.BANDWIDTH_BACKHAUL = int(rng.integers(4_000_000, 12_000_001))
    config.UAV_COMPUTING_CAPACITY[0] = int(rng.integers(5, 10)) * 10**9
    if len(env.uavs) > 1:
        config.UAV_COMPUTING_CAPACITY[1] = int(rng.integers(5, 14)) * 10**9

    source_xy = np.clip(config.MBS_POS[:2] + rng.uniform(-40.0, 40.0, size=2), [30.0, 30.0], [config.AREA_WIDTH - 30.0, config.AREA_HEIGHT - 30.0]).astype(np.float32)
    neighbor_xy = np.clip(source_xy + np.array([280.0, 20.0], dtype=np.float32), [30.0, 30.0], [config.AREA_WIDTH - 30.0, config.AREA_HEIGHT - 30.0])
    _set_source_and_neighbor_positions(env, rng, source_xy, neighbor_xy if len(env.uavs) > 1 else None)

    req_id = int(rng.integers(0, config.NUM_SERVICES))
    config.CPU_CYCLES_PER_BYTE[req_id] = int(rng.integers(3200, 6000))
    env.uavs[0]._current_service_request_count = int(rng.integers(10, 28))
    if len(env.uavs) > 1:
        env.uavs[1]._current_service_request_count = int(rng.integers(8, 20))

    request = _make_request(
        rng=rng,
        req_id=req_id,
        size_range=(2_000_000, 6_000_000),
        deadline_range=(0.5 * config.TIME_SLOT_DURATION, 1.2 * config.TIME_SLOT_DURATION),
    )
    covered_ues = max(env.uavs[0]._current_service_request_count + int(rng.integers(0, 4)), 1)
    return _compute_service_sample(env, request, covered_ues_count=covered_ues)


def _build_balanced_scenarios() -> list[tuple[ScenarioSpec, callable]]:
    return [
        (
            ScenarioSpec(
                name="local_cached_backhaul_bottleneck",
                intended_label=OFFLOAD_TARGET_LOCAL,
                description="Local cache hit with strongly throttled backhaul and fast local compute.",
            ),
            _scenario_local_cached_backhaul_bottleneck,
        ),
        (
            ScenarioSpec(
                name="local_compute_advantage",
                intended_label=OFFLOAD_TARGET_LOCAL,
                description="Local cache hit with faster local compute and an available but unattractive neighbor.",
            ),
            _scenario_local_compute_advantage,
        ),
        (
            ScenarioSpec(
                name="cooperative_neighbor_cached",
                intended_label=OFFLOAD_TARGET_COOPERATIVE,
                description="Heavy local queue with a nearby fast neighbor that already caches the service.",
            ),
            _scenario_cooperative_neighbor_cached,
        ),
        (
            ScenarioSpec(
                name="cooperative_queue_relief",
                intended_label=OFFLOAD_TARGET_COOPERATIVE,
                description="Heavy local queue relieved by a lightly loaded nearby neighbor under limited backhaul.",
            ),
            _scenario_cooperative_queue_relief,
        ),
        (
            ScenarioSpec(
                name="mbs_backhaul_advantage",
                intended_label=OFFLOAD_TARGET_MBS,
                description="Source UAV remains close to MBS with strong backhaul and heavy local compute demand.",
            ),
            _scenario_mbs_backhaul_advantage,
        ),
        (
            ScenarioSpec(
                name="mbs_large_job",
                intended_label=OFFLOAD_TARGET_MBS,
                description="Large service jobs near the MBS keep direct MBS offloading most attractive.",
            ),
            _scenario_mbs_large_job,
        ),
    ]


def _compute_heuristic_label(context: ServiceOffloadContext) -> int:
    if not context.cooperative_available:
        return OFFLOAD_TARGET_MBS if context.mbs_latency < context.local_latency else OFFLOAD_TARGET_LOCAL
    latencies = np.array([context.local_latency, context.cooperative_latency, context.mbs_latency], dtype=np.float32)
    return int(np.argmin(latencies))


def _collect_multi_scenario_samples(
    recorder: OffloadDatasetRecorder,
    *,
    per_class_target: int,
    max_attempts: int,
    seed: int,
) -> dict[str, object]:
    """Populate a recorder with the curated balanced template scenarios."""

    rng = np.random.default_rng(seed)
    base_snapshot = snapshot_config()
    config.SERVICE_OFFLOAD_POLICY = "heuristic"
    config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None

    env = Env()
    env.reset()

    scenario_entries = _build_balanced_scenarios()
    scenarios_by_label: dict[int, list[tuple[ScenarioSpec, callable]]] = {
        label: [entry for entry in scenario_entries if entry[0].intended_label == label] for label in range(OFFLOAD_NUM_CLASSES)
    }
    scenario_indices: dict[int, int] = {label: 0 for label in range(OFFLOAD_NUM_CLASSES)}
    attempt_counts: dict[str, int] = {spec.name: 0 for spec, _ in scenario_entries}
    accepted_counts: dict[str, int] = {spec.name: 0 for spec, _ in scenario_entries}
    raw_generated_label_counts: np.ndarray = np.zeros(OFFLOAD_NUM_CLASSES, dtype=np.int64)
    accepted_label_counts: np.ndarray = np.zeros(OFFLOAD_NUM_CLASSES, dtype=np.int64)

    try:
        for _ in range(max_attempts):
            deficits = np.maximum(per_class_target - accepted_label_counts, 0)
            if int(np.sum(deficits)) == 0:
                break

            target_label: int = int(np.argmax(deficits))
            candidate_scenarios = scenarios_by_label[target_label]
            scenario_position = scenario_indices[target_label] % len(candidate_scenarios)
            scenario_indices[target_label] += 1
            scenario_spec, scenario_sampler = candidate_scenarios[scenario_position]

            restore_config(base_snapshot)
            config.SERVICE_OFFLOAD_POLICY = "heuristic"
            config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None

            context, label = scenario_sampler(env, rng)
            raw_generated_label_counts[label] += 1
            attempt_counts[scenario_spec.name] += 1

            if accepted_label_counts[label] >= per_class_target:
                continue

            recorder.record(context, label, scenario_name=scenario_spec.name)
            accepted_counts[scenario_spec.name] += 1
            accepted_label_counts[label] += 1

        deficits = np.maximum(per_class_target - accepted_label_counts, 0)
        if int(np.sum(deficits)) != 0:
            raise RuntimeError(
                f"Multi-scenario sampling did not reach the per-class target. Remaining deficits: {deficits.tolist()}"
            )
    finally:
        restore_config(base_snapshot)

    return {
        "template_seed": seed,
        "template_per_class_target": int(per_class_target),
        "template_max_attempts": int(max_attempts),
        "template_raw_generated_label_counts": {
            OFFLOAD_TARGET_NAMES[idx]: int(raw_generated_label_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)
        },
        "template_accepted_label_counts": {
            OFFLOAD_TARGET_NAMES[idx]: int(accepted_label_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)
        },
        "template_scenario_attempt_counts": attempt_counts,
        "template_scenario_accepted_counts": accepted_counts,
        "template_scenario_descriptions": {spec.name: spec.description for spec, _ in scenario_entries},
    }


def _generate_procedural_contexts(
    recorder: OffloadDatasetRecorder,
    *,
    split: str,
    samples_per_class: int,
    seed: int,
) -> dict[str, object]:
    """Generate non-template procedural contexts aligned with the runtime rich feature family."""

    rng = np.random.default_rng(seed)
    collected = np.zeros(OFFLOAD_NUM_CLASSES, dtype=np.int64)

    max_file_size = max(float(np.max(config.FILE_SIZES)), 1.0)
    cpu_min = float(np.min(config.CPU_CYCLES_PER_BYTE))
    cpu_max = float(np.max(config.CPU_CYCLES_PER_BYTE))
    cpu_span = max(cpu_max - cpu_min, float(config.EPSILON))
    max_compute = max(float(np.max(config.UAV_COMPUTING_CAPACITY)), 1.0)
    request_size_span = max(float(config.MAX_INPUT_SIZE - config.MIN_INPUT_SIZE), 1.0)
    scenario_name = f"procedural_{split}"

    while np.any(collected < samples_per_class):
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
            raise ValueError(f"Unsupported procedural split: {split}")

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
        best_uav_uav_rate = 10.0 ** best_uav_uav_rate_log10 if cooperative_available else 0.0
        best_neighbor_mbs_rate = 10.0 ** best_neighbor_mbs_rate_log10 if cooperative_available else 0.0
        local_compute_share = max(float(config.EPSILON), local_compute_share_norm * max_compute)
        best_neighbor_compute_share = (
            max(float(config.EPSILON), best_neighbor_compute_share_norm * max_compute) if cooperative_available else 0.0
        )
        local_latency = (request_size / ue_uav_rate) + ((1.0 - float(local_cache_hit)) * file_size / uav_mbs_rate) + (cpu_cycles / local_compute_share)
        mbs_latency = (request_size / ue_uav_rate) + (request_size / uav_mbs_rate)
        cooperative_latency = float("inf")
        if cooperative_available:
            cooperative_latency = (
                (request_size / ue_uav_rate)
                + (request_size / max(best_uav_uav_rate, float(config.EPSILON)))
                + ((1.0 - best_neighbor_cache_belief) * file_size / max(best_neighbor_mbs_rate, float(config.EPSILON)))
                + (cpu_cycles / max(best_neighbor_compute_share, float(config.EPSILON)))
            )

        context = ServiceOffloadContext(
            local_latency=float(local_latency),
            cooperative_latency=float(cooperative_latency),
            mbs_latency=float(mbs_latency),
            deadline=float(deadline),
            priority=int(priority),
            local_cache_hit=bool(local_cache_hit),
            cooperative_available=bool(cooperative_available),
            local_queue_length=int(round(local_queue_fraction * float(config.MAX_ASSOCIATED_UES))),
            request_size=int(round(request_size)),
            service_file_size=int(round(file_size)),
            cpu_cycles_per_byte=float(cpu_density),
            neighbor_count=int(round(neighbor_count_fraction * max(float(config.NUM_UAVS - 1), 0.0))),
            ue_uav_rate=float(ue_uav_rate),
            uav_mbs_rate=float(uav_mbs_rate),
            best_uav_uav_rate=float(best_uav_uav_rate),
            best_neighbor_mbs_rate=float(best_neighbor_mbs_rate),
            local_compute_share=float(local_compute_share),
            best_neighbor_compute_share=float(best_neighbor_compute_share),
            best_neighbor_cache_belief=float(best_neighbor_cache_belief),
        )
        label = _compute_heuristic_label(context)
        if collected[label] >= samples_per_class:
            continue

        recorder.record(context, label, scenario_name=f"{scenario_name}_{OFFLOAD_TARGET_NAMES[label]}")
        collected[label] += 1

    return {
        f"{split}_procedural_seed": seed,
        f"{split}_procedural_samples_per_class": int(samples_per_class),
        f"{split}_procedural_label_counts": {
            OFFLOAD_TARGET_NAMES[idx]: int(collected[idx]) for idx in range(OFFLOAD_NUM_CLASSES)
        },
    }


def collect_random_rollout_dataset(
    *,
    output_path: str | Path,
    target_samples: int = 5000,
    max_steps: int = 20000,
    seed: int = config.SEED,
) -> dict[str, object]:
    """Roll the environment with random UAV trajectory actions and save imitation data."""

    np.random.seed(seed)
    config.SERVICE_OFFLOAD_POLICY = "heuristic"
    config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = None

    env = Env()
    env.reset()
    recorder = OffloadDatasetRecorder(default_scenario_name="random_rollout")

    step_in_episode: int = 0
    for _ in range(max_steps):
        actions = np.random.uniform(-1.0, 1.0, size=(config.NUM_UAVS, config.ACTION_DIM)).astype(np.float32)
        env.step(actions, sample_recorder=recorder)
        step_in_episode += 1

        if len(recorder.labels) >= target_samples:
            break
        if step_in_episode >= config.STEPS_PER_EPISODE:
            env.reset()
            step_in_episode = 0

    if not recorder.labels:
        raise RuntimeError("No service-request samples were collected. Increase max_steps or inspect environment settings.")

    label_counts = np.bincount(np.asarray(recorder.labels, dtype=np.int64), minlength=OFFLOAD_NUM_CLASSES)
    missing_classes: list[str] = [OFFLOAD_TARGET_NAMES[idx] for idx in range(OFFLOAD_NUM_CLASSES) if label_counts[idx] == 0]
    metadata: dict[str, object] = {
        "mode": "random_rollout",
        "seed": seed,
        "target_samples": int(target_samples),
        "max_steps": int(max_steps),
        "missing_classes": missing_classes,
        "notes": "Features are normalized; labels come from the runtime heuristic service offloading decision during random rollout.",
    }
    if missing_classes:
        metadata["warning"] = (
            "Some offloading classes were not observed in the collected heuristic data. "
            "The training pipeline will still run, but this dataset should be treated as a minimal closure baseline."
        )
    return _save_dataset(output_path, recorder, metadata)


def collect_multi_scenario_dataset(
    *,
    output_path: str | Path,
    per_class_target: int = 2000,
    max_attempts: int = 60000,
    seed: int = config.SEED,
) -> dict[str, object]:
    """Collect a near-balanced dataset by mixing targeted heuristic sampling scenarios."""

    recorder = OffloadDatasetRecorder(default_scenario_name="scenario_mixed")
    metadata: dict[str, object] = {
        "mode": "scenario_mixed",
        **_collect_multi_scenario_samples(
            recorder,
            per_class_target=per_class_target,
            max_attempts=max_attempts,
            seed=seed,
        ),
        "notes": (
            "This dataset mixes targeted heuristic sampling scenarios. "
            "Samples are accepted until each offloading class reaches the same target count."
        ),
    }
    return _save_dataset(output_path, recorder, metadata)


def collect_rich_candidate_dataset(
    *,
    output_path: str | Path,
    template_per_class_target: int = 2000,
    template_max_attempts: int = 60000,
    procedural_train_per_class: int = 1800,
    seed: int = config.SEED,
) -> dict[str, object]:
    """Collect the reduced-leakage runtime training dataset used by the richer learned policy."""

    recorder = OffloadDatasetRecorder(default_scenario_name="rich_candidate_mixed")
    metadata: dict[str, object] = {
        "mode": "rich_candidate_mixed",
        **_collect_multi_scenario_samples(
            recorder,
            per_class_target=template_per_class_target,
            max_attempts=template_max_attempts,
            seed=seed,
        ),
        **_generate_procedural_contexts(
            recorder,
            split="train",
            samples_per_class=procedural_train_per_class,
            seed=seed + 101,
        ),
        "notes": (
            "This runtime candidate dataset combines balanced heuristic template scenarios with procedurally generated "
            "non-template raw-state contexts. Both full-feature surrogate vectors and rich reduced-leakage vectors are saved."
        ),
    }
    return _save_dataset(output_path, recorder, metadata)


def collect_offload_dataset(
    *,
    output_path: str | Path,
    target_samples: int = 5000,
    max_steps: int = 20000,
    seed: int = config.SEED,
    mode: str = "random_rollout",
    per_class_target: int = 2000,
    max_attempts: int = 60000,
    procedural_train_per_class: int = 1800,
) -> dict[str, object]:
    if mode == "random_rollout":
        return collect_random_rollout_dataset(
            output_path=output_path,
            target_samples=target_samples,
            max_steps=max_steps,
            seed=seed,
        )
    if mode == "scenario_mixed":
        return collect_multi_scenario_dataset(
            output_path=output_path,
            per_class_target=per_class_target,
            max_attempts=max_attempts,
            seed=seed,
        )
    if mode == "rich_candidate_mixed":
        return collect_rich_candidate_dataset(
            output_path=output_path,
            template_per_class_target=per_class_target,
            template_max_attempts=max_attempts,
            procedural_train_per_class=procedural_train_per_class,
            seed=seed,
        )
    raise ValueError(f"Unsupported collection mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect request-level service offloading imitation data.")
    parser.add_argument("--output", type=str, default="offload_datasets/offload_dataset_minimal.npz", help="Path to the dataset .npz file.")
    parser.add_argument("--mode", type=str, default="random_rollout", choices=["random_rollout", "scenario_mixed", "rich_candidate_mixed"], help="Collection mode.")
    parser.add_argument("--samples", type=int, default=5000, help="Target number of service-request samples for random rollout mode.")
    parser.add_argument("--max_steps", type=int, default=20000, help="Maximum environment steps for random rollout mode.")
    parser.add_argument("--per_class_target", type=int, default=2000, help="Target number of samples per class for scenario-mixed mode.")
    parser.add_argument("--max_attempts", type=int, default=60000, help="Maximum synthetic sampling attempts for scenario-mixed mode.")
    parser.add_argument("--procedural_train_per_class", type=int, default=1800, help="Samples per class from procedural raw-state generation for rich_candidate_mixed mode.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed for dataset collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = collect_offload_dataset(
        output_path=args.output,
        target_samples=args.samples,
        max_steps=args.max_steps,
        seed=args.seed,
        mode=args.mode,
        per_class_target=args.per_class_target,
        max_attempts=args.max_attempts,
        procedural_train_per_class=args.procedural_train_per_class,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
