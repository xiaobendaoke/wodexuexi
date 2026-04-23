from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import warnings

import config
from environment import comm_model as comms
from environment.request_types import Request
from environment.user_equipments import UE
from marl_models.offload_policy import (
    OFFLOAD_TARGET_COOPERATIVE,
    OFFLOAD_TARGET_LOCAL,
    OFFLOAD_TARGET_MBS,
    ServiceOffloadContext,
    build_offload_policy,
)


def _get_belief_probability(file_id: int, neighbor_id: int) -> float:
    """Returns the estimated probability P_{v,i} that a neighbor has file_i."""
    rank: int = UE.id_to_rank_map[file_id]
    c_hat_v: float = config.UAV_STORAGE_CAPACITY[neighbor_id] / config.AVG_FILE_SIZE
    exponent: float = config.PROB_GAMMA * (rank - c_hat_v)
    probability: float = 1.0 / (1.0 + np.exp(exponent))
    return probability


def _get_computing_latency_and_energy(uav: "UAV", cpu_cycles: float) -> tuple[float, float]:
    """Calculate computing latency and energy for a UAV processing request."""
    assert uav._current_service_request_count > 0
    computing_capacity_per_request: float = config.UAV_COMPUTING_CAPACITY[uav.id] / uav._current_service_request_count
    latency: float = cpu_cycles / computing_capacity_per_request
    energy: float = config.K_CPU * cpu_cycles * (computing_capacity_per_request**2)
    return latency, energy


def _try_add_file_to_cache(uav: "UAV", file_id: int) -> None:
    """Try to add a file to UAV cache if there's enough space."""
    if uav._working_cache[file_id]:
        return
    used_space: int = int(np.sum(uav._working_cache * config.FILE_SIZES))
    if used_space + int(config.FILE_SIZES[file_id]) <= int(config.UAV_STORAGE_CAPACITY[uav.id]):
        uav._working_cache[file_id] = True


class UAV:
    _policy_cache: dict[tuple[str, str | None, float | None], dict[str, object | None]] = {}

    def __init__(self, uav_id: int) -> None:
        self.id: int = uav_id
        self.pos: np.ndarray = np.array(
            [
                np.random.uniform(0, config.AREA_WIDTH),
                np.random.uniform(0, config.AREA_HEIGHT),
                config.UAV_ALTITUDE,
            ],
            dtype=np.float32,
        )

        self._dist_moved: float = 0.0
        self._current_covered_ues: list[UE] = []
        self._neighbors: list[UAV] = []
        self._current_service_request_count: int = 0
        self._energy_current_slot: float = 0.0
        self.collision_violation: bool = False
        self.boundary_violation: bool = False

        # Cache and request tracking
        self.cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._working_cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._freq_counts: np.ndarray = np.zeros(config.NUM_FILES, dtype=np.float32)
        self._ema_scores: np.ndarray = np.zeros(config.NUM_FILES, dtype=np.float32)

        # Per-step service-request statistics
        self._service_request_count: int = 0
        self._service_offload_local_count: int = 0
        self._service_offload_cooperative_count: int = 0
        self._service_offload_mbs_count: int = 0
        self._service_learned_decision_count: int = 0
        self._service_heuristic_decision_count: int = 0
        self._service_fallback_count: int = 0
        self._service_predict_exception_fallback_count: int = 0

        self._uav_mbs_rate: float = 0.0
        policy_entry = self._get_service_offload_policy()
        self._service_offload_policy = policy_entry["policy"]
        self._service_offload_policy_requested: str = str(policy_entry["requested_policy"])
        self._service_offload_policy_loaded: bool = bool(policy_entry["loaded"])
        self._service_offload_policy_checkpoint_path: str | None = (
            str(policy_entry["checkpoint_path"]) if policy_entry["checkpoint_path"] is not None else None
        )
        self._service_offload_policy_feature_family: str | None = (
            str(policy_entry["feature_family"]) if policy_entry["feature_family"] is not None else None
        )
        self._service_offload_policy_load_error: str | None = (
            str(policy_entry["load_error"]) if policy_entry["load_error"] is not None else None
        )

    @classmethod
    def _get_service_offload_policy(cls) -> dict[str, object | None]:
        policy_name: str = getattr(config, "SERVICE_OFFLOAD_POLICY", "heuristic")
        if policy_name == "heuristic":
            return {
                "policy": None,
                "requested_policy": policy_name,
                "loaded": False,
                "checkpoint_path": None,
                "feature_family": None,
                "load_error": None,
            }
        checkpoint_path: str | None = getattr(config, "SERVICE_OFFLOAD_POLICY_CHECKPOINT", None)
        resolved_checkpoint_path: str | None = None
        if checkpoint_path is not None:
            resolved_checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        checkpoint_mtime: float | None = None
        if checkpoint_path is not None:
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.exists():
                checkpoint_mtime = checkpoint_file.stat().st_mtime
        cache_key: tuple[str, str | None, float | None] = (policy_name, checkpoint_path, checkpoint_mtime)
        if cache_key not in cls._policy_cache:
            try:
                policy = build_offload_policy(policy_name, checkpoint_path=checkpoint_path)
                checkpoint_used: str | None = resolved_checkpoint_path or str(getattr(policy, "checkpoint_path", checkpoint_path))
                feature_family: str | None = str(getattr(policy, "feature_family", None))
                cls._policy_cache[cache_key] = {
                    "policy": policy,
                    "requested_policy": policy_name,
                    "loaded": True,
                    "checkpoint_path": checkpoint_used,
                    "feature_family": feature_family,
                    "load_error": None,
                }
                print(
                    "[offload-audit] Loaded learned service offload policy "
                    f"checkpoint='{checkpoint_used}' feature_family='{feature_family}'."
                )
            except Exception as exc:
                cls._policy_cache[cache_key] = {
                    "policy": None,
                    "requested_policy": policy_name,
                    "loaded": False,
                    "checkpoint_path": resolved_checkpoint_path,
                    "feature_family": None,
                    "load_error": str(exc),
                }
                warnings.warn(
                    "[offload-audit] Service offload policy "
                    f"'{policy_name}' failed to load from checkpoint='{resolved_checkpoint_path}'. "
                    f"Falling back to heuristic. Details: {exc}",
                    RuntimeWarning,
                )
        return cls._policy_cache[cache_key]

    @property
    def energy(self) -> float:
        return self._energy_current_slot

    @property
    def current_covered_ues(self) -> list[UE]:
        return self._current_covered_ues

    @property
    def neighbors(self) -> list["UAV"]:
        return self._neighbors

    @property
    def service_request_count(self) -> int:
        return self._service_request_count

    @property
    def service_offload_local_count(self) -> int:
        return self._service_offload_local_count

    @property
    def service_offload_cooperative_count(self) -> int:
        return self._service_offload_cooperative_count

    @property
    def service_offload_mbs_count(self) -> int:
        return self._service_offload_mbs_count

    @property
    def service_offload_policy_requested(self) -> str:
        return self._service_offload_policy_requested

    @property
    def service_offload_policy_loaded(self) -> bool:
        return self._service_offload_policy_loaded

    @property
    def service_offload_policy_checkpoint_path(self) -> str | None:
        return self._service_offload_policy_checkpoint_path

    @property
    def service_offload_policy_feature_family(self) -> str | None:
        return self._service_offload_policy_feature_family

    @property
    def service_offload_policy_load_error(self) -> str | None:
        return self._service_offload_policy_load_error

    @property
    def service_learned_decision_count(self) -> int:
        return self._service_learned_decision_count

    @property
    def service_heuristic_decision_count(self) -> int:
        return self._service_heuristic_decision_count

    @property
    def service_fallback_count(self) -> int:
        return self._service_fallback_count

    @property
    def service_predict_exception_fallback_count(self) -> int:
        return self._service_predict_exception_fallback_count

    def reset_for_next_step(self) -> None:
        """Reset UAV state for a new step."""
        self._current_covered_ues = []
        self._neighbors = []
        self._current_service_request_count = 0
        self._freq_counts = np.zeros(config.NUM_FILES, dtype=np.float32)
        self._energy_current_slot = 0.0
        self._service_request_count = 0
        self._service_offload_local_count = 0
        self._service_offload_cooperative_count = 0
        self._service_offload_mbs_count = 0
        self._service_learned_decision_count = 0
        self._service_heuristic_decision_count = 0
        self._service_fallback_count = 0
        self._service_predict_exception_fallback_count = 0
        self.collision_violation = False
        self.boundary_violation = False

    def update_position(self, next_pos: np.ndarray) -> None:
        """Update the UAV's position to the new location chosen by the MARL agent."""
        new_pos: np.ndarray = np.append(next_pos, config.UAV_ALTITUDE)
        self._dist_moved = float(np.linalg.norm(new_pos - self.pos))
        self.pos = new_pos

    def set_neighbors(self, all_uavs: list["UAV"]) -> None:
        """Set neighboring UAVs within sensing range for this UAV."""
        self._neighbors = []
        for other_uav in all_uavs:
            if other_uav.id != self.id:
                distance = float(np.linalg.norm(self.pos - other_uav.pos))
                if distance <= config.UAV_SENSING_RANGE:
                    self._neighbors.append(other_uav)

    def calculate_initial_load(self) -> None:
        for ue in self._current_covered_ues:
            if ue.current_request.is_service:
                self._current_service_request_count += 1

    def process_requests(self, sample_recorder: Callable[[ServiceOffloadContext, int], None] | None = None) -> None:
        """Process requests while optionally recording heuristic service samples."""
        self._working_cache = self.cache.copy()
        self._uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(self.pos, config.MBS_POS))

        shuffled_indices: np.ndarray = np.random.permutation(len(self._current_covered_ues))
        for idx in shuffled_indices:
            ue: UE = self._current_covered_ues[idx]
            current_req: Request = ue.current_request
            if current_req.is_energy:
                self._process_energy_request(ue)
                continue

            ue_uav_rate: float = comms.calculate_ue_uav_rate(
                comms.calculate_channel_gain(ue.pos, self.pos),
                len(self._current_covered_ues),
            )

            if current_req.is_service:
                service_context, cooperative_uav = self._build_service_offload_context(current_req, ue_uav_rate)
                heuristic_target_idx, heuristic_target_uav = self._select_service_target_from_context(service_context, cooperative_uav)
                if sample_recorder is not None:
                    sample_recorder(service_context, heuristic_target_idx)
                best_target_idx, best_target_uav = self._select_service_offloading_target(
                    current_req,
                    ue_uav_rate,
                    context=service_context,
                    cooperative_uav=cooperative_uav,
                    heuristic_target_idx=heuristic_target_idx,
                    heuristic_target_uav=heuristic_target_uav,
                )
                self._service_request_count += 1
                self._record_service_offload_choice(best_target_idx)
            else:
                best_target_idx, best_target_uav = self._decide_offloading_target_heuristic(current_req, ue_uav_rate)

            self._freq_counts[current_req.req_id] += 1
            if best_target_idx == OFFLOAD_TARGET_COOPERATIVE and best_target_uav is not None:
                best_target_uav._freq_counts[current_req.req_id] += 1

            if current_req.is_service and best_target_idx != OFFLOAD_TARGET_LOCAL:
                # Optimistic relief: if this service is sent away, following users see the updated queue.
                self._current_service_request_count = max(0, self._current_service_request_count - 1)
                if best_target_idx == OFFLOAD_TARGET_COOPERATIVE and best_target_uav is not None:
                    best_target_uav._current_service_request_count += 1

            if current_req.is_service:
                self._process_service_request(ue, ue_uav_rate, best_target_idx, best_target_uav)
            else:
                self._process_content_request(ue, ue_uav_rate, best_target_idx, best_target_uav)

            assert ue.latency_current_request >= 0.0

    def _record_service_offload_choice(self, target_idx: int) -> None:
        if target_idx == OFFLOAD_TARGET_LOCAL:
            self._service_offload_local_count += 1
        elif target_idx == OFFLOAD_TARGET_COOPERATIVE:
            self._service_offload_cooperative_count += 1
        else:
            self._service_offload_mbs_count += 1

    def _build_service_offload_context(self, current_req: Request, ue_uav_rate: float) -> tuple[ServiceOffloadContext, "UAV" | None]:
        """Build the normalized policy context from the exact runtime heuristic state."""
        req_id: int = current_req.req_id
        service_load: int = max(self._current_service_request_count, 1)
        local_compute_share: float = float(config.UAV_COMPUTING_CAPACITY[self.id]) / float(service_load)
        local_latency: float = self._estimate_local_service_latency(current_req, ue_uav_rate)
        (
            cooperative_latency,
            cooperative_uav,
            best_uav_uav_rate,
            best_neighbor_mbs_rate,
            best_neighbor_compute_share,
            best_neighbor_cache_belief,
        ) = self._estimate_best_cooperative_service_candidate(current_req, ue_uav_rate)
        mbs_latency: float = self._estimate_mbs_service_latency(current_req, ue_uav_rate)

        context = ServiceOffloadContext(
            local_latency=local_latency,
            cooperative_latency=cooperative_latency,
            mbs_latency=mbs_latency,
            deadline=current_req.deadline,
            priority=current_req.priority,
            local_cache_hit=bool(self.cache[current_req.req_id]),
            cooperative_available=cooperative_uav is not None,
            local_queue_length=self._current_service_request_count,
            request_size=current_req.req_size,
            service_file_size=int(config.FILE_SIZES[req_id]),
            cpu_cycles_per_byte=float(config.CPU_CYCLES_PER_BYTE[req_id]),
            neighbor_count=len(self._neighbors),
            ue_uav_rate=ue_uav_rate,
            uav_mbs_rate=self._uav_mbs_rate,
            best_uav_uav_rate=best_uav_uav_rate,
            best_neighbor_mbs_rate=best_neighbor_mbs_rate,
            local_compute_share=local_compute_share,
            best_neighbor_compute_share=best_neighbor_compute_share,
            best_neighbor_cache_belief=best_neighbor_cache_belief,
        )
        return context, cooperative_uav

    def _select_service_target_from_context(
        self, context: ServiceOffloadContext, cooperative_uav: "UAV" | None
    ) -> tuple[int, "UAV" | None]:
        """Resolve the original latency-minimizing heuristic target from a precomputed context."""
        best_target_idx: int = OFFLOAD_TARGET_LOCAL
        best_target_uav: UAV | None = None
        best_exp_latency: float = context.local_latency

        if context.mbs_latency < best_exp_latency:
            best_exp_latency = context.mbs_latency
            best_target_idx = OFFLOAD_TARGET_MBS

        if cooperative_uav is not None and context.cooperative_latency < best_exp_latency:
            best_target_idx = OFFLOAD_TARGET_COOPERATIVE
            best_target_uav = cooperative_uav

        return best_target_idx, best_target_uav

    def _select_service_offloading_target(
        self,
        current_req: Request,
        ue_uav_rate: float,
        *,
        context: ServiceOffloadContext | None = None,
        cooperative_uav: "UAV" | None = None,
        heuristic_target_idx: int | None = None,
        heuristic_target_uav: "UAV" | None = None,
    ) -> tuple[int, "UAV" | None]:
        """Choose a service offload category, then resolve a cooperative UAV heuristically."""
        if context is None or heuristic_target_idx is None:
            context, cooperative_uav = self._build_service_offload_context(current_req, ue_uav_rate)
            heuristic_target_idx, heuristic_target_uav = self._select_service_target_from_context(context, cooperative_uav)

        if getattr(config, "SERVICE_OFFLOAD_POLICY", "heuristic") == "heuristic":
            self._service_heuristic_decision_count += 1
            return heuristic_target_idx, heuristic_target_uav

        if self._service_offload_policy is None:
            self._service_heuristic_decision_count += 1
            self._service_fallback_count += 1
            return heuristic_target_idx, heuristic_target_uav

        try:
            target_idx: int = int(self._service_offload_policy.predict(context))
        except Exception as exc:
            self._service_heuristic_decision_count += 1
            self._service_fallback_count += 1
            self._service_predict_exception_fallback_count += 1
            warnings.warn(
                "[offload-audit] Learned service offload predict() failed on "
                f"UAV={self.id} checkpoint='{self._service_offload_policy_checkpoint_path}' "
                f"feature_family='{self._service_offload_policy_feature_family}'. "
                f"Falling back to heuristic. Details: {exc}",
                RuntimeWarning,
            )
            return heuristic_target_idx, heuristic_target_uav

        if target_idx == OFFLOAD_TARGET_LOCAL:
            self._service_learned_decision_count += 1
            return OFFLOAD_TARGET_LOCAL, None
        if target_idx == OFFLOAD_TARGET_COOPERATIVE and cooperative_uav is not None:
            self._service_learned_decision_count += 1
            return OFFLOAD_TARGET_COOPERATIVE, cooperative_uav
        if target_idx == OFFLOAD_TARGET_MBS:
            self._service_learned_decision_count += 1
            return OFFLOAD_TARGET_MBS, None

        self._service_heuristic_decision_count += 1
        self._service_fallback_count += 1
        return heuristic_target_idx, heuristic_target_uav

    def _decide_offloading_target_heuristic(self, current_req: Request, ue_uav_rate: float) -> tuple[int, "UAV" | None]:
        """Original latency-based heuristic preserved as baseline and fallback."""
        if current_req.is_service:
            context, cooperative_uav = self._build_service_offload_context(current_req, ue_uav_rate)
            return self._select_service_target_from_context(context, cooperative_uav)

        best_exp_latency = self._estimate_local_content_latency(current_req, ue_uav_rate)
        best_target_idx = OFFLOAD_TARGET_LOCAL
        best_target_uav = None

        exp_mbs_latency = self._estimate_mbs_content_latency(current_req, ue_uav_rate)
        if exp_mbs_latency < best_exp_latency:
            best_exp_latency = exp_mbs_latency
            best_target_idx = OFFLOAD_TARGET_MBS

        exp_neighbor_latency, neighbor_uav = self._estimate_best_cooperative_content_latency(current_req, ue_uav_rate)
        if neighbor_uav is not None and exp_neighbor_latency < best_exp_latency:
            best_target_idx = OFFLOAD_TARGET_COOPERATIVE
            best_target_uav = neighbor_uav
        return best_target_idx, best_target_uav

    def _estimate_local_service_latency(self, current_req: Request, ue_uav_rate: float) -> float:
        req_size: int = current_req.req_size
        req_id: int = current_req.req_id
        file_size: int = int(config.FILE_SIZES[req_id])
        cpu_cycles: float = float(config.CPU_CYCLES_PER_BYTE[req_id]) * float(req_size)
        p_local: float = 1.0 if self.cache[req_id] else 0.0
        ue_uav_upload_latency: float = req_size / ue_uav_rate
        exp_fetch_latency: float = (1.0 - p_local) * (file_size / self._uav_mbs_rate)
        service_load: int = max(self._current_service_request_count, 1)
        est_comp_latency: float = cpu_cycles / (config.UAV_COMPUTING_CAPACITY[self.id] / service_load)
        return ue_uav_upload_latency + exp_fetch_latency + est_comp_latency

    def _estimate_mbs_service_latency(self, current_req: Request, ue_uav_rate: float) -> float:
        ue_uav_upload_latency: float = current_req.req_size / ue_uav_rate
        uav_mbs_upload_latency: float = current_req.req_size / self._uav_mbs_rate
        return ue_uav_upload_latency + uav_mbs_upload_latency

    def _estimate_best_cooperative_service_candidate(
        self,
        current_req: Request,
        ue_uav_rate: float,
    ) -> tuple[float, "UAV" | None, float, float, float, float]:
        """Estimate the best cooperative UAV and expose its raw runtime state for learned policies."""
        req_size: int = current_req.req_size
        req_id: int = current_req.req_id
        file_size: int = int(config.FILE_SIZES[req_id])
        cpu_cycles: float = float(config.CPU_CYCLES_PER_BYTE[req_id]) * float(req_size)
        ue_uav_upload_latency: float = req_size / ue_uav_rate

        best_latency: float = np.inf
        best_neighbor: UAV | None = None
        best_uav_uav_rate: float = 0.0
        best_neighbor_mbs_rate: float = 0.0
        best_neighbor_compute_share: float = 0.0
        best_neighbor_cache_belief: float = 0.0
        for neighbor in self._neighbors:
            belief_prob: float = _get_belief_probability(req_id, neighbor.id)
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, neighbor.pos))
            neighbor_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(neighbor.pos, config.MBS_POS))
            exp_neighbor_fetch_latency: float = (1.0 - belief_prob) * (file_size / neighbor_mbs_rate)
            neigh_load: int = max(neighbor._current_service_request_count + 1, 1)
            neighbor_compute_share: float = float(config.UAV_COMPUTING_CAPACITY[neighbor.id]) / float(neigh_load)
            est_comp_latency: float = cpu_cycles / neighbor_compute_share
            uav_uav_upload_latency: float = req_size / uav_uav_rate
            exp_neighbor_latency: float = ue_uav_upload_latency + uav_uav_upload_latency + exp_neighbor_fetch_latency + est_comp_latency
            if exp_neighbor_latency < best_latency:
                best_latency = exp_neighbor_latency
                best_neighbor = neighbor
                best_uav_uav_rate = uav_uav_rate
                best_neighbor_mbs_rate = neighbor_mbs_rate
                best_neighbor_compute_share = neighbor_compute_share
                best_neighbor_cache_belief = belief_prob

        return (
            best_latency,
            best_neighbor,
            best_uav_uav_rate,
            best_neighbor_mbs_rate,
            best_neighbor_compute_share,
            best_neighbor_cache_belief,
        )

    def _estimate_best_cooperative_service_latency(self, current_req: Request, ue_uav_rate: float) -> tuple[float, "UAV" | None]:
        best_latency, best_neighbor, _, _, _, _ = self._estimate_best_cooperative_service_candidate(current_req, ue_uav_rate)
        return best_latency, best_neighbor

    def _estimate_local_content_latency(self, current_req: Request, ue_uav_rate: float) -> float:
        req_id: int = current_req.req_id
        file_size: int = int(config.FILE_SIZES[req_id])
        p_local: float = 1.0 if self.cache[req_id] else 0.0
        ue_uav_download_latency: float = file_size / ue_uav_rate
        exp_fetch_latency: float = (1.0 - p_local) * (file_size / self._uav_mbs_rate)
        return exp_fetch_latency + ue_uav_download_latency

    def _estimate_mbs_content_latency(self, current_req: Request, ue_uav_rate: float) -> float:
        file_size: int = int(config.FILE_SIZES[current_req.req_id])
        uav_mbs_download_latency: float = file_size / self._uav_mbs_rate
        ue_uav_download_latency: float = file_size / ue_uav_rate
        return uav_mbs_download_latency + ue_uav_download_latency

    def _estimate_best_cooperative_content_latency(self, current_req: Request, ue_uav_rate: float) -> tuple[float, "UAV" | None]:
        req_id: int = current_req.req_id
        file_size: int = int(config.FILE_SIZES[req_id])
        ue_uav_download_latency: float = file_size / ue_uav_rate

        best_latency: float = np.inf
        best_neighbor: UAV | None = None
        for neighbor in self._neighbors:
            belief_prob: float = _get_belief_probability(req_id, neighbor.id)
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, neighbor.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(neighbor.pos, config.MBS_POS))
            uav_uav_download_latency: float = file_size / uav_uav_rate
            exp_neighbor_fetch_latency: float = (1.0 - belief_prob) * (file_size / uav_mbs_rate)
            exp_neighbor_latency: float = exp_neighbor_fetch_latency + uav_uav_download_latency + ue_uav_download_latency
            if exp_neighbor_latency < best_latency:
                best_latency = exp_neighbor_latency
                best_neighbor = neighbor

        return best_latency, best_neighbor

    def _process_service_request(self, ue: UE, ue_uav_rate: float, target_idx: int, target_uav: "UAV" | None) -> None:
        current_req: Request = ue.current_request
        req_size: int = current_req.req_size
        req_id: int = current_req.req_id
        assert req_id < config.NUM_SERVICES
        cpu_cycles: float = float(config.CPU_CYCLES_PER_BYTE[req_id]) * float(req_size)
        file_size: int = int(config.FILE_SIZES[req_id])

        ue_uav_upload_latency: float = req_size / ue_uav_rate
        ue.update_battery(0.0, ue_uav_upload_latency)
        if target_idx == OFFLOAD_TARGET_LOCAL:
            fetch_latency: float = 0.0
            if not self.cache[req_id]:
                fetch_latency = file_size / self._uav_mbs_rate
                _try_add_file_to_cache(self, req_id)

            comp_latency, comp_energy = _get_computing_latency_and_energy(self, cpu_cycles)
            ue.latency_current_request = ue_uav_upload_latency + fetch_latency + comp_latency
            self._energy_current_slot += comp_energy

        elif target_idx == OFFLOAD_TARGET_COOPERATIVE:
            assert target_uav is not None
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, target_uav.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(target_uav.pos, config.MBS_POS))
            uav_uav_upload_latency: float = req_size / uav_uav_rate

            fetch_latency = 0.0
            if not target_uav.cache[req_id]:
                fetch_latency = file_size / uav_mbs_rate
                _try_add_file_to_cache(target_uav, req_id)

            comp_latency, comp_energy = _get_computing_latency_and_energy(target_uav, cpu_cycles)
            ue.latency_current_request = ue_uav_upload_latency + uav_uav_upload_latency + fetch_latency + comp_latency
            target_uav._energy_current_slot += comp_energy
            _try_add_file_to_cache(self, req_id)

        else:
            uav_mbs_upload_latency: float = req_size / self._uav_mbs_rate
            ue.latency_current_request = ue_uav_upload_latency + uav_mbs_upload_latency
            _try_add_file_to_cache(self, req_id)

    def _process_content_request(self, ue: UE, ue_uav_rate: float, target_idx: int, target_uav: "UAV" | None) -> None:
        current_req: Request = ue.current_request
        req_id: int = current_req.req_id
        assert req_id >= config.NUM_SERVICES
        file_size: int = int(config.FILE_SIZES[req_id])

        ue_uav_download_latency: float = file_size / ue_uav_rate
        ue.update_battery(0.0, 0.0)
        if target_idx == OFFLOAD_TARGET_LOCAL:
            fetch_latency: float = 0.0
            if not self.cache[req_id]:
                fetch_latency = file_size / self._uav_mbs_rate
                _try_add_file_to_cache(self, req_id)

            ue.latency_current_request = fetch_latency + ue_uav_download_latency

        elif target_idx == OFFLOAD_TARGET_COOPERATIVE:
            assert target_uav is not None
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, target_uav.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(target_uav.pos, config.MBS_POS))
            uav_uav_download_latency: float = file_size / uav_uav_rate

            fetch_latency = 0.0
            if not target_uav.cache[req_id]:
                fetch_latency = file_size / uav_mbs_rate
                _try_add_file_to_cache(target_uav, req_id)

            ue.latency_current_request = fetch_latency + uav_uav_download_latency + ue_uav_download_latency
            _try_add_file_to_cache(self, req_id)

        else:
            uav_mbs_download_latency: float = file_size / self._uav_mbs_rate
            ue.latency_current_request = uav_mbs_download_latency + ue_uav_download_latency
            _try_add_file_to_cache(self, req_id)

    def _process_energy_request(self, ue: UE) -> None:
        """Process an emergency energy request from a UE."""
        channel_gain: float = comms.calculate_channel_gain(self.pos, ue.pos)
        harv_energy: float = config.WPT_EFFICIENCY * config.WPT_TRANSMIT_POWER * channel_gain * config.TIME_SLOT_DURATION
        ue.update_battery(harv_energy, 0.0)
        ue.latency_current_request = 0.0

    def update_ema_and_cache(self) -> None:
        """Update EMA scores and cache reactively."""
        self._ema_scores = config.GDSF_SMOOTHING_FACTOR * self._freq_counts + (1 - config.GDSF_SMOOTHING_FACTOR) * self._ema_scores
        self.cache = self._working_cache.copy()

    def gdsf_cache_update(self) -> None:
        """Update cache using the GDSF caching policy at a longer timescale."""
        priority_scores: np.ndarray = self._ema_scores / config.FILE_SIZES
        sorted_file_ids: np.ndarray = np.argsort(-priority_scores)
        self.cache = np.zeros(config.NUM_FILES, dtype=bool)
        used_space = 0.0
        for file_id in sorted_file_ids:
            file_size = config.FILE_SIZES[file_id]
            if used_space + file_size <= config.UAV_STORAGE_CAPACITY[self.id]:
                self.cache[file_id] = True
                used_space += file_size
            else:
                break

    def update_energy_consumption(self) -> None:
        """Update UAV energy consumption for the current time slot."""
        time_moving: float = self._dist_moved / config.UAV_SPEED
        time_hovering: float = config.TIME_SLOT_DURATION - time_moving
        fly_energy: float = config.POWER_MOVE * time_moving + config.POWER_HOVER * time_hovering
        self._energy_current_slot += fly_energy
        has_energy_request: bool = any(ue.current_request.is_energy for ue in self._current_covered_ues)
        if has_energy_request:
            self._energy_current_slot += config.WPT_TRANSMIT_POWER * config.TIME_SLOT_DURATION
