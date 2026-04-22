import numpy as np

import config
from environment.uavs import UAV
from environment.user_equipments import UE


class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0
        self._last_step_stats: dict[str, float] = {}

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    @property
    def last_step_stats(self) -> dict[str, float]:
        return self._last_step_stats

    def reset(self, initial_positions: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """Resets the environment to an initial state and returns the initial observations."""
        if getattr(config, "USE_HOTSPOTS", False):
            UE.generate_hotspots()

        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]

        if initial_positions is not None:
            for i, uav in enumerate(self._uavs):
                uav.pos[:2] = initial_positions[i]

        self._time_step = 0
        self._last_step_stats = {}
        return self._get_obs()

    def step(self, actions: np.ndarray, sample_recorder=None) -> tuple[list[np.ndarray], list[float], dict[str, float]]:
        """Execute one time step of the simulation.

        sample_recorder is an optional callable used by the request-level
        classifier dataset pipeline to record heuristic service decisions.
        """
        self._time_step += 1

        for uav in self._uavs:
            uav.calculate_initial_load()

        for uav in self._uavs:
            uav.process_requests(sample_recorder=sample_recorder)

        for ue in self._ues:
            if not ue.assigned:
                ue.update_battery(0.0, 0.0)
            ue.update_service_coverage(self._time_step)

        for uav in self._uavs:
            uav.update_ema_and_cache()
            uav.update_energy_consumption()

        rewards, metrics = self._get_rewards_and_metrics()
        self._last_step_stats = metrics.copy()

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        for ue in self._ues:
            ue.update_position()

        for uav in self._uavs:
            uav.reset_for_next_step()

        self._apply_actions_to_env(actions)

        next_obs: list[np.ndarray] = self._get_obs()
        return next_obs, rewards, metrics

    def _get_obs(self) -> list[np.ndarray]:
        """Gets the local observation for each UAV agent."""
        for ue in self._ues:
            ue.generate_request()
        self._associate_ues_to_uavs()
        for uav in self._uavs:
            uav.set_neighbors(self._uavs)

        all_obs: list[np.ndarray] = []
        for uav in self._uavs:
            own_pos: np.ndarray = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
            own_cache: np.ndarray = uav.cache.astype(np.float32)
            own_state: np.ndarray = np.concatenate([own_pos, own_cache])

            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_OBS_DIM), dtype=np.float32)
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS]
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                neighbor_states[i, :] = relative_pos

            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, config.UE_OBS_DIM), dtype=np.float32)
            ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos[:2] - u.pos[:2])))[: config.MAX_ASSOCIATED_UES]
            for i, ue in enumerate(ues):
                delta_pos: np.ndarray = (ue.pos[:2] - uav.pos[:2]) / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
                request = ue.current_request
                norm_type: float = float(request.req_type) / 2.0
                norm_id: float = float(request.req_id) / float(config.NUM_FILES)
                norm_size: float = float(request.req_size) / float(config.MAX_INPUT_SIZE)
                norm_deadline: float = float(request.deadline) / float(config.SERVICE_DEADLINE_MAX) if request.is_service else 0.0
                norm_priority: float = float(request.priority) / float(config.SERVICE_PRIORITY_MAX) if request.is_service else 0.0
                norm_battery: float = ue.battery_level / config.UE_BATTERY_CAPACITY
                request_info: np.ndarray = np.array(
                    [norm_type, norm_size, norm_id, norm_deadline, norm_priority, norm_battery],
                    dtype=np.float32,
                )
                ue_states[i, :] = np.concatenate([delta_pos, request_info])

            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            assert obs.size == config.OBS_DIM_SINGLE, f"Observation dimension mismatch: {obs.size} != {config.OBS_DIM_SINGLE}"
            all_obs.append(obs)

        return all_obs

    def _apply_actions_to_env(self, actions: np.ndarray) -> None:
        """Calculates next positions and resolves potential collisions iteratively."""
        current_positions: np.ndarray = np.array([uav.pos[:2] for uav in self._uavs], dtype=np.float32)
        max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION

        delta_vec_raw: np.ndarray = np.array(actions, dtype=np.float32)
        raw_magnitude: np.ndarray = np.linalg.norm(delta_vec_raw, axis=1, keepdims=True)

        clipped_magnitude: np.ndarray = np.minimum(raw_magnitude, 1.0)
        distances: np.ndarray = clipped_magnitude * max_dist
        denom: np.ndarray = raw_magnitude + float(config.EPSILON)
        directions: np.ndarray = delta_vec_raw / denom
        delta_pos: np.ndarray = directions * distances

        proposed_positions: np.ndarray = current_positions + delta_pos

        min_boundary_gap: float = config.UAV_COVERAGE_RADIUS / 2.0
        for i, uav in enumerate(self._uavs):
            if not (
                min_boundary_gap <= proposed_positions[i, 0] <= config.AREA_WIDTH - min_boundary_gap
                and min_boundary_gap <= proposed_positions[i, 1] <= config.AREA_HEIGHT - min_boundary_gap
            ):
                uav.boundary_violation = True
        next_positions: np.ndarray = np.clip(
            proposed_positions,
            [min_boundary_gap, min_boundary_gap],
            [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap],
        )

        min_sep_sq: float = config.MIN_UAV_SEPARATION**2
        for _ in range(config.COLLISION_AVOIDANCE_ITERATIONS + 1):
            collision_detected_in_iter: bool = False
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    pos_i: np.ndarray = next_positions[i]
                    pos_j: np.ndarray = next_positions[j]
                    dist_sq: float = np.sum((pos_i - pos_j) ** 2)
                    if dist_sq < min_sep_sq:
                        self._uavs[i].collision_violation = True
                        self._uavs[j].collision_violation = True
                        collision_detected_in_iter = True
                        dist: float = np.sqrt(dist_sq) if dist_sq > 0 else config.EPSILON
                        overlap: float = config.MIN_UAV_SEPARATION - dist
                        direction: np.ndarray = (pos_i - pos_j) / dist
                        next_positions[i] += direction * overlap * 0.5
                        next_positions[j] -= direction * overlap * 0.5
            if not collision_detected_in_iter:
                break

        final_positions: np.ndarray = np.clip(
            next_positions,
            [min_boundary_gap, min_boundary_gap],
            [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap],
        )
        for i, uav in enumerate(self._uavs):
            uav.update_position(final_positions[i])

    def _associate_ues_to_uavs(self) -> None:
        """Assign each UE to at most one UAV, resolving overlaps by choosing the closest UAV."""
        for ue in self._ues:
            covering_uavs: list[tuple[UAV, float]] = []
            for uav in self._uavs:
                distance: float = float(np.linalg.norm(uav.pos[:2] - ue.pos[:2]))
                if distance <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance))

            if not covering_uavs:
                continue
            best_uav, _ = min(covering_uavs, key=lambda x: x[1])
            best_uav.current_covered_ues.append(ue)
            ue.assigned = True

    def _get_rewards_and_metrics(self) -> tuple[list[float], dict[str, float]]:
        """Return the reward and tracked metrics for the current step."""
        total_latency: float = sum(
            ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues
        )
        total_energy: float = sum(uav.energy for uav in self._uavs)
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues], dtype=np.float32)
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))
        offline_count: int = sum(1 for ue in self._ues if ue.battery_level < config.UE_CRITICAL_THRESHOLD)
        offline_rate: float = offline_count / config.NUM_UES

        total_service_requests_generated: int = sum(1 for ue in self._ues if ue.current_request.is_service)
        total_service_requests_processed: int = sum(uav.service_request_count for uav in self._uavs)
        deadline_satisfied_count: int = sum(1 for ue in self._ues if ue.is_service_deadline_satisfied())
        total_local_offloads: int = sum(uav.service_offload_local_count for uav in self._uavs)
        total_cooperative_offloads: int = sum(uav.service_offload_cooperative_count for uav in self._uavs)
        total_mbs_offloads: int = sum(uav.service_offload_mbs_count for uav in self._uavs)
        assert total_local_offloads + total_cooperative_offloads + total_mbs_offloads == total_service_requests_processed

        # Convention:
        # - deadline_satisfaction_rate and mbs_load_ratio are normalized by all generated service requests.
        # - offloading ratios are normalized by processed service requests only.
        # - when there are no requests for a denominator, the ratio is defined as 0.0.
        deadline_satisfaction_rate: float = 0.0
        mbs_load_ratio: float = 0.0
        if total_service_requests_generated > 0:
            deadline_satisfaction_rate = deadline_satisfied_count / total_service_requests_generated
            mbs_load_ratio = total_mbs_offloads / total_service_requests_generated

        offloading_ratio_local: float = 0.0
        offloading_ratio_cooperative: float = 0.0
        offloading_ratio_mbs: float = 0.0
        if total_service_requests_processed > 0:
            offloading_ratio_local = total_local_offloads / total_service_requests_processed
            offloading_ratio_cooperative = total_cooperative_offloads / total_service_requests_processed
            offloading_ratio_mbs = total_mbs_offloads / total_service_requests_processed
            assert np.isclose(offloading_ratio_local + offloading_ratio_cooperative + offloading_ratio_mbs, 1.0)

        r_fairness: float = config.ALPHA_3 * np.log(jfi + config.EPSILON)
        r_latency: float = config.ALPHA_1 * np.log(total_latency + config.EPSILON)
        r_energy: float = config.ALPHA_2 * np.log(total_energy + config.EPSILON)
        r_offline: float = config.ALPHA_4 * np.log(1.0 + offline_rate)
        reward: float = r_fairness - r_latency - r_energy - r_offline
        rewards: list[float] = [reward] * config.NUM_UAVS
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]

        metrics: dict[str, float] = {
            "latency": total_latency,
            "energy": total_energy,
            "fairness": jfi,
            "offline_rate": offline_rate,
            "deadline_satisfaction_rate": deadline_satisfaction_rate,
            "offloading_ratio_local": offloading_ratio_local,
            "offloading_ratio_cooperative": offloading_ratio_cooperative,
            "offloading_ratio_mbs": offloading_ratio_mbs,
            "mbs_load_ratio": mbs_load_ratio,
            "service_requests_generated": float(total_service_requests_generated),
            "service_requests_processed": float(total_service_requests_processed),
            "service_offloads_local": float(total_local_offloads),
            "service_offloads_cooperative": float(total_cooperative_offloads),
            "service_offloads_mbs": float(total_mbs_offloads),
        }
        return rewards, metrics
