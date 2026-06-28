from __future__ import annotations

"""
中文注释说明：environment/env.py

文件作用：
    定义多无人机移动边缘计算仿真环境，维护无人机、用户设备、任务请求、通信链路和奖励计算。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - Env: 仿真环境对象，承载无人机、用户设备、任务请求和奖励计算。

主要依赖：
    numpy, config, environment

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

import numpy as np

import config
from environment.uavs import UAV
from environment.user_equipments import UE


# 类 Env：仿真环境对象，承载无人机、用户设备、任务请求和奖励计算。
class Env:
    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑。
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0
        self._last_step_stats: dict[str, float] = {}
        self._last_runtime_audit: dict[str, object] = {}
        self._last_admission_stats: dict[str, int] = {}
        self._episode_runtime_audit_totals: dict[str, int] = self._make_empty_runtime_audit_totals()

    # 函数 uavs：关键函数，承载本模块的一段可复用实验逻辑。
    @property
    def uavs(self) -> list[UAV]:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self._uavs

    # 函数 ues：关键函数，承载本模块的一段可复用实验逻辑。
    @property
    def ues(self) -> list[UE]:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self._ues

    # 函数 last_step_stats：关键函数，承载本模块的一段可复用实验逻辑。
    @property
    def last_step_stats(self) -> dict[str, float]:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self._last_step_stats

    # 函数 last_runtime_audit：关键函数，承载本模块的一段可复用实验逻辑。
    @property
    def last_runtime_audit(self) -> dict[str, object]:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self._last_runtime_audit

    # 函数 _make_empty_runtime_audit_totals：关键函数，承载本模块的一段可复用实验逻辑。
    @staticmethod
    def _make_empty_runtime_audit_totals() -> dict[str, int]:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {
            "fallback_count": 0,
            "predict_exception_fallback_count": 0,
            "learned_decision_count": 0,
            "heuristic_decision_count": 0,
        }

    # 函数 reset：重置环境或对象状态，开始新的回合，主要参数：initial_positions。
    def reset(self, initial_positions: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """Resets the environment to an initial state and returns the initial observations."""
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if getattr(config, "USE_HOTSPOTS", False):
            UE.generate_hotspots()

        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if initial_positions is not None:
            # 循环处理：遍历 (i, uav) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for i, uav in enumerate(self._uavs):
                uav.pos[:2] = initial_positions[i]

        self._time_step = 0
        self._last_step_stats = {}
        self._last_runtime_audit = {}
        self._last_admission_stats = {}
        self._episode_runtime_audit_totals = self._make_empty_runtime_audit_totals()
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self._get_obs()

    # 函数 step：推进环境一个时间步并返回状态转移结果，主要参数：actions, sample_recorder。
    def step(
        self,
        actions: np.ndarray,
        sample_recorder=None,
        offloading_actions: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[float], dict[str, float]]:
        """Execute one time step of the simulation.

        sample_recorder is an optional callable used by the request-level
        classifier dataset pipeline to record heuristic service decisions.
        """
        self._time_step += 1

        # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for uav in self._uavs:
            uav._current_service_request_count = 0
            uav.calculate_initial_load()

        # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for uav_idx, uav in enumerate(self._uavs):
            uav_offload_actions = None
            if offloading_actions is not None:
                uav_offload_actions = np.asarray(offloading_actions[uav_idx], dtype=np.int64)
            uav.process_requests(sample_recorder=sample_recorder, offload_actions=uav_offload_actions)

        # 循环处理：遍历 ue 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for ue in self._ues:
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if not ue.assigned:
                ue.update_battery(0.0, 0.0)
            ue.update_service_coverage(self._time_step)

        # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for uav in self._uavs:
            uav.update_ema_and_cache()
            uav.update_energy_consumption()

        rewards, metrics = self._get_rewards_and_metrics()
        self._last_step_stats = metrics.copy()
        self._last_runtime_audit = self._collect_runtime_audit()

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for uav in self._uavs:
                uav.gdsf_cache_update()

        # 循环处理：遍历 ue 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for ue in self._ues:
            ue.update_position()

        # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for uav in self._uavs:
            uav.reset_for_next_step()

        self._apply_actions_to_env(actions)

        next_obs: list[np.ndarray] = self._get_obs()
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return next_obs, rewards, metrics

    def get_offloading_obs_and_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower-layer request-level observations and legal action masks."""
        for uav in self._uavs:
            uav._current_service_request_count = 0
            uav.calculate_initial_load()

        all_obs = np.zeros((config.NUM_UAVS, config.OFFLOAD_OBS_DIM_SINGLE), dtype=np.float32)
        all_masks = np.zeros(
            (config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV, config.OFFLOAD_NUM_ACTIONS),
            dtype=np.float32,
        )
        coop_masked_count = 0
        mbs_masked_count = 0

        for uav_idx, uav in enumerate(self._uavs):
            uav_mbs_rate = getattr(uav, "_uav_mbs_rate", 0.0)
            if uav_mbs_rate <= 0.0:
                from environment import comm_model as comms

                uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(uav.pos, config.MBS_POS))
                uav._uav_mbs_rate = float(uav_mbs_rate)

            own_pos = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
            own_queue = float(getattr(uav, "_current_service_request_count", 0)) / max(float(config.MAX_ASSOCIATED_UES), 1.0)
            own_neighbor_count = float(len(uav.neighbors)) / max(float(config.MAX_UAV_NEIGHBORS), 1.0)
            own_mbs_rate = np.log10(max(float(uav_mbs_rate), float(config.EPSILON))) / 10.0
            own_features = np.array([own_pos[0], own_pos[1], own_queue, own_neighbor_count, own_mbs_rate], dtype=np.float32)

            request_features = np.zeros(
                (config.MAX_OFFLOAD_REQUESTS_PER_UAV, config.OFFLOAD_REQUEST_FEATURE_DIM),
                dtype=np.float32,
            )
            service_ues = [ue for ue in uav.current_covered_ues if ue.current_request.is_service]
            service_ues = sorted(service_ues, key=lambda ue: float(np.linalg.norm(uav.pos[:2] - ue.pos[:2])))
            service_ues = service_ues[: config.MAX_OFFLOAD_REQUESTS_PER_UAV]

            for req_idx, ue in enumerate(service_ues):
                request = ue.current_request
                from environment import comm_model as comms

                ue_uav_rate = comms.calculate_ue_uav_rate(
                    comms.calculate_channel_gain(ue.pos, uav.pos),
                    max(len(uav.current_covered_ues), 1),
                )
                context, _ = uav._build_service_offload_context(request, ue_uav_rate)
                local_ratio = self._safe_ratio(context.local_latency, context.deadline)
                coop_ratio = self._safe_ratio(context.cooperative_latency, context.deadline)
                mbs_ratio = self._safe_ratio(context.mbs_latency, context.deadline)
                delta_pos = (ue.pos[:2] - uav.pos[:2]) / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
                request_features[req_idx] = np.array(
                    [
                        1.0,
                        float(request.req_size) / float(config.MAX_INPUT_SIZE),
                        float(request.req_id) / float(max(config.NUM_SERVICES, 1)),
                        float(request.deadline) / float(config.SERVICE_DEADLINE_MAX),
                        float(request.priority) / float(config.SERVICE_PRIORITY_MAX),
                        float(context.local_cache_hit),
                        float(context.cooperative_available),
                        float(context.local_queue_length) / max(float(config.MAX_ASSOCIATED_UES), 1.0),
                        local_ratio,
                        coop_ratio,
                        mbs_ratio,
                        float(context.best_neighbor_compute_share) / max(float(np.max(config.UAV_COMPUTING_CAPACITY)), 1.0),
                        float(delta_pos[0]),
                        float(delta_pos[1]),
                    ],
                    dtype=np.float32,
                )
                all_masks[uav_idx, req_idx, config.OFFLOAD_ACTION_LOCAL] = 1.0
                all_masks[uav_idx, req_idx, config.OFFLOAD_ACTION_MBS] = 1.0
                best_noncoop_latency = min(float(context.local_latency), float(context.mbs_latency))
                local_compute_share = max(float(context.local_compute_share), float(config.EPSILON))
                req_size = float(request.req_size)
                file_size = float(config.FILE_SIZES[request.req_id])
                cpu_cycles = float(config.CPU_CYCLES_PER_BYTE[request.req_id]) * req_size
                ue_uav_upload_latency = req_size * float(config.BITS_PER_BYTE) / max(float(ue_uav_rate), float(config.EPSILON))
                for neighbor in uav.neighbors:
                    if neighbor.id == uav_idx or neighbor.id < 0 or neighbor.id >= config.NUM_UAVS:
                        continue
                    belief_prob = uav._get_service_neighbor_cache_belief(request.req_id, neighbor)
                    uav_uav_rate = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(uav.pos, neighbor.pos))
                    neighbor_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(neighbor.pos, config.MBS_POS))
                    exp_neighbor_fetch_latency = (1.0 - belief_prob) * file_size * float(config.BITS_PER_BYTE) / max(
                        float(neighbor_mbs_rate),
                        float(config.EPSILON),
                    )
                    neighbor_load = max(int(getattr(neighbor, "_current_service_request_count", 0)) + 1, 1)
                    neighbor_compute_share = float(config.UAV_COMPUTING_CAPACITY[neighbor.id]) / float(neighbor_load)
                    cooperative_latency = (
                        ue_uav_upload_latency
                        + req_size * float(config.BITS_PER_BYTE) / max(float(uav_uav_rate), float(config.EPSILON))
                        + exp_neighbor_fetch_latency
                        + cpu_cycles / max(neighbor_compute_share, float(config.EPSILON))
                    )
                    coop_compute_ratio = float(neighbor_compute_share) / local_compute_share
                    coop_deadline_ratio = float(cooperative_latency) / max(float(context.deadline), float(config.EPSILON))
                    coop_relative_latency = float(cooperative_latency) / max(best_noncoop_latency, float(config.EPSILON))
                    coop_feasible = (
                        np.isfinite(float(cooperative_latency))
                        and coop_deadline_ratio <= float(config.OFFLOAD_COOP_MAX_DEADLINE_RATIO)
                        and coop_relative_latency <= float(config.OFFLOAD_COOP_MAX_RELATIVE_LATENCY)
                        and coop_compute_ratio >= float(config.OFFLOAD_COOP_MIN_COMPUTE_SHARE_RATIO)
                    )
                    action_idx = int(config.OFFLOAD_ACTION_COOP_BASE + neighbor.id)
                    if getattr(config, "OFFLOAD_MASK_MODE", "quality") != "quality" or coop_feasible:
                        all_masks[uav_idx, req_idx, action_idx] = 1.0
                    else:
                        coop_masked_count += 1
                if (
                    getattr(config, "OFFLOAD_MASK_MODE", "quality") == "quality"
                    and context.mbs_latency >= float(config.OFFLOAD_LATENCY_RATIO_CLIP) * max(context.deadline, config.EPSILON)
                ):
                    all_masks[uav_idx, req_idx, config.OFFLOAD_ACTION_MBS] = 0.0
                    mbs_masked_count += 1
                if np.sum(all_masks[uav_idx, req_idx]) <= 0.0:
                    all_masks[uav_idx, req_idx, config.OFFLOAD_ACTION_LOCAL] = 1.0

            all_obs[uav_idx] = np.concatenate([own_features, request_features.reshape(-1)])

        self._last_runtime_audit["step_coop_masked_count"] = int(coop_masked_count)
        self._last_runtime_audit["step_mbs_masked_count"] = int(mbs_masked_count)
        return all_obs, all_masks

    @staticmethod
    def _safe_ratio(value: float, denominator: float) -> float:
        ratio = float(value) / max(float(denominator), float(config.EPSILON))
        return float(np.clip(ratio / float(config.OFFLOAD_LATENCY_RATIO_CLIP), 0.0, 1.0))

    # 函数 _get_obs：关键函数，承载本模块的一段可复用实验逻辑。
    def _get_obs(self) -> list[np.ndarray]:
        """Gets the local observation for each UAV agent."""
        # 循环处理：遍历 ue 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for ue in self._ues:
            ue.generate_request()
        self._associate_ues_to_uavs()
        # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for uav in self._uavs:
            uav.set_neighbors(self._uavs)

        all_obs: list[np.ndarray] = []
        # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for uav in self._uavs:
            own_pos: np.ndarray = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
            own_cache: np.ndarray = uav.cache.astype(np.float32)
            own_state: np.ndarray = np.concatenate([own_pos, own_cache])

            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_OBS_DIM), dtype=np.float32)
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS]
            # 循环处理：遍历 (i, neighbor) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                neighbor_states[i, :] = relative_pos

            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, config.UE_OBS_DIM), dtype=np.float32)
            ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos[:2] - u.pos[:2])))[: config.MAX_ASSOCIATED_UES]
            # 循环处理：遍历 (i, ue) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
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

        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return all_obs

    # 函数 _apply_actions_to_env：仿真环境对象，承载无人机、用户设备、任务请求和奖励计算，主要参数：actions。
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
        # 循环处理：遍历 (i, uav) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for i, uav in enumerate(self._uavs):
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
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
        # 循环处理：遍历 _ 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for _ in range(config.COLLISION_AVOIDANCE_ITERATIONS + 1):
            collision_detected_in_iter: bool = False
            # 循环处理：遍历 i 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for i in range(config.NUM_UAVS):
                # 循环处理：遍历 j 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
                for j in range(i + 1, config.NUM_UAVS):
                    pos_i: np.ndarray = next_positions[i]
                    pos_j: np.ndarray = next_positions[j]
                    dist_sq: float = np.sum((pos_i - pos_j) ** 2)
                    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                    if dist_sq < min_sep_sq:
                        self._uavs[i].collision_violation = True
                        self._uavs[j].collision_violation = True
                        collision_detected_in_iter = True
                        dist: float = np.sqrt(dist_sq) if dist_sq > 0 else config.EPSILON
                        overlap: float = config.MIN_UAV_SEPARATION - dist
                        direction: np.ndarray = (pos_i - pos_j) / dist
                        next_positions[i] += direction * overlap * 0.5
                        next_positions[j] -= direction * overlap * 0.5
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if not collision_detected_in_iter:
                break

        final_positions: np.ndarray = np.clip(
            next_positions,
            [min_boundary_gap, min_boundary_gap],
            [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap],
        )
        # 循环处理：遍历 (i, uav) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for i, uav in enumerate(self._uavs):
            uav.update_position(final_positions[i])

    # 函数 _associate_ues_to_uavs：关键函数，承载本模块的一段可复用实验逻辑。
    def _associate_ues_to_uavs(self) -> None:
        """Assign each UE to at most one UAV, with optional service fallback admission."""
        stats = {
            "service_requests_generated": 0,
            "naturally_covered_service_requests": 0,
            "service_requests_uncovered": 0,
            "forced_service_admissions": 0,
        }
        # 循环处理：遍历 ue 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for ue in self._ues:
            is_service_request = bool(ue.current_request.is_service)
            if is_service_request:
                stats["service_requests_generated"] += 1
            covering_uavs: list[tuple[UAV, float]] = []
            nearest_uav: UAV | None = None
            nearest_distance: float = float("inf")
            # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
            for uav in self._uavs:
                distance: float = float(np.linalg.norm(uav.pos[:2] - ue.pos[:2]))
                if distance < nearest_distance:
                    nearest_uav = uav
                    nearest_distance = distance
                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if distance <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance))

            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if not covering_uavs:
                if is_service_request:
                    stats["service_requests_uncovered"] += 1
                    if getattr(config, "FORCE_SERVICE_ADMISSION", False) and nearest_uav is not None:
                        nearest_uav.current_covered_ues.append(ue)
                        ue.assigned = True
                        stats["forced_service_admissions"] += 1
                continue
            if is_service_request:
                stats["naturally_covered_service_requests"] += 1
            best_uav, _ = min(covering_uavs, key=lambda x: x[1])
            best_uav.current_covered_ues.append(ue)
            ue.assigned = True
        self._last_admission_stats = stats

    # 函数 _get_rewards_and_metrics：关键函数，承载本模块的一段可复用实验逻辑。
    def _get_rewards_and_metrics(self) -> tuple[list[float], dict[str, float]]:
        """Return the reward and tracked metrics for the current step."""
        total_latency: float = sum(
            ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues
        )
        total_energy: float = sum(uav.energy for uav in self._uavs)
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues], dtype=np.float32)
        jfi: float = 0.0
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))
        offline_count: int = sum(1 for ue in self._ues if ue.battery_level < config.UE_CRITICAL_THRESHOLD)
        offline_rate: float = offline_count / config.NUM_UES

        total_service_requests_generated: int = sum(1 for ue in self._ues if ue.current_request.is_service)
        total_service_requests_processed: int = sum(uav.service_request_count for uav in self._uavs)
        deadline_satisfied_count: int = sum(1 for ue in self._ues if ue.is_service_deadline_satisfied())
        natural_service_requests: int = int(self._last_admission_stats.get("naturally_covered_service_requests", 0))
        uncovered_service_requests: int = int(self._last_admission_stats.get("service_requests_uncovered", 0))
        forced_service_admissions: int = int(self._last_admission_stats.get("forced_service_admissions", 0))
        total_local_offloads: int = sum(uav.service_offload_local_count for uav in self._uavs)
        total_cooperative_offloads: int = sum(uav.service_offload_cooperative_count for uav in self._uavs)
        total_mbs_offloads: int = sum(uav.service_offload_mbs_count for uav in self._uavs)
        total_learned_decisions: int = sum(uav.service_learned_decision_count for uav in self._uavs)
        total_heuristic_decisions: int = sum(uav.service_heuristic_decision_count for uav in self._uavs)
        total_fallbacks: int = sum(uav.service_fallback_count for uav in self._uavs)
        total_predict_exception_fallbacks: int = sum(uav.service_predict_exception_fallback_count for uav in self._uavs)
        assert total_local_offloads + total_cooperative_offloads + total_mbs_offloads == total_service_requests_processed
        assert total_learned_decisions + total_heuristic_decisions == total_service_requests_processed

        # Convention:
        # - deadline_satisfaction_rate and mbs_load_ratio are normalized by all generated service requests.
        # - offloading ratios are normalized by processed service requests only.
        # - when there are no requests for a denominator, the ratio is defined as 0.0.
        deadline_satisfaction_rate: float = 0.0
        mbs_load_ratio: float = 0.0
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if total_service_requests_generated > 0:
            deadline_satisfaction_rate = deadline_satisfied_count / total_service_requests_generated
            mbs_load_ratio = total_mbs_offloads / total_service_requests_generated
        forced_admission_ratio: float = forced_service_admissions / max(float(total_service_requests_generated), float(config.EPSILON))
        natural_coverage_service_ratio: float = natural_service_requests / max(float(total_service_requests_generated), float(config.EPSILON))

        offloading_ratio_local: float = 0.0
        offloading_ratio_cooperative: float = 0.0
        offloading_ratio_mbs: float = 0.0
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if total_service_requests_processed > 0:
            offloading_ratio_local = total_local_offloads / total_service_requests_processed
            offloading_ratio_cooperative = total_cooperative_offloads / total_service_requests_processed
            offloading_ratio_mbs = total_mbs_offloads / total_service_requests_processed
            assert np.isclose(offloading_ratio_local + offloading_ratio_cooperative + offloading_ratio_mbs, 1.0)

        # Linear normalized reward (replaces legacy log-form):
        #   reward = W_FAIR*jfi - W_LAT*norm_lat - W_ENERGY*norm_energy - W_OFFLINE*offline_rate + W_DSR*dsr - W_MBS*mbs_load
        norm_latency: float = total_latency / (config.NUM_UES * config.NON_SERVED_LATENCY_PENALTY + config.EPSILON)
        norm_energy: float = total_energy / (config.NUM_UAVS * config.REWARD_NORM_ENERGY_REF + config.EPSILON)
        r_fairness: float = config.REWARD_W_FAIR * jfi
        r_latency: float = config.REWARD_W_LAT * norm_latency
        r_energy: float = config.REWARD_W_ENERGY * norm_energy
        r_offline: float = config.REWARD_W_OFFLINE * offline_rate
        r_dsr: float = config.REWARD_W_DSR * deadline_satisfaction_rate
        r_mbs: float = 0.5 * mbs_load_ratio  # lightweight MBS load penalty for upper layer
        reward: float = r_fairness - r_latency - r_energy - r_offline + r_dsr - r_mbs
        rewards: list[float] = [reward] * config.NUM_UAVS
        # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for uav in self._uavs:
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]

        # 有效能效 = 满足截止时间的请求数 / UAV总能耗
        effective_energy_efficiency: float = 0.0
        if total_energy > config.EPSILON:
            effective_energy_efficiency = float(deadline_satisfied_count) / total_energy

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
            "effective_energy_efficiency": effective_energy_efficiency,
            "service_requests_generated": float(total_service_requests_generated),
            "service_requests_processed": float(total_service_requests_processed),
            "service_requests_uncovered": float(uncovered_service_requests),
            "forced_service_admissions": float(forced_service_admissions),
            "forced_admission_ratio": float(forced_admission_ratio),
            "natural_coverage_service_ratio": float(natural_coverage_service_ratio),
            "service_offloads_local": float(total_local_offloads),
            "service_offloads_cooperative": float(total_cooperative_offloads),
            "service_offloads_mbs": float(total_mbs_offloads),
            "service_learned_decision_count": float(total_learned_decisions),
            "service_heuristic_decision_count": float(total_heuristic_decisions),
            "service_fallback_count": float(total_fallbacks),
            "service_predict_exception_fallback_count": float(total_predict_exception_fallbacks),
            "service_offload_policy_loaded": float(all(uav.service_offload_policy_loaded for uav in self._uavs)),
        }
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return rewards, metrics

    # 函数 _collect_runtime_audit：关键函数，承载本模块的一段可复用实验逻辑。
    def _collect_runtime_audit(self) -> dict[str, object]:
        """Aggregate per-step and cumulative service-offloading audit fields for the current episode."""

        step_fallback_count: int = sum(uav.service_fallback_count for uav in self._uavs)
        step_predict_exception_fallback_count: int = sum(
            uav.service_predict_exception_fallback_count for uav in self._uavs
        )
        step_learned_decision_count: int = sum(uav.service_learned_decision_count for uav in self._uavs)
        step_heuristic_decision_count: int = sum(uav.service_heuristic_decision_count for uav in self._uavs)

        self._episode_runtime_audit_totals["fallback_count"] += step_fallback_count
        self._episode_runtime_audit_totals["predict_exception_fallback_count"] += step_predict_exception_fallback_count
        self._episode_runtime_audit_totals["learned_decision_count"] += step_learned_decision_count
        self._episode_runtime_audit_totals["heuristic_decision_count"] += step_heuristic_decision_count

        first_uav: UAV | None = self._uavs[0] if self._uavs else None
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {
            "service_offload_policy_requested": first_uav.service_offload_policy_requested if first_uav is not None else "heuristic",
            "service_offload_policy_loaded": bool(all(uav.service_offload_policy_loaded for uav in self._uavs)),
            "service_offload_policy_checkpoint_path": (
                first_uav.service_offload_policy_checkpoint_path if first_uav is not None else None
            ),
            "service_offload_policy_feature_family": (
                first_uav.service_offload_policy_feature_family if first_uav is not None else None
            ),
            "service_offload_policy_load_error": (
                first_uav.service_offload_policy_load_error if first_uav is not None else None
            ),
            "step_service_fallback_count": step_fallback_count,
            "step_service_predict_exception_fallback_count": step_predict_exception_fallback_count,
            "step_service_learned_decision_count": step_learned_decision_count,
            "step_service_heuristic_decision_count": step_heuristic_decision_count,
            "episode_service_fallback_count": self._episode_runtime_audit_totals["fallback_count"],
            "episode_service_predict_exception_fallback_count": self._episode_runtime_audit_totals[
                "predict_exception_fallback_count"
            ],
            "episode_service_learned_decision_count": self._episode_runtime_audit_totals["learned_decision_count"],
            "episode_service_heuristic_decision_count": self._episode_runtime_audit_totals["heuristic_decision_count"],
        }
