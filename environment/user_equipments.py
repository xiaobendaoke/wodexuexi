"""
中文注释说明：environment/user_equipments.py

文件作用：
    定义用户设备实体及其任务请求生成、位置和资源需求等属性。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - UE: 用户设备对象，产生任务请求并等待服务。

主要依赖：
    config, numpy, environment

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

import config
import numpy as np
from environment.request_types import Request


# 类 UE：用户设备对象，产生任务请求并等待服务。
class UE:
    all_ids: np.ndarray
    global_ranks: np.ndarray
    id_to_rank_map: dict[int, int]
    global_probs: np.ndarray
    hotspot_centers: list[np.ndarray]

    # 函数 initialize_ue_class：用户设备对象，产生任务请求并等待服务，主要参数：cls。
    @classmethod
    def initialize_ue_class(cls, rng=None) -> None:
        rng = np.random if rng is None else rng
        cls.all_ids = np.arange(config.NUM_FILES)  # Assume IDs 0 to NUM_SERVICES-1 are Services, rest are Contents
        cls.global_ranks = np.arange(1, config.NUM_FILES + 1)
        rng.shuffle(cls.global_ranks)  # Currently random ranks assigned
        cls.id_to_rank_map = dict(zip(cls.all_ids, cls.global_ranks))  # Mapping from ID to rank
        zipf_denom: float = np.sum(1 / cls.global_ranks**config.ZIPF_BETA)
        cls.global_probs = (1 / cls.global_ranks**config.ZIPF_BETA) / zipf_denom

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if getattr(config, "USE_HOTSPOTS", False):
            cls.generate_hotspots(rng)

    # 函数 generate_hotspots：通信链路速率，主要参数：cls。
    @classmethod
    def generate_hotspots(cls, rng=None) -> None:
        """Bounded whole-layout rejection: never accept an invalid last draw."""
        rng = np.random if rng is None else rng
        for _ in range(1000):
            centers = []
            for _ in range(config.NUM_HOTSPOTS):
                for _ in range(100):
                    candidate = np.array([
                        rng.uniform(config.HOTSPOT_RADIUS, config.AREA_WIDTH - config.HOTSPOT_RADIUS),
                        rng.uniform(config.HOTSPOT_RADIUS, config.AREA_HEIGHT - config.HOTSPOT_RADIUS),
                    ], dtype=np.float32)
                    if all(np.linalg.norm(candidate - c) >= config.HOTSPOT_SEPARATION for c in centers):
                        centers.append(candidate)
                        break
                else:
                    break
            if len(centers) == config.NUM_HOTSPOTS:
                cls.hotspot_centers = centers
                return
        raise ValueError("Cannot generate a feasible hotspot layout for this configuration")

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：ue_id。
    def __init__(self, ue_id: int, rng=None, mobility_rng=None, population=None) -> None:
        self.rng = np.random if rng is None else rng
        self.mobility_rng = self.rng if mobility_rng is None else mobility_rng
        self.all_ids = (population["all_ids"] if population else UE.all_ids).copy()
        self.global_probs = (population["global_probs"] if population else UE.global_probs).copy()
        centers = population.get("hotspot_centers", []) if population else getattr(UE, "hotspot_centers", [])
        self.hotspot_centers = [c.copy() for c in centers]
        self.id: int = ue_id
        self.pos: np.ndarray = np.array([self.mobility_rng.uniform(0, config.AREA_WIDTH), self.mobility_rng.uniform(0, config.AREA_HEIGHT), 0.0], dtype=np.float32)
        self.is_hotspot_user = getattr(config, "USE_HOTSPOTS", False) and (self.id < config.NUM_UES * getattr(config, "HOTSPOT_UE_PROB", 0.0))
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if self.is_hotspot_user:
            self.pos[:2] = self._get_position_in_hotspot()

        self.current_request: Request = Request.service(0, 0, config.TIME_SLOT_DURATION, config.SERVICE_PRIORITY_MIN)
        self.latency_current_request: float = 0.0  # Latency for the current request
        self.assigned: bool = False

        # Random Waypoint Model
        self._waypoint: np.ndarray
        self._wait_time: int
        self._set_new_waypoint()  # Initialize first waypoint

        # Fairness Tracking
        self._successful_requests: int = 0
        self.service_coverage: float = 0.0

    # 函数 update_position：更新模型、环境或统计量的状态。
    def update_position(self) -> None:
        """Updates the UE's position for one time slot as per the Random Waypoint model."""
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if self._wait_time > 0:
            self._wait_time -= 1
            # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
            return

        direction_vec: np.ndarray = self._waypoint - self.pos[:2]
        distance_to_waypoint: float = float(np.linalg.norm(direction_vec))

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if config.UE_MAX_DIST >= distance_to_waypoint:  # Reached the waypoint
            self.pos[:2] = self._waypoint
            self._set_new_waypoint()
        else:  # Move towards the waypoint
            move_vector = (direction_vec / distance_to_waypoint) * config.UE_MAX_DIST
            self.pos[:2] += move_vector

    # 函数 _get_position_in_hotspot：关键函数，承载本模块的一段可复用实验逻辑。
    def _get_position_in_hotspot(self) -> np.ndarray:
        """Generates a random position strictly within this UE's assigned hotspot."""
        angle: float = self.mobility_rng.uniform(0, 2 * np.pi)
        r: float = config.HOTSPOT_RADIUS * np.sqrt(self.mobility_rng.uniform(0, 1))
        offset: np.ndarray = r * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        center: np.ndarray = self.hotspot_centers[self.id % config.NUM_HOTSPOTS]
        pos: np.ndarray = np.clip(center + offset, [0, 0], [config.AREA_WIDTH, config.AREA_HEIGHT])
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return pos.astype(np.float32)

    # 函数 generate_request：用户设备产生的任务请求。
    def generate_request(self) -> None:
        """Generates a new explicit request object for the current time slot."""
        req_id = int(self.rng.choice(self.all_ids, p=self.global_probs))
        req_size = int(self.rng.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE))
        deadline = float(self.rng.uniform(config.SERVICE_DEADLINE_MIN, config.SERVICE_DEADLINE_MAX))
        priority = int(self.rng.randint(config.SERVICE_PRIORITY_MIN, config.SERVICE_PRIORITY_MAX + 1))
        self.latent_request = (req_id, req_size, deadline, priority)
        if req_id < config.NUM_SERVICES:
            self.current_request = Request.service(req_size=req_size, req_id=req_id, deadline=deadline, priority=priority)
        else:
            self.current_request = Request.content(req_id=req_id)
        self.latency_current_request = 0.0
        self.assigned = False

    # 函数 update_service_coverage：更新模型、环境或统计量的状态，主要参数：current_time_step_t。
    def update_service_coverage(self, current_time_step_t: int) -> None:
        """Updates the fairness metric without mixing in service deadlines."""
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if self.is_successful_for_fairness():
            self._successful_requests += 1

        assert current_time_step_t > 0
        self.service_coverage = self._successful_requests / current_time_step_t

    # 函数 is_successful_for_fairness：关键函数，承载本模块的一段可复用实验逻辑。
    def is_successful_for_fairness(self) -> bool:
        """Legacy fairness success definition kept separate from deadline satisfaction."""
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.assigned and self.latency_current_request <= config.TIME_SLOT_DURATION

    # 函数 is_service_deadline_satisfied：关键函数，承载本模块的一段可复用实验逻辑。
    def is_service_deadline_satisfied(self) -> bool:
        """Deadline satisfaction is only defined for service requests."""
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.current_request.is_service and self.assigned and self.latency_current_request <= self.current_request.deadline

    # 函数 _set_new_waypoint：关键函数，承载本模块的一段可复用实验逻辑。
    def _set_new_waypoint(self):
        """Set a new destination, speed, and wait time as per the Random Waypoint model."""
        # If hotspots are active, the new waypoint MUST also be inside the hotspot!
        if self.is_hotspot_user:
            self._waypoint = self._get_position_in_hotspot()
        else:
            self._waypoint = np.array([self.mobility_rng.uniform(0, config.AREA_WIDTH), self.mobility_rng.uniform(0, config.AREA_HEIGHT)], dtype=np.float32)

        self._wait_time = self.mobility_rng.randint(0, config.UE_MAX_WAIT_TIME + 1)
