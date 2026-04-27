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
    def initialize_ue_class(cls) -> None:
        cls.all_ids = np.arange(config.NUM_FILES)  # Assume IDs 0 to NUM_SERVICES-1 are Services, rest are Contents
        cls.global_ranks = np.arange(1, config.NUM_FILES + 1)
        np.random.shuffle(cls.global_ranks)  # Currently random ranks assigned
        cls.id_to_rank_map = dict(zip(cls.all_ids, cls.global_ranks))  # Mapping from ID to rank
        zipf_denom: float = np.sum(1 / cls.global_ranks**config.ZIPF_BETA)
        cls.global_probs = (1 / cls.global_ranks**config.ZIPF_BETA) / zipf_denom

        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if getattr(config, "USE_HOTSPOTS", False):
            cls.generate_hotspots()

    # 函数 generate_hotspots：通信链路速率，主要参数：cls。
    @classmethod
    def generate_hotspots(cls) -> None:
        """Randomizes the locations of the hotspots across the map."""
        cls.hotspot_centers = []
        max_retries = 100  # Safety limit for rejection sampling

        # 循环处理：遍历 _ 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for _ in range(config.NUM_HOTSPOTS):
            valid: bool = False
            new_center: np.ndarray = np.zeros(2, dtype=np.float32)
            retries: int = 0
            # 循环控制：在条件满足期间持续推进采样、训练或搜索流程。
            while not valid and retries < max_retries:
                hx: float = np.random.uniform(config.HOTSPOT_RADIUS, config.AREA_WIDTH - config.HOTSPOT_RADIUS)
                hy: float = np.random.uniform(config.HOTSPOT_RADIUS, config.AREA_HEIGHT - config.HOTSPOT_RADIUS)
                new_center = np.array([hx, hy], dtype=np.float32)

                # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                if not cls.hotspot_centers:
                    valid = True
                else:
                    distances: np.ndarray = np.linalg.norm(np.array(cls.hotspot_centers) - new_center, axis=1)
                    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
                    if np.min(distances) > config.HOTSPOT_SEPARATION:
                        valid = True
                retries += 1

            cls.hotspot_centers.append(new_center)

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：ue_id。
    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), 0.0], dtype=np.float32)
        self.is_hotspot_user = getattr(config, "USE_HOTSPOTS", False) and (self.id < config.NUM_UES * getattr(config, "HOTSPOT_UE_PROB", 0.0))
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if self.is_hotspot_user:
            self.pos[:2] = self._get_position_in_hotspot()

        self.battery_level: float = np.random.uniform(0.6, 1.0) * config.UE_BATTERY_CAPACITY  # Start at capacity between 60% to 100%

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
        angle: float = np.random.uniform(0, 2 * np.pi)
        r: float = config.HOTSPOT_RADIUS * np.sqrt(np.random.uniform(0, 1))
        offset: np.ndarray = r * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        center: np.ndarray = UE.hotspot_centers[self.id % config.NUM_HOTSPOTS]
        pos: np.ndarray = np.clip(center + offset, [0, 0], [config.AREA_WIDTH, config.AREA_HEIGHT])
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return pos.astype(np.float32)

    # 函数 generate_request：用户设备产生的任务请求。
    def generate_request(self) -> None:
        """Generates a new explicit request object for the current time slot."""

        # Check for Emergency Energy Request
        if self.battery_level < config.UE_CRITICAL_THRESHOLD:
            self.current_request = Request.energy()
            self.latency_current_request = 0.0
            self.assigned = False
            # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
            return

        req_id: int = np.random.choice(UE.all_ids, p=UE.global_probs)
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if req_id < config.NUM_SERVICES:
            req_size: int = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE)
            deadline: float = float(np.random.uniform(config.SERVICE_DEADLINE_MIN, config.SERVICE_DEADLINE_MAX))
            priority: int = int(np.random.randint(config.SERVICE_PRIORITY_MIN, config.SERVICE_PRIORITY_MAX + 1))
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
            self._waypoint = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)], dtype=np.float32)

        self._wait_time = np.random.randint(0, config.UE_MAX_WAIT_TIME + 1)

    # 函数 update_battery：更新模型、环境或统计量的状态，主要参数：harv_energy, ue_transmit_time。
    def update_battery(self, harv_energy: float, ue_transmit_time: float) -> None:
        """Updates battery level based on consumption and harvesting."""
        consumed_energy: float = config.UE_STATIC_POWER * config.TIME_SLOT_DURATION
        consumed_energy += config.TRANSMIT_POWER * ue_transmit_time
        self.battery_level = min(config.UE_BATTERY_CAPACITY, self.battery_level - consumed_energy + harv_energy)
        self.battery_level = max(0.0, self.battery_level)
