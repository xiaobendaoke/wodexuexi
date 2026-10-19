"""
中文注释说明：environment/request_types.py

文件作用：
    定义任务请求类型和服务类型，用于描述任务大小、计算强度、时延约束和缓存需求。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - REQUEST_TYPE_SERVICE: 用户设备产生的任务请求。
    - REQUEST_TYPE_CONTENT: 用户设备产生的任务请求。
    - REQUEST_TYPE_ENERGY: 用户设备产生的任务请求。
    - Request: 用户设备产生的任务请求。

主要依赖：
    dataclasses

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

from dataclasses import dataclass


# 关键变量 REQUEST_TYPE_SERVICE：用户设备产生的任务请求。
REQUEST_TYPE_SERVICE: int = 0
# 关键变量 REQUEST_TYPE_CONTENT：用户设备产生的任务请求。
REQUEST_TYPE_CONTENT: int = 1
# 关键变量 REQUEST_TYPE_ENERGY：用户设备产生的任务请求。
REQUEST_TYPE_ENERGY: int = 2


# 类 Request：用户设备产生的任务请求。
@dataclass
class Request:
    """Explicit request structure used across the environment.

    Deadline and priority are meaningful for service requests only.
    Content and emergency energy requests keep default placeholder values.
    """

    req_type: int
    req_size: int
    req_id: int
    deadline: float = 0.0
    priority: int = 0

    # 函数 is_service：关键函数，承载本模块的一段可复用实验逻辑。
    @property
    def is_service(self) -> bool:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.req_type == REQUEST_TYPE_SERVICE

    # 函数 is_content：关键函数，承载本模块的一段可复用实验逻辑。
    @property
    def is_content(self) -> bool:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.req_type == REQUEST_TYPE_CONTENT

    # 函数 is_energy：执行、移动或通信过程产生的能耗。
    @property
    def is_energy(self) -> bool:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.req_type == REQUEST_TYPE_ENERGY

    # 函数 service：关键函数，承载本模块的一段可复用实验逻辑，主要参数：cls, req_size, req_id, deadline, priority。
    @classmethod
    def service(cls, req_size: int, req_id: int, deadline: float, priority: int) -> "Request":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return cls(
            req_type=REQUEST_TYPE_SERVICE,
            req_size=req_size,
            req_id=req_id,
            deadline=deadline,
            priority=priority,
        )

    # 函数 content：关键函数，承载本模块的一段可复用实验逻辑，主要参数：cls, req_id。
    @classmethod
    def content(cls, req_id: int) -> "Request":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return cls(req_type=REQUEST_TYPE_CONTENT, req_size=0, req_id=req_id)

    # 函数 energy：执行、移动或通信过程产生的能耗，主要参数：cls。
    @classmethod
    def energy(cls) -> "Request":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return cls(req_type=REQUEST_TYPE_ENERGY, req_size=0, req_id=0)
