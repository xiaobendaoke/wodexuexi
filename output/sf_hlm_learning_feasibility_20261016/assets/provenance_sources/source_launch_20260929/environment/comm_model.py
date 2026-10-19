"""
中文注释说明：environment/comm_model.py

文件作用：
    实现通信模型相关计算，包括距离、信道增益、传输速率和链路开销估计。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - calculate_channel_gain(): 关键函数，承载本模块的一段可复用实验逻辑。
    - calculate_ue_uav_rate(): 无人机对象，包含位置、电量、计算资源和缓存服务。
    - calculate_uav_mbs_rate(): 无人机对象，包含位置、电量、计算资源和缓存服务。
    - calculate_uav_uav_rate(): 无人机对象，包含位置、电量、计算资源和缓存服务。

主要依赖：
    config, numpy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

import config
import numpy as np


# 函数 calculate_channel_gain：关键函数，承载本模块的一段可复用实验逻辑，主要参数：pos1, pos2。
def calculate_channel_gain(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculates channel gain based on the free-space path loss model."""
    distance_sq: float = np.sum((pos1 - pos2) ** 2)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return (config.G_CONSTS_PRODUCT) / (distance_sq + config.EPSILON)


# 函数 calculate_ue_uav_rate：无人机对象，包含位置、电量、计算资源和缓存服务，主要参数：channel_gain, num_associated_ues。
def calculate_ue_uav_rate(channel_gain: float, num_associated_ues: int) -> float:
    """Calculates data rate between a UE and a UAV."""
    assert num_associated_ues != 0
    bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return bandwidth_per_ue * np.log2(1 + snr)


# 函数 calculate_uav_mbs_rate：无人机对象，包含位置、电量、计算资源和缓存服务，主要参数：channel_gain。
def calculate_uav_mbs_rate(channel_gain: float) -> float:
    """Calculates data rate between a UAV and the MBS."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return config.BANDWIDTH_BACKHAUL * np.log2(1 + snr)


# 函数 calculate_uav_uav_rate：无人机对象，包含位置、电量、计算资源和缓存服务，主要参数：channel_gain。
def calculate_uav_uav_rate(channel_gain: float) -> float:
    """Calculates data rate between two UAVs."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return config.BANDWIDTH_INTER * np.log2(1 + snr)
