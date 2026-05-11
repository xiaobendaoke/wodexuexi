"""
中文注释说明：config.py

文件作用：
    集中定义多无人机移动边缘计算实验的环境参数、训练超参数、奖励权重和任务卸载配置，是其他脚本读取实验设定的核心配置入口。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - MODEL: 当前训练或测试的多智能体模型实例。
    - SEED: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - STEPS_PER_EPISODE: 训练或测试的回合编号。
    - LOG_FREQ: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - IMG_FREQ: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - TEST_LOG_FREQ: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - TEST_IMG_FREQ: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - MBS_POS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - NUM_UAVS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - NUM_UES: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - AREA_WIDTH: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - AREA_HEIGHT: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - TIME_SLOT_DURATION: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - UE_MAX_DIST: 用户设备对象，产生任务请求并等待服务。
    - UE_MAX_WAIT_TIME: 用户设备对象，产生任务请求并等待服务。
    - USE_HOTSPOTS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - NUM_HOTSPOTS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - HOTSPOT_RADIUS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。

主要依赖：
    numpy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import numpy as np

# Training Parameters
# Thesis-oriented default: use the learning-based upper-layer controller as the default entry model.
MODEL: str = "attention_mappo"  # options: 'maddpg', 'matd3', 'mappo', 'masac', 'attention_<model>', 'random'
# 关键变量 SEED：全局常量或配置项，会影响环境规模、训练过程或实验输出。
SEED: int = 42  # random seed for reproducibility
np.random.seed(SEED)  # set numpy random seed
# 关键变量 STEPS_PER_EPISODE：训练或测试的回合编号。
STEPS_PER_EPISODE: int = 1000  # total T
# 关键变量 LOG_FREQ：全局常量或配置项，会影响环境规模、训练过程或实验输出。
LOG_FREQ: int = 1  # episodes
# 关键变量 IMG_FREQ：全局常量或配置项，会影响环境规模、训练过程或实验输出。
IMG_FREQ: int = 1000  # steps
# 关键变量 TEST_LOG_FREQ：全局常量或配置项，会影响环境规模、训练过程或实验输出。
TEST_LOG_FREQ: int = 1  # episodes (for testing)
# 关键变量 TEST_IMG_FREQ：全局常量或配置项，会影响环境规模、训练过程或实验输出。
TEST_IMG_FREQ: int = 100  # steps (for testing)

# Simulation Parameters
MBS_POS: np.ndarray = np.array([350.0, 350.0, 30.0], dtype=np.float32)  # (X_mbs, Y_mbs, Z_mbs) in meters
# 关键变量 NUM_UAVS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
NUM_UAVS: int = 5  # U
# 关键变量 NUM_UES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
NUM_UES: int = 100  # M
# 关键变量 AREA_WIDTH：全局常量或配置项，会影响环境规模、训练过程或实验输出。
AREA_WIDTH: int = 700  # X_max in meters
# 关键变量 AREA_HEIGHT：全局常量或配置项，会影响环境规模、训练过程或实验输出。
AREA_HEIGHT: int = 700  # Y_max in meters
# 关键变量 TIME_SLOT_DURATION：全局常量或配置项，会影响环境规模、训练过程或实验输出。
TIME_SLOT_DURATION: float = 1.0  # tau in seconds
# 关键变量 UE_MAX_DIST：用户设备对象，产生任务请求并等待服务。
UE_MAX_DIST: float = 15.0  # d_max^UE in meters
# 关键变量 UE_MAX_WAIT_TIME：用户设备对象，产生任务请求并等待服务。
UE_MAX_WAIT_TIME: int = 10  # in time slots

# 关键变量 USE_HOTSPOTS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
USE_HOTSPOTS: bool = True  # thesis runs: encourage non-uniform traffic so trajectory/offloading coupling is visible
# 关键变量 NUM_HOTSPOTS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
NUM_HOTSPOTS: int = 2  # number of hotspots
# 关键变量 HOTSPOT_RADIUS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
HOTSPOT_RADIUS: float = 100.0  # radius of each hotspot in meters
assert NUM_HOTSPOTS * HOTSPOT_RADIUS * 2 <= min(AREA_WIDTH, AREA_HEIGHT), "Hotspots cannot fit in the area without overlap."
# 关键变量 HOTSPOT_SEPARATION：全局常量或配置项，会影响环境规模、训练过程或实验输出。
HOTSPOT_SEPARATION: float = 400.0  # minimum separation between hotspots in meters
assert HOTSPOT_SEPARATION >= 2 * HOTSPOT_RADIUS, "Hotspot separation must be at least twice the hotspot radius to avoid overlap."
# 关键变量 HOTSPOT_UE_PROB：用户设备对象，产生任务请求并等待服务。
HOTSPOT_UE_PROB: float = 0.85  # slightly stronger clustering to create cooperative-routing opportunities

# UAV Parameters
UAV_ALTITUDE: int = 100  # H in meters
# 关键变量 UAV_SPEED：无人机对象，包含位置、电量、计算资源和缓存服务。
UAV_SPEED: float = 15.0  # v^UAV in m/s
# 关键变量 UAV_STORAGE_CAPACITY：无人机对象，包含位置、电量、计算资源和缓存服务。
UAV_STORAGE_CAPACITY: np.ndarray = np.random.choice(np.arange(120 * 10**6, 180 * 10**6, 10**6), size=NUM_UAVS).astype(np.int64)  # S_u in bytes
# 关键变量 UAV_COMPUTING_CAPACITY：无人机对象，包含位置、电量、计算资源和缓存服务。
UAV_COMPUTING_CAPACITY: np.ndarray = np.random.choice(np.arange(40 * 10**9, 90 * 10**9, 5 * 10**9), size=NUM_UAVS).astype(np.int64)  # F_u in cycles/sec
# 关键变量 UAV_SENSING_RANGE：无人机对象，包含位置、电量、计算资源和缓存服务。
UAV_SENSING_RANGE: float = 460.0  # R^sense in meters; keeps cooperative UAVs visible in thesis runtime experiments
# 关键变量 UAV_COVERAGE_RADIUS：无人机对象，包含位置、电量、计算资源和缓存服务。
UAV_COVERAGE_RADIUS: float = 100.0  # R in meters
# 关键变量 MIN_UAV_SEPARATION：无人机对象，包含位置、电量、计算资源和缓存服务。
MIN_UAV_SEPARATION: float = 200.0  # d_min in meters
assert np.all(UAV_STORAGE_CAPACITY > 0)
assert np.all(UAV_COMPUTING_CAPACITY > 0)
assert UAV_COVERAGE_RADIUS * 2 <= MIN_UAV_SEPARATION
assert UAV_SENSING_RANGE >= MIN_UAV_SEPARATION

# Collisions and Penalties
COLLISION_AVOIDANCE_ITERATIONS: int = 20  # number of iterations to resolve collisions
# 关键变量 COLLISION_PENALTY：全局常量或配置项，会影响环境规模、训练过程或实验输出。
COLLISION_PENALTY: float = 10.0  # penalty per collision
# 关键变量 BOUNDARY_PENALTY：全局常量或配置项，会影响环境规模、训练过程或实验输出。
BOUNDARY_PENALTY: float = 10.0  # penalty for going out of bounds
# 关键变量 NON_SERVED_LATENCY_PENALTY：全局常量或配置项，会影响环境规模、训练过程或实验输出。
NON_SERVED_LATENCY_PENALTY: float = 20.0  # penalty in latency for non-served requests
# IMPORTANT : Reconfigurable, should try for various values including : NUM_UAVS - 1 and NUM_UES
MAX_UAV_NEIGHBORS: int = max(0, NUM_UAVS - 1)
# 关键变量 MAX_ASSOCIATED_UES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
MAX_ASSOCIATED_UES: int = min(30, NUM_UES // NUM_UAVS + 10)
assert MAX_UAV_NEIGHBORS >= 0 and MAX_UAV_NEIGHBORS <= NUM_UAVS - 1
assert MAX_ASSOCIATED_UES >= 1 and MAX_ASSOCIATED_UES <= NUM_UES

# 关键变量 POWER_MOVE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
POWER_MOVE: float = 100.0  # P_move in Watts
# 关键变量 POWER_HOVER：全局常量或配置项，会影响环境规模、训练过程或实验输出。
POWER_HOVER: float = 80.0  # P_hover in Watts

# Request Parameters
NUM_SERVICES: int = 25  # S
# 关键变量 NUM_CONTENTS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
NUM_CONTENTS: int = 50  # K
# 关键变量 NUM_FILES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
NUM_FILES: int = NUM_SERVICES + NUM_CONTENTS  # S + K
# 关键变量 CPU_CYCLES_PER_BYTE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
CPU_CYCLES_PER_BYTE: np.ndarray = np.random.randint(1200, 2800, size=NUM_SERVICES)  # omega_s_m
# 关键变量 FILE_SIZES：全局常量或配置项，会影响环境规模、训练过程或实验输出。
FILE_SIZES: np.ndarray = np.random.randint(10**6, 5 * 10**6, size=NUM_FILES).astype(np.int64)  # in bytes
# 关键变量 MIN_INPUT_SIZE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
MIN_INPUT_SIZE: int = 1 * 10**6  # in bytes
# 关键变量 MAX_INPUT_SIZE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
MAX_INPUT_SIZE: int = 5 * 10**6  # in bytes
# 关键变量 ZIPF_BETA：全局常量或配置项，会影响环境规模、训练过程或实验输出。
ZIPF_BETA: float = 0.8  # beta^Zipf
# 关键变量 K_CPU：全局常量或配置项，会影响环境规模、训练过程或实验输出。
K_CPU: float = 1e-27  # CPU capacitance coefficient
# 关键变量 SERVICE_DEADLINE_MIN：全局常量或配置项，会影响环境规模、训练过程或实验输出。
SERVICE_DEADLINE_MIN: float = 0.65 * TIME_SLOT_DURATION  # moderate deadline pressure for richer local/cooperative/MBS trade-offs
# 关键变量 SERVICE_DEADLINE_MAX：全局常量或配置项，会影响环境规模、训练过程或实验输出。
SERVICE_DEADLINE_MAX: float = 2.10 * TIME_SLOT_DURATION  # enough slack for local/cooperative decisions to be feasible
# 关键变量 SERVICE_PRIORITY_MIN：全局常量或配置项，会影响环境规模、训练过程或实验输出。
SERVICE_PRIORITY_MIN: int = 1
# 关键变量 SERVICE_PRIORITY_MAX：全局常量或配置项，会影响环境规模、训练过程或实验输出。
SERVICE_PRIORITY_MAX: int = 3
# Labels for the request-level learned offload policy can be generated either
# from the original latency-only heuristic or from a thesis-oriented multi-objective oracle.
OFFLOAD_LABEL_MODE: str = "enhanced_oracle"  # options: "heuristic", "enhanced_oracle"
# 关键变量 OFFLOAD_ORACLE_DEADLINE_WEIGHT：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_ORACLE_DEADLINE_WEIGHT: float = 1.25
# 关键变量 OFFLOAD_ORACLE_MBS_LOAD_WEIGHT：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OFFLOAD_ORACLE_MBS_LOAD_WEIGHT: float = 0.065
# 关键变量 OFFLOAD_ORACLE_QUEUE_WEIGHT：用户设备对象，产生任务请求并等待服务。
OFFLOAD_ORACLE_QUEUE_WEIGHT: float = 0.06
# 关键变量 OFFLOAD_ORACLE_COOP_QUEUE_RELIEF_BONUS：用户设备对象，产生任务请求并等待服务。
OFFLOAD_ORACLE_COOP_QUEUE_RELIEF_BONUS: float = 0.02
# Default to heuristic so old configs and baselines retain the original behavior unless explicitly switched.
SERVICE_OFFLOAD_POLICY: str = "heuristic"  # options: "heuristic", "learned"
# 关键变量 SERVICE_OFFLOAD_POLICY_CHECKPOINT：全局常量或配置项，会影响环境规模、训练过程或实验输出。
SERVICE_OFFLOAD_POLICY_CHECKPOINT: str | None = None  # checkpoint for the standalone request-level classifier

# Hierarchical lower-layer MAPPO offloading defaults.
# Each UAV is a lower-layer agent and makes at most MAX_OFFLOAD_REQUESTS_PER_UAV
# request-level decisions per environment step.
OFFLOAD_MODEL_NAME: str = "constrained_attention_offload_mappo"
MAX_OFFLOAD_REQUESTS_PER_UAV: int = MAX_ASSOCIATED_UES
OFFLOAD_NUM_ACTIONS: int = 3  # 0: local UAV, 1: cooperative UAV, 2: MBS
OFFLOAD_LATENCY_RATIO_CLIP: float = 10.0
OFFLOAD_REQUEST_FEATURE_DIM: int = 14
OFFLOAD_OBS_DIM_SINGLE: int = 5 + (MAX_OFFLOAD_REQUESTS_PER_UAV * OFFLOAD_REQUEST_FEATURE_DIM)
OFFLOAD_REWARD_DEADLINE_WEIGHT: float = 3.0
OFFLOAD_REWARD_LATENCY_WEIGHT: float = 0.35
OFFLOAD_REWARD_ENERGY_WEIGHT: float = 0.10
OFFLOAD_REWARD_MBS_WEIGHT: float = 0.35
OFFLOAD_REWARD_COOP_BONUS: float = 0.08
OFFLOAD_REWARD_SUCCESS_BONUS: float = 0.40
OFFLOAD_REWARD_SCALING_FACTOR: float = 1.0
OFFLOAD_CONSTRAINT_MODE: str = "lagrange"  # options: "none", "lagrange"
OFFLOAD_MASK_MODE: str = "quality"  # options: "quality", "none"
OFFLOAD_USE_ATTENTION: bool = True
OFFLOAD_DSR_TARGET: float = 0.234
OFFLOAD_MBS_LOAD_CEILING: float = 0.0589
OFFLOAD_LAGRANGE_LR: float = 0.05
OFFLOAD_LAGRANGE_MAX: float = 20.0
OFFLOAD_COOP_MAX_DEADLINE_RATIO: float = 1.50
OFFLOAD_COOP_MAX_RELATIVE_LATENCY: float = 1.35
OFFLOAD_COOP_MIN_COMPUTE_SHARE_RATIO: float = 0.25

# Caching Parameters
T_CACHE_UPDATE_INTERVAL: int = 50  # T_cache
# 关键变量 GDSF_SMOOTHING_FACTOR：全局常量或配置项，会影响环境规模、训练过程或实验输出。
GDSF_SMOOTHING_FACTOR: float = 0.75  # beta^gdsf

# Probabilistic Caching Parameters
AVG_FILE_SIZE: float = float(np.mean(FILE_SIZES))
# 关键变量 PROB_GAMMA：全局常量或配置项，会影响环境规模、训练过程或实验输出。
PROB_GAMMA: float = 0.5  # gamma

# Communication Parameters
G_CONSTS_PRODUCT: float = 2.2846 * 1.42 * 1e-4  # G_0 * g_0
# 关键变量 TRANSMIT_POWER：全局常量或配置项，会影响环境规模、训练过程或实验输出。
TRANSMIT_POWER: float = 0.5  # P^comm in Watts
# 关键变量 AWGN：全局常量或配置项，会影响环境规模、训练过程或实验输出。
AWGN: float = 1e-13  # sigma^2
# 关键变量 BANDWIDTH_INTER：全局常量或配置项，会影响环境规模、训练过程或实验输出。
BANDWIDTH_INTER: int = 20 * 10**6  # B^inter in Hz
# 关键变量 BANDWIDTH_EDGE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
BANDWIDTH_EDGE: int = 40 * 10**6  # B^edge in Hz
# 关键变量 BANDWIDTH_BACKHAUL：全局常量或配置项，会影响环境规模、训练过程或实验输出。
BANDWIDTH_BACKHAUL: int = 750_000  # B^backhaul in Hz; constrained backhaul reduces the all-to-MBS collapse

# WPT Parameters
UE_BATTERY_CAPACITY: float = 100.0  # B_max in Joules
# 关键变量 UE_CRITICAL_THRESHOLD：用户设备对象，产生任务请求并等待服务。
UE_CRITICAL_THRESHOLD: float = 0.3 * UE_BATTERY_CAPACITY  # B_low in Joules
# 关键变量 WPT_TRANSMIT_POWER：全局常量或配置项，会影响环境规模、训练过程或实验输出。
WPT_TRANSMIT_POWER: float = 50.0  # P^WPT in Watts (actual UAV WPT power cost)
# 关键变量 WPT_HARVEST_GAIN：全局常量或配置项，会影响环境规模、训练过程或实验输出。
WPT_HARVEST_GAIN: float = 1e5  # equivalent WPT harvest gain for the simplified channel model
# 关键变量 WPT_EFFICIENCY：全局常量或配置项，会影响环境规模、训练过程或实验输出。
WPT_EFFICIENCY: float = 0.6  # eta (energy harvesting efficiency, 60%)
# 关键变量 UE_STATIC_POWER：用户设备对象，产生任务请求并等待服务。
UE_STATIC_POWER: float = 0.05  # Idle power consumption in Watts

# Model Parameters
# Reward formula: reward = ALPHA_3*log(fairness) - ALPHA_1*log(latency) - ALPHA_2*log(energy) - ALPHA_4*log(1+offline_rate)
# Then scaled by REWARD_SCALING_FACTOR.
ALPHA_1 = 1.0  # weightage for latency (negative term, higher = stronger penalty for latency)
# 关键变量 ALPHA_2：全局常量或配置项，会影响环境规模、训练过程或实验输出。
ALPHA_2 = 0.4  # weightage for energy (negative term, lower priority than latency)
# 关键变量 ALPHA_3：全局常量或配置项，会影响环境规模、训练过程或实验输出。
ALPHA_3 = 2.0  # weightage for fairness (positive term, encourage equal service)
# 关键变量 ALPHA_4：全局常量或配置项，会影响环境规模、训练过程或实验输出。
ALPHA_4 = 50.0  # weightage for offline rate (negative term, penalizes UEs running out of battery)
# 关键变量 REWARD_SCALING_FACTOR：当前时间步或当前回合的奖励值。
REWARD_SCALING_FACTOR: float = 0.01  # scaling factor for rewards (prevents exploding values)

# 关键变量 SELF_OBS_DIM：全局常量或配置项，会影响环境规模、训练过程或实验输出。
SELF_OBS_DIM: int = 2 + NUM_FILES  # pos (2) + cache (NUM_FILES)
# 关键变量 REQUEST_OBS_DIM：用户设备产生的任务请求。
REQUEST_OBS_DIM: int = 5  # type, size, id, deadline, priority
# 关键变量 UE_OBS_DIM：用户设备对象，产生任务请求并等待服务。
UE_OBS_DIM: int = 2 + REQUEST_OBS_DIM + 1  # pos (2) + request signature (5) + battery level (1)
# 关键变量 NEIGHBOR_OBS_DIM：全局常量或配置项，会影响环境规模、训练过程或实验输出。
NEIGHBOR_OBS_DIM: int = 2  # pos (2)
# 关键变量 OBS_DIM_SINGLE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
OBS_DIM_SINGLE: int = SELF_OBS_DIM + (MAX_UAV_NEIGHBORS * NEIGHBOR_OBS_DIM) + (MAX_ASSOCIATED_UES * UE_OBS_DIM)
# 关键变量 ACTION_DIM：智能体输出的动作，通常包含移动方向、服务缓存或任务卸载相关决策。
ACTION_DIM: int = 2  # angle, distance from [-1, 1]
# 关键变量 MLP_HIDDEN_DIM：全局常量或配置项，会影响环境规模、训练过程或实验输出。
MLP_HIDDEN_DIM: int = 128

# 关键变量 ACTOR_LR：全局常量或配置项，会影响环境规模、训练过程或实验输出。
ACTOR_LR: float = 3e-4
# 关键变量 CRITIC_LR：全局常量或配置项，会影响环境规模、训练过程或实验输出。
CRITIC_LR: float = 3e-4
# 关键变量 DISCOUNT_FACTOR：全局常量或配置项，会影响环境规模、训练过程或实验输出。
DISCOUNT_FACTOR: float = 0.99  # gamma
# 关键变量 UPDATE_FACTOR：全局常量或配置项，会影响环境规模、训练过程或实验输出。
UPDATE_FACTOR: float = 0.012  # tau
# 关键变量 MAX_GRAD_NORM：全局常量或配置项，会影响环境规模、训练过程或实验输出。
MAX_GRAD_NORM: float = 0.5  # maximum norm for gradient clipping to prevent exploding gradients
# 关键变量 LOG_STD_MAX：全局常量或配置项，会影响环境规模、训练过程或实验输出。
LOG_STD_MAX: float = 2  # maximum log standard deviation for stochastic policies
# 关键变量 LOG_STD_MIN：全局常量或配置项，会影响环境规模、训练过程或实验输出。
LOG_STD_MIN: float = -20  # minimum log standard deviation for stochastic policies
# 关键变量 EPSILON：全局常量或配置项，会影响环境规模、训练过程或实验输出。
EPSILON: float = 1e-9  # small value to prevent division by zero

# Off-policy algorithm hyperparameters
REPLAY_BUFFER_SIZE: int = 10**6  # B
# 关键变量 REPLAY_BATCH_SIZE：从数据集或经验池中取出的一个训练批次。
REPLAY_BATCH_SIZE: int = 128  # minibatch size
# 关键变量 INITIAL_RANDOM_STEPS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
INITIAL_RANDOM_STEPS: int = 5000  # steps of random actions for exploration
# 关键变量 LEARN_FREQ：全局常量或配置项，会影响环境规模、训练过程或实验输出。
LEARN_FREQ: int = 10  # steps to learn after

# Gaussian Noise Parameters (for MADDPG and MATD3)
INITIAL_NOISE_SCALE: float = 0.2
# 关键变量 MIN_NOISE_SCALE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
MIN_NOISE_SCALE: float = 0.01
# 关键变量 NOISE_DECAY_RATE：通信链路速率。
NOISE_DECAY_RATE: float = 0.995

# MATD3 Specific Hyperparameters
POLICY_UPDATE_FREQ: int = 2  # delayed policy update frequency
# 关键变量 TARGET_POLICY_NOISE：全局常量或配置项，会影响环境规模、训练过程或实验输出。
TARGET_POLICY_NOISE: float = 0.25  # standard deviation of target policy smoothing noise.
# 关键变量 NOISE_CLIP：全局常量或配置项，会影响环境规模、训练过程或实验输出。
NOISE_CLIP: float = 0.5  # range to clip target policy smoothing noise

# MAPPO Specific Hyperparameters
PPO_ROLLOUT_LENGTH: int = STEPS_PER_EPISODE  # number of steps to collect per rollout before updating
# 关键变量 PPO_GAE_LAMBDA：全局常量或配置项，会影响环境规模、训练过程或实验输出。
PPO_GAE_LAMBDA: float = 0.95  # lambda parameter for GAE
# 关键变量 PPO_EPOCHS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
PPO_EPOCHS: int = 8  # number of epochs to run on the collected rollout data
# 关键变量 PPO_BATCH_SIZE：从数据集或经验池中取出的一个训练批次。
PPO_BATCH_SIZE: int = 256  # size of mini-batches to use during the update step
# 关键变量 PPO_CLIP_EPS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
PPO_CLIP_EPS: float = 0.2  # clipping parameter (epsilon) for the PPO surrogate objective
# 关键变量 PPO_ENTROPY_COEF：全局常量或配置项，会影响环境规模、训练过程或实验输出。
PPO_ENTROPY_COEF: float = 0.005  # slightly lower entropy helps attention-MAPPO stabilize later in training

# MASAC Specific Hyperparameters
ALPHA_LR: float = 3e-4  # learning rate for the entropy temperature alpha

# Attention Hyperparameters
ATTN_HIDDEN_DIM: int = 64  # Embedding size for internal attention representations
# 关键变量 ATTN_NUM_HEADS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
ATTN_NUM_HEADS: int = 8  # Number of attention heads
assert ATTN_HIDDEN_DIM % ATTN_NUM_HEADS == 0, f"ATTN_HIDDEN_DIM ({ATTN_HIDDEN_DIM}) must be divisible by ATTN_NUM_HEADS ({ATTN_NUM_HEADS})"
