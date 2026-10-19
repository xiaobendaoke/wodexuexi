"""
中文注释说明：utils/plot_snapshots.py

文件作用：
    可视化环境快照或实验过程中的状态分布，辅助检查仿真行为。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - plot_snapshot(): 关键函数，承载本模块的一段可复用实验逻辑。

主要依赖：
    environment, config, matplotlib, numpy, os

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from environment.env import Env
import config
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# from matplotlib import colormaps as cm  # trajectory tracking code, comment if not needed
import numpy as np
import os


# Trajectory tracking code, comment if not needed
# class TrajectoryTracker:
#     def __init__(self) -> None:
#         self.paths: dict[int, list[np.ndarray]] = {}

#     def update(self, env: Env) -> None:
#         for uav in env.uavs:
#             if uav.id not in self.paths:
#                 self.paths[uav.id] = []
#             self.paths[uav.id].append(uav.pos.copy())

#     def reset(self, env: Env) -> None:
#         self.paths = {}
#         self.update(env)

#     def get_path(self, uav_id: int) -> np.ndarray:
#         return np.array(self.paths.get(uav_id, []))


# # Global tracker instance
# tracker = TrajectoryTracker()


# def update_trajectories(env: Env) -> None:
#     tracker.update(env)


# def reset_trajectories(env: Env) -> None:
#     tracker.reset(env)


# 函数 plot_snapshot：关键函数，承载本模块的一段可复用实验逻辑，主要参数：env, progress_step, step, save_dir, timestamp, initial。
def plot_snapshot(env: Env, progress_step: int, step: int, save_dir: str, timestamp: str, initial: bool = False) -> None:
    """Generates and saves a plot of the current environment state."""
    save_path: str = f"{save_dir}/state_images_{timestamp}/episode_{progress_step:04d}"
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal")
    ax.set_title(f"Simulation Snapshot at Episode: {progress_step}, Step: {step}")
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")

    # Plot UEs
    service_ues_pos: np.ndarray = np.array([ue.pos for ue in env.ues if ue.current_request.is_service], dtype=np.float32)
    content_ues_pos: np.ndarray = np.array([ue.pos for ue in env.ues if ue.current_request.is_content], dtype=np.float32)
    energy_ues_pos: np.ndarray = np.array([ue.pos for ue in env.ues if ue.current_request.is_energy], dtype=np.float32)

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if service_ues_pos.size > 0:
        ax.scatter(service_ues_pos[:, 0], service_ues_pos[:, 1], c="blue", marker=".", alpha=0.6, label="UE (Service Req)")

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if content_ues_pos.size > 0:
        ax.scatter(content_ues_pos[:, 0], content_ues_pos[:, 1], c="green", marker=".", alpha=0.6, label="UE (Content Req)")

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if energy_ues_pos.size > 0:
        ax.scatter(energy_ues_pos[:, 0], energy_ues_pos[:, 1], c="purple", marker=".", alpha=0.6, label="UE (Energy Req)")

    # Plot UAV trajectories, comment if not needed
    # cmap = cm["plasma"]
    # colors = cmap(np.linspace(0, 1, len(env.uavs)))
    # for uav, color in zip(env.uavs, colors):
    #     path = tracker.get_path(uav.id)
    #     if len(path) > 1:
    #         ax.plot(path[:, 0], path[:, 1], color=color, linestyle="-", linewidth=1.5, alpha=0.7)

    plotted_labels: set = set()

    # Plot UAVs and their connections
    for uav in env.uavs:
        # UAV position (red square)
        lbl = "UAV" if "UAV" not in plotted_labels else ""
        ax.scatter(uav.pos[0], uav.pos[1], c="red", marker="s", s=100, label=lbl, zorder=5)
        plotted_labels.add("UAV")

        # UAV coverage radius
        coverage_circle: Circle = Circle((uav.pos[0], uav.pos[1]), config.UAV_COVERAGE_RADIUS, color="red", alpha=0.1, label="Service Range" if "Service Range" not in plotted_labels else "")
        ax.add_patch(coverage_circle)
        plotted_labels.add("Service Range")

        # Lines to covered UEs (green)
        for ue in uav.current_covered_ues:
            lbl = "UE Association" if "UE Association" not in plotted_labels else ""
            ax.plot([uav.pos[0], ue.pos[0]], [uav.pos[1], ue.pos[1]], "g-", lw=0.5, alpha=0.5, label=lbl)
            plotted_labels.add("UE Association")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.9)

    # Save the figure
    if initial:
        plt.savefig(f"{save_path}/initial.png")
    else:
        plt.savefig(f"{save_path}/step_{step:04d}.png")

    plt.close(fig)

    # reset_trajectories(env)  # tracking code, comment if not needed
