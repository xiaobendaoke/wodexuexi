"""
中文注释说明：utils/plot_logs.py

文件作用：
    读取单次实验日志并生成训练或测试曲线图。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - plot_metric(): 关键函数，承载本模块的一段可复用实验逻辑。
    - generate_plots(): 通信链路速率。

主要依赖：
    os, json, matplotlib, numpy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


# 函数 plot_metric：关键函数，承载本模块的一段可复用实验逻辑，主要参数：x, y, xlabel, ylabel, title, output_path。
def plot_metric(
    x: list,
    y: list,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
    smoothing_window: int = 5,
) -> None:
    """Plot a single metric with optional smoothing and save it."""
    plt.figure(figsize=(12, 6))

    # Apply smoothing using moving average if window > 1
    if len(y) > smoothing_window and smoothing_window > 1:
        y_smooth = np.convolve(y, np.ones(smoothing_window) / smoothing_window, mode="valid")
        x_smooth = x[: len(y_smooth)]
        plt.plot(x_smooth, y_smooth, linewidth=2, label="Smoothed", color="#1f77b4")
        plt.plot(x, y, alpha=0.3, linestyle="--", label="Raw", color="#1f77b4")
    else:
        plt.plot(x, y, linewidth=2, color="#1f77b4")

    plt.xlabel(xlabel, fontsize=12, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend() if len(y) > smoothing_window and smoothing_window > 1 else None
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# 函数 generate_plots：通信链路速率，主要参数：log_file, output_dir, output_file_prefix, timestamp, smoothing_window。
def generate_plots(log_file: str, output_dir: str, output_file_prefix: str, timestamp: str, smoothing_window: int = 5) -> None:
    """Generate all required plots from JSON log file."""

    # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
    with open(log_file, "r") as file:
        log_data: list[dict] = json.load(file)

    os.makedirs(output_dir, exist_ok=True)

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not log_data:
        print(f"ERROR: Log file is empty: {log_file}")
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return

    # Extract data
    parameters: dict = {
        "episode": [entry.get("episode") for entry in log_data],
        "reward": [entry.get("reward") for entry in log_data],
        "latency": [entry.get("latency") for entry in log_data],
        "energy": [entry.get("energy") for entry in log_data],
        "fairness": [entry.get("fairness") for entry in log_data],
        "offline_rate": [entry.get("offline_rate") for entry in log_data],
        "deadline_satisfaction_rate": [entry.get("deadline_satisfaction_rate") for entry in log_data],
        "offloading_ratio_local": [entry.get("offloading_ratio_local") for entry in log_data],
        "offloading_ratio_cooperative": [entry.get("offloading_ratio_cooperative") for entry in log_data],
        "offloading_ratio_mbs": [entry.get("offloading_ratio_mbs") for entry in log_data],
        "mbs_load_ratio": [entry.get("mbs_load_ratio") for entry in log_data],
        "actor_loss": [entry.get("actor_loss") for entry in log_data],
        "critic_loss": [entry.get("critic_loss") for entry in log_data],
        "entropy_loss": [entry.get("entropy_loss") for entry in log_data],
        "alpha_loss": [entry.get("alpha_loss") for entry in log_data],
    }

    x_data = parameters["episode"]

    # Plot environment metrics
    metrics_to_plot = [
        "reward",
        "latency",
        "energy",
        "fairness",
        "offline_rate",
        "deadline_satisfaction_rate",
        "offloading_ratio_local",
        "offloading_ratio_cooperative",
        "offloading_ratio_mbs",
        "mbs_load_ratio",
    ]
    # 循环处理：遍历 metric 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for metric in metrics_to_plot:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if any(v is not None for v in parameters[metric]):
            y_data = [v if v is not None else np.nan for v in parameters[metric]]
            # Remove NaN values for cleaner plots
            valid_indices = [i for i, v in enumerate(y_data) if not np.isnan(v)]
            x_filtered = [x_data[i] for i in valid_indices]
            y_filtered = [y_data[i] for i in valid_indices]

            title = f"{metric.replace('_', ' ').title()} vs Episode"
            output_path = os.path.join(output_dir, f"{output_file_prefix}_{metric}_{timestamp}.png")
            plot_metric(x_filtered, y_filtered, "Episode", metric.title(), title, output_path, smoothing_window)

    # Plot loss curves (if available)
    loss_metrics = ["actor_loss", "critic_loss", "entropy_loss", "alpha_loss"]
    # 循环处理：遍历 metric 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for metric in loss_metrics:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if any(v is not None for v in parameters[metric]):
            y_data = [v if v is not None else np.nan for v in parameters[metric]]
            valid_indices = [i for i, v in enumerate(y_data) if not np.isnan(v)]
            x_filtered = [x_data[i] for i in valid_indices]
            y_filtered = [y_data[i] for i in valid_indices]

            title = f"{metric.replace('_', ' ').title()} vs Episode"
            output_path = os.path.join(output_dir, f"{output_file_prefix}_{metric}_{timestamp}.png")
            plot_metric(x_filtered, y_filtered, "Episode", metric.title(), title, output_path, smoothing_window)

    print(f"All plots saved to {output_dir}\n")
