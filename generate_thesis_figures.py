"""
中文注释说明：generate_thesis_figures.py

文件作用：
    根据实验日志和结果文件生成论文所需图表。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - ROOT: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - FIG_DIR: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - UPPER_LOGS: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - LOWER_SUMMARY: 实验汇总信息，最终写入报告或 manifest 文件。
    - load_json(): 加载模型参数或实验数据。
    - mean(): 关键函数，承载本模块的一段可复用实验逻辑。
    - percent(): 关键函数，承载本模块的一段可复用实验逻辑。
    - save_current(): 保存模型参数或实验结果。
    - draw_box(): 关键函数，承载本模块的一段可复用实验逻辑。
    - draw_arrow(): 关键函数，承载本模块的一段可复用实验逻辑。
    - plot_framework(): 关键函数，承载本模块的一段可复用实验逻辑。
    - upper_metrics(): 关键函数，承载本模块的一段可复用实验逻辑。
    - plot_upper_comparison(): 关键函数，承载本模块的一段可复用实验逻辑。
    - lower_overall(): 关键函数，承载本模块的一段可复用实验逻辑。
    - plot_lower_runtime(): 关键函数，承载本模块的一段可复用实验逻辑。
    - plot_mbs_tradeoff(): 关键函数，承载本模块的一段可复用实验逻辑。
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    json, pathlib, matplotlib, numpy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# 关键变量 ROOT：全局常量或配置项，会影响环境规模、训练过程或实验输出。
ROOT = Path(__file__).resolve().parent
# 关键变量 FIG_DIR：全局常量或配置项，会影响环境规模、训练过程或实验输出。
FIG_DIR = ROOT / "docs" / "figures"
# 关键变量 UPPER_LOGS：全局常量或配置项，会影响环境规模、训练过程或实验输出。
UPPER_LOGS = {
    "注意力-MAPPO": ROOT
    / "results"
    / "full_runs"
    / "wpt_fix_thesis_run"
    / "test_logs"
    / "attention_mappo"
    / "log_data_20260427_142720_attention_mappo.json",
    "无协调贪心": ROOT
    / "results"
    / "full_runs"
    / "wpt_fix_thesis_run"
    / "test_logs"
    / "uncoordinated_greedy"
    / "log_data_20260427_154340_uncoordinated_greedy.json",
}
# 关键变量 LOWER_SUMMARY：实验汇总信息，最终写入报告或 manifest 文件。
LOWER_SUMMARY = (
    ROOT
    / "results"
    / "full_offload_experiments"
    / "wpt_fix_thesis_run_offload"
    / "experiment_summary.json"
)


# 函数 load_json：加载模型参数或实验数据，主要参数：path。
def load_json(path: Path) -> dict:
    # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
    with path.open("r", encoding="utf-8") as f:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return json.load(f)


# 函数 mean：关键函数，承载本模块的一段可复用实验逻辑，主要参数：values。
def mean(values: list[float]) -> float:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return float(np.mean(np.asarray(values, dtype=float)))


# 函数 percent：关键函数，承载本模块的一段可复用实验逻辑，主要参数：value。
def percent(value: float) -> float:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return value * 100.0


# 函数 save_current：保存模型参数或实验结果，主要参数：name。
def save_current(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close()


# 函数 draw_box：关键函数，承载本模块的一段可复用实验逻辑，主要参数：ax, xy, width, height, text, color。
def draw_box(ax, xy, width, height, text, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="#111827",
        wrap=True,
    )


# 函数 draw_arrow：关键函数，承载本模块的一段可复用实验逻辑，主要参数：ax, start, end。
def draw_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.4,
            color="#374151",
        )
    )


# 函数 plot_framework：关键函数，承载本模块的一段可复用实验逻辑。
def plot_framework() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.07, 0.66), 0.26, 0.18, "上层\n轨迹控制\n注意力-MAPPO", "#dbeafe")
    draw_box(ax, (0.37, 0.66), 0.26, 0.18, "系统状态\n链路 / 队列\n缓存 / 负载", "#e0f2fe")
    draw_box(ax, (0.67, 0.66), 0.26, 0.18, "下层\n请求级卸载\n本地 / 协作 / MBS", "#dcfce7")

    draw_box(ax, (0.08, 0.22), 0.22, 0.16, "UAV 位置\n覆盖范围\n邻居链路", "#fef3c7")
    draw_box(ax, (0.39, 0.22), 0.22, 0.16, "服务请求\n截止期\n优先级", "#fde68a")
    draw_box(ax, (0.70, 0.22), 0.22, 0.16, "性能指标\n时延 / 能耗\nMBS 负载", "#fee2e2")

    draw_arrow(ax, (0.33, 0.75), (0.37, 0.75))
    draw_arrow(ax, (0.63, 0.75), (0.67, 0.75))
    draw_arrow(ax, (0.20, 0.66), (0.19, 0.38))
    draw_arrow(ax, (0.50, 0.66), (0.50, 0.38))
    draw_arrow(ax, (0.78, 0.66), (0.81, 0.38))
    draw_arrow(ax, (0.70, 0.30), (0.61, 0.30))
    draw_arrow(ax, (0.39, 0.30), (0.30, 0.30))

    ax.text(
        0.5,
        0.94,
        "多无人机 MEC 双层协同优化框架",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    save_current("fig_framework.png")


# 函数 upper_metrics：关键函数，承载本模块的一段可复用实验逻辑。
def upper_metrics() -> dict[str, dict[str, float]]:
    output = {}
    # 循环处理：遍历 (name, path) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for name, path in UPPER_LOGS.items():
        data = load_json(path)
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if isinstance(data, dict):
            data = [data]
        output[name] = {
            "奖励": mean([row["reward"] for row in data]),
            "平均时延": mean([row["latency"] for row in data]),
            "截止期满足率": percent(mean([row["deadline_satisfaction_rate"] for row in data])),
            "公平性": mean([row["fairness"] for row in data]),
        }
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return output


# 函数 plot_upper_comparison：关键函数，承载本模块的一段可复用实验逻辑。
def plot_upper_comparison() -> None:
    metrics = upper_metrics()
    names = list(metrics)
    keys = ["奖励", "平均时延", "截止期满足率", "公平性"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.ravel()
    colors = ["#2563eb", "#f97316"]

    # 循环处理：遍历 (ax, key) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for ax, key in zip(axes, keys):
        values = [metrics[name][key] for name in names]
        ax.bar(names, values, color=colors, width=0.56)
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.25)
        # 循环处理：遍历 (idx, value) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for idx, value in enumerate(values):
            label = f"{value:.2f}" if abs(value) < 10000 else f"{value:.1f}"
            ax.text(idx, value, label, ha="center", va="bottom", fontsize=8)

    fig.suptitle("上层轨迹控制性能对比", fontsize=14, fontweight="bold")
    save_current("fig_upper_comparison.png")


# 函数 lower_overall：关键函数，承载本模块的一段可复用实验逻辑。
def lower_overall() -> dict:
    summary = load_json(LOWER_SUMMARY)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return summary["runtime_comparison"]["overall"]


# 函数 plot_lower_runtime：关键函数，承载本模块的一段可复用实验逻辑。
def plot_lower_runtime() -> None:
    overall = lower_overall()
    policies = [
        "heuristic_offloading",
        "surrogate_baseline",
        "rich_reduced_runtime_policy",
    ]
    labels = ["启发式", "代理基线", "精简特征策略"]
    metrics = [
        ("latency", "平均时延", lambda x: x),
        ("deadline_satisfaction_rate", "截止期满足率（%）", percent),
        ("offloading_ratio_mbs", "MBS 卸载比例（%）", percent),
        ("mbs_load_ratio", "MBS 负载比例（%）", percent),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    axes = axes.ravel()
    colors = ["#6b7280", "#2563eb", "#16a34a"]

    # 循环处理：遍历 (ax, (field, title, transform)) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for ax, (field, title, transform) in zip(axes, metrics):
        values = [transform(overall[p][field]["mean"]) for p in policies]
        ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        # 循环处理：遍历 (idx, value) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("下层任务卸载策略在线性能对比", fontsize=14, fontweight="bold")
    save_current("fig_lower_runtime.png")


# 函数 plot_mbs_tradeoff：关键函数，承载本模块的一段可复用实验逻辑。
def plot_mbs_tradeoff() -> None:
    summary = load_json(LOWER_SUMMARY)
    overall = summary["runtime_comparison"]["overall"]
    baseline = overall["heuristic_offloading"]
    policies = {
        "代理基线": overall["surrogate_baseline"],
        "精简特征策略": overall["rich_reduced_runtime_policy"],
    }

    points = []
    # 循环处理：遍历 (name, data) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for name, data in policies.items():
        mbs_reduction = (
            baseline["mbs_load_ratio"]["mean"] - data["mbs_load_ratio"]["mean"]
        ) / baseline["mbs_load_ratio"]["mean"] * 100
        deadline_loss = (
            baseline["deadline_satisfaction_rate"]["mean"]
            - data["deadline_satisfaction_rate"]["mean"]
        ) / baseline["deadline_satisfaction_rate"]["mean"] * 100
        energy_increase = (
            data["energy"]["mean"] - baseline["energy"]["mean"]
        ) / baseline["energy"]["mean"] * 100
        points.append((name, mbs_reduction, deadline_loss, energy_increase))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#2563eb", "#16a34a"]
    # 循环处理：遍历 (color, (name, x, y, size_metric)) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for color, (name, x, y, size_metric) in zip(colors, points):
        ax.scatter(x, y, s=140 + size_metric * 18, color=color, alpha=0.82, edgecolor="#111827")
        ax.text(x + 0.8, y + 0.05, f"{name}\n能耗 +{size_metric:.1f}%", fontsize=9)

    ax.set_xlabel("相对启发式策略的 MBS 负载降低幅度（%）")
    ax.set_ylabel("相对启发式策略的截止期满足率损失（%）")
    ax.set_title("MBS 减负与服务质量代价权衡", fontweight="bold")
    ax.grid(alpha=0.28)
    ax.set_xlim(0, max(p[1] for p in points) + 12)
    ax.set_ylim(0, max(p[2] for p in points) + 1.5)
    save_current("fig_mbs_tradeoff.png")


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main() -> None:
    plot_framework()
    plot_upper_comparison()
    plot_lower_runtime()
    plot_mbs_tradeoff()
    print(f"论文图表已生成到：{FIG_DIR}")


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
