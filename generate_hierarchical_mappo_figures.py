from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "docs" / "figures"


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_current(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=240, bbox_inches="tight")
    plt.close()


def moving_average(values: list[float], window: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = min(window, arr.size)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: arr.size]


def plot_training_curves(log_path: Path) -> None:
    rows = load_json(log_path)
    if isinstance(rows, dict):
        rows = [rows]
    reward = [float(row.get("reward", row.get("avg_reward", 0.0))) for row in rows]
    dsr = [float(row.get("deadline_satisfaction_rate", 0.0)) * 100.0 for row in rows]
    mbs = [float(row.get("mbs_load_ratio", 0.0)) * 100.0 for row in rows]
    energy = [float(row.get("energy", 0.0)) for row in rows]
    x = np.arange(1, len(reward) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8))
    panels = [
        (reward, "系统奖励", axes[0, 0]),
        (dsr, "截止时间满足率（%）", axes[0, 1]),
        (mbs, "宏基站负载比例（%）", axes[1, 0]),
        (energy, "系统能耗", axes[1, 1]),
    ]
    for values, ylabel, ax in panels:
        ax.plot(x, values, alpha=0.25, color="#2563eb")
        ax.plot(x, moving_average(values), linewidth=2.0, color="#1d4ed8")
        ax.set_xlabel("训练记录点")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle("双层注意力 MAPPO 训练过程", fontweight="bold")
    save_current("fig_hierarchical_training_curves.png")


def metric_from_policy(summary: dict, policy: str, metric: str) -> float:
    item = summary.get(policy, {})
    if isinstance(item.get(metric), dict):
        return float(item[metric].get("mean", 0.0))
    return float(item.get(metric, 0.0))


def plot_policy_comparison(summary_path: Path) -> None:
    summary = load_json(summary_path)
    policies = list(summary.get("overall", summary).keys())
    source = summary.get("overall", summary)
    labels = [p.replace("_", "\n") for p in policies]
    metrics = [
        ("latency", "平均时延", lambda x: x),
        ("energy", "能耗", lambda x: x),
        ("deadline_satisfaction_rate", "截止时间满足率（%）", lambda x: x * 100.0),
        ("mbs_load_ratio", "宏基站负载比例（%）", lambda x: x * 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2))
    colors = ["#64748b" if "hierarchical" not in p else "#dc2626" for p in policies]
    for ax, (metric, ylabel, transform) in zip(axes.ravel(), metrics):
        values = [transform(metric_from_policy(source, p, metric)) for p in policies]
        ax.bar(np.arange(len(policies)), values, color=colors)
        ax.set_xticks(np.arange(len(policies)), labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("不同轨迹-卸载方法主性能对比", fontweight="bold")
    save_current("fig_hierarchical_policy_comparison.png")


def plot_mbs_dsr_tradeoff(summary_path: Path) -> None:
    summary = load_json(summary_path)
    source = summary.get("overall", summary)
    policies = list(source.keys())
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for policy in policies:
        x = metric_from_policy(source, policy, "mbs_load_ratio") * 100.0
        y = metric_from_policy(source, policy, "deadline_satisfaction_rate") * 100.0
        color = "#dc2626" if "hierarchical" in policy else "#2563eb"
        ax.scatter(x, y, s=70, color=color)
        ax.text(x, y, " " + policy.replace("_", " "), fontsize=8, va="center")
    ax.set_xlabel("宏基站负载比例（越低越好，%）")
    ax.set_ylabel("截止时间满足率（越高越好，%）")
    ax.set_title("宏基站减负与服务稳定性权衡", fontweight="bold")
    ax.grid(alpha=0.25)
    save_current("fig_hierarchical_mbs_dsr_tradeoff.png")


def plot_request_destination(summary_path: Path) -> None:
    summary = load_json(summary_path)
    source = summary.get("overall", summary)
    policies = list(source.keys())
    local = [metric_from_policy(source, p, "offloading_ratio_local") * 100.0 for p in policies]
    coop = [metric_from_policy(source, p, "offloading_ratio_cooperative") * 100.0 for p in policies]
    mbs = [metric_from_policy(source, p, "offloading_ratio_mbs") * 100.0 for p in policies]
    x = np.arange(len(policies))
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(x, local, label="本地无人机", color="#22c55e")
    ax.bar(x, coop, bottom=local, label="协作无人机", color="#3b82f6")
    ax.bar(x, mbs, bottom=np.asarray(local) + np.asarray(coop), label="宏基站", color="#f97316")
    ax.set_xticks(x, [p.replace("_", "\n") for p in policies], rotation=20, ha="right")
    ax.set_ylabel("请求去向比例（%）")
    ax.set_title("请求执行位置结构", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_current("fig_hierarchical_request_destination.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for hierarchical MAPPO experiments.")
    parser.add_argument("--train_log", type=str, default=None, help="Path to hierarchical training log JSON.")
    parser.add_argument("--comparison_summary", type=str, default=None, help="Path to policy comparison summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_log:
        plot_training_curves(Path(args.train_log))
    if args.comparison_summary:
        summary = Path(args.comparison_summary)
        plot_policy_comparison(summary)
        plot_mbs_dsr_tradeoff(summary)
        plot_request_destination(summary)
    print(f"双层 MAPPO 图已生成到：{FIG_DIR}")


if __name__ == "__main__":
    main()
