from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import config
from analyze_experiment_statistics import build_statistics, extract_policy_units
from analyze_offload_classifier_quality import analyze_classifier_quality

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "docs" / "figures"
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
UPPER_TRAIN_LOGS = {
    "注意力-MAPPO": ROOT
    / "results"
    / "full_runs"
    / "wpt_fix_thesis_run"
    / "train_logs"
    / "attention_mappo"
    / "log_data_20260427_142720_attention_mappo.json",
    "无协调贪心": ROOT
    / "results"
    / "full_runs"
    / "wpt_fix_thesis_run"
    / "train_logs"
    / "uncoordinated_greedy"
    / "log_data_20260427_154340_uncoordinated_greedy.json",
}
LOWER_SUMMARY = (
    ROOT
    / "results"
    / "full_offload_experiments"
    / "wpt_fix_thesis_run_offload"
    / "experiment_summary.json"
)
OFFLOAD_DATASET = (
    ROOT
    / "results"
    / "full_offload_experiments"
    / "wpt_fix_thesis_run_offload"
    / "datasets"
    / "offload_dataset_runtime_candidate.npz"
)
OFFLOAD_SURROGATE_CHECKPOINT = (
    ROOT
    / "results"
    / "full_offload_experiments"
    / "wpt_fix_thesis_run_offload"
    / "checkpoints"
    / "offload_policy_surrogate_runtime.pt"
)
CLASSIFIER_QUALITY_SUMMARY = (
    ROOT
    / "results"
    / "full_offload_experiments"
    / "wpt_fix_thesis_run_offload"
    / "reports"
    / "classifier_quality_summary.json"
)
LOWER_STATS_SUMMARY = (
    ROOT
    / "results"
    / "full_offload_experiments"
    / "wpt_fix_thesis_run_offload"
    / "reports"
    / "runtime_offload_policy_statistics.json"
)

POLICIES = [
    "heuristic_offloading",
    "surrogate_baseline",
    "rich_reduced_runtime_policy",
]
POLICY_LABELS = {
    "heuristic_offloading": "启发式",
    "surrogate_baseline": "Oracle引导策略",
    "rich_reduced_runtime_policy": "精简特征策略",
}
POLICY_COLORS = {
    "heuristic_offloading": "#6b7280",
    "surrogate_baseline": "#2563eb",
    "rich_reduced_runtime_policy": "#16a34a",
}
SCENARIO_LABELS = {
    "default_uniform": "均匀",
    "hotspot_deadline_stress": "热点+紧截止期",
    "local_cache_friendly": "本地缓存友好",
    "cooperative_friendly": "协作友好",
    "mbs_heavy_jobs": "MBS优势",
}


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_current(name: str, dpi: int = 240) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=dpi, bbox_inches="tight")
    plt.close()


def percent(value: float) -> float:
    return value * 100.0


def format_value(value: float) -> str:
    if abs(value) >= 10000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_box(ax, xy, width, height, text, color, fontsize=10, linewidth=1.2):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=linewidth,
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
        fontsize=fontsize,
        color="#111827",
        linespacing=1.25,
    )


def rolling_mean(values: list[float] | np.ndarray, window: int = 15) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr
    resolved_window = min(int(window), int(arr.size))
    kernel = np.ones(resolved_window, dtype=float) / float(resolved_window)
    padded = np.pad(arr, (resolved_window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: arr.size]


def ci_width_from_stat(stat: dict, transform=lambda x: x) -> tuple[float, float, float]:
    mean = transform(float(stat["mean"]))
    low = transform(float(stat["ci_low"]))
    high = transform(float(stat["ci_high"]))
    return mean, max(mean - low, 0.0), max(high - mean, 0.0)


def draw_arrow(ax, start, end, color="#374151", connectionstyle="arc3,rad=0.0", style="->"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=15,
            linewidth=1.5,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def plot_framework() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.03, 0.72), 0.20, 0.15, "系统实体\nUE / UAV / MBS\n缓存 / 队列 / 能量", "#f3f4f6", fontsize=9)
    draw_box(ax, (0.30, 0.72), 0.22, 0.15, "上层决策环\nAttention-MAPPO\nstate: UAV/邻居/UE\n动作: 方向 + 距离", "#dbeafe", fontsize=9)
    draw_box(ax, (0.60, 0.72), 0.20, 0.15, "环境更新\n覆盖 / 链路 / 队列\nMBS负载 / 能量", "#e0f2fe", fontsize=9)
    draw_box(ax, (0.82, 0.72), 0.15, 0.15, "输出指标\nreward / latency\nDSR / fairness", "#fee2e2", fontsize=8)

    draw_box(ax, (0.12, 0.40), 0.22, 0.15, "请求到达\nservice requests\n输入大小 / deadline\n优先级", "#fef3c7", fontsize=9)
    draw_box(ax, (0.43, 0.40), 0.24, 0.15, "下层决策环\noracle-guided policy\nfeatures: 候选时延\n队列 / MBS负载 / 协作收益", "#dcfce7", fontsize=8.5)
    draw_box(ax, (0.75, 0.40), 0.18, 0.15, "请求去向\nlocal UAV\ncooperative UAV\nMBS fallback", "#ede9fe", fontsize=9)
    draw_box(ax, (0.28, 0.12), 0.44, 0.13, "离线增强 oracle 标签\nlatency + deadline penalty + queue pressure + MBS load penalty + coop relief", "#f8fafc", fontsize=8.5)

    draw_arrow(ax, (0.23, 0.795), (0.30, 0.795))
    draw_arrow(ax, (0.52, 0.795), (0.60, 0.795))
    draw_arrow(ax, (0.80, 0.795), (0.82, 0.795))
    draw_arrow(ax, (0.66, 0.72), (0.23, 0.50), color="#0f766e", connectionstyle="arc3,rad=0.16")
    draw_arrow(ax, (0.34, 0.475), (0.43, 0.475))
    draw_arrow(ax, (0.67, 0.475), (0.75, 0.475))
    draw_arrow(ax, (0.84, 0.40), (0.70, 0.25), color="#7c3aed", connectionstyle="arc3,rad=-0.08")
    draw_arrow(ax, (0.50, 0.25), (0.50, 0.40), color="#7c3aed")
    draw_arrow(ax, (0.80, 0.52), (0.72, 0.72), color="#dc2626", connectionstyle="arc3,rad=-0.20")
    draw_arrow(ax, (0.43, 0.72), (0.42, 0.55), color="#2563eb")

    ax.text(0.5, 0.94, "双层协同优化端到端流程", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.035, "时间尺度: 上层每时隙控制 UAV 运动；下层在请求级选择执行位置；执行反馈改变下一时隙队列、链路和 MBS 负载", ha="center", va="center", fontsize=9, color="#4b5563")
    save_current("fig_framework.png")


def upper_metric_samples() -> dict[str, dict[str, list[float]]]:
    output: dict[str, dict[str, list[float]]] = {}
    fields = {
        "奖励": ("reward", lambda x: x),
        "平均时延": ("latency", lambda x: x),
        "截止期满足率（%）": ("deadline_satisfaction_rate", percent),
        "公平性": ("fairness", lambda x: x),
    }
    for name, path in UPPER_LOGS.items():
        rows = load_json(path)
        if isinstance(rows, dict):
            rows = [rows]
        output[name] = {}
        for label, (field, transform) in fields.items():
            output[name][label] = [transform(float(row[field])) for row in rows]
    return output


def plot_upper_comparison() -> None:
    samples = upper_metric_samples()
    names = list(samples)
    metrics = list(next(iter(samples.values())).keys())
    colors = ["#2563eb", "#f97316"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        means = [float(np.mean(samples[name][metric])) for name in names]
        stds = [float(np.std(samples[name][metric])) for name in names]
        ax.bar(names, means, yerr=stds, color=colors, width=0.55, capsize=5, alpha=0.92)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(means):
            va = "bottom" if value >= 0 else "top"
            offset = max(abs(value) * 0.02, 0.01)
            y = value + offset if value >= 0 else value - offset
            ax.text(idx, y, format_value(value), ha="center", va=va, fontsize=8)

    fig.suptitle("上层轨迹控制测试性能对比（均值 ± 标准差）", fontsize=14, fontweight="bold")
    save_current("fig_upper_comparison.png")


def runtime_summary() -> dict:
    return load_json(LOWER_SUMMARY)["runtime_comparison"]


def lower_overall() -> dict:
    return runtime_summary()["overall"]


def lower_statistics() -> dict:
    if LOWER_STATS_SUMMARY.exists():
        return load_json(LOWER_STATS_SUMMARY)
    source_summary = load_json(LOWER_SUMMARY)
    policy_units = extract_policy_units(source_summary)
    return build_statistics(policy_units, reference="heuristic_offloading", confidence=0.95, bootstrap_samples=3000)


def metric_stat(data: dict, policy: str, field: str, transform=lambda x: x) -> tuple[float, float]:
    stat = data[policy][field]
    return transform(float(stat["mean"])), abs(transform(float(stat["std"])))


def plot_lower_runtime() -> None:
    overall = lower_overall()
    metrics = [
        ("latency", "平均时延", lambda x: x),
        ("deadline_satisfaction_rate", "截止期满足率（%）", percent),
        ("offloading_ratio_mbs", "MBS 卸载比例（%）", percent),
        ("mbs_load_ratio", "MBS 负载比例（%）", percent),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    axes = axes.ravel()
    x = np.arange(len(POLICIES))
    labels = [POLICY_LABELS[p] for p in POLICIES]
    colors = [POLICY_COLORS[p] for p in POLICIES]

    for ax, (field, title, transform) in zip(axes, metrics):
        means, stds = zip(*(metric_stat(overall, p, field, transform) for p in POLICIES))
        ax.bar(x, means, yerr=stds, color=colors, width=0.58, capsize=5, alpha=0.92)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(means):
            ax.text(idx, value + max(abs(value) * 0.025, 0.05), format_value(value), ha="center", fontsize=8)

    fig.suptitle("下层任务卸载策略在线性能对比（多场景/多随机种子，均值 ± 标准差）", fontsize=14, fontweight="bold")
    save_current("fig_lower_runtime.png")


def plot_mbs_tradeoff() -> None:
    overall = lower_overall()
    baseline = overall["heuristic_offloading"]
    points = []
    for policy in POLICIES[1:]:
        data = overall[policy]
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
        points.append((policy, mbs_reduction, deadline_loss, energy_increase))

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.axhline(0, color="#9ca3af", linewidth=1)
    ax.axvline(0, color="#9ca3af", linewidth=1)
    for policy, x, y, size_metric in points:
        size = 180 + max(size_metric, 0.0) * 20
        color = POLICY_COLORS[policy]
        ax.scatter(x, y, s=size, color=color, alpha=0.82, edgecolor="#111827", linewidth=1.2)
        ax.text(x + 1.1, y + 0.08, f"{POLICY_LABELS[policy]}\n能耗 +{size_metric:.1f}%", fontsize=9)

    ax.annotate("更理想区域\n减负更高、损失更低", xy=(48, 0.35), xytext=(25, 0.2), arrowprops={"arrowstyle": "->", "color": "#374151"}, fontsize=9)
    ax.set_xlabel("相对启发式策略的 MBS 负载降低幅度（%）")
    ax.set_ylabel("相对启发式策略的截止期满足率损失（%）")
    ax.set_title("MBS 减负与服务质量代价权衡（气泡大小表示能耗增加）", fontweight="bold")
    ax.grid(alpha=0.28)
    ax.set_xlim(0, max(p[1] for p in points) + 14)
    ax.set_ylim(0, max(p[2] for p in points) + 1.4)
    save_current("fig_mbs_tradeoff.png")


def plot_seedwise_delta_ci() -> None:
    stats = lower_statistics()
    paired = stats.get("paired_vs_reference", {})
    policies = [policy for policy in POLICIES[1:] if policy in paired]
    metrics = [
        ("latency", "时延差值"),
        ("deadline_satisfaction_rate", "DSR差值（百分点）"),
        ("mbs_load_ratio", "MBS负载差值（百分点）"),
    ]
    transforms = {
        "latency": lambda x: x,
        "deadline_satisfaction_rate": percent,
        "mbs_load_ratio": percent,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    for ax, (metric, title) in zip(axes, metrics):
        y = np.arange(len(policies))
        means = []
        lower = []
        upper = []
        for policy in policies:
            stat = paired[policy][metric]
            transform = transforms[metric]
            mean = transform(float(stat["mean"]))
            ci_low = transform(float(stat["ci_low"]))
            ci_high = transform(float(stat["ci_high"]))
            means.append(mean)
            lower.append(mean - ci_low)
            upper.append(ci_high - mean)
        colors = [POLICY_COLORS[policy] for policy in policies]
        ax.axvline(0, color="#6b7280", linewidth=1)
        ax.barh(y, means, xerr=[lower, upper], color=colors, alpha=0.9, capsize=4)
        ax.set_yticks(y, [POLICY_LABELS[policy] for policy in policies])
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("相对启发式策略的成对随机种子差值（均值与95%置信区间）", fontsize=14, fontweight="bold")
    save_current("fig_seedwise_delta_ci.png")


def plot_mbs_dsr_confidence_tradeoff() -> None:
    stats = lower_statistics()
    paired = stats.get("paired_vs_reference", {})
    policies = [policy for policy in POLICIES[1:] if policy in paired]
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.axhline(0, color="#9ca3af", linewidth=1)
    ax.axvline(0, color="#9ca3af", linewidth=1)
    for policy in policies:
        mbs = paired[policy]["mbs_load_ratio"]
        dsr = paired[policy]["deadline_satisfaction_rate"]
        x = -percent(float(mbs["mean"]))
        xerr = np.array([[abs(x - -percent(float(mbs["ci_high"])))], [abs(-percent(float(mbs["ci_low"])) - x)]])
        y = -percent(float(dsr["mean"]))
        yerr = np.array([[abs(y - -percent(float(dsr["ci_high"])))], [abs(-percent(float(dsr["ci_low"])) - y)]])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=9,
            linewidth=1.6,
            capsize=4,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
        ax.text(x + 0.35, y + 0.08, POLICY_LABELS[policy], fontsize=9)
    ax.set_xlabel("MBS 负载降低幅度（百分点，95% CI）")
    ax.set_ylabel("截止期满足率损失（百分点，95% CI）")
    ax.set_title("MBS 减负与服务质量损失的置信区间", fontweight="bold")
    ax.grid(alpha=0.28)
    ax.legend(loc="best")
    save_current("fig_mbs_dsr_confidence_tradeoff.png")


def is_non_dominated(points: list[dict[str, float]]) -> list[bool]:
    flags = []
    for idx, point in enumerate(points):
        dominated = False
        for other_idx, other in enumerate(points):
            if idx == other_idx:
                continue
            no_worse = other["mbs_load"] <= point["mbs_load"] and other["dsr"] >= point["dsr"]
            strictly_better = other["mbs_load"] < point["mbs_load"] or other["dsr"] > point["dsr"]
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def plot_pareto_tradeoff_scatter() -> None:
    summary = runtime_summary()
    points: list[dict[str, float | str]] = []
    for scenario, scenario_details in summary["per_scenario"].items():
        for policy, policy_details in scenario_details.items():
            aggregate = policy_details["aggregate"]
            points.append(
                {
                    "scenario": str(scenario),
                    "policy": str(policy),
                    "mbs_load": percent(float(aggregate["mbs_load_ratio"]["mean"])),
                    "mbs_ratio": percent(float(aggregate["offloading_ratio_mbs"]["mean"])),
                    "dsr": percent(float(aggregate["deadline_satisfaction_rate"]["mean"])),
                    "latency": float(aggregate["latency"]["mean"]),
                    "energy": float(aggregate["energy"]["mean"]),
                }
            )
    flags = is_non_dominated(points)  # minimize MBS load, maximize DSR

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    for policy in POLICIES:
        xs = [float(p["mbs_load"]) for p in points if p["policy"] == policy]
        ys = [float(p["dsr"]) for p in points if p["policy"] == policy]
        sizes = [70 + max(float(p["latency"]) / 25.0, 0.0) for p in points if p["policy"] == policy]
        ax.scatter(xs, ys, s=sizes, color=POLICY_COLORS[policy], alpha=0.70, edgecolor="#111827", linewidth=0.6, label=POLICY_LABELS[policy])

    frontier = [p for p, flag in zip(points, flags) if flag]
    frontier = sorted(frontier, key=lambda p: float(p["mbs_load"]))
    if frontier:
        ax.plot([float(p["mbs_load"]) for p in frontier], [float(p["dsr"]) for p in frontier], color="#111827", linestyle="--", linewidth=1.2, label="非支配边界")
        for p in frontier[:6]:
            ax.text(float(p["mbs_load"]) + 0.25, float(p["dsr"]) + 0.25, POLICY_LABELS[str(p["policy"])], fontsize=8)

    ax.set_xlabel("MBS 负载比例（%）")
    ax.set_ylabel("截止期满足率 DSR（%）")
    ax.set_title("MBS 依赖与服务成功率的 Pareto-style 权衡图", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    save_current("fig_pareto_tradeoff_scatter.png")


def plot_training_convergence() -> None:
    metrics = [
        ("reward", "Reward"),
        ("latency", "Latency"),
        ("energy", "Energy"),
        ("deadline_satisfaction_rate", "DSR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    axes = axes.ravel()
    for ax, (field, title) in zip(axes, metrics):
        for name, path in UPPER_TRAIN_LOGS.items():
            if not path.exists():
                continue
            rows = load_json(path)
            episodes = np.asarray([float(row["episode"]) for row in rows], dtype=float)
            values = np.asarray([float(row[field]) for row in rows], dtype=float)
            if field == "deadline_satisfaction_rate":
                values = values * 100.0
            smoothed = rolling_mean(values, window=15)
            ax.plot(episodes, smoothed, linewidth=1.8, label=name)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
    axes[3].set_ylabel("Percent" if metrics[3][0] == "deadline_satisfaction_rate" else "")
    axes[0].legend(loc="best")
    fig.suptitle("上层训练收敛曲线（单次正式训练，15-episode rolling mean）", fontsize=14, fontweight="bold")
    save_current("fig_training_convergence.png")


def plot_paired_seed_slope() -> None:
    summary = load_json(LOWER_SUMMARY)
    policy_units = extract_policy_units(summary)
    reference = "heuristic_offloading"
    candidate = "surrogate_baseline"
    metrics = [
        ("mbs_load_ratio", "MBS负载比例（%）", percent, "lower"),
        ("offloading_ratio_mbs", "MBS卸载比例（%）", percent, "lower"),
        ("deadline_satisfaction_rate", "DSR（%）", percent, "higher"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8))
    for ax, (metric, title, transform, direction) in zip(axes, metrics):
        ref_values = [transform(unit[metric]) for unit in policy_units[reference]]
        cand_values = [transform(unit[metric]) for unit in policy_units[candidate]]
        for idx, (ref, cand) in enumerate(zip(ref_values, cand_values, strict=False)):
            color = "#16a34a" if ((cand < ref) if direction == "lower" else (cand > ref)) else "#dc2626"
            ax.plot([0, 1], [ref, cand], color=color, alpha=0.75, linewidth=1.4)
            ax.scatter([0, 1], [ref, cand], color=color, s=28)
            ax.text(1.03, cand, f"s{idx + 1}", fontsize=7, va="center")
        ax.set_xlim(-0.15, 1.28)
        ax.set_xticks([0, 1], ["启发式", "Oracle引导"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("同随机种子配对改进图", fontsize=14, fontweight="bold")
    save_current("fig_paired_seed_slope.png")


def plot_request_destination_stack() -> None:
    stats = lower_statistics()
    metric_names = [
        ("offloading_ratio_local", "本地 UAV", "#2563eb"),
        ("offloading_ratio_cooperative", "协作 UAV", "#16a34a"),
        ("offloading_ratio_mbs", "MBS", "#dc2626"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    y = np.arange(len(POLICIES))
    left = np.zeros(len(POLICIES), dtype=float)
    for metric, label, color in metric_names:
        values = [percent(float(stats["metrics"][policy][metric]["mean"])) for policy in POLICIES]
        ax.barh(y, values, left=left, color=color, label=label, alpha=0.88)
        for idx, value in enumerate(values):
            if value >= 5:
                ax.text(left[idx] + value / 2, idx, f"{value:.1f}%", ha="center", va="center", fontsize=8, color="white")
        left += np.asarray(values)
    ax.set_yticks(y, [POLICY_LABELS[p] for p in POLICIES])
    ax.set_xlim(0, 100)
    ax.set_xlabel("服务请求去向占比（%）")
    ax.set_title("请求去向结构图（local / cooperative / MBS）", fontweight="bold")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.20))
    ax.grid(axis="x", alpha=0.20)
    save_current("fig_request_destination_stack.png")


def classifier_quality() -> dict:
    if CLASSIFIER_QUALITY_SUMMARY.exists():
        return load_json(CLASSIFIER_QUALITY_SUMMARY)
    return analyze_classifier_quality(
        dataset_path=OFFLOAD_DATASET,
        checkpoint_path=OFFLOAD_SURROGATE_CHECKPOINT,
        output_json=CLASSIFIER_QUALITY_SUMMARY,
        split="validation",
        val_ratio=0.2,
        seed=42,
        num_bins=10,
        device="cpu",
    )


def plot_offload_classifier_quality() -> None:
    quality = classifier_quality()
    matrix = np.asarray(quality["confusion_matrix"], dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    bins = quality["calibration_bins"]
    class_names = quality["class_names"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    ax = axes[0]
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)), class_names)
    ax.set_yticks(np.arange(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Oracle label")
    ax.set_title("归一化混淆矩阵")
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized[i, j] * 100:.1f}%\n({int(matrix[i, j])})", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    confidences = [float(item["confidence"]) for item in bins if int(item["count"]) > 0]
    accuracies = [float(item["accuracy"]) for item in bins if int(item["count"]) > 0]
    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1.2)
    ax.plot(confidences, accuracies, marker="o", color="#2563eb", linewidth=1.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration curve")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.92,
        f"Acc={quality['accuracy']:.3f}\nMacro-F1={quality['macro_f1']:.3f}\nECE={quality['ece']:.3f}\nBrier={quality['brier']:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    fig.suptitle("下层 oracle-guided 卸载器分类质量", fontsize=14, fontweight="bold")
    save_current("fig_offload_classifier_quality.png")


def scenario_metric(summary: dict, scenario: str, policy: str, field: str, transform=lambda x: x) -> tuple[float, float]:
    stat = summary["per_scenario"][scenario][policy]["aggregate"][field]
    return transform(float(stat["mean"])), abs(transform(float(stat["std"])))


def plot_scenario_sensitivity() -> None:
    summary = runtime_summary()
    scenarios = list(summary["per_scenario"].keys())
    x = np.arange(len(scenarios))
    width = 0.23

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
    panels = [
        ("deadline_satisfaction_rate", "截止期满足率（%）", percent),
        ("mbs_load_ratio", "MBS 负载比例（%）", percent),
        ("offloading_ratio_cooperative", "协作卸载比例（%）", percent),
    ]

    for ax, (field, ylabel, transform) in zip(axes, panels):
        for offset, policy in zip([-width, 0.0, width], POLICIES):
            means, stds = zip(*(scenario_metric(summary, s, policy, field, transform) for s in scenarios))
            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy],
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_title("下层卸载策略多场景敏感性（均值 ± 标准差）", fontsize=14, fontweight="bold")
    axes[0].legend(ncol=3, loc="best")
    axes[-1].set_xticks(x, [SCENARIO_LABELS.get(s, s) for s in scenarios], rotation=12, ha="right")
    axes[-1].set_xlabel("场景压力类型")
    save_current("fig_scenario_sensitivity.png")


def representative_scene(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hotspots = np.array([[190.0, 195.0], [515.0, 505.0]], dtype=float)
    ue_points = []
    hotspot_count = int(config.NUM_UES * config.HOTSPOT_UE_PROB)
    for idx in range(config.NUM_UES):
        if idx < hotspot_count:
            center = hotspots[idx % len(hotspots)]
            radius = config.HOTSPOT_RADIUS * np.sqrt(rng.uniform(0.0, 1.0))
            angle = rng.uniform(0.0, 2.0 * np.pi)
            point = center + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            point = rng.uniform([0.0, 0.0], [config.AREA_WIDTH, config.AREA_HEIGHT])
        ue_points.append(np.clip(point, [0.0, 0.0], [config.AREA_WIDTH, config.AREA_HEIGHT]))

    start = np.array(
        [
            [110.0, 565.0],
            [245.0, 475.0],
            [350.0, 350.0],
            [455.0, 225.0],
            [590.0, 135.0],
        ],
        dtype=float,
    )[: config.NUM_UAVS]
    targets = np.array(
        [
            [190.0, 225.0],
            [315.0, 330.0],
            [420.0, 385.0],
            [515.0, 505.0],
            [560.0, 370.0],
        ],
        dtype=float,
    )[: config.NUM_UAVS]
    return hotspots, np.asarray(ue_points), start, targets


def plot_trajectory_coverage() -> None:
    hotspots, ue_points, start, targets = representative_scene()
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal")
    ax.grid(alpha=0.22, linestyle="--")

    ax.scatter(ue_points[:, 0], ue_points[:, 1], s=14, color="#111827", alpha=0.32, label="UE")
    for idx, hotspot in enumerate(hotspots, 1):
        ax.add_patch(Circle(hotspot, config.HOTSPOT_RADIUS, edgecolor="#f97316", facecolor="#fed7aa", alpha=0.22, linewidth=1.5))
        ax.text(hotspot[0], hotspot[1] + config.HOTSPOT_RADIUS + 12, f"热点{idx}", ha="center", fontsize=9, color="#9a3412")

    mbs = np.asarray(config.MBS_POS[:2], dtype=float)
    ax.scatter(mbs[0], mbs[1], marker="s", s=180, color="#dc2626", edgecolor="#111827", label="MBS", zorder=5)
    ax.text(mbs[0] + 12, mbs[1] + 12, "MBS", fontsize=10, fontweight="bold", color="#991b1b")

    colors = ["#2563eb", "#16a34a", "#7c3aed", "#f97316", "#0891b2"]
    for idx, (s, t) in enumerate(zip(start, targets), 1):
        color = colors[(idx - 1) % len(colors)]
        control = np.array([(s[0] + t[0]) / 2, (s[1] + t[1]) / 2 + 55 * ((-1) ** idx)])
        curve_t = np.linspace(0, 1, 60)
        curve = (1 - curve_t)[:, None] ** 2 * s + 2 * (1 - curve_t)[:, None] * curve_t[:, None] * control + curve_t[:, None] ** 2 * t
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.0, alpha=0.9)
        ax.scatter(s[0], s[1], marker="^", s=85, color=color, edgecolor="#111827", zorder=6)
        ax.scatter(t[0], t[1], marker="o", s=95, color=color, edgecolor="#111827", zorder=6)
        ax.add_patch(Circle(t, config.UAV_COVERAGE_RADIUS, edgecolor=color, facecolor=color, alpha=0.10, linewidth=1.5))
        ax.text(t[0] + 8, t[1] + 8, f"UAV{idx}", fontsize=8, color=color, fontweight="bold")

    draw_arrow(ax, tuple(targets[1]), tuple(targets[2]), color="#16a34a", connectionstyle="arc3,rad=-0.12")
    draw_arrow(ax, tuple(targets[3]), tuple(mbs), color="#facc15", connectionstyle="arc3,rad=0.18")
    ax.text(250, 355, "协作链路", color="#15803d", fontsize=9)
    ax.text(455, 425, "MBS卸载", color="#a16207", fontsize=9)

    ax.scatter([], [], marker="^", s=85, color="#6b7280", label="UAV起点")
    ax.scatter([], [], marker="o", s=85, color="#6b7280", label="UAV终点/覆盖")
    ax.set_xlabel("X 位置（m）")
    ax.set_ylabel("Y 位置（m）")
    ax.set_title("多 UAV 轨迹与覆盖演化示意（由实验配置参数生成）", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.92)
    save_current("fig_trajectory_coverage.png")


def latest_spatial_trace_path() -> Path | None:
    trace_root = ROOT / "results" / "joint_experiments"
    if not trace_root.exists():
        return None
    candidates = sorted(trace_root.glob("*/spatial_traces/*_spatial_trace.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def plot_uav_hotspot_fallback_heatmap() -> None:
    trace_path = latest_spatial_trace_path()
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal")
    ax.grid(alpha=0.20, linestyle="--")

    if trace_path is not None:
        trace = load_json(trace_path)
        ue_points: list[list[float]] = []
        uav_tracks: dict[int, list[list[float]]] = {}
        fallback_points: list[list[float]] = []
        for record in trace.get("records", [])[:4]:
            for step in record.get("steps", []):
                ue_points.extend(step.get("ue_positions", []))
                for idx, pos in enumerate(step.get("uav_positions", [])):
                    uav_tracks.setdefault(idx, []).append(pos)
                metrics = step.get("metrics", {})
                if float(metrics.get("offloading_ratio_mbs", 0.0)) > 0.0:
                    fallback_points.extend(step.get("ue_positions", [])[:: max(len(step.get("ue_positions", [])) // 12, 1)])
        if ue_points:
            ue_arr = np.asarray(ue_points, dtype=float)
            ax.hist2d(ue_arr[:, 0], ue_arr[:, 1], bins=36, range=[[0, config.AREA_WIDTH], [0, config.AREA_HEIGHT]], cmap="YlOrRd", alpha=0.68)
        colors = ["#2563eb", "#16a34a", "#7c3aed", "#f97316", "#0891b2"]
        for idx, track in uav_tracks.items():
            arr = np.asarray(track, dtype=float)
            if arr.size:
                ax.plot(arr[:, 0], arr[:, 1], color=colors[idx % len(colors)], linewidth=1.8, label=f"UAV{idx + 1}")
                ax.scatter(arr[-1, 0], arr[-1, 1], color=colors[idx % len(colors)], s=50, edgecolor="#111827")
        if fallback_points:
            fb = np.asarray(fallback_points, dtype=float)
            ax.scatter(fb[:, 0], fb[:, 1], marker="x", color="#dc2626", s=38, label="MBS fallback sampled")
        subtitle = f"真实空间 trace: {trace_path.parent.parent.name}"
    else:
        hotspots, ue_points, start, targets = representative_scene(seed=7)
        ax.hist2d(ue_points[:, 0], ue_points[:, 1], bins=32, range=[[0, config.AREA_WIDTH], [0, config.AREA_HEIGHT]], cmap="YlOrRd", alpha=0.68)
        colors = ["#2563eb", "#16a34a", "#7c3aed", "#f97316", "#0891b2"]
        for idx, (s, t) in enumerate(zip(start, targets), 1):
            ax.plot([s[0], t[0]], [s[1], t[1]], color=colors[(idx - 1) % len(colors)], linewidth=1.8, label=f"UAV{idx}")
            ax.add_patch(Circle(t, config.UAV_COVERAGE_RADIUS, edgecolor=colors[(idx - 1) % len(colors)], facecolor="none", linewidth=1.2, alpha=0.75))
        fallback = np.asarray([[640, 610], [615, 560], [80, 90], [110, 70]], dtype=float)
        ax.scatter(fallback[:, 0], fallback[:, 1], marker="x", color="#dc2626", s=48, label="MBS fallback示意")
        subtitle = "示意图: 尚未发现真实 spatial_trace 文件"

    mbs = np.asarray(config.MBS_POS[:2], dtype=float)
    ax.scatter(mbs[0], mbs[1], marker="s", s=160, color="#991b1b", edgecolor="#111827", label="MBS")
    ax.set_xlabel("X 位置（m）")
    ax.set_ylabel("Y 位置（m）")
    ax.set_title(f"UAV 轨迹与热点覆盖热图\n{subtitle}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    save_current("fig_uav_hotspot_fallback_heatmap.png")


def load_mbs_penalty_sensitivity_summary() -> dict | None:
    summary_path = ROOT / "results" / "full_offload_experiments" / "mbs_penalty_sensitivity_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = load_json(summary_path)
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("runs"):
        return None
    return data


def plot_hyperparameter_sensitivity_template() -> None:
    scan_summary = load_mbs_penalty_sensitivity_summary()
    if scan_summary is not None:
        runs = sorted(scan_summary["runs"], key=lambda item: float(item["weight"]))
        weights = np.array([float(item["weight"]) for item in runs], dtype=float)

        panels = [
            ("deadline_satisfaction_rate", "截止期满足率（%）", percent),
            ("mbs_load_ratio", "MBS 负载比例（%）", percent),
            ("energy", "能耗", lambda x: x),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
        for ax, (metric, ylabel, transform) in zip(axes, panels):
            for policy, marker in [
                ("surrogate_baseline", "o"),
                ("rich_reduced_runtime_policy", "s"),
                ("heuristic_offloading", "^"),
            ]:
                values = [
                    transform(float(item.get(policy, {}).get(metric, np.nan)))
                    for item in runs
                ]
                ax.plot(
                    weights,
                    values,
                    marker=marker,
                    linewidth=1.9,
                    color=POLICY_COLORS.get(policy, "#6b7280"),
                    label=POLICY_LABELS.get(policy, policy),
                )
            ax.set_xlabel("MBS load penalty weight")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        axes[0].set_title("MBS penalty 敏感性扫描", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=8)
        fig.text(
            0.5,
            0.01,
            "其他 oracle 权重保持默认；该图用于附录说明 MBS 减负权重变化下的服务质量与负载权衡。",
            ha="center",
            fontsize=9,
            color="#475569",
        )
        save_current("fig_hyperparameter_sensitivity_template.png")
        return

    mbs_weights = np.array([0.00, 0.03, 0.065, 0.10, 0.14])
    queue_weights = np.array([0.00, 0.04, 0.08, 0.12])
    grid = np.full((queue_weights.size, mbs_weights.size), np.nan)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.imshow(np.zeros_like(grid), cmap="Greys", vmin=0, vmax=1, alpha=0.12)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="#cbd5e1", linewidth=1))
            ax.text(x, y, "待扫描", ha="center", va="center", fontsize=9, color="#475569")
    ax.set_xticks(np.arange(mbs_weights.size), [f"{v:.3f}" for v in mbs_weights])
    ax.set_yticks(np.arange(queue_weights.size), [f"{v:.2f}" for v in queue_weights])
    ax.set_xlabel("MBS load penalty weight")
    ax.set_ylabel("Queue pressure weight")
    ax.set_title("超参数/权重敏感性热图模板（需要额外网格扫描结果）", fontweight="bold")
    ax.text(
        0.5,
        -0.25,
        "该图为占位模板，不作为实验结论；填充方式: 每个格运行同 seed 评估并写入 DSR / MBS load / energy 均值与CI。",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#475569",
    )
    save_current("fig_hyperparameter_sensitivity_template.png")


def write_figure_catalog() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        ("fig_framework.png", "端到端系统流程图", "主文强烈建议保留", "双层协同优化框架：上层负责 UAV 轨迹与协同控制，下层负责请求级卸载决策。"),
        ("fig_pareto_tradeoff_scatter.png", "Pareto-style 权衡图", "主文建议保留", "不同策略与场景在 MBS 依赖与服务成功率之间形成权衡边界。"),
        ("fig_training_convergence.png", "训练收敛曲线", "主文或附录", "上层控制器训练过程的 reward、latency、energy 和 DSR rolling mean。"),
        ("fig_paired_seed_slope.png", "配对种子改进图", "主文建议保留", "同一随机种子下，oracle-guided 策略相对启发式的指标变化。"),
        ("fig_uav_hotspot_fallback_heatmap.png", "UAV 轨迹与热点覆盖热图", "当前为 trace/示意混合，谨慎使用", "空间 trace 或配置示意展示 UAV 轨迹、UE 热点与 MBS fallback 位置。"),
        ("fig_request_destination_stack.png", "请求去向结构图", "主文建议保留", "下层卸载策略改变 local、cooperative UAV 与 MBS 的请求流向占比。"),
        ("fig_offload_classifier_quality.png", "下层分类质量图", "主文或附录强烈建议", "oracle-guided 卸载器与 oracle 标签的一致性及概率校准表现。"),
        ("fig_hyperparameter_sensitivity_template.png", "超参数敏感性图", "附录", "若存在 MBS penalty 扫描结果则展示真实敏感性；否则生成待扫描模板。"),
    ]
    lines = [
        "# Candidate Figure Catalog",
        "",
        "| File | Type | Recommendation | Caption suggestion |",
        "| --- | --- | --- | --- |",
    ]
    for file_name, figure_type, recommendation, caption in rows:
        lines.append(f"| `{file_name}` | {figure_type} | {recommendation} | {caption} |")
    (FIG_DIR / "FIGURE_CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    plot_framework()
    plot_trajectory_coverage()
    plot_uav_hotspot_fallback_heatmap()
    plot_upper_comparison()
    plot_training_convergence()
    plot_lower_runtime()
    plot_scenario_sensitivity()
    plot_pareto_tradeoff_scatter()
    plot_mbs_tradeoff()
    plot_seedwise_delta_ci()
    plot_paired_seed_slope()
    plot_mbs_dsr_confidence_tradeoff()
    plot_request_destination_stack()
    plot_offload_classifier_quality()
    plot_hyperparameter_sensitivity_template()
    write_figure_catalog()
    print(f"论文图表已生成到：{FIG_DIR}")


if __name__ == "__main__":
    main()
