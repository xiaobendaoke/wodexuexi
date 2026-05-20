#!/usr/bin/env python3
"""Generate Chinese, paper-style README figures for the UAV-MEC draft."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "docs" / "figures"
MAIN_STATS = ROOT / "results" / "joint_experiments" / "paper_revised_full_20260518_main_merged" / "statistics.json"
ABLATION_STATS = ROOT / "results" / "joint_experiments" / "paper_revised_full_20260518_ablation_merged" / "statistics.json"
MAIN_WORKLOAD42 = ROOT / "results" / "joint_experiments" / "paper_revised_full_20260518_main_workload42"
TRACE_CANDIDATES = [
    ROOT / "results" / "joint_experiments" / "readme_spatial_trace_seed42" / "spatial_traces" / "uncoordinated_greedy__heuristic_spatial_trace.json",
    ROOT / "results" / "joint_experiments" / "readme_spatial_trace_seed42" / "spatial_traces" / "full_hierarchical_marl_spatial_trace.json",
    ROOT / "results" / "joint_experiments" / "readme_spatial_trace_seed42" / "spatial_trace.json",
]

METHOD_LABELS = {
    "uncoordinated_greedy__heuristic": "非协同+启发式",
    "attention_mappo__heuristic": "轨迹MAPPO+启发式",
    "uncoordinated_greedy__lower_mappo": "非协同+下层MAPPO",
    "attention_mappo__lower_mappo": "轨迹MAPPO+下层MAPPO",
    "full_hierarchical_marl": "完整双层MARL",
}

SHORT_LABELS = {
    "uncoordinated_greedy__heuristic": "基线",
    "attention_mappo__heuristic": "仅上层",
    "uncoordinated_greedy__lower_mappo": "仅下层",
    "attention_mappo__lower_mappo": "上下层",
    "full_hierarchical_marl": "完整双层",
}

ABLATION_LABELS = {
    "lower_full": "完整下层",
    "lower_no_mask": "无Mask",
    "lower_no_lagrange": "无Lagrange",
    "lower_no_attention": "无Attention",
}

COLORS = {
    "blue": "#3867D6",
    "orange": "#E58A2A",
    "green": "#3A8F5B",
    "red": "#C94C4C",
    "purple": "#7B61A8",
    "teal": "#288B8B",
    "gray": "#68707A",
    "dark": "#20242A",
    "light": "#F6F7F9",
}

SERIES_COLORS = [COLORS["gray"], COLORS["purple"], COLORS["orange"], COLORS["green"], COLORS["blue"]]


def available_font_names() -> set[str]:
    return {f.name for f in fm.fontManager.ttflist}


def setup_style() -> None:
    preferred = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    installed = available_font_names()
    family = next((name for name in preferred if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": [family] + preferred,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(stats: dict[str, Any], method: str, name: str, field: str = "mean") -> float:
    return float(stats["metrics"][method][name][field])


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#586069", lw: float = 1.4) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color, shrinkA=5, shrinkB=5))


def box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, color: str, fontsize: int = 10) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.015",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color + "12",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["dark"], linespacing=1.25)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=11, weight="bold")


def fig_system_model() -> None:
    rng = np.random.default_rng(42)
    hotspots = np.array([[160, 540], [530, 540], [170, 170], [530, 170], [350, 350]], dtype=float)
    ue_pos = np.vstack([np.clip(rng.normal(c, [55, 45], size=(20, 2)), 20, 680) for c in hotspots])
    uav_pos = np.array([[150, 520], [520, 540], [185, 185], [520, 160], [350, 355]], dtype=float)
    mbs = np.array([350, 350], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 700)
    ax.set_aspect("equal")
    ax.set_title("系统场景与链路模型")
    ax.set_xlabel("x 坐标 (m)")
    ax.set_ylabel("y 坐标 (m)")
    ax.scatter(ue_pos[:, 0], ue_pos[:, 1], s=13, color="#7A828C", alpha=0.62, label="用户设备 UE")
    for i, pos in enumerate(uav_pos):
        ax.add_patch(Circle(pos, 100, fill=False, color=COLORS["blue"], linewidth=1.0, linestyle="--", alpha=0.8))
        ax.scatter(pos[0], pos[1], marker="^", s=125, color=COLORS["blue"], edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(pos[0] + 8, pos[1] + 8, f"UAV{i}", fontsize=9, weight="bold")
    ax.scatter(mbs[0], mbs[1], marker="s", s=160, color=COLORS["red"], edgecolor="white", linewidth=0.8, label="宏基站 MBS", zorder=5)
    ax.text(mbs[0] + 10, mbs[1] - 10, "MBS", fontsize=10, weight="bold")
    for pos in uav_pos:
        arrow(ax, tuple(pos), tuple(mbs), COLORS["red"], 1.0)
    for a, b in [(0, 4), (1, 4), (2, 4), (3, 4)]:
        ax.plot([uav_pos[a, 0], uav_pos[b, 0]], [uav_pos[a, 1], uav_pos[b, 1]], color=COLORS["teal"], linewidth=1.4, alpha=0.75)
    ax.text(18, 675, "UE-UAV接入 / UAV-UAV协作 / UAV-MBS回传", fontsize=9, color=COLORS["dark"], bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D0D7DE", "lw": 0.8})
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.11))
    save(fig, "fig_system_model")


def fig_latency_process() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    ax.axis("off")
    lanes = [
        ("本地执行", ["UE上传\n8D/R_ue-uav", "缓存未命中\n取回服务/文件", "UAV计算\nC/f_uav"], COLORS["blue"]),
        ("协作执行", ["UE上传\n8D/R_ue-uav", "UAV间转发\n8D/R_uav-uav", "缓存未命中\n协作取回", "协作UAV计算\nC/f_coop"], COLORS["green"]),
        ("MBS执行", ["UE上传\n8D/R_ue-uav", "回传链路\n8D/R_backhaul", "MBS计算\nC/f_mbs"], COLORS["red"]),
    ]
    for r, (name, steps, col) in enumerate(lanes):
        y = 0.74 - r * 0.28
        ax.text(0.035, y + 0.06, name, ha="left", va="center", fontsize=11, weight="bold", color=col)
        xs = np.linspace(0.18, 0.82, len(steps))
        for i, (txt, x) in enumerate(zip(steps, xs)):
            box(ax, (x, y), (0.14, 0.12), txt, col, 9)
            if i < len(steps) - 1:
                arrow(ax, (x + 0.14, y + 0.06), (xs[i + 1], y + 0.06), COLORS["gray"], 1.2)
    ax.text(0.5, 0.055, "传输时延统一按 bytes 到 bit/s 换算：T_tx = 8D / R", ha="center", fontsize=10, weight="bold")
    ax.set_title("时延处理流程", y=0.96)
    save(fig, "fig_latency_process")


def fig_system_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    ax.axis("off")
    items = [
        ("状态生成\nUAV/UE/请求", COLORS["blue"]),
        ("下层卸载\n请求级动作", COLORS["green"]),
        ("服务执行\n时延/能耗/DSR", COLORS["orange"]),
        ("上层轨迹\n连续控制", COLORS["purple"]),
        ("位置更新\n进入下一时隙", COLORS["teal"]),
    ]
    xs = np.linspace(0.055, 0.81, len(items))
    for i, ((txt, col), x) in enumerate(zip(items, xs)):
        box(ax, (x, 0.45), (0.14, 0.25), txt, col, 9)
        if i < len(items) - 1:
            arrow(ax, (x + 0.14, 0.575), (xs[i + 1], 0.575), COLORS["gray"], 1.3)
    arrow(ax, (0.88, 0.43), (0.055, 0.43), COLORS["gray"], 1.2)
    ax.text(0.47, 0.22, "每步统计：奖励、能耗、截止期满足率 DSR、MBS负载、公平性、离线率", ha="center", fontsize=9)
    ax.set_title("单时隙系统工作流程", y=0.93)
    save(fig, "fig_system_workflow")


def fig_hmarl_framework() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.axis("off")
    box(ax, (0.05, 0.60), (0.22, 0.20), "上层轨迹控制\nAttention-MAPPO\n连续二维动作", COLORS["purple"], 9)
    box(ax, (0.05, 0.22), (0.22, 0.20), "下层请求卸载\n约束式MAPPO\n离散卸载动作", COLORS["green"], 9)
    box(ax, (0.36, 0.63), (0.18, 0.14), "UAV观测\n位置/邻居/覆盖/负载", COLORS["blue"], 9)
    box(ax, (0.36, 0.25), (0.18, 0.14), "请求观测\nattention + mask\nLagrange惩罚", COLORS["orange"], 9)
    box(ax, (0.67, 0.54), (0.22, 0.20), "共享MEC环境\n链路/缓存/电池\nMBS负载", COLORS["teal"], 9)
    box(ax, (0.67, 0.20), (0.22, 0.15), "指标与奖励\nDSR/能耗/公平性\n离线率", COLORS["red"], 9)
    for p1, p2, col in [
        ((0.27, 0.70), (0.36, 0.70), COLORS["purple"]),
        ((0.54, 0.70), (0.67, 0.64), COLORS["purple"]),
        ((0.27, 0.32), (0.36, 0.32), COLORS["green"]),
        ((0.54, 0.32), (0.67, 0.28), COLORS["green"]),
        ((0.78, 0.54), (0.78, 0.35), COLORS["gray"]),
        ((0.67, 0.26), (0.27, 0.26), COLORS["gray"]),
    ]:
        arrow(ax, p1, p2, col, 1.3)
    ax.text(0.5, 0.08, "训练采用 CTDE；下层动作：0=本地，1=MBS，2+i=协作UAV i，当前 |A|=7", ha="center", fontsize=9)
    ax.set_title("质量感知约束式双层 MARL 框架", y=0.96)
    save(fig, "fig_hmarl_framework")


def trace_files() -> list[Path]:
    files = [p for p in TRACE_CANDIDATES if p.exists()]
    files.extend(sorted((ROOT / "results" / "joint_experiments" / "readme_spatial_trace_seed42" / "spatial_traces").glob("*_spatial_trace.json")))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in files:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def load_spatial_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in trace_files():
        try:
            payload = load_json(path)
        except Exception:
            continue
        for rec in payload.get("records", []):
            combo = rec.get("combo") or payload.get("combo")
            if combo and combo not in records and rec.get("steps"):
                records[str(combo)] = rec
    return records


def plot_trace_panel(ax: plt.Axes, rec: dict[str, Any], title: str) -> None:
    steps = rec.get("steps", [])
    ue = np.array(steps[0]["ue_positions"], dtype=float)
    trajectories = np.array([step["uav_positions"] for step in steps], dtype=float)
    ax.scatter(ue[:, 0], ue[:, 1], s=10, color="#7A828C", alpha=0.38, label="UE")
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"], COLORS["purple"]]
    for i in range(trajectories.shape[1]):
        pts = trajectories[:, i, :]
        ax.plot(pts[:, 0], pts[:, 1], color=palette[i], linewidth=1.7, label=f"UAV{i}")
        ax.scatter(pts[0, 0], pts[0, 1], marker="o", s=32, color=palette[i], edgecolor="white", zorder=4)
        ax.scatter(pts[-1, 0], pts[-1, 1], marker="^", s=62, color=palette[i], edgecolor="white", zorder=5)
        ax.add_patch(Circle(pts[-1], 100, fill=False, linestyle="--", linewidth=0.8, color=palette[i], alpha=0.42))
    ax.scatter([350], [350], marker="s", s=72, color="#111827", label="MBS")
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 700)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x 坐标 (m)")
    ax.set_ylabel("y 坐标 (m)")


def fig_trajectory_before_after() -> bool:
    records = load_spatial_records()
    base = records.get("uncoordinated_greedy__heuristic")
    full = records.get("full_hierarchical_marl")
    if not base or not full:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), sharex=True, sharey=True)
    plot_trace_panel(axes[0], base, "基线：非协同+启发式")
    plot_trace_panel(axes[1], full, "完整双层 MARL")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("固定 seed 真实 rollout 的 UAV 轨迹对比", fontsize=13, weight="bold")
    save(fig, "fig_trajectory_before_after")
    return True


def fig_service_coverage_fairness() -> bool:
    records = load_spatial_records()
    base = records.get("uncoordinated_greedy__heuristic")
    full = records.get("full_hierarchical_marl")
    if not base or not full:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), sharex=True, sharey=True)
    sc = None
    for ax, rec, title in [(axes[0], base, "基线覆盖"), (axes[1], full, "完整双层MARL覆盖")]:
        steps = rec["steps"]
        ue = np.array(steps[0]["ue_positions"], dtype=float)
        coverage_values = []
        for step in steps:
            uavs = np.array(step["uav_positions"], dtype=float)
            dist = np.sqrt(((ue[:, None, :] - uavs[None, :, :]) ** 2).sum(axis=2))
            coverage_values.append((dist.min(axis=1) <= 100).astype(float))
        coverage = np.mean(np.vstack(coverage_values), axis=0)
        sc = ax.scatter(ue[:, 0], ue[:, 1], c=coverage, cmap="viridis", s=18, vmin=0, vmax=1)
        final_uavs = np.array(steps[-1]["uav_positions"], dtype=float)
        for pos in final_uavs:
            ax.add_patch(Circle(pos, 100, fill=False, linestyle="--", color="white", linewidth=1.0, alpha=0.85))
            ax.scatter(pos[0], pos[1], marker="^", s=68, color=COLORS["red"], edgecolor="white")
        ax.set_xlim(0, 700)
        ax.set_ylim(0, 700)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x 坐标 (m)")
        ax.set_ylabel("y 坐标 (m)")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.86, label="采样步覆盖比例")
    fig.suptitle("固定 seed 真实 rollout 的 UE 覆盖公平性", fontsize=13, weight="bold")
    save(fig, "fig_service_coverage_fairness")
    return True


def fig_action_mask_decision() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.axis("off")
    columns = ["本地", "MBS", "协作0", "协作1", "协作2", "协作3", "协作4"]
    valid = [True, False, True, False, True, False, True]
    x0, y0, w, h = 0.08, 0.50, 0.11, 0.18
    for i, (col, ok) in enumerate(zip(columns, valid)):
        x = x0 + i * (w + 0.015)
        color = COLORS["green"] if ok else COLORS["red"]
        ax.add_patch(Rectangle((x, y0), w, h, facecolor=color + "16", edgecolor=color, linewidth=1.2))
        ax.text(x + w / 2, y0 + h * 0.63, col, ha="center", va="center", fontsize=9)
        ax.text(x + w / 2, y0 + h * 0.25, "可选" if ok else "屏蔽", ha="center", va="center", fontsize=8, color=color, weight="bold")
    box(ax, (0.40, 0.18), (0.20, 0.15), "采样动作\n协作UAV2", COLORS["green"], 9)
    arrow(ax, (x0 + 4 * (w + 0.015) + w / 2, y0), (0.50, 0.33), COLORS["green"], 1.4)
    ax.text(0.5, 0.82, "请求级动作空间：|A| = 2 + N = 7（N=5）", ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.08, "统计 cooperative ratio 时，将所有 2+i 协作动作统一汇总为协作卸载。", ha="center", fontsize=9)
    save(fig, "fig_action_mask_decision")


def apply_axis_labels(ax: plt.Axes, values: Iterable[float]) -> None:
    vals = list(values)
    lo, hi = min(vals), max(vals)
    if lo >= 0:
        ax.set_ylim(0, hi * 1.18 if hi else 1.0)
    elif hi <= 0:
        ax.set_ylim(lo * 1.12, 0)
    else:
        span = hi - lo
        ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)


def fig_main_comparison(stats: dict[str, Any]) -> None:
    methods = list(METHOD_LABELS)
    metrics = [
        ("reward", "综合奖励", 1.0),
        ("energy", "能耗 (×10^6)", 1e-6),
        ("latency", "时延 (×10^6)", 1e-6),
        ("fairness", "公平性", 1.0),
        ("deadline_satisfaction_rate", "截止期满足率 DSR", 1.0),
        ("mbs_load_ratio", "MBS负载比例", 1.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    for ax, (m, title, scale) in zip(axes.flat, metrics):
        means = [metric(stats, method, m, "mean") * scale for method in methods]
        stds = [metric(stats, method, m, "std") * scale for method in methods]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, capsize=3, color=SERIES_COLORS, linewidth=0.6, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[v] for v in methods], rotation=0, ha="center")
        ax.set_title(title)
        apply_axis_labels(ax, [v + s for v, s in zip(means, stds)] + [v - s for v, s in zip(means, stds)] + [0])
    fig.suptitle("主实验多指标对比（mean ± std, N=30）", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig_main_comparison")


def fig_offloading_distribution(stats: dict[str, Any]) -> None:
    methods = list(METHOD_LABELS)
    local = np.array([metric(stats, m, "offloading_ratio_local") for m in methods])
    coop = np.array([metric(stats, m, "offloading_ratio_cooperative") for m in methods])
    mbs = np.array([metric(stats, m, "offloading_ratio_mbs") for m in methods])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(methods))
    ax.bar(x, local, label="本地执行", color=COLORS["blue"])
    ax.bar(x, coop, bottom=local, label="协作执行", color=COLORS["green"])
    ax.bar(x, mbs, bottom=local + coop, label="MBS执行", color=COLORS["red"])
    for i, v in enumerate(mbs):
        ax.text(i, min(0.98, local[i] + coop[i] + v + 0.015), f"MBS {v*100:.1f}%", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[v] for v in methods])
    ax.set_ylabel("卸载比例")
    ax.set_title("卸载分布对比")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig_offloading_distribution")


def fig_ablation_comparison(stats: dict[str, Any]) -> None:
    methods = list(ABLATION_LABELS)
    metrics = [
        ("deadline_satisfaction_rate", "DSR", 1.0),
        ("offloading_ratio_mbs", "MBS卸载比例", 1.0),
        ("offloading_ratio_cooperative", "协作卸载比例", 1.0),
        ("energy", "能耗 (×10^6)", 1e-6),
        ("mbs_load_ratio", "MBS负载比例", 1.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8))
    for ax, (m, title, scale) in zip(axes.flat[:5], metrics):
        means = [metric(stats, method, m, "mean") * scale for method in methods]
        stds = [metric(stats, method, m, "std") * scale for method in methods]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, capsize=3, color=[COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["purple"]], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_LABELS[v] for v in methods], rotation=15, ha="right")
        ax.set_title(title)
        apply_axis_labels(ax, [v + s for v, s in zip(means, stds)] + [v - s for v, s in zip(means, stds)] + [0])
    axes.flat[5].axis("off")
    axes.flat[5].text(0.5, 0.55, "消融目标：分析 mask、Lagrange、attention\n对卸载分布和能耗-负载权衡的影响", ha="center", va="center", fontsize=11)
    fig.suptitle("下层消融实验对比（mean ± std, N=30）", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig_ablation_comparison")


def find_log_dirs(root: Path, combo: str) -> list[Path]:
    return sorted(root.glob(f"test_logs/{combo}/*/log_data_*.json"))


def load_log_series(combo: str, root: Path = MAIN_WORKLOAD42) -> list[dict[str, Any]]:
    paths = find_log_dirs(root, combo)
    return load_json(paths[-1]) if paths else []


def moving_average(values: list[float], window: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def fig_lagrange_dsr_mbs_tradeoff() -> None:
    data = load_log_series("full_hierarchical_marl")
    keys = [("lambda_dsr", "λ_DSR"), ("lambda_mbs", "λ_MBS"), ("deadline_satisfaction_rate", "DSR"), ("mbs_load_ratio", "MBS负载")]
    if data and all(k in data[0] for k, _ in keys):
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.8))
        x = [d.get("episode", i + 1) for i, d in enumerate(data)]
        for ax, (k, title), col in zip(axes.flat, keys, [COLORS["purple"], COLORS["orange"], COLORS["green"], COLORS["red"]]):
            y = [float(d.get(k, 0.0)) for d in data]
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, color=col)
            ax.set_title(title)
            ax.set_xlabel("评估 episode")
        fig.suptitle("Lagrange 约束动态与 DSR-MBS 权衡（workload42 评估日志）", fontsize=13, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, "fig_lagrange_dsr_mbs_tradeoff")
        return

    stats = load_json(MAIN_STATS)
    methods = list(METHOD_LABELS)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for method, col in zip(methods, SERIES_COLORS):
        ax.scatter(metric(stats, method, "mbs_load_ratio"), metric(stats, method, "deadline_satisfaction_rate"), s=84, label=METHOD_LABELS[method], color=col)
    ax.set_xlabel("MBS负载比例")
    ax.set_ylabel("截止期满足率 DSR")
    ax.set_title("DSR-MBS 工作点")
    ax.legend(fontsize=8)
    save(fig, "fig_lagrange_dsr_mbs_tradeoff")


def fig_hierarchical_training_curves() -> None:
    combos = ["uncoordinated_greedy__heuristic", "attention_mappo__heuristic", "full_hierarchical_marl"]
    keys = [("reward", "综合奖励"), ("energy", "能耗 (×10^6)"), ("fairness", "公平性"), ("deadline_satisfaction_rate", "DSR")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for ax, (key, title) in zip(axes.flat, keys):
        for combo, col in zip(combos, [COLORS["gray"], COLORS["purple"], COLORS["blue"]]):
            data = load_log_series(combo)
            if not data or key not in data[0]:
                continue
            x = [d.get("episode", i + 1) for i, d in enumerate(data)]
            y = [float(d.get(key, 0.0)) for d in data]
            if key == "energy":
                y = [v / 1e6 for v in y]
            ax.plot(x, moving_average(y, 3), linewidth=1.9, label=METHOD_LABELS[combo], color=col)
        ax.set_title(title)
        ax.set_xlabel("评估 episode")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("固定 workload42 评估曲线", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_hierarchical_training_curves")


def generate_all() -> None:
    setup_style()
    main_stats = load_json(MAIN_STATS)
    ablation_stats = load_json(ABLATION_STATS)
    fig_system_model()
    fig_latency_process()
    fig_system_workflow()
    fig_hmarl_framework()
    have_traj = fig_trajectory_before_after()
    have_cov = fig_service_coverage_fairness()
    if not have_traj or not have_cov:
        print("WARN: spatial trace missing; trajectory/coverage figures were not regenerated.")
    fig_action_mask_decision()
    fig_main_comparison(main_stats)
    fig_offloading_distribution(main_stats)
    fig_ablation_comparison(ablation_stats)
    fig_lagrange_dsr_mbs_tradeoff()
    fig_hierarchical_training_curves()


if __name__ == "__main__":
    generate_all()
