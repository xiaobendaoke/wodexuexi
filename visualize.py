"""
visualize.py — Multi-UAV MEC System Visualization

Modes:
  python visualize.py --mode snapshot   Save single-frame PNG snapshot
  python visualize.py --mode animate    Save simulation animation GIF
  python visualize.py --mode results    Bar charts from experiment metrics JSON
"""

from __future__ import annotations

import argparse
import json
import io
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import config
from environment.env import Env
from environment.user_equipments import UE
from environment.uavs import UAV

# ── CJK font setup ──────────────────────────────────────────────────────────
_CJK_FONT = None
for _name in ("WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
               "Noto Sans CJK SC", "SimHei"):
    try:
        from matplotlib.font_manager import findfont
        if findfont(_name, fallback_to_default=False):
            _CJK_FONT = _name
            break
    except Exception:
        continue

if _CJK_FONT is not None:
    matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    matplotlib.rcParams["font.monospace"] = ["WenQuanYi Micro Hei Mono",
                                              "DejaVu Sans Mono"]
    matplotlib.rcParams["axes.unicode_minus"] = False

# ── Color palette ───────────────────────────────────────────────────────────
C_UAV_FREE    = "#378ADD"
C_UAV_BUSY    = "#1D9E75"
C_UE_NORMAL   = "#66C2A5"
C_UE_LOW      = "#FDC086"
C_UE_CRITICAL = "#E74C3C"
C_MBS         = "#8B0000"
C_COMM_UAV    = "#AFA9EC"
C_COMM_MBS    = "#B07C8C"
C_COVERAGE    = "#C8CBD0"
C_HOTSPOT     = "#FFF3CD"
C_BG          = "#F8F8F6"
C_SERVICE     = "#EF9F27"
C_CONTENT     = "#56B4E9"
C_ENERGY      = "#CC79A7"
C_ASSOC       = "#B8B8B8"


def _ensure_env_initialized() -> None:
    if not hasattr(UE, "global_probs") or UE.global_probs is None:
        UE.initialize_ue_class()


def _randomize_ue_batteries(env: Env, seed: int) -> None:
    """Visualization helper: spread UE batteries across [5, B_max] so the
    snapshot shows a richer mix of normal, low, and critical/energy-request UEs."""
    rng = np.random.default_rng(seed + 1)
    for ue in env.ues:
        ue.battery_level = float(rng.uniform(5, config.UE_BATTERY_CAPACITY))


# ── Snapshot ────────────────────────────────────────────────────────────────

def snapshot(
    out_path: str = "docs/figures/fig_system_snapshot.png",
    seed: int = 42,
    n_ticks: int = 500,
    warmup_ticks: int = 0,
    dpi: int = 200,
) -> None:
    """Render a top-down 2D snapshot of the Multi-UAV MEC system."""
    _ensure_env_initialized()
    env = Env()
    env.reset(initial_positions=None)
    _randomize_ue_batteries(env, seed)

    rng = np.random.default_rng(seed)

    # Warmup phase (silent, for battery drain / energy request emergence)
    for _ in range(warmup_ticks):
        actions = rng.uniform(-1, 1, (config.NUM_UAVS, 2))
        env.get_offloading_obs_and_masks()
        offload_actions = np.full(
            (config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV),
            config.OFFLOAD_ACTION_LOCAL,
            dtype=np.int64,
        )
        env.step(actions, offloading_actions=offload_actions)
    if warmup_ticks > 0:
        print(f"[visualize] Warmup complete: {warmup_ticks} ticks")

    for _ in range(n_ticks):
        actions = rng.uniform(-1, 1, (config.NUM_UAVS, 2))
        env.get_offloading_obs_and_masks()
        offload_actions = np.full(
            (config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV),
            config.OFFLOAD_ACTION_LOCAL,
            dtype=np.int64,
        )
        env.step(actions, offloading_actions=offload_actions)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"多无人机MEC系统快照  |  t={env._time_step}  |  "
        f"{config.NUM_UAVS}架无人机, {config.NUM_UES}个用户设备",
        fontsize=12, fontweight="bold", pad=12,
    )

    _draw_system_state(ax, env)

    # 图例 — 使用 Line2D + marker 确保形状与图中 scatter 一致
    legend_elements = [
        Line2D([0], [0], marker="o", color=C_UAV_FREE, linestyle="None",
               markersize=10, markeredgecolor="white", markeredgewidth=1.0,
               label="无人机(空闲)"),
        Line2D([0], [0], marker="o", color=C_UAV_BUSY, linestyle="None",
               markersize=10, markeredgecolor="white", markeredgewidth=1.0,
               label="无人机(服务中)"),
        Line2D([0], [0], marker="o", color=C_UE_NORMAL, linestyle="None",
               markersize=8, label="用户设备(电量正常)"),
        Line2D([0], [0], marker="o", color=C_UE_LOW, linestyle="None",
               markersize=8, label="用户设备(电量偏低)"),
        Line2D([0], [0], marker="$\mathbf{!}$", color=C_UE_CRITICAL,
               linestyle="None", markersize=10,
               label="用户设备(电量临界/offline风险)"),
        Line2D([0], [0], color=C_MBS, marker="s", linestyle="None",
               markersize=12, label="宏基站(MBS)"),
        Line2D([0], [0], color=C_SERVICE, marker="^", linestyle="None",
               markersize=8, label="服务请求"),
        Line2D([0], [0], color=C_CONTENT, marker="s", linestyle="None",
               markersize=8, label="内容请求"),
        Line2D([0], [0], color=C_ENERGY, marker="D", linestyle="None",
               markersize=8, label="能量请求(WPT)"),
        Line2D([0], [0], color=C_COMM_UAV, lw=0.8, ls="--",
               label="无人机间链路"),
        Line2D([0], [0], color=C_ASSOC, lw=0.25,
               label="用户-无人机关联"),
    ]
    if config.USE_HOTSPOTS:
        legend_elements.insert(-2, mpatches.Patch(color=C_HOTSPOT, alpha=0.35,
                                                   label="热点区域"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.85, ncol=2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"[visualize] Snapshot saved to {out_path}")


def _draw_system_state(ax: plt.Axes, env: Env) -> None:
    """Draw complete system state onto the given Axes."""
    area_w, area_h = config.AREA_WIDTH, config.AREA_HEIGHT

    # ── Hotspot regions ──
    if config.USE_HOTSPOTS and hasattr(UE, "hotspot_centers"):
        for center in UE.hotspot_centers:
            ax.add_patch(plt.Circle(
                (float(center[0]), float(center[1])),
                config.HOTSPOT_RADIUS,
                facecolor=C_HOTSPOT, edgecolor="#E0C060",
                lw=1.0, alpha=0.35, zorder=0,
            ))

    # ── MBS ──
    mbs_x, mbs_y = float(config.MBS_POS[0]), float(config.MBS_POS[1])
    ax.scatter(mbs_x, mbs_y, s=300, marker="s", color=C_MBS, zorder=5,
               edgecolors="white", linewidths=2.0)
    ax.text(mbs_x, mbs_y + 20, "宏基站", fontsize=11, ha="center",
            fontweight="bold", color=C_MBS)

    # ── Communication links ──
    uav_positions = {u.id: (float(u.pos[0]), float(u.pos[1]))
                     for u in env.uavs}
    drawn_uav_pairs = set()

    for u in env.uavs:
        ux, uy = uav_positions[u.id]
        # UAV-MBS link (dotted)
        ax.plot([ux, mbs_x], [uy, mbs_y], color=C_COMM_MBS, lw=0.3,
                ls=":", alpha=0.35, zorder=1)
        # UAV-UAV neighbor links (dashed)
        for neighbor in u.neighbors:
            nid = neighbor.id if isinstance(neighbor, UAV) else neighbor
            pair = tuple(sorted((u.id, nid)))
            if pair in drawn_uav_pairs:
                continue
            drawn_uav_pairs.add(pair)
            if nid in uav_positions:
                vx, vy = uav_positions[nid]
                ax.plot([ux, vx], [uy, vy], color=C_COMM_UAV, lw=0.8,
                        ls="--", alpha=0.5, zorder=2)

    # ── UEs ──
    for ue in env.ues:
        ux, uy = float(ue.pos[0]), float(ue.pos[1])

        # Color by battery ratio
        battery_ratio = ue.battery_level / max(config.UE_BATTERY_CAPACITY, 1.0)
        if battery_ratio > 0.4:
            ue_color = C_UE_NORMAL
        elif battery_ratio > 0.1:
            ue_color = C_UE_LOW
        else:
            ue_color = C_UE_CRITICAL

        # Shape by request type (service / content / energy)
        req = ue.current_request
        if req.is_energy:
            marker, req_color = "D", C_ENERGY
        elif req.is_service:
            marker, req_color = "^", C_SERVICE
        else:
            marker, req_color = "s", C_CONTENT

        ax.scatter(ux, uy, s=18, color=ue_color, zorder=3, alpha=0.7)
        if ue.assigned:
            ax.scatter(ux, uy + 8, s=50, marker=marker, color=req_color,
                       zorder=4, edgecolors="white", linewidths=0.5, alpha=0.9)

        # UE-UAV association lines
        if ue.assigned:
            covering = [u for u in env.uavs if ue in u.current_covered_ues]
            if covering:
                cu = covering[0]
                ax.plot([ux, float(cu.pos[0])], [uy, float(cu.pos[1])],
                        color=C_ASSOC, lw=0.25, alpha=0.25, zorder=1)

        # Critical battery mark
        if battery_ratio <= 0.1:
            ax.text(ux - 4, uy - 8, "!", fontsize=6, color=C_UE_CRITICAL,
                    ha="center", fontweight="bold", zorder=6)

    # ── UAVs (drawn last to stay on top) ──
    for u in env.uavs:
        ux, uy = float(u.pos[0]), float(u.pos[1])
        is_busy = len(u.current_covered_ues) > 0
        uav_color = C_UAV_BUSY if is_busy else C_UAV_FREE

        # Coverage radius circle
        ax.add_patch(plt.Circle(
            (ux, uy), config.UAV_COVERAGE_RADIUS,
            fill=False, edgecolor=C_COVERAGE, lw=0.6, alpha=0.4, zorder=1,
        ))

        # UAV marker
        ax.scatter(ux, uy, s=220, marker="o", color=uav_color, zorder=6,
                   edgecolors="white", linewidths=1.5)
        ax.text(ux, uy, f"U{u.id}", fontsize=8, ha="center", va="center",
                color="white", fontweight="bold", zorder=7)

        # Service request load (computed live from covered UEs)
        request_count = sum(
            1 for ue in u.current_covered_ues
            if ue.current_request.is_service
        )
        ax.text(ux, uy - 18, f"负载:{request_count}", fontsize=6,
                ha="center", color="#555", zorder=7)

        # Cache ratio label
        cache_ratio = float(np.mean(u.cache.astype(float)))
        ax.text(ux, uy - 28, f"缓存:{cache_ratio:.0%}", fontsize=5.5,
                ha="center", color="#888", zorder=7)

    # ── Boundary box ──
    ax.plot([0, area_w, area_w, 0, 0], [0, 0, area_h, area_h, 0],
            color="#999", lw=1.2, zorder=0)

    # ── Stats footer ──
    ue_battery_levels = [ue.battery_level for ue in env.ues]
    n_service = sum(1 for ue in env.ues if ue.current_request.is_service)
    n_content = sum(1 for ue in env.ues
                    if not ue.current_request.is_service
                    and not ue.current_request.is_energy)
    n_energy = sum(1 for ue in env.ues if ue.current_request.is_energy)
    avg_bat = np.mean(ue_battery_levels) if ue_battery_levels else 0.0

    stats = (
        f"服务请求: {n_service}  |  内容请求: {n_content}  |  "
        f"能量请求: {n_energy}  |  平均电量: {avg_bat:.1f}J"
    )
    ax.text(area_w / 2, -15, stats, fontsize=8, ha="center",
            color="#555")


# ── Animation ───────────────────────────────────────────────────────────────

def animate(
    n_ticks: int = 200,
    seed: int = 42,
    save_path: str = "docs/figures/fig_system_animation.gif",
    dpi: int = 100,
    fps: int = 10,
    warmup_ticks: int = 0,
) -> None:
    """Render an animated GIF of the simulation over `n_ticks` steps."""
    _ensure_env_initialized()
    env = Env()
    env.reset(initial_positions=None)
    _randomize_ue_batteries(env, seed)
    rng = np.random.default_rng(seed)

    # Warmup phase (silent)
    for _ in range(warmup_ticks):
        actions = rng.uniform(-1, 1, (config.NUM_UAVS, 2))
        env.get_offloading_obs_and_masks()
        offload_actions = np.full(
            (config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV),
            config.OFFLOAD_ACTION_LOCAL,
            dtype=np.int64,
        )
        env.step(actions, offloading_actions=offload_actions)
    if warmup_ticks > 0:
        print(f"[visualize] Warmup complete: {warmup_ticks} ticks")

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, config.AREA_WIDTH)
    ax.set_ylim(0, config.AREA_HEIGHT)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    from PIL import Image
    frames = []

    print(f"[visualize] Generating animation ({n_ticks} frames)...")
    cum_dsr_vals: list[float] = []
    for tick in range(n_ticks):
        actions = rng.uniform(-1, 1, (config.NUM_UAVS, 2))
        env.get_offloading_obs_and_masks()
        offload_actions = np.full(
            (config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV),
            config.OFFLOAD_ACTION_LOCAL,
            dtype=np.int64,
        )
        _obs, _rewards, metrics = env.step(
            actions, offloading_actions=offload_actions,
        )
        cum_dsr_vals.append(metrics.get("deadline_satisfaction_rate", 0))

        ax.cla()
        ax.set_facecolor(C_BG)
        ax.set_xlim(0, config.AREA_WIDTH)
        ax.set_ylim(0, config.AREA_HEIGHT)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        avg_dsr = float(np.mean(cum_dsr_vals))
        processed = int(metrics.get("service_requests_processed", 0))
        generated = int(metrics.get("service_requests_generated", 0))
        ax.set_title(
            f"多无人机MEC仿真  |  t={env._time_step}/{n_ticks}  |  "
            f"累积DSR={avg_dsr:.1%}  |  "
            f"已处理 {processed}/{generated}",
            fontsize=11,
        )

        _draw_system_state(ax, env)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=C_BG)
        buf.seek(0)
        frames.append(Image.open(buf))

        if tick % 50 == 0:
            print(f"  Frame {tick}/{n_ticks} captured...")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    duration = int(1000 / fps)
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    plt.close(fig)
    print(f"[visualize] Animation saved to {save_path} "
          f"({len(frames)} frames, {fps} FPS)")


# ── Results Comparison Charts ───────────────────────────────────────────────

def plot_results(metrics_path: str = "results/final/metrics.json") -> None:
    """Plot 4-panel benchmark comparison from experiment metrics JSON."""
    metrics_file = Path(metrics_path)
    if not metrics_file.exists():
        print(f"[visualize] Metrics file not found: {metrics_path}")
        print("[visualize] Run experiments first, then retry.")
        return

    with open(metrics_file) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    if not summary:
        print("[visualize] No 'summary' field in metrics JSON.")
        return

    names = list(summary.keys())
    bar_colors = ["#378ADD", "#E74C3C", "#1D9E75", "#9B59B6", "#EF9F27"]

    def _safe(d, key, default=0.0):
        return float(d.get(key, default))

    dsr      = [_safe(summary[n], "deadline_satisfaction_rate") * 100 for n in names]
    energy   = [_safe(summary[n], "avg_energy_per_step", 0) for n in names]
    fairness = [_safe(summary[n], "fairness", 0) for n in names]
    mbs_load = [_safe(summary[n], "mbs_load_ratio", 0) * 100 for n in names]

    labels = [n.replace("_", " ").title() for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), facecolor="white")
    fig.suptitle("多无人机MEC: 策略对比", fontsize=13,
                 fontweight="bold", y=1.01)

    def _bar(ax, vals, title, ylabel, ymax=None):
        bars = ax.bar(labels, vals, color=bar_colors[:len(names)],
                      width=0.5, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        if ymax:
            ax.set_ylim(0, ymax)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + (ymax or max(vals) * 0.01),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
        ax.tick_params(axis="y", labelsize=9)

    _bar(axes[0], dsr,      "截止时间满足率(DSR)",       "DSR (%)", ymax=105)
    _bar(axes[1], energy,   "每步平均能耗",               "能耗 (J)",
         ymax=max(energy) * 1.3 if energy else 1.0)
    _bar(axes[2], fairness, "Jain公平性指数",              "JFI", ymax=1.1)
    _bar(axes[3], mbs_load, "宏基站负载率",                "MBS负载 (%)",
         ymax=max(mbs_load) * 1.3 if mbs_load else 1.0)

    out_path = Path(metrics_path).parent / "benchmark_comparison.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Comparison chart saved to {out_path}")


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-UAV MEC System Visualization Tool"
    )
    parser.add_argument("--mode", choices=["snapshot", "animate", "results"],
                        default="snapshot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=80,
                        help="Simulation steps for snapshot/animation")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Warmup ticks before rendering (battery drain)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output image resolution")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path")
    parser.add_argument("--metrics", type=str,
                        default="results/final/metrics.json",
                        help="Metrics JSON path for --mode results")
    parser.add_argument("--fps", type=int, default=10,
                        help="Animation frames per second (--mode animate)")
    args = parser.parse_args()

    matplotlib.use("Agg")

    if args.mode == "snapshot":
        out = args.output or "docs/figures/fig_system_snapshot.png"
        snapshot(out_path=out, seed=args.seed, n_ticks=args.ticks,
                 warmup_ticks=args.warmup, dpi=args.dpi)
    elif args.mode == "animate":
        out = args.output or "docs/figures/fig_system_animation.gif"
        animate(n_ticks=args.ticks, seed=args.seed, save_path=out,
                warmup_ticks=args.warmup, dpi=args.dpi, fps=args.fps)
    elif args.mode == "results":
        plot_results(metrics_path=args.metrics)


if __name__ == "__main__":
    main()
