"""Draw the 2x2 fixed confirmation trajectory and slot-500 coverage comparison.

Top row: all 1000 executed trajectory segments per UAV plus start/end positions.
Bottom row: slot-500 (before motion) 100 m physical coverage circles and service requests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qa-tools", type=Path, required=True)
    p.add_argument(
        "--preview",
        action="store_true",
        help="render a local review copy off the experiment server",
    )
    args = p.parse_args()
    if platform.system() != "Linux" and not args.preview:
        raise RuntimeError(
            "Scientific rendering runs on the experiment server; "
            "pass --preview for a local review copy"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    sys.path.insert(0, str(args.qa_tools))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    from thesis_figure_style import setup_thesis_style

    summary = json.loads((args.data / "replay_summary.json").read_text(encoding="utf-8"))
    source_path = args.data / "paired_replay.npz"
    if (
        summary["status"] != "PASS"
        or hashlib.sha256(source_path.read_bytes()).hexdigest() != summary["source_data_sha256"]
    ):
        raise ValueError("Source supplement checksum/verification failed")
    data = np.load(source_path)
    args.output.mkdir(parents=True, exist_ok=True)

    chosen_font = setup_thesis_style(font_size=9.0)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [chosen_font, "Songti SC", "SimSun", "Times New Roman"],
            "font.sans-serif": [chosen_font, "Songti SC", "SimSun", "Times New Roman"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.6,
            "font.size": 9.0,
        }
    )

    colors = ["#3D6FB6", "#D68B32", "#4B9A85", "#A07EB5", "#C86166"]
    methods = ("lower_no_attention", "lower_mlp_upper_uncovered_obs")
    top_titles = ("(a) 传统局部观测：完整轨迹", "(b) 扩展观测：完整轨迹")
    bot_titles = ("(c) 传统局部观测：覆盖快照", "(d) 扩展观测：覆盖快照")
    slot = int(summary["snapshot_step_zero_based"])
    users = data["ue_positions"]
    hotspots = data["hotspot_centers"]
    hist, _, _ = np.histogram2d(
        users[:, :, 0].ravel(),
        users[:, :, 1].ravel(),
        bins=35,
        range=((0, 700), (0, 700)),
    )
    assert int(hist.sum()) == 100000

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.25), dpi=600)
    box_w = 165.6 / 475.2
    box_h = 165.6 / 450.0
    left_xs = (0.085, 0.575)
    top_y = 1.0 - (23.4 + 165.6) / 450.0
    bot_y = 1.0 - (230.4 + 165.6) / 450.0

    for j, lx in enumerate(left_xs):
        axes[0, j].set_position([lx, top_y, box_w, box_h])
        axes[1, j].set_position([lx, bot_y, box_w, box_h])
    top_axes = list(axes[0])
    bot_axes = list(axes[1])

    def circle(ax, center, radius, color, linestyle, width, zorder):
        theta = np.linspace(0, 2 * np.pi, 1201)
        xy = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
        visible = ((xy >= 0) & (xy <= 700)).all(axis=1)
        xy[~visible] = np.nan
        ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=linestyle, lw=width, zorder=zorder)

    def style_ax(ax, title: str, show_xlabel: bool) -> None:
        ax.set(
            xlim=(0, 700),
            ylim=(0, 700),
            xticks=(0, 200, 400, 600),
            yticks=(0, 200, 400, 600),
            xlabel="水平位置 $x$ / m" if show_xlabel else "",
            ylabel="水平位置 $y$ / m",
        )
        ax.set_title(title, fontsize=9, loc="left", pad=6)
        ax.tick_params(labelsize=9, length=2.5, pad=3)
        ax.grid(False)

    for ax, method, title in zip(top_axes, methods, top_titles):
        positions = data[method + "_uav_positions"]
        assert positions.shape == (1001, 5, 2)
        for center in hotspots:
            circle(ax, center, 100, "#A9AEB7", (0, (3.7, 1.6)), 0.8, 2)
        for u, color in enumerate(colors):
            xy = positions[:, u]
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=0.82, alpha=0.88, zorder=3)
            ax.scatter(
                *xy[0],
                s=24,
                marker="o",
                facecolors="white",
                edgecolors=color,
                linewidths=1.1,
                zorder=7,
            )
            ax.scatter(
                *xy[-1],
                s=21,
                marker="s",
                facecolors=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=8,
            )
        style_ax(ax, title, show_xlabel=False)

    for ax, method, title in zip(bot_axes, methods, bot_titles):
        positions = data[method + "_uav_positions"]
        for center in hotspots:
            circle(ax, center, 100, "#A9AEB7", (0, (3.7, 1.6)), 0.8, 2)
        for u, color in enumerate(colors):
            xy = positions[:, u]
            circle(ax, xy[slot], 100, color, "-", 1.0, 4)
            ax.scatter(
                *xy[slot],
                s=26,
                marker="^",
                facecolors=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=8,
            )
        service = data[method + "_service"][slot]
        admitted = data[method + "_admitted"][slot]
        ax.scatter(
            users[slot, admitted, 0],
            users[slot, admitted, 1],
            s=15,
            marker="o",
            facecolors="#272B31",
            edgecolors="white",
            linewidths=0.3,
            zorder=6,
        )
        outside = service & ~admitted
        ax.scatter(
            users[slot, outside, 0],
            users[slot, outside, 1],
            s=17,
            marker="x",
            color="#8A9099",
            linewidths=0.8,
            zorder=6,
        )
        style_ax(ax, title, show_xlabel=True)

    top_handles = [Line2D([], [], color=c, lw=1.5, label=f"UAV {i + 1}") for i, c in enumerate(colors)] + [
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#404751",
            markersize=4.8,
            ls="",
            label="起点",
        ),
        Line2D([], [], marker="s", color="#404751", markersize=4.8, ls="", label="终点"),
    ]
    bot_handles = [
        Line2D([], [], marker="^", color="#404751", markersize=5.4, ls="", label="UAV位置"),
        Line2D([], [], marker="o", color="#272B31", markersize=5.4, ls="", label="已接纳服务请求"),
        Line2D([], [], marker="x", color="#8A9099", markersize=5.4, ls="", label="未接纳服务请求"),
        Line2D([], [], color="#A9AEB7", ls=(0, (3.7, 1.6)), lw=0.8, label="热点边界"),
    ]
    fig.legend(
        handles=top_handles,
        ncol=7,
        loc="center",
        bbox_to_anchor=(0.504, 0.535),
        fontsize=9,
        frameon=False,
        columnspacing=1.25,
        handlelength=1.3,
        handletextpad=0.45,
    )
    fig.legend(
        handles=bot_handles,
        ncol=4,
        loc="center",
        bbox_to_anchor=(0.504, 0.042),
        fontsize=9,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.5,
        handletextpad=0.5,
    )

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=args.output / "ch3_trajectory.alignment.json",
        overlay_svg=args.output / "ch3_trajectory.alignment.svg",
        strict=True,
    )
    for stem in ("ch3_trajectory_comparison", "图3-10_固定配对回合轨迹与覆盖对比"):
        fig.savefig(args.output / f"{stem}.pdf", facecolor="white")
        fig.savefig(args.output / f"{stem}.svg", facecolor="white")
        fig.savefig(args.output / f"{stem}.png", dpi=600, facecolor="white")
        fig.savefig(args.output / f"{stem}.tiff", dpi=600, facecolor="white")
    plt.close(fig)

    with (args.output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "method",
            "mean_uav_path_m",
            "hotspot_admission",
            "global_admission",
            "conditional_satisfaction",
            "generated",
            "admitted",
            "successful",
            "hotspot_generated",
            "hotspot_admitted",
            "dsr",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            writer.writerow(
                {
                    key: method if key == "method" else summary["methods"][method][key]
                    for key in fields
                }
            )
    with (args.output / "trajectory_positions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("method", "state_index", "uav_id", "x_m", "y_m"))
        for method in methods:
            for t, frame in enumerate(data[method + "_uav_positions"]):
                for u, xy in enumerate(frame):
                    writer.writerow((method, t, u + 1, *xy))
    with (args.output / "snapshot_requests.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("method", "slot", "ue_id", "x_m", "y_m", "is_service", "admitted"))
        for method in methods:
            for u, xy in enumerate(users[slot]):
                writer.writerow(
                    (
                        method,
                        int(summary["displayed_slot"]),
                        u,
                        *xy,
                        int(data[method + "_service"][slot, u]),
                        int(data[method + "_admitted"][slot, u]),
                    )
                )
    np.savetxt(args.output / "shared_ue_density.csv", hist.T, delimiter=",", fmt="%d")
    metadata = dict(
        kind="paired spatial case",
        independent_pairs=1,
        uncertainty="None; descriptive fixed case",
        selection=summary["selection"],
        snapshot=f"slot {summary['displayed_slot']} before motion; zero-based step {slot}",
        trajectories="All 1000 executed segments per UAV plus initial state; no smoothing or downsampling",
        layout="Top: complete trajectories only. Bottom: slot-500 physical coverage and actual requests, no trajectories.",
        density="Removed redundant UE density layer for clarity; complete UE positions remain in the unchanged NPZ source.",
        claim="Describe deployment differences; do not infer population flight-distance reduction from this case",
        fonts={"text": chosen_font, "math": "Times New Roman", "size_pt": 9},
        source_data_sha256=summary["source_data_sha256"],
        plotting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
