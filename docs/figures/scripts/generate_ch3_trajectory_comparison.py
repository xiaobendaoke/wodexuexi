"""Draw the fixed confirmation pair from verified executed-state supplements.

Build anew; inherit only the thesis typography and restrained UAV palette.
All 1000 executed segments and all 100000 UE-time observations are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qa-tools", type=Path, required=True)
    p.add_argument("--preview", action="store_true",
                   help="render a local review copy off the experiment server")
    args = p.parse_args()
    if platform.system() != "Linux" and not args.preview:
        raise RuntimeError("Scientific rendering runs on the experiment server; "
                           "pass --preview for a local review copy")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    sys.path.insert(0, str(args.qa_tools))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    from thesis_figure_style import setup_thesis_style

    summary = json.loads((args.data / "replay_summary.json").read_text(encoding="utf-8"))
    source_path = args.data / "paired_replay.npz"
    if summary["status"] != "PASS" or hashlib.sha256(source_path.read_bytes()).hexdigest() != summary["source_data_sha256"]:
        raise ValueError("Source supplement checksum/verification failed")
    data = np.load(source_path)
    args.output.mkdir(parents=True, exist_ok=True)
    setup_thesis_style(font_size=8)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.6})
    colors = ["#3D6FB6", "#D68B32", "#4B9A85", "#A07EB5", "#C86166"]
    methods = ("lower_no_attention", "lower_mlp_upper_uncovered_obs")
    titles = ("(a) 传统局部观测", "(b) 预接纳扩展观测")
    slot = summary["snapshot_step_zero_based"]
    users = data["ue_positions"]
    hotspots = data["hotspot_centers"]
    hist, xe, ye = np.histogram2d(users[:, :, 0].ravel(), users[:, :, 1].ravel(),
                                bins=35, range=((0, 700), (0, 700)))
    assert int(hist.sum()) == 100000
    cmap = LinearSegmentedColormap.from_list("ue_density", ["#FFFFFF", "#DDE1E6"])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.05), dpi=600)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.215, top=0.945, wspace=0.16)

    def circle(ax, center, radius, color, linestyle, width, zorder):
        # Display only the portion inside the service region; retain true radius.
        theta = np.linspace(0, 2 * np.pi, 1201)
        xy = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
        visible = ((xy >= 0) & (xy <= 700)).all(axis=1)
        xy[~visible] = np.nan
        ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=linestyle, lw=width, zorder=zorder)

    for ax, method, title in zip(axes, methods, titles):
        positions = data[method + "_uav_positions"]
        assert positions.shape == (1001, 5, 2)
        ax.imshow(hist.T, origin="lower", extent=(0, 700, 0, 700), cmap=cmap,
                  vmin=0, vmax=hist.max(), interpolation="bilinear", zorder=0)
        for center in hotspots:
            circle(ax, center, 100, "#8A9099", (0, (4, 3)), 0.7, 2)
        for u, color in enumerate(colors):
            xy = positions[:, u]
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=0.95, alpha=0.9, zorder=3)
            ax.scatter(*xy[0], s=28, marker="o", facecolors="white", edgecolors=color,
                       linewidths=1.1, zorder=7)
            ax.scatter(*xy[-1], s=24, marker="s", facecolors=color, edgecolors="white",
                       linewidths=0.5, zorder=8)
            circle(ax, xy[slot], 100, color, "-", 0.7, 4)
            # A single direction arrow keeps the heading readable without crowding.
            k = 620
            delta = xy[k + 1] - xy[k]
            if np.linalg.norm(delta) > 1.0:
                ax.annotate("", xy=xy[k + 1], xytext=xy[k],
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.8,
                                            mutation_scale=7, shrinkA=0, shrinkB=0), zorder=5)
        service = data[method + "_service"][slot]
        admitted = data[method + "_admitted"][slot]
        ax.scatter(users[slot, admitted, 0], users[slot, admitted, 1], s=15, marker="o",
                   facecolors="#272B31", edgecolors="white", linewidths=0.3, zorder=6)
        outside = service & ~admitted
        ax.scatter(users[slot, outside, 0], users[slot, outside, 1], s=17, marker="x",
                   color="#8A9099", linewidths=0.8, zorder=6)
        ax.set(xlim=(0, 700), ylim=(0, 700), xticks=(0, 200, 400, 600), yticks=(0, 200, 400, 600),
               xlabel="水平位置 $x$ / m", ylabel="水平位置 $y$ / m")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=8, loc="left", pad=6)
        ax.tick_params(labelsize=8, length=2.5, pad=3)
    handles = [Line2D([], [], color=c, lw=1.5, label=f"UAV {i + 1}") for i, c in enumerate(colors)] + [
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#404751", ls="", label="起点"),
        Line2D([], [], marker="s", color="#404751", ls="", label="终点"),
        Line2D([], [], color="#404751", lw=0.85, label="100 m覆盖圆"),
        Line2D([], [], marker="o", color="#272B31", ls="", label="已接纳服务请求"),
        Line2D([], [], marker="x", color="#8A9099", ls="", label="未接纳服务请求"),
        Line2D([], [], color="#8A9099", ls="--", lw=0.8, label="预设热点边界"),
        Patch(facecolor="#DDE1E6", edgecolor="none", label="UE出现密度"),
    ]
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               fontsize=8, frameon=False, columnspacing=1.6, handlelength=1.5,
               handletextpad=0.5, markerscale=0.9, labelspacing=0.9)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=args.output / "ch3_trajectory.alignment.json",
                                      overlay_svg=args.output / "ch3_trajectory.alignment.svg", strict=True)
    fig.savefig(args.output / "ch3_trajectory_comparison.pdf", facecolor="white")
    fig.savefig(args.output / "ch3_trajectory_comparison.svg", facecolor="white")
    fig.savefig(args.output / "ch3_trajectory_comparison.png", dpi=600, facecolor="white")
    plt.close(fig)
    with (args.output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["method", "mean_uav_path_m", "hotspot_admission", "global_admission",
                  "generated", "admitted", "hotspot_generated", "hotspot_admitted", "dsr"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            writer.writerow({key: method if key == "method" else summary["methods"][method][key] for key in fields})
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
                writer.writerow((method, 500, u, *xy, int(data[method + "_service"][slot, u]),
                                 int(data[method + "_admitted"][slot, u])))
    np.savetxt(args.output / "shared_ue_density.csv", hist.T, delimiter=",", fmt="%d")
    metadata = dict(kind="paired spatial case", independent_pairs=1, uncertainty="None; descriptive fixed case",
                    selection=summary["selection"], snapshot="slot 500 before motion; zero-based step 499",
                    trajectories="All 1000 executed segments per UAV plus initial state; no smoothing or downsampling",
                    density="All 100000 UE-time observations, shared 35x35 grid, 20m bins; linear common greyscale",
                    claim="Describe deployment differences; do not infer population flight-distance reduction from this case",
                    background_density_max=int(hist.max()), source_data_sha256=summary["source_data_sha256"],
                    plotting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "figure_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
