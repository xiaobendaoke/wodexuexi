"""Shared thesis figure style and lightweight text-space QA.

The style intentionally follows the compact multi-panel look of the baseline
matrix figures: white background, light grid, thin axes, small serif Chinese
text, and restrained method colors.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.text import Text


FONT_CANDIDATES = [
    "SimSun",
    "宋体",
    "AR PL UMing CN",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "DejaVu Serif",
    "DejaVu Sans",
    "serif",
]

PALETTE = {
    "text": "#22252A",
    "grid": "#D5D9E2",
    "axis": "#333333",
    "baseline": "#717784",
    "upper": "#6F7DBE",
    "lower": "#53A6A6",
    "combined": "#E2A64A",
    "full": "#3D6FB6",
    "accent": "#C85A54",
    "green": "#3A8D5A",
    "orange": "#D8A448",
    "red": "#C75E5A",
    "purple": "#7C70B2",
    "blue_light": "#9EC4DF",
    "green_light": "#8FCB98",
}

BASELINE_COLORS = {
    "random": "#9AA0A6",
    "uniform": "#B08968",
    "ippo": "#4C78A8",
    "vanilla_mappo": "#F58518",
    "joint_mappo": "#7F7F7F",
    "proposed": "#54A24B",
    "heuristic": "#9AA0A6",
}

METHOD_LABELS_CN = {
    "random": "随机策略",
    "uniform": "均匀卸载",
    "ippo": "IPPO",
    "vanilla_mappo": "普通\nMAPPO",
    "joint_mappo": "联合\nMAPPO",
    "proposed": "本文方法",
    "heuristic": "启发式",
}


def selected_chinese_font() -> str:
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in installed:
            return name
    return "DejaVu Sans"


def setup_thesis_style(font_size: float = 8.0, legend_frameon: bool = False) -> str:
    """Apply the shared compact thesis style and return the selected font."""

    family = selected_chinese_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family] + FONT_CANDIDATES,
            "font.sans-serif": [family] + FONT_CANDIDATES,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": legend_frameon,
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "mathtext.fontset": "dejavuserif",
        }
    )
    return family


def soften_axes(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=PALETTE["grid"], linewidth=0.55, alpha=0.78)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.7)


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def _visible_texts(fig: plt.Figure) -> list[Text]:
    texts: list[Text] = []
    for item in fig.findobj(match=Text):
        if not item.get_visible():
            continue
        if not item.get_text().strip():
            continue
        texts.append(item)
    return texts


def qa_text_space(fig: plt.Figure, name: str, *, overlap_threshold: float = 0.38) -> dict[str, object]:
    """Return lightweight text-space QA diagnostics for a rendered figure."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans).expanded(1.01, 1.03)
    texts = _visible_texts(fig)
    text_boxes = []
    overflow = []
    for text in texts:
        bbox = text.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        text_boxes.append((text, bbox))
        if (
            bbox.x0 < fig_bbox.x0 - 2
            or bbox.y0 < fig_bbox.y0 - 2
            or bbox.x1 > fig_bbox.x1 + 2
            or bbox.y1 > fig_bbox.y1 + 2
        ):
            overflow.append(text.get_text())

    overlaps = []
    for i, (text_a, box_a) in enumerate(text_boxes):
        for text_b, box_b in text_boxes[i + 1 :]:
            x0 = max(box_a.x0, box_b.x0)
            y0 = max(box_a.y0, box_b.y0)
            x1 = min(box_a.x1, box_b.x1)
            y1 = min(box_a.y1, box_b.y1)
            if x1 <= x0 or y1 <= y0:
                continue
            inter = (x1 - x0) * (y1 - y0)
            min_area = min(box_a.width * box_a.height, box_b.width * box_b.height)
            if min_area > 0 and inter / min_area >= overlap_threshold:
                overlaps.append([text_a.get_text(), text_b.get_text()])

    text_area = sum(b.width * b.height for _, b in text_boxes)
    figure_area = max(fig_bbox.width * fig_bbox.height, 1.0)
    text_area_ratio = text_area / figure_area
    meaningful_overflow = [t for t in overflow if not re.fullmatch(r"[-+]?\d+(\.\d+)?", t.strip())]
    return {
        "figure": name,
        "text_count": len(text_boxes),
        "overflow_count": len(overflow),
        "overflow_text": overflow[:12],
        "overlap_count": len(overlaps),
        "overlaps": overlaps[:12],
        "text_area_ratio": round(float(text_area_ratio), 4),
        "status": "needs_review" if meaningful_overflow or text_area_ratio > 0.22 else "ok",
    }


def write_qa_report(report_path: Path, result: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous: list[dict[str, object]] = []
    if report_path.exists():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = []
    previous = [row for row in previous if row.get("figure") != result.get("figure")]
    previous.append(result)
    report_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")


def save_pub(
    fig: plt.Figure,
    name: str,
    output_dirs: Iterable[Path],
    *,
    formats: tuple[str, ...] = ("pdf", "svg", "png", "tiff"),
    png_dpi: int = 450,
    tiff_dpi: int = 600,
    pad_inches: float = 0.045,
    qa_report: Path | None = None,
) -> dict[str, object]:
    """Save publication outputs and optionally append a text-space QA result."""

    qa = qa_text_space(fig, name)
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in formats:
            dpi = tiff_dpi if ext == "tiff" else png_dpi if ext == "png" else None
            fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=pad_inches, dpi=dpi)
    if qa_report is not None:
        write_qa_report(qa_report, qa)
    plt.close(fig)
    return qa
