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
    "Songti SC",
    "AR PL UMing CN",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Times New Roman",
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

# 全书统一状态与机制图语义色板（遵循四条绘图红线）：
# 绿色=已覆盖/按时完成；橙色=已接纳超时；红色=未覆盖/超期淘汰；
# 实线=前向推理与真实执行；虚线=训练监督、蒸馏目标与梯度回传。
SEMANTIC_PALETTE = {
    "covered_ok": "#3A8D5A",
    "admitted_miss": "#D8A448",
    "uncovered_expired": "#C75E5A",
    "module_blue": "#3D6FB6",
    "module_teal": "#53A6A6",
    "module_purple": "#7C70B2",
    "neutral_slate": "#5A6673",
    "panel_bg": "#F7F9FC",
    "border_light": "#CBD2DC",
    "text_dark": "#22252A",
}

# 2026-11 全书 23 图重构设计系统常量（匹配中大硕士模板 A4 版芯 \textwidth = 146.6 mm）
TEXTWIDTH_MM: float = 146.6
TEXTWIDTH_IN: float = TEXTWIDTH_MM / 25.4
TEXTWIDTH_PT: float = TEXTWIDTH_IN * 72.0
COMPACT_WIDTH_MM: float = 132.0
COMPACT_WIDTH_IN: float = COMPACT_WIDTH_MM / 25.4

REDESIGN_PALETTE = {
    "hero": "#005A9C",          # 本文核心方法（扩展需求观测 / MC-ITR / 联合微调与 Top-6 MPC）
    "hero_light": "#DCEAF7",    # 核心方法浅底色填充
    "hero_mid": "#2B7BBA",      # 核心次要系列（如微调零前瞻与微调 MPC 区分）
    "comparator": "#D55E00",    # 核心强基线对标（纯关联基线 / Iterative-Base / 在线规划基线）
    "comp_light": "#FCE8D8",    # 强基线浅底色
    "distilled": "#009E73",     # 递进中间态（Greedy / 两轮规划蒸馏）
    "dist_light": "#DCF3EC",    # 中间态浅底色
    "ablation": "#6A51A3",      # 消融对照（cf=0 纯在策略暖启动 / 去期望反馈 / 分解式）
    "abl_light": "#ECE7F6",     # 消融浅底色
    "prior": "#6E7781",         # 初始先验 / Heuristic / 弱基线
    "muted": "#AEB6C0",         # 背景细轨迹 / 未覆盖灰色部分
    "crimson": "#C43C39",       # 安全红线 (d_min=200m / 1.0s 时隙预算) 与反向梯度箭头
    "amber": "#E69F00",         # 已接纳超时损失 (A-S)
    "success": "#2E8B57",       # 按时完成服务 (S)
    "ink": "#1F2328",           # 统一深色正文墨色
    "grid": "#E2E6EC",          # 极浅辅助网格线
    "panel_bg": "#F7F9FC",      # 面板/容器极浅灰蓝衬底
    "border": "#CBD2DC",        # 容器淡边框
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


def resolve_qa_scripts_dir(repo_root: Path) -> Path:
    """Locate bundled or installed nature-figure QA scripts without hardcoded machine paths."""
    candidates = [
        repo_root / "tools" / "figure_qa",
        Path.home() / ".gemini" / "config" / "skills" / "nature-figure" / "scripts",
        Path.home() / ".codex" / "skills" / "nature-figure" / "scripts",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError("Unable to locate nature-figure QA scripts directory.")


def setup_thesis_style(font_size: float = 8.0, legend_frameon: bool = True) -> str:
    """Apply the shared enclosed-frame engineering thesis style and return the selected font."""

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
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.95,
            "axes.edgecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.pad": 3.5,
            "ytick.major.pad": 3.5,
            "legend.frameon": legend_frameon,
            "legend.fancybox": False,
            "legend.edgecolor": "#4A525D",
            "legend.facecolor": "white",
            "legend.framealpha": 0.96,
            "patch.linewidth": 0.75,
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.cal": "STIXGeneral:italic",
            "mathtext.fallback": "stix",
        }
    )
    return family


def soften_axes(ax: plt.Axes, axis: str = "both") -> None:
    if axis in ("x", "y", "both"):
        ax.grid(True, axis=axis, color=PALETTE["grid"], linewidth=0.55, alpha=0.82)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.95)
        ax.spines[side].set_edgecolor(PALETTE["text"])
    has_x = len(ax.get_xticks()) > 0
    has_y = len(ax.get_yticks()) > 0
    ax.tick_params(
        axis="x",
        direction="in",
        length=3.0,
        width=0.8,
        top=has_x,
        bottom=has_x,
    )
    ax.tick_params(
        axis="y",
        direction="in",
        length=3.0,
        width=0.8,
        left=has_y,
        right=has_y,
    )


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

    import io

    qa = qa_text_space(fig, name)
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in formats:
            dpi = tiff_dpi if ext == "tiff" else png_dpi if ext == "png" else None
            buf = io.BytesIO()
            fig.savefig(buf, format=ext, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi)
            payload = buf.getvalue()
            target = out_dir / f"{name}.{ext}"
            for attempt in range(4):
                try:
                    target.write_bytes(payload)
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    import time

                    time.sleep(0.2 * (attempt + 1))
    if qa_report is not None:
        write_qa_report(qa_report, qa)
    plt.close(fig)
    return qa


def setup_redesign_style(font_size: float = 8.0) -> str:
    """Apply the unified 146.6 mm 1:1 thesis redesign style and return selected CJK font."""
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
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.80,
            "axes.edgecolor": REDESIGN_PALETTE["ink"],
            "axes.labelcolor": REDESIGN_PALETTE["ink"],
            "xtick.color": REDESIGN_PALETTE["ink"],
            "ytick.color": REDESIGN_PALETTE["ink"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.pad": 2.8,
            "ytick.major.pad": 2.8,
            "legend.frameon": False,
            "patch.linewidth": 0.75,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.cal": "STIXGeneral:italic",
            "mathtext.fallback": "stix",
        }
    )
    return family


def style_redesign_axis(
    ax: plt.Axes,
    *,
    grid: str = "y",
    box: bool = False,
) -> None:
    """Format quantitative or spatial axis cleanly with subtle grid and crisp spines."""
    if grid in ("x", "y", "both"):
        ax.grid(True, axis=grid, color=REDESIGN_PALETTE["grid"], linewidth=0.55, alpha=0.85, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.80)
        ax.spines[side].set_edgecolor(REDESIGN_PALETTE["ink"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(box)
        if box:
            ax.spines[side].set_linewidth(0.80)
            ax.spines[side].set_edgecolor(REDESIGN_PALETTE["ink"])
    ax.tick_params(
        axis="both",
        direction="in" if box else "out",
        length=2.8,
        width=0.75,
        top=False,
        right=False,
    )


def add_panel_tag(
    ax: plt.Axes,
    tag: str,
    *,
    x_offset_pt: float = -2.0,
    y_offset_pt: float = 5.0,
    fontsize: float = 9.0,
) -> None:
    """Place a panel tag such as '(a)' at a fixed physical point offset above top-left."""
    from matplotlib.transforms import ScaledTranslation

    offset = ScaledTranslation(
        x_offset_pt / 72.0,
        y_offset_pt / 72.0,
        ax.figure.dpi_scale_trans,
    )
    ax.text(
        0.0,
        1.0,
        tag,
        transform=ax.transAxes + offset,
        fontsize=fontsize,
        fontweight="bold",
        color=REDESIGN_PALETTE["ink"],
        ha="left",
        va="bottom",
    )


def export_redesign_figure(
    fig: plt.Figure,
    stem: str,
    *,
    repo_root: Path,
    output_dirs: Iterable[Path],
    qa_dir: Path,
    axes: list[plt.Axes] | None = None,
    row_groups: list[object] | None = None,
    column_groups: list[object] | None = None,
    exclude_axes: Iterable[plt.Axes] = (),
    exemptions: list[dict[str, object]] | None = None,
    png_dpi: int = 450,
    min_glyph_pt: float = 5.0,
) -> dict[str, object]:
    """Export 1:1 physical-width Matplotlib figure and run full alignment + PDF text + collision QA."""
    import io
    import subprocess
    import sys

    qa_scripts = resolve_qa_scripts_dir(repo_root)
    if str(qa_scripts) not in sys.path:
        sys.path.insert(0, str(qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    qa_dir.mkdir(parents=True, exist_ok=True)
    align_json = qa_dir / f"{stem}.alignment.json"
    align_svg = qa_dir / f"{stem}.alignment.svg"

    target_axes = list(axes) if axes is not None else list(fig.axes)
    align_report = require_matplotlib_panel_alignment(
        fig,
        axes=target_axes,
        row_groups=row_groups,
        column_groups=column_groups,
        exclude_axes=exclude_axes,
        exemptions=exemptions or [],
        json_out=align_json,
        overlay_svg=align_svg if len(target_axes) >= 2 else None,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )

    written_pdf: Path | None = None
    written_svg: Path | None = None
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            dpi = png_dpi if ext == "png" else None
            buf = io.BytesIO()
            fig.savefig(buf, format=ext, facecolor="white", dpi=dpi)
            payload = buf.getvalue()
            if ext == "svg":
                # Strip DTD and normalize Matplotlib's 1-pt-user-unit 'px' CSS label so SVG tools read true pt sizes
                payload = re.sub(rb"<!DOCTYPE[^>]*>\s*", b"", payload)
                payload = re.sub(rb"(font:\s*(?:italic\s+|bold\s+|700\s+|600\s+)*)(\d+(?:\.\d+)?)px\b", rb"\g<1>\2pt", payload)
                payload = re.sub(rb"(font-size:\s*)(\d+(?:\.\d+)?)px\b", rb"\g<1>\2pt", payload)
            target = out_dir / f"{stem}.{ext}"
            target.write_bytes(payload)
            if ext == "pdf" and written_pdf is None:
                written_pdf = target
            if ext == "svg" and written_svg is None:
                written_svg = target
    plt.close(fig)

    assert written_pdf is not None
    res_text = subprocess.run(
        [
            sys.executable,
            str(qa_scripts / "audit_pdf_text.py"),
            str(written_pdf),
            "--min-pt",
            str(min_glyph_pt),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    (qa_dir / f"{stem}.pdf_text.json").write_text(res_text.stdout, encoding="utf-8")
    if res_text.returncode != 0:
        raise RuntimeError(f"PDF glyph size floor failed for {stem}:\n{res_text.stdout}\n{res_text.stderr}")

    col_json = qa_dir / f"{stem}.collision.json"
    res_col = subprocess.run(
        [
            sys.executable,
            str(qa_scripts / "audit_figure_collisions.py"),
            str(written_pdf),
            "--json-out",
            str(col_json),
        ],
        capture_output=True,
        text=True,
    )
    if res_col.returncode == 1 or res_col.returncode == 2:
        raise RuntimeError(f"Collision audit failed for {stem} (code {res_col.returncode}):\n{res_col.stdout}\n{res_col.stderr}")
    col_data = json.loads(col_json.read_text(encoding="utf-8"))
    return {
        "stem": stem,
        "alignment_verdict": align_report.get("verdict"),
        "collision_verdict": col_data.get("verdict"),
        "collision_summary": col_data.get("summary", {}),
    }
