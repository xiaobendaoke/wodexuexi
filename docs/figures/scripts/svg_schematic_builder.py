"""Exact physical-point vector schematic builder for the 23-figure thesis redesign.

Uses a top-down physical point coordinate grid (1 unit = 1 pt = 1/72 in) on the exact
SYSU thesis textwidth canvas (146.6 mm = 415.56 pt), exporting live editable SVG +
embedded-TrueType PDF + 450 DPI PNG via Python/Matplotlib and verifying every output
against SVG schematic rules, PDF glyph size floor (>= 5.0 pt), and PDF collision checks.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from thesis_figure_style import (
    REDESIGN_PALETTE,
    TEXTWIDTH_MM,
    export_redesign_figure,
    resolve_qa_scripts_dir,
    setup_redesign_style,
)


class SchematicCanvas:
    """Top-down physical-point vector schematic canvas (x right, y down, 1 unit = 1 pt)."""

    def __init__(self, height_mm: float, width_mm: float = TEXTWIDTH_MM, font_size: float = 8.0) -> None:
        setup_redesign_style(font_size=font_size)
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.w = self.width_mm * 72.0 / 25.4
        self.h = self.height_mm * 72.0 / 25.4
        self.fig = plt.figure(figsize=(self.width_mm / 25.4, self.height_mm / 25.4))
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax.set_xlim(0.0, self.w)
        self.ax.set_ylim(self.h, 0.0)
        self.ax.axis("off")
        self.ax.add_patch(
            Rectangle((0.0, 0.0), self.w, self.h, facecolor="#FFFFFF", edgecolor="none", zorder=0)
        )

    def panel_tag(self, x: float, y: float, tag: str, fontsize: float = 9.0) -> None:
        """Draw a bold subpanel tag such as '(a)' at top-left coordinate (x, y)."""
        self.ax.text(
            x,
            y,
            tag,
            fontsize=fontsize,
            fontweight="bold",
            color=REDESIGN_PALETTE["ink"],
            ha="left",
            va="top",
            zorder=6,
        )

    def container(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str = "",
        *,
        stroke: str = REDESIGN_PALETTE["border"],
        fill: str = REDESIGN_PALETTE["panel_bg"],
        lw: float = 0.90,
        ls: str = "-",
        r: float = 4.5,
        title_color: str | None = None,
        title_size: float = 8.5,
        title_align: str = "left",
        zorder: float = 1.0,
    ) -> None:
        """Draw a clean stage or layer container with top headroom for its bold header."""
        patch = FancyBboxPatch(
            (x + r, y + r),
            max(1.0, w - 2 * r),
            max(1.0, h - 2 * r),
            boxstyle=f"round,pad={r}",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=lw,
            linestyle=ls,
            zorder=zorder,
        )
        self.ax.add_patch(patch)
        if title:
            tx = x + 8.0 if title_align == "left" else x + w / 2.0
            self.ax.text(
                tx,
                y + 6.5,
                title,
                fontsize=title_size,
                fontweight="bold",
                color=title_color or REDESIGN_PALETTE["ink"],
                ha=title_align,
                va="top",
                zorder=zorder + 0.5,
            )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str = "",
        *,
        fill: str = "#FFFFFF",
        stroke: str = REDESIGN_PALETTE["border"],
        lw: float = 0.90,
        ls: str = "-",
        r: float = 3.6,
        bold_title: bool = False,
        title_color: str = REDESIGN_PALETTE["ink"],
        sub_color: str = REDESIGN_PALETTE["ink"],
        title_size: float = 8.1,
        sub_size: float = 7.6,
        zorder: float = 2.5,
    ) -> None:
        """Draw a module card with explicit separate baselines for title and one-line job."""
        patch = FancyBboxPatch(
            (x + r, y + r),
            max(1.0, w - 2 * r),
            max(1.0, h - 2 * r),
            boxstyle=f"round,pad={r}",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=lw,
            linestyle=ls,
            zorder=zorder,
        )
        self.ax.add_patch(patch)
        cx = x + w / 2.0
        title_lines = [ln for ln in title.split("\n") if ln != ""]
        sub_lines = [ln for ln in subtitle.split("\n") if ln != ""] if subtitle else []
        total_lines = len(title_lines) + len(sub_lines)
        if total_lines <= 1:
            self.ax.text(
                cx,
                y + h / 2.0,
                title,
                fontsize=title_size,
                fontweight="bold" if bold_title else "normal",
                color=title_color,
                ha="center",
                va="center",
                zorder=zorder + 0.5,
            )
        else:
            pad_v = max(8.5, h * 0.25)
            span = max(8.0, h - 2.0 * pad_v)
            step = span / max(1, total_lines - 1)
            all_specs = [(ln, title_size, bold_title, title_color) for ln in title_lines] + [
                (ln, sub_size, False, sub_color) for ln in sub_lines
            ]
            for idx, (ln, fsz, is_b, col) in enumerate(all_specs):
                ly = y + pad_v + idx * step
                self.ax.text(
                    cx,
                    ly,
                    ln,
                    fontsize=fsz,
                    fontweight="bold" if is_b else "normal",
                    color=col,
                    ha="center",
                    va="center",
                    zorder=zorder + 0.5,
                )

    def pill(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        fill: str = REDESIGN_PALETTE["hero_light"],
        stroke: str = REDESIGN_PALETTE["hero"],
        text_color: str = REDESIGN_PALETTE["ink"],
        fontsize: float = 7.6,
        bold: bool = False,
        lw: float = 0.75,
        zorder: float = 3.5,
    ) -> None:
        """Draw a compact rounded badge for tensor dimensions or metric callouts."""
        r = min(h * 0.42, 3.2)
        patch = FancyBboxPatch(
            (x + r, y + r),
            max(1.0, w - 2 * r),
            max(1.0, h - 2 * r),
            boxstyle=f"round,pad={r}",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=lw,
            zorder=zorder,
        )
        self.ax.add_patch(patch)
        self.ax.text(
            x + w / 2.0,
            y + h / 2.0,
            text,
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=text_color,
            ha="center",
            va="center",
            zorder=zorder + 0.5,
        )

    def step_badge(
        self,
        cx: float,
        cy: float,
        label: str,
        *,
        fill: str = REDESIGN_PALETTE["hero"],
        text_color: str = "#FFFFFF",
        r: float = 6.2,
        fontsize: float = 7.6,
        zorder: float = 4.5,
    ) -> None:
        """Draw a numbered step circle beside a primary directed edge."""
        self.ax.add_patch(
            Circle((cx, cy), radius=r, facecolor=fill, edgecolor="none", zorder=zorder)
        )
        self.ax.text(
            cx,
            cy + 0.3,
            label,
            fontsize=fontsize,
            fontweight="bold",
            color=text_color,
            ha="center",
            va="center",
            zorder=zorder + 0.5,
        )

    def text(
        self,
        x: float,
        y: float,
        s: str,
        *,
        fontsize: float = 7.8,
        color: str = REDESIGN_PALETTE["ink"],
        bold: bool = False,
        ha: str = "center",
        va: str = "center",
        linespacing: float = 1.25,
        zorder: float = 4.0,
    ) -> None:
        """Draw live vector text at (x, y) in physical points."""
        self.ax.text(
            x,
            y,
            s,
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=color,
            ha=ha,
            va=va,
            linespacing=linespacing,
            zorder=zorder,
        )

    def _draw_arrowhead(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str,
        head_len: float = 5.4,
        head_half_w: float = 2.4,
        zorder: float = 3.2,
    ) -> tuple[float, float]:
        """Draw a crisp filled triangle arrowhead ending exactly at (x2, y2) and return shortened shaft end."""
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return x2, y2
        ux, uy = dx / dist, dy / dist
        px, py = -uy, ux
        bx, by = x2 - ux * head_len, y2 - uy * head_len
        pts = [
            (x2, y2),
            (bx + px * head_half_w, by + py * head_half_w),
            (bx - px * head_half_w, by - py * head_half_w),
        ]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=zorder))
        shaft_end_x = x2 - ux * (head_len * 0.82)
        shaft_end_y = y2 - uy * (head_len * 0.82)
        return shaft_end_x, shaft_end_y

    def arrow(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        *,
        color: str = REDESIGN_PALETTE["ink"],
        lw: float = 1.05,
        ls: str = "-",
        rad: float = 0.0,
        head_len: float = 5.2,
        head_half_w: float = 2.3,
        zorder: float = 3.0,
    ) -> None:
        """Draw a straight or smooth-curved directed connector."""
        if abs(rad) > 1e-4:
            patch = FancyArrowPatch(
                p1,
                p2,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=lw,
                linestyle=ls,
                color=color,
                zorder=zorder,
            )
            self.ax.add_patch(patch)
            return
        sx, sy = self._draw_arrowhead(
            p1[0], p1[1], p2[0], p2[1], color=color, head_len=head_len, head_half_w=head_half_w, zorder=zorder + 0.1
        )
        self.ax.plot([p1[0], sx], [p1[1], sy], color=color, linewidth=lw, linestyle=ls, solid_capstyle="butt", zorder=zorder)

    def ortho_arrow(
        self,
        pts: Sequence[tuple[float, float]],
        *,
        color: str = REDESIGN_PALETTE["ink"],
        lw: float = 1.05,
        ls: str = "-",
        head_len: float = 5.2,
        head_half_w: float = 2.3,
        zorder: float = 3.0,
    ) -> None:
        """Draw an orthogonal polyline connector ending in a crisp triangle arrowhead."""
        if len(pts) < 2:
            return
        sx, sy = self._draw_arrowhead(
            pts[-2][0],
            pts[-2][1],
            pts[-1][0],
            pts[-1][1],
            color=color,
            head_len=head_len,
            head_half_w=head_half_w,
            zorder=zorder + 0.1,
        )
        xs = [p[0] for p in pts[:-1]] + [sx]
        ys = [p[1] for p in pts[:-1]] + [sy]
        self.ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, solid_joinstyle="round", zorder=zorder)

    def line(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        *,
        color: str = REDESIGN_PALETTE["border"],
        lw: float = 0.85,
        ls: str = "-",
        zorder: float = 1.8,
    ) -> None:
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw, linestyle=ls, zorder=zorder)

    def legend_row(
        self,
        y: float,
        items: Sequence[tuple[str, str, str]],
        *,
        x_start: float = 18.0,
        x_end: float | None = None,
        fontsize: float = 7.6,
    ) -> None:
        """Draw a compact horizontal semantic legend row: (kind_or_ls, color, label)."""
        end = x_end if x_end is not None else self.w - 14.0
        n = max(1, len(items))
        slot = (end - x_start) / n
        for idx, (style, color, label) in enumerate(items):
            lx = x_start + idx * slot
            if style.startswith("box:"):
                fill = style.split(":", 1)[1]
                self.ax.add_patch(
                    Rectangle((lx, y - 3.6), 11.0, 7.2, facecolor=fill, edgecolor=color, linewidth=0.85, zorder=3.5)
                )
                tx = lx + 14.5
            else:
                self.arrow((lx, y), (lx + 15.0, y), color=color, lw=1.15, ls=style, head_len=4.4, head_half_w=1.9)
                tx = lx + 18.0
            self.text(tx, y, label, fontsize=fontsize, ha="left", va="center")

    def export(
        self,
        stem: str,
        *,
        repo_root: Path,
        output_dirs: Iterable[Path],
        qa_dir: Path,
    ) -> dict[str, object]:
        """Export SVG/PDF/PNG and run SVG schematic audit + PDF text audit + PDF collision audit."""
        out_list = list(output_dirs)
        result = export_redesign_figure(
            self.fig,
            stem,
            repo_root=repo_root,
            output_dirs=out_list,
            qa_dir=qa_dir,
            axes=[self.ax],
        )
        qa_scripts = resolve_qa_scripts_dir(repo_root)
        if str(qa_scripts) not in sys.path:
            sys.path.insert(0, str(qa_scripts))
        import audit_svg_schematic

        if not hasattr(audit_svg_schematic, "math"):
            audit_svg_schematic.math = math  # Work around missing stdlib import in upstream auditor
        svg_path = out_list[0] / f"{stem}.svg"
        svg_report = audit_svg_schematic.audit_svg(
            svg_path.read_bytes(),
            min_pt=5.0,
            max_pt=11.5,
            max_width_mm=183.0,
        )
        svg_data = svg_report
        (qa_dir / f"{stem}.svg_audit.json").write_text(
            json.dumps(svg_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fail_findings = [f for f in svg_data["findings"] if f["severity"] == "FAIL"]
        if not svg_data.get("auditable") or fail_findings:
            raise RuntimeError(f"SVG schematic audit failed for {stem}: {svg_data}")
        result["svg_audit_text_count"] = svg_data.get("text_count")
        result["svg_min_pt"] = svg_data.get("minimum_found_pt")
        return result
