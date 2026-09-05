"""Deterministic SVG figures drawn directly from the evidence documents.

No plotting library, no font metrics and no timestamps: the same documents
always produce the same bytes, so a clean clone can prove that the committed
figures were drawn from the committed evidence. Nothing here reads an image,
so no dataset pixel can reach a published figure. Every position is computed
from a value read out of a document, and every printed label is either a name
or an exact integer count; no result is rounded into a label.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from drivemetrics.report.builder import load_json_object

FIGURE_NAMES: tuple[str, ...] = ("paired-differences", "small-tertile-critical-misses")
INTERVAL_METRICS: tuple[str, ...] = ("miou", "critical_recall")
METRIC_LABELS: dict[str, str] = {"miou": "mean IoU", "critical_recall": "critical-class recall"}
#: How the approved models are named on a figure. Any other name passes through.
DISPLAY_NAMES: dict[str, str] = {
    "segformer_b2": "SegFormer-B2",
    "upernet_convnextv2_tiny": "ConvNeXtV2-Tiny",
    "upernet_dinov2_small": "DINOv2-Small",
}
#: One colour per position in the baseline order; cycles if there are more models.
PALETTE: tuple[str, ...] = ("#1f3a5f", "#c0392b", "#7f8c8d")
#: Candidate tick multipliers, coarsest last so that one always fits.
TICK_MULTIPLIERS: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0)
MAX_TICKS = 8

WIDTH = 760
LABEL_WIDTH = 330
MARGIN_RIGHT = 24
ROW_HEIGHT = 26
BAR_HEIGHT = 14
GROUP_GAP = 10
TOP = 48
BOTTOM = 44
FONT = 'font-family="Helvetica, Arial, sans-serif" font-size="12"'


@dataclass(frozen=True)
class FiguresResult:
    """Where the two figures were written."""

    figure_paths: tuple[Path, ...]


def display_name(model: str) -> str:
    """Return the figure name of a model, or the model's own name when it has none."""

    return DISPLAY_NAMES.get(model, model)


def _tick_step(span: float) -> float:
    """Return the finest round step that puts at most eight ticks across the span."""

    magnitude = 10.0 ** math.floor(math.log10(span))
    fitting = [
        magnitude * multiplier
        for multiplier in TICK_MULTIPLIERS
        if span / (magnitude * multiplier) <= MAX_TICKS
    ]
    return fitting[0]


def _scale(low: float, high: float) -> tuple[float, float, float]:
    """Return an axis that covers the data and zero, snapped to round ticks."""

    step = _tick_step(max(high, 0.0) - min(low, 0.0))
    return math.floor(min(low, 0.0) / step) * step, math.ceil(max(high, 0.0) / step) * step, step


def _ticks(x_min: float, x_max: float, step: float) -> list[float]:
    return [x_min + index * step for index in range(round((x_max - x_min) / step) + 1)]


def _px(value: float, x_min: float, x_max: float) -> float:
    return LABEL_WIDTH + (value - x_min) / (x_max - x_min) * (WIDTH - LABEL_WIDTH - MARGIN_RIGHT)


def _open(height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(title)}">\n'
        f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>\n'
        f'<text x="{LABEL_WIDTH}" y="20" {FONT} font-weight="bold">{escape(title)}</text>\n'
    )


def _axis(y: float, x_min: float, x_max: float, step: float) -> str:
    parts = [
        f'<line x1="{LABEL_WIDTH}" y1="{y:.2f}" x2="{WIDTH - MARGIN_RIGHT}" y2="{y:.2f}" '
        'stroke="#333333" stroke-width="1"/>\n'
    ]
    for tick in _ticks(x_min, x_max, step):
        x = _px(tick, x_min, x_max)
        parts.append(
            f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{y + 5:.2f}" stroke="#333333"/>\n'
            f'<text x="{x:.2f}" y="{y + 18:.2f}" {FONT} text-anchor="middle">{tick + 0.0:g}</text>\n'
        )
    return "".join(parts)


def paired_difference_svg(rankings: dict[str, Any]) -> str:
    """Draw every pairwise bootstrap interval under its metric, filled when it excludes zero."""

    rows: list[tuple[str, dict[str, Any] | None]] = []
    for metric in INTERVAL_METRICS:
        rows.append((metric, None))
        rows.extend((metric, entry) for entry in rankings["separability"][metric])
    pairs = [entry for _, entry in rows if entry is not None]
    x_min, x_max, step = _scale(
        min(float(entry["low"]) for entry in pairs),
        max(float(entry["high"]) for entry in pairs),
    )
    height = TOP + ROW_HEIGHT * len(rows) + BOTTOM
    parts = [
        _open(height, "Paired differences with bootstrap intervals"),
        f'<text x="{LABEL_WIDTH}" y="36" {FONT} fill="#555555">'
        "filled: interval excludes zero; hollow: interval includes zero</text>\n",
    ]
    zero = _px(0.0, x_min, x_max)
    parts.append(
        f'<line class="zero" x1="{zero:.2f}" y1="{TOP}" x2="{zero:.2f}" '
        f'y2="{TOP + ROW_HEIGHT * len(rows)}" stroke="#999999" stroke-dasharray="4 3"/>\n'
    )
    for position, (metric, entry) in enumerate(rows):
        y = TOP + ROW_HEIGHT * position + ROW_HEIGHT / 2
        if entry is None:
            parts.append(
                f'<text x="8" y="{y + 4:.2f}" {FONT} font-weight="bold">'
                f"{escape(METRIC_LABELS[metric])}</text>\n"
            )
            continue
        label = f"{display_name(str(entry['left']))} minus {display_name(str(entry['right']))}"
        kind = "excludes-zero" if bool(entry["excludes_zero"]) else "includes-zero"
        fill = "#1f3a5f" if kind == "excludes-zero" else "#ffffff"
        parts.append(
            f'<g class="row {kind}">\n'
            f'<text x="{LABEL_WIDTH - 8}" y="{y + 4:.2f}" {FONT} text-anchor="end">'
            f"{escape(label)}</text>\n"
            f'<line x1="{_px(float(entry["low"]), x_min, x_max):.2f}" y1="{y:.2f}" '
            f'x2="{_px(float(entry["high"]), x_min, x_max):.2f}" y2="{y:.2f}" '
            'stroke="#1f3a5f" stroke-width="2"/>\n'
            f'<circle cx="{_px(float(entry["estimate"]), x_min, x_max):.2f}" cy="{y:.2f}" '
            f'r="4.5" fill="{fill}" stroke="#1f3a5f" stroke-width="2"/>\n'
            "</g>\n"
        )
    parts.append(_axis(TOP + ROW_HEIGHT * len(rows) + 8, x_min, x_max, step))
    parts.append("</svg>\n")
    return "".join(parts)


def small_tertile_miss_svg(extended: dict[str, Any], model_order: Sequence[str]) -> str:
    """Draw each class's critical-miss rate on the smallest tertile, one bar per model.

    Classes are ordered by the leading model's miss rate so the worst-served classes
    read first; every bar is labelled with the exact counts it was drawn from.
    """

    blocks = extended["instances"]
    if "not_computed" in blocks:
        raise ValueError(f"instance coverage was not computed: {blocks['not_computed']}")

    def small(model: str, class_name: str) -> dict[str, Any]:
        return dict(blocks[model]["by_class"][class_name]["by_tertile"]["small"])

    def rate(model: str, class_name: str) -> float:
        block = small(model, class_name)
        return float(block["critical_misses"]) / float(block["instance_count"])

    leading = model_order[0]
    classes = sorted(blocks[leading]["by_class"], key=lambda name: (-rate(leading, name), name))
    group_height = BAR_HEIGHT * len(model_order) + GROUP_GAP
    axis_y = TOP + group_height * len(classes) + 6
    height = axis_y + BOTTOM + 14 * len(model_order)
    plot_width = WIDTH - LABEL_WIDTH - MARGIN_RIGHT - 64
    parts = [
        _open(height, "Critical misses on the smallest-tertile instances, by class"),
        f'<text x="{LABEL_WIDTH}" y="36" {FONT} fill="#555555">'
        "bar: share of instances with less than half their footprint recovered; "
        "label: misses / instances</text>\n",
    ]
    for row, class_name in enumerate(classes):
        y_top = TOP + group_height * row
        parts.append(
            f'<text x="{LABEL_WIDTH - 8}" y="{y_top + group_height / 2:.2f}" {FONT} '
            f'text-anchor="end">{escape(class_name)}</text>\n'
        )
        for index, model in enumerate(model_order):
            block = small(model, class_name)
            bar_width = rate(model, class_name) * plot_width
            bar_y = y_top + BAR_HEIGHT * index
            parts.append(
                f'<rect class="bar" x="{LABEL_WIDTH}" y="{bar_y:.2f}" width="{bar_width:.2f}" '
                f'height="{BAR_HEIGHT - 2}" fill="{PALETTE[index % len(PALETTE)]}"/>\n'
                f'<text x="{LABEL_WIDTH + bar_width + 6:.2f}" y="{bar_y + BAR_HEIGHT - 4:.2f}" '
                f"{FONT}>{int(block['critical_misses'])}/{int(block['instance_count'])}</text>\n"
            )
    parts.append(
        f'<line x1="{LABEL_WIDTH}" y1="{axis_y}" x2="{LABEL_WIDTH + plot_width}" y2="{axis_y}" '
        'stroke="#333333"/>\n'
    )
    for quarter in range(5):
        tick_x = LABEL_WIDTH + plot_width * quarter / 4
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{axis_y}" x2="{tick_x:.2f}" y2="{axis_y + 5}" '
            'stroke="#333333"/>\n'
            f'<text x="{tick_x:.2f}" y="{axis_y + 18}" {FONT} text-anchor="middle">'
            f"{quarter * 25}%</text>\n"
        )
    for index, model in enumerate(model_order):
        legend_y = axis_y + 36 + index * 14
        parts.append(
            f'<rect x="{LABEL_WIDTH}" y="{legend_y - 9}" width="10" height="10" '
            f'fill="{PALETTE[index % len(PALETTE)]}"/>\n'
            f'<text x="{LABEL_WIDTH + 14}" y="{legend_y}" {FONT}>'
            f"{escape(display_name(model))}</text>\n"
        )
    parts.append("</svg>\n")
    return "".join(parts)


def write_figures(artifacts_dir: Path, output_dir: Path) -> FiguresResult:
    """Draw both figures from the committed documents and write them as LF-terminated SVG."""

    rankings = load_json_object(artifacts_dir / "rankings.json")
    extended = load_json_object(artifacts_dir / "extended-metrics.json")
    order = [str(model) for model in rankings["comparisons"][0]["baseline_order"]]
    drawn = {
        "paired-differences": paired_difference_svg(rankings),
        "small-tertile-critical-misses": small_tertile_miss_svg(extended, order),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in FIGURE_NAMES:
        path = output_dir / f"{name}.svg"
        path.write_text(drawn[name], encoding="utf-8", newline="\n")
        paths.append(path)
    return FiguresResult(figure_paths=tuple(paths))
