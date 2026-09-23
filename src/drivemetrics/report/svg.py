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

FIGURE_NAMES: tuple[str, ...] = (
    "paired-differences",
    "small-tertile-critical-misses",
    "headline-top-two",
    "miou-gap-by-class",
)
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
#: Left edge of the header on figures whose title is too long for the label column.
HEADER_X = 8
#: Row pitch of the per-class figure, which draws one bar per class.
CLASS_ROW_HEIGHT = 20
#: First bar row of the per-class figure, below its two subtitle lines and legend.
CLASS_ROWS_TOP = 84
#: The risk profile whose critical classes are highlighted on the per-class figure.
CRITICAL_PROFILE = "vru_priority"
CRITICAL_FILL = "#c0392b"
OTHER_FILL = "#7f8c8d"
TOTAL_FILL = "#1f3a5f"
#: A decomposition whose bars do not add up to the published difference is refused.
SUM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class FiguresResult:
    """Where the evidence figures were written."""

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


def _open(height: int, title: str, title_x: int = LABEL_WIDTH) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(title)}">\n'
        f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>\n'
        f'<text x="{title_x}" y="20" {FONT} font-weight="bold">{escape(title)}</text>\n'
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


def _zero_line(x: float, y1: int, y2: int) -> str:
    return (
        f'<line class="zero" x1="{x:.2f}" y1="{y1}" x2="{x:.2f}" '
        f'y2="{y2}" stroke="#999999" stroke-dasharray="4 3"/>\n'
    )


def _interval_row(label: str, entry: dict[str, Any], y: float, x_min: float, x_max: float) -> str:
    """Draw one interval with its estimate, filled when the interval excludes zero."""

    kind = "excludes-zero" if bool(entry["excludes_zero"]) else "includes-zero"
    fill = "#1f3a5f" if kind == "excludes-zero" else "#ffffff"
    return (
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
        _zero_line(_px(0.0, x_min, x_max), TOP, TOP + ROW_HEIGHT * len(rows)),
    ]
    for position, (metric, entry) in enumerate(rows):
        y = TOP + ROW_HEIGHT * position + ROW_HEIGHT / 2
        if entry is None:
            parts.append(
                f'<text x="8" y="{y + 4:.2f}" {FONT} font-weight="bold">'
                f"{escape(METRIC_LABELS[metric])}</text>\n"
            )
            continue
        label = f"{display_name(str(entry['left']))} minus {display_name(str(entry['right']))}"
        parts.append(_interval_row(label, entry, y, x_min, x_max))
    parts.append(_axis(TOP + ROW_HEIGHT * len(rows) + 8, x_min, x_max, step))
    parts.append("</svg>\n")
    return "".join(parts)


def top_two_intervals(rankings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return, per interval metric, the one comparison between the top two models.

    The top two are the first two models of the baseline order. Entries are matched
    by model names, never by their position in the list, and keep their own
    left-minus-right orientation, which is the sign the claims publish. Every
    metric must compare the pair in the same orientation, so one title can name it.
    """

    top_two = {str(model) for model in rankings["comparisons"][0]["baseline_order"][:2]}
    selected: dict[str, dict[str, Any]] = {}
    for metric in INTERVAL_METRICS:
        matches = [
            entry
            for entry in rankings["separability"][metric]
            if {str(entry["left"]), str(entry["right"])} == top_two
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {metric} interval between {sorted(top_two)}, found {len(matches)}"
            )
        selected[metric] = dict(matches[0])
    orientations = {(str(entry["left"]), str(entry["right"])) for entry in selected.values()}
    if len(orientations) != 1:
        raise ValueError(f"the top-two intervals disagree on orientation: {sorted(orientations)}")
    return selected


def headline_top_two_svg(rankings: dict[str, Any]) -> str:
    """Draw the top two models' intervals on an axis that covers only them and zero.

    The all-pairs figure must also fit the comparisons against the weakest model, which
    squeezes the headline pair into a few pixels beside zero; this figure zooms in on it.
    """

    entries = top_two_intervals(rankings)
    first = entries[INTERVAL_METRICS[0]]
    left, right = display_name(str(first["left"])), display_name(str(first["right"]))
    x_min, x_max, step = _scale(
        min(float(entry["low"]) for entry in entries.values()),
        max(float(entry["high"]) for entry in entries.values()),
    )
    rows_bottom = TOP + ROW_HEIGHT * len(INTERVAL_METRICS)
    parts = [
        _open(
            rows_bottom + BOTTOM,
            f"Paired differences between the top two models, {left} minus {right}",
            HEADER_X,
        ),
        f'<text x="{HEADER_X}" y="36" {FONT} fill="#555555">'
        "filled: interval excludes zero; hollow: interval includes zero</text>\n",
        _zero_line(_px(0.0, x_min, x_max), TOP, rows_bottom),
    ]
    for position, metric in enumerate(INTERVAL_METRICS):
        y = TOP + ROW_HEIGHT * position + ROW_HEIGHT / 2
        parts.append(_interval_row(METRIC_LABELS[metric], entries[metric], y, x_min, x_max))
    parts.append(_axis(rows_bottom + 8, x_min, x_max, step))
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


def miou_gap_by_class_svg(metrics: dict[str, Any], left: str, right: str) -> str:
    """Split the mean IoU difference between two models into one bar per class.

    Mean IoU weighs every class equally, so a class contributes its IoU difference
    divided by the number of classes and the bars add up to the mean IoU difference;
    a decomposition that does not add up is refused. The critical classes of the
    vulnerable-road-user profile are highlighted and combined into one row, so the
    reader does not have to add bars by eye. The values are seed-averaged point
    estimates: no per-class interval exists, so none is drawn.
    """

    per_class = metrics["per_class"]
    names = [str(name) for name in per_class["class_names"]]
    critical = {
        int(index) for index in metrics["risk_profiles"][CRITICAL_PROFILE]["critical_class_ids"]
    }
    bars = sorted(
        ((float(a) - float(b)) / len(names), index, name)
        for index, (name, a, b) in enumerate(
            zip(
                names,
                per_class["by_model"][left]["iou"],
                per_class["by_model"][right]["iou"],
                strict=True,
            )
        )
    )
    critical_sum = math.fsum(value for value, index, _ in bars if index in critical)
    other_sum = math.fsum(value for value, index, _ in bars if index not in critical)
    total = float(metrics["metrics"][left]["miou"]) - float(metrics["metrics"][right]["miou"])
    if abs(critical_sum + other_sum - total) > SUM_TOLERANCE:
        raise ValueError(
            f"per-class contributions sum to {critical_sum + other_sum!r}, "
            f"but the mean IoU difference is {total!r}"
        )
    combined = (
        ("critical classes, combined", critical_sum, CRITICAL_FILL),
        ("other classes, combined", other_sum, OTHER_FILL),
        ("mean IoU difference", total, TOTAL_FILL),
    )
    values = [value for value, _, _ in bars] + [value for _, value, _ in combined]
    x_min, x_max, step = _scale(min(values), max(values))
    zero = _px(0.0, x_min, x_max)

    def bar(css: str, label: str, value: float, fill: str, y_top: float) -> str:
        end = _px(value, x_min, x_max)
        return (
            f'<text x="{LABEL_WIDTH - 8}" y="{y_top + 14:.2f}" {FONT} text-anchor="end">'
            f"{escape(label)}</text>\n"
            f'<rect class="{css}" x="{min(zero, end):.2f}" y="{y_top + 4:.2f}" '
            f'width="{abs(end - zero):.2f}" height="{CLASS_ROW_HEIGHT - 8}" fill="{fill}"/>\n'
        )

    separator = CLASS_ROWS_TOP + CLASS_ROW_HEIGHT * len(bars) + 5
    rows_bottom = separator + 5 + CLASS_ROW_HEIGHT * len(combined)
    title = (
        "Per-class contribution to the mean IoU difference, "
        f"{display_name(left)} minus {display_name(right)}"
    )
    parts = [
        _open(rows_bottom + 8 + BOTTOM, title, HEADER_X),
        f'<text x="{HEADER_X}" y="36" {FONT} fill="#555555">each bar: the class IoU '
        "difference divided by the number of classes; the bars sum to the mean IoU "
        "difference</text>\n",
        f'<text x="{HEADER_X}" y="52" {FONT} fill="#555555">seed-averaged point estimates '
        "from metrics.json; no per-class interval was computed</text>\n",
        f'<rect x="{HEADER_X}" y="59" width="10" height="10" fill="{CRITICAL_FILL}"/>\n'
        f'<text x="{HEADER_X + 14}" y="68" {FONT}>critical classes (vulnerable road users)'
        "</text>\n",
        f'<rect x="{HEADER_X + 292}" y="59" width="10" height="10" fill="{OTHER_FILL}"/>\n'
        f'<text x="{HEADER_X + 306}" y="68" {FONT}>other classes</text>\n',
        _zero_line(zero, CLASS_ROWS_TOP, rows_bottom),
    ]
    for row, (value, index, name) in enumerate(bars):
        highlighted = index in critical
        parts.append(
            bar(
                "bar critical" if highlighted else "bar",
                name,
                value,
                CRITICAL_FILL if highlighted else OTHER_FILL,
                CLASS_ROWS_TOP + CLASS_ROW_HEIGHT * row,
            )
        )
    parts.append(
        f'<line x1="{HEADER_X}" y1="{separator}" x2="{WIDTH - MARGIN_RIGHT}" y2="{separator}" '
        'stroke="#cccccc"/>\n'
    )
    for row, (label, value, fill) in enumerate(combined):
        parts.append(bar("combined", label, value, fill, separator + 5 + CLASS_ROW_HEIGHT * row))
    parts.append(_axis(rows_bottom + 8, x_min, x_max, step))
    parts.append("</svg>\n")
    return "".join(parts)


def write_figures(artifacts_dir: Path, output_dir: Path) -> FiguresResult:
    """Draw every evidence figure from the committed documents as LF-terminated SVG."""

    rankings = load_json_object(artifacts_dir / "rankings.json")
    extended = load_json_object(artifacts_dir / "extended-metrics.json")
    metrics = load_json_object(artifacts_dir / "metrics.json")
    order = [str(model) for model in rankings["comparisons"][0]["baseline_order"]]
    miou = top_two_intervals(rankings)["miou"]
    drawn = {
        "paired-differences": paired_difference_svg(rankings),
        "small-tertile-critical-misses": small_tertile_miss_svg(extended, order),
        "headline-top-two": headline_top_two_svg(rankings),
        "miou-gap-by-class": miou_gap_by_class_svg(metrics, str(miou["left"]), str(miou["right"])),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in FIGURE_NAMES:
        path = output_dir / f"{name}.svg"
        path.write_text(drawn[name], encoding="utf-8", newline="\n")
        paths.append(path)
    return FiguresResult(figure_paths=tuple(paths))
