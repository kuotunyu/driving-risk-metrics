"""Contracts for the two deterministic SVG figures drawn from the evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PROTOCOL_HASH = "a" * 64
MANIFEST_HASH = "b" * 64


def pair(left: str, right: str, low: float, high: float) -> dict[str, Any]:
    return {
        "left": left,
        "right": right,
        "low": low,
        "high": high,
        "estimate": (low + high) / 2,
        "excludes_zero": low > 0 or high < 0,
    }


RANKINGS: dict[str, Any] = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "baseline_metric": "miou",
    "comparisons": [
        {
            "metric_name": "critical_recall",
            "baseline_order": ["alpha", "beta"],
            "comparison_order": ["alpha", "beta"],
            "reversal_observed": False,
        }
    ],
    "separability": {
        "miou": [pair("alpha", "beta", -0.02, 0.01)],
        "critical_recall": [pair("alpha", "beta", -0.04, -0.01)],
        "pixel_accuracy": [pair("alpha", "beta", -0.001, 0.001)],
    },
}


def small(misses: int, count: int) -> dict[str, Any]:
    return {
        "by_tertile": {
            "small": {
                "critical_misses": misses,
                "instance_count": count,
                "mean_correct_fraction": 0.5,
            }
        },
        "critical_misses": misses,
        "instance_count": count,
        "mean_correct_fraction": 0.5,
    }


EXTENDED: dict[str, Any] = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "instances": {
        "alpha": {"by_class": {"car": small(3, 7), "person": small(17, 18)}},
        "beta": {"by_class": {"car": small(5, 7), "person": small(18, 18)}},
    },
}

NOT_COMPUTED: dict[str, Any] = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "instances": {"not_computed": "the frozen area tertiles were not supplied"},
}


def load_svg_module() -> ModuleType:
    try:
        from drivemetrics.report import svg
    except ImportError:
        pytest.fail("drivemetrics.report.svg is missing", pytrace=False)
    return svg


def write_documents(directory: Path, *, extended: dict[str, Any] = EXTENDED) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rankings.json").write_text(json.dumps(RANKINGS, sort_keys=True), "utf-8")
    (directory / "extended-metrics.json").write_text(json.dumps(extended, sort_keys=True), "utf-8")
    return directory


def test_every_pair_is_one_row_under_its_metric_marked_by_whether_it_excludes_zero() -> None:
    """A pair drawn without its zero verdict would let a reader infer significance by eye.

    Pairs are grouped under a heading per metric rather than repeating the metric in
    every label, because a label long enough to name two models and a metric does
    not fit beside the plot and a clipped label is a result nobody can read.
    """

    svg = load_svg_module()

    figure = svg.paired_difference_svg(RANKINGS)

    assert figure.count('class="row') == 2
    assert figure.count("includes-zero") == 1
    assert figure.count("excludes-zero") == 1
    assert figure.count("alpha minus beta") == 2
    assert figure.index(">mean IoU<") < figure.index(">critical-class recall<")
    assert "pixel_accuracy" not in figure


def test_known_models_are_shown_by_their_display_names() -> None:
    """The figure is read by people; an unknown name passes through unchanged."""

    svg = load_svg_module()

    assert svg.display_name("segformer_b2") == "SegFormer-B2"
    assert svg.display_name("upernet_convnextv2_tiny") == "ConvNeXtV2-Tiny"
    assert svg.display_name("upernet_dinov2_small") == "DINOv2-Small"
    assert svg.display_name("alpha") == "alpha"


def test_the_zero_line_is_always_drawn() -> None:
    """Without the zero line an interval cannot be read against the null difference."""

    svg = load_svg_module()

    assert 'class="zero"' in svg.paired_difference_svg(RANKINGS)


def test_tick_steps_put_between_three_and_eight_ticks_on_any_span() -> None:
    """A span with two ticks hides the scale; one with twenty is unreadable."""

    svg = load_svg_module()

    for span in (0.007, 0.05, 0.16, 0.4, 0.9, 3.7, 9.9, 95.0):
        ticks = span / svg._tick_step(span)
        assert 3 < ticks <= 8, span


def test_every_bar_is_labelled_with_the_exact_counts_it_was_drawn_from() -> None:
    """A bar without its counts is a rounded result; the counts are the result."""

    svg = load_svg_module()

    figure = svg.small_tertile_miss_svg(EXTENDED, ["alpha", "beta"])

    assert figure.count('class="bar"') == 4
    for label in (">3/7<", ">17/18<", ">5/7<", ">18/18<"):
        assert label in figure


def test_classes_are_ordered_by_the_leading_models_miss_rate() -> None:
    """The worst-served class must read first, and the order must not depend on names."""

    svg = load_svg_module()

    figure = svg.small_tertile_miss_svg(EXTENDED, ["alpha", "beta"])

    assert figure.index(">person<") < figure.index(">car<")


def test_the_legend_follows_the_baseline_order() -> None:
    """The legend order is the ranking order, so the reader can match bars to models."""

    svg = load_svg_module()

    figure = svg.small_tertile_miss_svg(EXTENDED, ["beta", "alpha"])

    assert figure.index(">beta<") < figure.index(">alpha<")


def test_an_instance_block_that_was_not_computed_refuses_to_draw() -> None:
    """An empty chart would read as zero misses, which is the opposite of unknown."""

    svg = load_svg_module()

    with pytest.raises(ValueError, match=r"^instance coverage was not computed: the frozen"):
        svg.small_tertile_miss_svg(NOT_COMPUTED, ["alpha"])


def test_two_runs_write_byte_identical_figures(tmp_path: Path) -> None:
    """A figure that differs between builds cannot be proved to come from the evidence."""

    svg = load_svg_module()
    first = svg.write_figures(write_documents(tmp_path / "one"), tmp_path / "out-one")
    second = svg.write_figures(write_documents(tmp_path / "two"), tmp_path / "out-two")

    assert [path.name for path in first.figure_paths] == [
        "paired-differences.svg",
        "small-tertile-critical-misses.svg",
    ]
    for left, right in zip(first.figure_paths, second.figure_paths, strict=True):
        assert left.read_bytes() == right.read_bytes()
        assert b"\r" not in left.read_bytes()


def test_a_missing_document_fails_closed(tmp_path: Path) -> None:
    """Drawing from a partial evidence set would publish a figure nothing can audit."""

    svg = load_svg_module()
    directory = write_documents(tmp_path / "docs")
    (directory / "extended-metrics.json").unlink()

    with pytest.raises(FileNotFoundError, match=r"^report input is missing:"):
        svg.write_figures(directory, tmp_path / "out")


def test_no_figure_embeds_an_image(tmp_path: Path) -> None:
    """The release ships no dataset pixels, and a figure is the easiest place to leak one."""

    svg = load_svg_module()
    result = svg.write_figures(write_documents(tmp_path / "docs"), tmp_path / "out")

    for path in result.figure_paths:
        text = path.read_text(encoding="utf-8")
        assert "<image" not in text
        assert "data:" not in text
        assert text.startswith("<svg ")


def test_report_package_exports_the_figure_entry_points() -> None:
    """The CLI reaches the figures through the package, like the report."""

    import drivemetrics.report as report

    svg = load_svg_module()
    assert report.write_figures is svg.write_figures
    assert report.FiguresResult is svg.FiguresResult
