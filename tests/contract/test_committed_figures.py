"""The committed figures must be exactly what the committed evidence draws.

The figure module promises that a clean clone can prove the figures came from
the evidence. This test keeps that promise: it redraws every figure from
docs/evidence and compares the bytes with docs/figures. It also checks, on the
released metrics, the per-class decomposition that the README describes in
words beside the mean IoU gap figure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from drivemetrics.report.svg import FIGURE_NAMES, top_two_intervals, write_figures

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "docs/evidence/bdd100k_semseg_v1"
FIGURES = REPO_ROOT / "docs/figures"


def test_every_committed_figure_is_redrawn_byte_for_byte(tmp_path: Path) -> None:
    """A committed figure that the evidence does not redraw cannot be traced to it."""

    result = write_figures(EVIDENCE, tmp_path)

    expected = [f"{name}.svg" for name in FIGURE_NAMES]
    assert [path.name for path in result.figure_paths] == expected
    assert sorted(path.name for path in FIGURES.glob("*.svg")) == sorted(expected)
    for path in result.figure_paths:
        assert path.read_bytes() == (FIGURES / path.name).read_bytes(), path.name


def test_the_critical_classes_carry_more_than_the_whole_mean_iou_gap() -> None:
    """The README sentence beside the gap figure, checked on the released metrics."""

    metrics = json.loads((EVIDENCE / "metrics.json").read_text(encoding="utf-8"))
    rankings = json.loads((EVIDENCE / "rankings.json").read_text(encoding="utf-8"))
    miou = top_two_intervals(rankings)["miou"]
    left, right = str(miou["left"]), str(miou["right"])
    names = metrics["per_class"]["class_names"]
    critical = set(metrics["risk_profiles"]["vru_priority"]["critical_class_ids"])
    contributions = [
        (a - b) / len(names)
        for a, b in zip(
            metrics["per_class"]["by_model"][left]["iou"],
            metrics["per_class"]["by_model"][right]["iou"],
            strict=True,
        )
    ]
    critical_sum = math.fsum(c for index, c in enumerate(contributions) if index in critical)
    other_sum = math.fsum(c for index, c in enumerate(contributions) if index not in critical)
    total = metrics["metrics"][left]["miou"] - metrics["metrics"][right]["miou"]

    assert (left, right) == ("segformer_b2", "upernet_convnextv2_tiny")
    assert len(names) == 19
    assert len(critical) == 4
    assert critical_sum < total < 0
    assert other_sum > 0
