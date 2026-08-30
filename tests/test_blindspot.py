"""Tests for the blind-spot rate.

The metric's value comes from being un-gameable in a specific way: it cannot be
raised by doing well on images that do not contain the hazard. These tests pin
that property down, along with the threshold behaviour that keeps it honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivemetrics import blind_spot_curve, blind_spot_rate, collect_counts
from drivemetrics.taxonomy import class_index

PED = class_index("Pedestrian")
ROAD = class_index("Road")
BIC = class_index("Bicyclist")


def _image(ped_pixels: int, recovered: int, size: int = 40):
    """One image with `ped_pixels` pedestrian pixels, `recovered` predicted right."""
    target = np.full((size, size), ROAD, dtype=np.int64)
    pred = np.full((size, size), ROAD, dtype=np.int64)
    flat_t = target.reshape(-1)
    flat_p = pred.reshape(-1)
    flat_t[:ped_pixels] = PED
    flat_p[:recovered] = PED
    return target, pred


def test_missed_hazard_counts_as_blind():
    t, p = _image(ped_pixels=100, recovered=0)
    stats = blind_spot_rate([collect_counts("img", t, p)])
    assert stats["Pedestrian"].present_images == 1
    assert stats["Pedestrian"].blind_images == 1
    assert stats["Pedestrian"].blind_rate == pytest.approx(1.0)


def test_recovered_hazard_is_not_blind():
    t, p = _image(ped_pixels=100, recovered=100)
    stats = blind_spot_rate([collect_counts("img", t, p)])
    assert stats["Pedestrian"].blind_images == 0
    assert stats["Pedestrian"].blind_rate == pytest.approx(0.0)


def test_partial_recovery_above_threshold_is_not_blind():
    """The bar is 'registered the hazard at all', not 'segmented it well'."""
    t, p = _image(ped_pixels=100, recovered=20)  # recall 0.2 > default 0.10
    stats = blind_spot_rate([collect_counts("img", t, p)])
    assert stats["Pedestrian"].blind_images == 0


def test_token_recovery_below_threshold_is_still_blind():
    t, p = _image(ped_pixels=100, recovered=2)  # recall 0.02 < 0.10
    stats = blind_spot_rate([collect_counts("img", t, p)])
    assert stats["Pedestrian"].blind_images == 1


def test_images_without_the_hazard_are_not_counted():
    """The property that makes this metric hard to game.

    Ninety-nine perfectly segmented road-only images cannot dilute a single
    total pedestrian failure, because they never contained a pedestrian.
    """
    counts = []
    for i in range(99):
        t = np.full((40, 40), ROAD, dtype=np.int64)
        counts.append(collect_counts(f"road-{i}", t, t.copy()))
    t, p = _image(ped_pixels=100, recovered=0)
    counts.append(collect_counts("hazard", t, p))

    stats = blind_spot_rate(counts)
    assert stats["Pedestrian"].present_images == 1
    assert stats["Pedestrian"].blind_rate == pytest.approx(1.0)


def test_min_gt_pixels_excludes_annotation_slivers():
    """A three-pixel pedestrian is not evidence of a blind spot."""
    t, p = _image(ped_pixels=3, recovered=0)
    counts = [collect_counts("tiny", t, p)]
    assert blind_spot_rate(counts, min_gt_pixels=50)["Pedestrian"].present_images == 0
    assert blind_spot_rate(counts, min_gt_pixels=1)["Pedestrian"].present_images == 1


def test_blind_image_ids_are_recorded_for_the_failure_gallery():
    t_bad, p_bad = _image(ped_pixels=100, recovered=0)
    t_ok, p_ok = _image(ped_pixels=100, recovered=100)
    counts = [collect_counts("bad", t_bad, p_bad), collect_counts("ok", t_ok, p_ok)]
    stats = blind_spot_rate(counts)
    assert stats["Pedestrian"].blind_image_ids == ["bad"]


def test_absent_class_yields_nan_not_zero():
    """No bicyclists anywhere means 'no evidence', not 'perfect score'."""
    t, p = _image(ped_pixels=100, recovered=100)
    stats = blind_spot_rate([collect_counts("img", t, p)], classes=[PED, BIC])
    assert stats["Bicyclist"].present_images == 0
    assert np.isnan(stats["Bicyclist"].blind_rate)


def test_rate_is_the_ratio_over_several_images():
    counts = []
    for i in range(4):
        recovered = 0 if i < 3 else 100
        t, p = _image(ped_pixels=100, recovered=recovered)
        counts.append(collect_counts(f"i{i}", t, p))
    stats = blind_spot_rate(counts)
    assert stats["Pedestrian"].blind_rate == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# The threshold curve
# ---------------------------------------------------------------------------


def test_curve_is_monotonically_non_decreasing():
    """Raising the bar can only ever create more blind images, never fewer."""
    counts = []
    for i, recovered in enumerate([0, 5, 20, 50, 90, 100]):
        t, p = _image(ped_pixels=100, recovered=recovered)
        counts.append(collect_counts(f"i{i}", t, p))
    curve = blind_spot_curve(counts)["Pedestrian"]
    rates = [r for _, r in curve]
    assert all(b >= a - 1e-12 for a, b in zip(rates, rates[1:]))


def test_curve_endpoints_are_meaningful():
    counts = []
    for i, recovered in enumerate([0, 50, 100]):
        t, p = _image(ped_pixels=100, recovered=recovered)
        counts.append(collect_counts(f"i{i}", t, p))
    curve = dict(blind_spot_curve(counts)["Pedestrian"])
    # At threshold 0 nothing can fail: recall >= 0 always holds.
    assert curve[0.0] == pytest.approx(0.0)
    # At threshold 1.0 only the perfectly recovered image survives.
    assert curve[1.0] == pytest.approx(2 / 3)


def test_curve_exposes_token_predictions():
    """A model emitting a few pixels looks fine at a lax threshold and collapses
    at a strict one — which is exactly what the curve is for."""
    counts = []
    for i in range(10):
        t, p = _image(ped_pixels=100, recovered=12)  # just over the 0.10 bar
        counts.append(collect_counts(f"i{i}", t, p))
    curve = dict(blind_spot_curve(counts)["Pedestrian"])
    assert curve[0.10] == pytest.approx(0.0)
    assert curve[0.50] == pytest.approx(1.0)


def test_counts_ignore_void_pixels():
    from drivemetrics import IGNORE_INDEX

    target = np.full((10, 10), IGNORE_INDEX, dtype=np.int64)
    target[0, :20] = PED
    pred = np.full((10, 10), ROAD, dtype=np.int64)
    counts = collect_counts("img", target, pred)
    assert counts.gt.sum() == int((target != IGNORE_INDEX).sum())
