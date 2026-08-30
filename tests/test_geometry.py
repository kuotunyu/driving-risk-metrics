"""Tests for the flat-ground IPM model and distance-stratified evaluation.

Geometry code fails silently — a sign error or a swapped axis produces plausible
numbers that are simply wrong, and no downstream metric will complain. These
tests therefore check the model against closed-form values computed by hand,
not against its own past output.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivemetrics import CAMVID_DEFAULT_CAMERA, CameraModel, StratifiedEvaluator
from drivemetrics.geometry.ipm import (
    DEFAULT_BANDS,
    band_labels,
    band_map,
    distance_map,
    row_to_distance,
)
from drivemetrics.taxonomy import class_index

PED = class_index("Pedestrian")
ROAD = class_index("Road")


# ---------------------------------------------------------------------------
# The projection itself
# ---------------------------------------------------------------------------


def test_horizon_row_is_infinitely_far():
    cam = CAMVID_DEFAULT_CAMERA
    assert np.isinf(row_to_distance(np.array([cam.horizon_row]), cam)[0])


def test_above_the_horizon_is_infinite_not_negative():
    """Rays above the horizon never meet the ground; they must not wrap around
    to a negative or spuriously large finite distance."""
    cam = CAMVID_DEFAULT_CAMERA
    rows = np.array([0.0, cam.horizon_row - 1.0])
    dist = row_to_distance(rows, cam)
    assert np.all(np.isinf(dist))


def test_distance_decreases_monotonically_down_the_image():
    cam = CAMVID_DEFAULT_CAMERA
    rows = np.arange(int(cam.horizon_row) + 1, cam.image_height, dtype=np.float64)
    dist = row_to_distance(rows, cam)
    assert np.all(np.diff(dist) < 0)
    assert np.all(dist > 0)


def test_closed_form_45_degrees():
    """At 45 degrees below the horizon the ground distance equals camera height.

    Chosen because tan(45 deg) = 1 exactly, so the expected value is the camera
    height with no floating-point slack — a genuine independent check rather
    than a restatement of the implementation.
    """
    cam = CameraModel(
        horizon_row=100.0, focal_px=200.0, height_m=1.5, image_height=400, image_width=400
    )
    # A 45 degree depression needs (v - horizon) == focal_px.
    row = np.array([100.0 + 200.0])
    assert row_to_distance(row, cam)[0] == pytest.approx(1.5)


def test_closed_form_arbitrary_angle():
    cam = CameraModel(
        horizon_row=50.0, focal_px=300.0, height_m=1.2, image_height=400, image_width=400
    )
    row = 50.0 + 150.0  # tan(alpha) = 150/300 = 0.5
    expected = 1.2 / 0.5
    assert row_to_distance(np.array([row]), cam)[0] == pytest.approx(expected)


def test_doubling_camera_height_doubles_distance():
    base = CameraModel(
        horizon_row=100.0, focal_px=200.0, height_m=1.0, image_height=400, image_width=400
    )
    tall = CameraModel(
        horizon_row=100.0, focal_px=200.0, height_m=2.0, image_height=400, image_width=400
    )
    rows = np.array([250.0, 300.0, 380.0])
    assert np.allclose(row_to_distance(rows, tall), 2 * row_to_distance(rows, base))


def test_distance_map_is_constant_along_rows():
    """The flat-ground model makes distance a function of row alone."""
    dmap = distance_map(CAMVID_DEFAULT_CAMERA)
    finite = dmap[np.isfinite(dmap).all(axis=1)]
    assert np.all(finite == finite[:, :1])


# ---------------------------------------------------------------------------
# Camera model validation
# ---------------------------------------------------------------------------


def test_rejects_nonsense_parameters():
    with pytest.raises(ValueError, match="focal_px"):
        CameraModel(horizon_row=10, focal_px=0, height_m=1.5, image_height=100, image_width=100)
    with pytest.raises(ValueError, match="height_m"):
        CameraModel(horizon_row=10, focal_px=100, height_m=-1, image_height=100, image_width=100)
    with pytest.raises(ValueError, match="horizon_row"):
        CameraModel(horizon_row=500, focal_px=100, height_m=1.5, image_height=100, image_width=100)


def test_uniform_rescale_preserves_geometry():
    cam = CAMVID_DEFAULT_CAMERA
    half = cam.scaled_to(cam.image_height // 2, cam.image_width // 2)
    # The same physical point should map to the same distance at both scales.
    row_full = cam.horizon_row + 100
    row_half = half.horizon_row + 50
    assert row_to_distance(np.array([row_full]), cam)[0] == pytest.approx(
        row_to_distance(np.array([row_half]), half)[0]
    )


def test_non_uniform_rescale_is_refused():
    """Changing aspect ratio changes the implied focal length; refuse to guess."""
    with pytest.raises(ValueError, match="non-uniform"):
        CAMVID_DEFAULT_CAMERA.scaled_to(180, 480)


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------


def test_band_labels_render_readably():
    assert band_labels((0.0, 15.0, 40.0, float("inf"))) == ("0-15m", "15-40m", "40m+")


def test_sky_is_excluded_from_every_band():
    """Pixels above the horizon must be -1, not silently pooled into 'far'."""
    bmap = band_map(CAMVID_DEFAULT_CAMERA)
    horizon = int(CAMVID_DEFAULT_CAMERA.horizon_row)
    assert np.all(bmap[:horizon] == -1)
    assert np.any(bmap[horizon:] >= 0)


def test_bands_are_ordered_near_to_far_down_the_image():
    bmap = band_map(CAMVID_DEFAULT_CAMERA)
    rows_per_band = [np.where(bmap[:, 0] == i)[0] for i in range(len(DEFAULT_BANDS) - 1)]
    # Band 0 is nearest, so it must occupy the lowest rows in the image.
    assert rows_per_band[0].min() > rows_per_band[-1].max()


def test_band_edges_must_increase():
    with pytest.raises(ValueError, match="strictly increasing"):
        band_map(CAMVID_DEFAULT_CAMERA, (0.0, 40.0, 15.0))


# ---------------------------------------------------------------------------
# Stratified evaluation
# ---------------------------------------------------------------------------


def _blank(cam):
    return np.full((cam.image_height, cam.image_width), ROAD, dtype=np.int64)


def test_stratified_isolates_a_near_failure_from_a_far_one():
    """The whole point: the same number of wrong pixels, at different ranges."""
    cam = CAMVID_DEFAULT_CAMERA
    bmap = band_map(cam)
    near_rows = np.where(bmap[:, 0] == 0)[0]
    far_rows = np.where(bmap[:, 0] == 2)[0]

    # A pedestrian missed up close.
    t_near = _blank(cam)
    t_near[near_rows[0] : near_rows[0] + 5, :10] = PED
    p_near = _blank(cam)

    # An identically sized pedestrian missed far away.
    t_far = _blank(cam)
    t_far[far_rows[0] : far_rows[0] + 5, :10] = PED
    p_far = _blank(cam)

    near_res = StratifiedEvaluator(cam).update(t_near, p_near).result()
    far_res = StratifiedEvaluator(cam).update(t_far, p_far).result()

    labels = band_labels(DEFAULT_BANDS)
    # The near failure shows up only in the near band, and vice versa.
    assert near_res.per_band[labels[0]]["support"]["Pedestrian"] == 50
    assert near_res.per_band[labels[2]]["support"]["Pedestrian"] == 0
    assert far_res.per_band[labels[2]]["support"]["Pedestrian"] == 50
    assert far_res.per_band[labels[0]]["support"]["Pedestrian"] == 0


def test_stratified_bands_partition_the_ground_pixels():
    """No ground pixel counted twice, none lost."""
    cam = CAMVID_DEFAULT_CAMERA
    t = _blank(cam)
    ev = StratifiedEvaluator(cam).update(t, t.copy())
    res = ev.result()
    total = sum(b["pixels"] for b in res.per_band.values())
    expected = int(np.sum(band_map(cam) >= 0))
    assert total == expected


def test_stratified_refuses_a_shape_mismatch():
    """Silently resizing masks would corrupt the distance assignment."""
    ev = StratifiedEvaluator(CAMVID_DEFAULT_CAMERA)
    with pytest.raises(ValueError, match="does not match the camera model"):
        ev.update(np.zeros((100, 100), dtype=np.int64), np.zeros((100, 100), dtype=np.int64))


def test_result_carries_the_camera_assumptions():
    """A stratified number must never be quotable without its geometry."""
    res = StratifiedEvaluator(CAMVID_DEFAULT_CAMERA).update(
        _blank(CAMVID_DEFAULT_CAMERA), _blank(CAMVID_DEFAULT_CAMERA)
    ).result()
    assert "assumed" in res.camera["source"]
    assert res.camera["horizon_row"] == CAMVID_DEFAULT_CAMERA.horizon_row
