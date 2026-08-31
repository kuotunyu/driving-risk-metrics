"""Contracts for normalized image-space exposure bands."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest


def load_spatial_module() -> ModuleType:
    try:
        from drivemetrics.metrics import spatial
    except ImportError:
        pytest.fail("drivemetrics.metrics.spatial is missing", pytrace=False)
    return spatial


@pytest.mark.parametrize(
    ("height", "expected"),
    [
        (9, [0, 0, 0, 1, 1, 1, 2, 2, 2]),
        (8, [0, 0, 0, 1, 1, 1, 2, 2]),
        (7, [0, 0, 0, 1, 1, 2, 2]),
        (2, [0, 1]),
        (1, [0]),
    ],
)
def test_height_remainders_are_assigned_top_then_middle_then_bottom(
    height: int, expected: list[int]
) -> None:
    spatial = load_spatial_module()

    result = spatial.normalized_image_bands(height)

    np.testing.assert_array_equal(result, np.array(expected, dtype=np.int64))
    assert result.dtype == np.int64
    assert result.shape == (height,)


@pytest.mark.parametrize("height", [0, -1, True, 1.5])
def test_normalized_image_bands_rejects_nonpositive_or_noninteger_height(
    height: object,
) -> None:
    spatial = load_spatial_module()

    with pytest.raises((TypeError, ValueError), match="positive integer"):
        spatial.normalized_image_bands(height)  # type: ignore[arg-type]


def test_metrics_package_exports_normalized_image_bands() -> None:
    import drivemetrics.metrics as metrics

    assert metrics.normalized_image_bands is load_spatial_module().normalized_image_bands
