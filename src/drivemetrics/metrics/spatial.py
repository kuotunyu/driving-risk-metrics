"""Normalized image-band exposure; this is image space, not distance or depth."""

from __future__ import annotations

import numpy as np

from drivemetrics.metrics.confusion import Int64Array

NORMALIZED_IMAGE_BAND_NAMES = ("top", "middle", "bottom")


def normalized_image_bands(height: int) -> Int64Array:
    """Return top/middle/bottom row IDs with deterministic remainder placement.

    The output metric is named ``normalized_image_band``. Its values describe
    normalized image rows only and must never be interpreted as physical depth
    or metric distance.
    """

    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")

    base, remainder = divmod(height, len(NORMALIZED_IMAGE_BAND_NAMES))
    counts = [base + int(index < remainder) for index in range(3)]
    return np.repeat(np.arange(3, dtype=np.int64), counts)
