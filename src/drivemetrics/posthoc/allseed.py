"""One verified read of every released prediction artifact, for the all-seed analysis.

The analysis is pre-registered in ``docs/posthoc/allseed-v1/analysis-plan.md``.
It re-aggregates the per-image artifacts that the released evaluation wrote for
the nine formal runs; no model is run and no released number is changed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
UInt16Array = npt.NDArray[np.uint16]
BoolArray = npt.NDArray[np.bool_]

ANALYSIS_PLAN = "docs/posthoc/allseed-v1/analysis-plan.md"
#: Top-label ECE bins. 65,535 = 15 x 4,369, so the bins line up exactly with the
#: stored 65,536-level confidence grid and no stored level straddles a bin edge.
TOPLABEL_BINS = 15
_CONFIDENCE_SCALE = 65535
_LEVELS_PER_BIN = _CONFIDENCE_SCALE // TOPLABEL_BINS


def toplabel_ece_components(q16: UInt16Array, correct: BoolArray) -> Float64Array:
    """Per-bin pixel count, confidence sum and correct count of one image's top-1 predictions."""

    if q16.shape != correct.shape:
        raise ValueError("the confidences and the correctness flags must pair pixel for pixel")
    levels = q16.astype(np.int64)
    bins = np.minimum(levels // _LEVELS_PER_BIN, TOPLABEL_BINS - 1)
    stats = np.zeros((TOPLABEL_BINS, 3), dtype=np.float64)
    stats[:, 0] = np.bincount(bins, minlength=TOPLABEL_BINS)
    stats[:, 1] = np.bincount(bins, weights=levels / _CONFIDENCE_SCALE, minlength=TOPLABEL_BINS)
    stats[:, 2] = np.bincount(bins[correct], minlength=TOPLABEL_BINS)
    return stats


def toplabel_ece(stats: Float64Array) -> float:
    """Pixel-weighted mean gap between accuracy and confidence over the bins."""

    total = float(stats[:, 0].sum())
    if total == 0:
        raise ValueError("the histogram holds no pixel to compute an error over")
    return float(np.abs(stats[:, 2] - stats[:, 1]).sum() / total)
