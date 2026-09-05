"""Selective risk-coverage curves and the fixed risk-coverage area convention."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
Int64Array = npt.NDArray[np.int64]
Float64Array = npt.NDArray[np.float64]


def _validate_curve_inputs(confidence: Float64Array, correctness: BoolArray) -> None:
    if not isinstance(confidence, np.ndarray) or confidence.dtype != np.float64:
        raise ValueError("confidence must be a float64 array")
    if not isinstance(correctness, np.ndarray) or correctness.dtype != np.bool_:
        raise ValueError("correctness must be a boolean array")
    if confidence.ndim != 1 or correctness.ndim != 1:
        raise ValueError("confidence and correctness must be one-dimensional")
    if confidence.size != correctness.size:
        raise ValueError("confidence and correctness must have the same length")
    if confidence.size == 0:
        raise ValueError("confidence must contain at least one sample")
    if not np.all(np.isfinite(confidence)):
        raise ValueError("confidence values must be finite")


def _validate_area_inputs(coverage: Float64Array, risk: Float64Array) -> None:
    for name, array in (("coverage", coverage), ("risk", risk)):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise ValueError(f"{name} must be a float64 array")
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if coverage.size != risk.size:
        raise ValueError("coverage and risk must have the same length")
    if coverage.size < 2:
        raise ValueError("coverage must contain at least two points")
    if not np.all(np.diff(coverage) > 0.0):
        raise ValueError("coverage must be strictly increasing")
    if coverage[0] <= 0.0 or coverage[-1] > 1.0:
        raise ValueError("coverage values must lie in (0, 1]")
    if not np.all(np.isfinite(risk)) or np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("risk values must lie in [0, 1]")


def selective_risk_curve(
    confidence: Float64Array,
    correctness: BoolArray,
) -> tuple[Float64Array, Float64Array]:
    """Return the coverage grid ``1/N..1`` and the selective risk at each coverage.

    Samples are accepted in descending confidence order. Ties keep their original
    index order, so equally confident samples never move the curve because of the
    sort implementation. Only the ordering of ``confidence`` is interpreted; its
    scale and units are never assumed.
    """

    _validate_curve_inputs(confidence, correctness)
    sample_count = confidence.size
    order = np.argsort(-confidence, kind="stable")
    accepted = np.arange(1, sample_count + 1, dtype=np.float64)
    coverage = accepted / float(sample_count)
    risk = np.cumsum(~correctness[order], dtype=np.float64) / accepted
    return coverage, risk


def area_under_risk_coverage(coverage: Float64Array, risk: Float64Array) -> float:
    """Return the risk-coverage area under one fixed convention.

    The trapezoid rule is applied over the supplied coverage grid and divided by
    the covered span ``coverage[-1] - coverage[0]``. The result is a
    coverage-weighted mean selective risk in ``[0, 1]``, so curves built from
    different sample counts stay comparable: a fully correct model scores 0 and a
    fully incorrect model scores 1. A single coverage point has no span and is
    rejected instead of being divided by zero.
    """

    _validate_area_inputs(coverage, risk)
    span = float(coverage[-1] - coverage[0])
    area = float(np.trapezoid(risk, coverage)) / span
    # The validation above pins risk inside [0, 1] and coverage strictly increasing
    # inside (0, 1], so this coverage-weighted mean is mathematically inside [0, 1]
    # too. Only the trapezoid sum's rounding can push it out, and only by a unit in
    # the last place; clamping removes that and can hide no real excursion.
    return min(1.0, max(0.0, area))


def selective_risk_from_histogram(
    counts: Int64Array,
    correct_counts: Int64Array,
) -> tuple[Float64Array, Float64Array]:
    """Rebuild the risk-coverage curve from a confidence histogram, at bin boundaries.

    A locked cohort is nine hundred million pixels, and the artifacts retain a
    quantized histogram rather than a per-pixel confidence. That is enough for an
    exact curve wherever the acceptance threshold falls ON a bin boundary: every
    pixel above it is accepted and every pixel below it is not, so coverage and
    risk are both determined by the counts.

    Inside a bin the curve is NOT recoverable, because the pixels there are
    indistinguishable at the stored precision. The published curve is therefore
    defined at bin boundaries and nowhere else, which is a narrower claim than a
    per-pixel curve and is the honest one for this evidence.

    Bins are read most confident first. An empty bin contributes no point: it is
    not a coverage level anything can be evaluated at.
    """

    counts_array = np.asarray(counts, dtype=np.int64)
    correct_array = np.asarray(correct_counts, dtype=np.int64)
    if counts_array.ndim != 1 or counts_array.shape != correct_array.shape:
        raise ValueError("counts and correct counts must be flat arrays of the same length")
    if np.any(counts_array < 0) or np.any(correct_array < 0):
        raise ValueError("histogram counts must be nonnegative")
    if np.any(correct_array > counts_array):
        raise ValueError("correct counts must not exceed the bin counts")
    total = int(counts_array.sum())
    if total == 0:
        raise ValueError("the histogram contains no pixel to build a curve from")

    descending = counts_array[::-1]
    descending_correct = correct_array[::-1]
    occupied = descending > 0
    accepted = np.cumsum(descending[occupied], dtype=np.float64)
    wrong = np.cumsum((descending - descending_correct)[occupied], dtype=np.float64)
    return accepted / float(total), wrong / accepted
