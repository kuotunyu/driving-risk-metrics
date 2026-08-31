"""Selective risk-coverage curves and the fixed risk-coverage area convention."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
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
    return float(np.trapezoid(risk, coverage)) / span
