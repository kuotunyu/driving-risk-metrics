"""Replay of the released two-stage bootstrap draws, for intervals built from replicates.

The released :func:`~drivemetrics.analysis.bootstrap.two_stage_paired_bootstrap_statistic`
draws, for every resample, the images with replacement and then the seeds of
each model, and reports a 95% interval. The all-seed analysis needs the same
intervals at a second confidence level, and AURC intervals over 65,536-level
confidence histograms that the released function would have to hold in memory
for every resample. Both are computed here from run-level replicates of the SAME
draws: the generator is called in exactly the released order, the per-resample
sums use exactly the released arithmetic, and the seed stage and percentiles
are the released ones. The tests pin the equality bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from drivemetrics.analysis.bootstrap import (
    _combine_groups,
    _model_run_groups,
    _validate_combine,
    _validate_draw_settings,
    _validate_labels,
)

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.intp]

#: Integers up to this bound are represented exactly in float64, so a BLAS
#: product of integer-valued arrays below it is exact whatever order it sums in.
EXACT_FLOAT_INTEGER_LIMIT = 2**53


@dataclass(frozen=True)
class TwoStageDraws:
    """Every resample's image weights and, per model group, the runs its seed stage drew."""

    image_weights: Float64Array
    seed_draws: tuple[IndexArray, ...]
    groups: tuple[IndexArray, ...]


@dataclass(frozen=True)
class Interval:
    """One percentile interval and the confidence it was read at."""

    estimate: float
    low: float
    high: float
    confidence: float


def draw_two_stage(
    image_count: int,
    model_seed_ids: tuple[int, ...],
    *,
    resamples: int = 5000,
    seed: int = 20260831,
) -> TwoStageDraws:
    """Make the released function's draws, in its order, without computing anything."""

    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count <= 0:
        raise ValueError("image_count must be a positive integer")
    _validate_labels(model_seed_ids, len(model_seed_ids))
    _validate_draw_settings(resamples, seed)
    groups = _model_run_groups(model_seed_ids)
    generator = np.random.default_rng(seed)
    weights = np.empty((resamples, image_count), dtype=np.float64)
    seed_draws = [np.empty((resamples, group.size), dtype=np.intp) for group in groups]
    for index in range(resamples):
        image_draw = generator.integers(0, image_count, size=image_count)
        weights[index] = np.bincount(image_draw, minlength=image_count).astype(np.float64)
        for position, group in enumerate(groups):
            seed_draws[position][index] = group[generator.integers(0, group.size, size=group.size)]
    return TwoStageDraws(image_weights=weights, seed_draws=tuple(seed_draws), groups=groups)


def _checked_run_values(produced: object, run_count: int) -> Float64Array:
    values = np.asarray(produced, dtype=np.float64)
    if values.shape != (run_count,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"the statistic must return one finite value per run, expected {(run_count,)}"
        )
    return values


def run_replicates(
    draws: TwoStageDraws,
    components: Float64Array,
    statistic: Callable[[Float64Array], Float64Array],
) -> Float64Array:
    """Each resample's per-run statistic, summed exactly as the released function sums."""

    if components.ndim != 3 or components.shape[1] != draws.image_weights.shape[1]:
        raise ValueError("components must be run by image by component, over the drawn images")
    run_count = components.shape[0]
    replicates = np.empty((draws.image_weights.shape[0], run_count), dtype=np.float64)
    for index, image_weights in enumerate(draws.image_weights):
        summed = np.sum(components * image_weights[None, :, None], axis=1)
        replicates[index] = _checked_run_values(statistic(summed), run_count)
    return replicates


def histogram_run_replicates(
    draws: TwoStageDraws,
    counts: Int64Array,
    correct: Int64Array,
    statistic: Callable[[Int64Array, Int64Array], float],
    *,
    chunk: int = 250,
) -> Float64Array:
    """One run's statistic of the pooled histograms for every resample, pooled exactly.

    Pooling a resample is the image-weight vector times the image-by-level
    histograms. Every value is an integer below 2**53, so the matrix product is
    exact in float64 in any summation order, and the pooled histograms are the
    ones the released summation would give.
    """

    if counts.shape != correct.shape or counts.ndim != 2:
        raise ValueError("counts and correct must be image-by-level histograms of one shape")
    if counts.shape[0] != draws.image_weights.shape[1]:
        raise ValueError("the histograms must cover the drawn images")
    largest = int(counts.max(initial=0)) * int(draws.image_weights.max(initial=0))
    if largest * counts.shape[0] >= EXACT_FLOAT_INTEGER_LIMIT:
        raise ValueError("the pooled histograms could not be represented exactly in float64")
    dense_counts = counts.astype(np.float64)
    dense_correct = correct.astype(np.float64)
    resamples = draws.image_weights.shape[0]
    replicates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, chunk):
        weights = draws.image_weights[start : start + chunk]
        pooled_counts = np.rint(weights @ dense_counts).astype(np.int64)
        pooled_correct = np.rint(weights @ dense_correct).astype(np.int64)
        for offset in range(weights.shape[0]):
            replicates[start + offset] = statistic(pooled_counts[offset], pooled_correct[offset])
    return replicates


def interval_from_replicates(
    replicates: Float64Array,
    estimate_runs: Float64Array,
    model_seed_ids: tuple[int, ...],
    draws: TwoStageDraws,
    *,
    combine: str,
    confidence: float,
) -> Interval:
    """Apply the released seed stage and percentiles to run-level replicates."""

    _validate_combine(combine)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if replicates.shape != (draws.image_weights.shape[0], len(model_seed_ids)):
        raise ValueError("the replicates must hold one row per resample and one column per run")
    combined = np.empty(replicates.shape[0], dtype=np.float64)
    for index in range(replicates.shape[0]):
        drawn = tuple(seed_draw[index] for seed_draw in draws.seed_draws)
        combined[index] = _combine_groups(replicates[index], drawn, combine)
    tail_percent = (1.0 - confidence) * 50.0
    low, high = np.percentile(combined, [tail_percent, 100.0 - tail_percent])
    return Interval(
        estimate=_combine_groups(estimate_runs, draws.groups, combine),
        low=float(low),
        high=float(high),
        confidence=confidence,
    )
