"""Deterministic two-stage paired bootstrap over validation images and seeds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]

BOOTSTRAP_CONFIDENCE = 0.95


@dataclass(frozen=True)
class BootstrapInterval:
    """One percentile interval with the exact settings that reproduce it."""

    estimate: float
    low: float
    high: float
    confidence: float
    resamples: int
    seed: int


def _validate_labels(model_seed_ids: tuple[int, ...], run_count: int) -> None:
    if len(model_seed_ids) != run_count:
        raise ValueError("model_seed_ids must label exactly one model per run")
    for model_id in model_seed_ids:
        if isinstance(model_id, bool) or not isinstance(model_id, int):
            raise TypeError("model_seed_ids must contain integers")


def _validate_draw_settings(resamples: int, seed: int) -> None:
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _validate_inputs(
    values: Float64Array,
    model_seed_ids: tuple[int, ...],
    resamples: int,
    seed: int,
) -> None:
    if not isinstance(values, np.ndarray) or values.dtype != np.float64:
        raise ValueError("values must be a float64 array")
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional run-by-image array")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("values must contain at least one run and one image")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    _validate_labels(model_seed_ids, values.shape[0])
    _validate_draw_settings(resamples, seed)


def _model_run_groups(model_seed_ids: tuple[int, ...]) -> tuple[IndexArray, ...]:
    labels = np.asarray(model_seed_ids, dtype=np.int64)
    return tuple(np.flatnonzero(labels == model_id) for model_id in np.unique(labels))


def _nested_mean(run_means: Float64Array, groups: tuple[IndexArray, ...]) -> float:
    return float(np.mean([float(run_means[group].mean()) for group in groups]))


def two_stage_paired_bootstrap_statistic(
    components: Float64Array,
    model_seed_ids: tuple[int, ...],
    statistic: Callable[[Float64Array], Float64Array],
    *,
    resamples: int = 5000,
    seed: int = 20260831,
) -> BootstrapInterval:
    """Bootstrap a metric that is a function of summed per-image components.

    Cohort mIoU is a ratio of summed confusions, not a mean of per-image mIoU.
    Those are different numbers, and a bootstrap over per-image scalars answers
    the second question while looking like it answered the first. This estimator
    resamples the components, sums them, and only then applies ``statistic``, so
    a ratio metric is recomputed the way the cohort metric is defined.

    ``components`` is run by image by component, and ``statistic`` maps the
    summed run-by-component array to one value per run. The image and seed draws
    are generated in exactly the same order as
    :func:`two_stage_paired_bootstrap`, so for a linear statistic the two agree
    to the last bit and their intervals can be reported side by side.
    """

    if not isinstance(components, np.ndarray) or components.dtype != np.float64:
        raise ValueError("components must be a float64 array")
    if components.ndim != 3:
        raise ValueError("components must be a three-dimensional run-by-image-by-component array")
    if components.shape[0] == 0 or components.shape[1] == 0 or components.shape[2] == 0:
        raise ValueError("components must contain at least one run, image, and component")
    if not np.all(np.isfinite(components)):
        raise ValueError("components must be finite")
    _validate_labels(model_seed_ids, components.shape[0])
    _validate_draw_settings(resamples, seed)

    groups = _model_run_groups(model_seed_ids)
    image_count = components.shape[1]
    generator = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        image_draw = generator.integers(0, image_count, size=image_count)
        image_weights = np.bincount(image_draw, minlength=image_count).astype(np.float64)
        summed = np.sum(components * image_weights[None, :, None], axis=1)
        run_values = _applied_statistic(statistic, summed, components.shape[0])
        seed_draws = tuple(
            group[generator.integers(0, group.size, size=group.size)] for group in groups
        )
        replicates[index] = _nested_mean(run_values, seed_draws)

    tail_percent = (1.0 - BOOTSTRAP_CONFIDENCE) * 50.0
    low, high = np.percentile(replicates, [tail_percent, 100.0 - tail_percent])
    estimate = _applied_statistic(statistic, np.sum(components, axis=1), components.shape[0])
    return BootstrapInterval(
        estimate=_nested_mean(estimate, groups),
        low=float(low),
        high=float(high),
        confidence=BOOTSTRAP_CONFIDENCE,
        resamples=resamples,
        seed=seed,
    )


def _applied_statistic(
    statistic: Callable[[Float64Array], Float64Array],
    summed: Float64Array,
    run_count: int,
) -> Float64Array:
    """Apply the caller statistic and refuse anything that is not usable."""

    produced = np.asarray(statistic(summed), dtype=np.float64)
    if produced.shape != (run_count,):
        raise ValueError(
            f"statistic must return one value per run, expected {(run_count,)} "
            f"but got {produced.shape}"
        )
    if not np.all(np.isfinite(produced)):
        raise ValueError("statistic must return finite values")
    return produced


def two_stage_paired_bootstrap(
    values: Float64Array,
    model_seed_ids: tuple[int, ...],
    *,
    resamples: int = 5000,
    seed: int = 20260831,
) -> BootstrapInterval:
    """Return the paired image-and-seed percentile interval for a run-by-image metric.

    ``values`` holds one row per model/seed run and one column per validation
    image, and ``model_seed_ids`` names the model each run belongs to. Every
    replicate draws the validation images once and applies that single draw to
    all runs, so paired differences between runs survive resampling. Seeds are
    then drawn with replacement inside each model, in ascending model-ID order.
    The estimand is the unweighted mean over models of the mean over each model
    seeds of that run mean over images, so a model with more seeds cannot
    dominate the result. Each replicate reduces the resampled images with an
    explicit NumPy summation rather than a BLAS product, so the interval does not
    depend on the local linear-algebra backend. The reported bounds are the
    linearly interpolated percentiles of the replicate distribution at
    ``BOOTSTRAP_CONFIDENCE``.
    """

    _validate_inputs(values, model_seed_ids, resamples, seed)
    groups = _model_run_groups(model_seed_ids)
    image_count = values.shape[1]
    generator = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        image_draw = generator.integers(0, image_count, size=image_count)
        image_weights = np.bincount(image_draw, minlength=image_count).astype(np.float64)
        run_means = np.sum(values * image_weights, axis=1) / image_count
        seed_draws = tuple(
            group[generator.integers(0, group.size, size=group.size)] for group in groups
        )
        replicates[index] = _nested_mean(run_means, seed_draws)

    tail_percent = (1.0 - BOOTSTRAP_CONFIDENCE) * 50.0
    low, high = np.percentile(replicates, [tail_percent, 100.0 - tail_percent])
    return BootstrapInterval(
        estimate=_nested_mean(values.mean(axis=1), groups),
        low=float(low),
        high=float(high),
        confidence=BOOTSTRAP_CONFIDENCE,
        resamples=resamples,
        seed=seed,
    )
