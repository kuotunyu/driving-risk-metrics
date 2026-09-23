"""Contracts for replaying the released bootstrap draws.

The all-seed analysis needs intervals at more than one confidence level, and for
AURC over 65,536-level histograms that the released function cannot hold in
memory. Both are computed from run-level replicates of the SAME draws the
released two-stage bootstrap makes, so every new interval is the one the
released function would give. These tests pin that equality bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivemetrics.analysis.aggregate import _signed_difference_statistic
from drivemetrics.analysis.bootstrap import two_stage_paired_bootstrap_statistic
from drivemetrics.metrics.selective import area_under_risk_coverage, selective_risk_from_histogram
from drivemetrics.posthoc import draws

PAIR = (0, 0, 0, 1, 1, 1)
LEVEL = (0, 0, 0)


def ratio_statistic(summed: np.ndarray) -> np.ndarray:
    """A genuinely non-linear statistic of summed components."""

    return summed[:, 0] / summed[:, 1:].sum(axis=1)


def integer_components(runs: int, images: int, width: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 50, size=(runs, images, width)).astype(np.float64)


def released_and_replayed(components, labels, statistic, combine):
    released = two_stage_paired_bootstrap_statistic(
        components, labels, statistic, combine=combine, resamples=300, seed=7
    )
    replay = draws.draw_two_stage(components.shape[1], labels, resamples=300, seed=7)
    interval = draws.interval_from_replicates(
        draws.run_replicates(replay, components, statistic),
        statistic(np.sum(components, axis=1)),
        labels,
        replay,
        combine=combine,
        confidence=0.95,
    )
    return released, interval


def test_the_replayed_draws_reproduce_a_released_paired_interval_exactly() -> None:
    components = integer_components(6, 40, 5)
    signed = _signed_difference_statistic(ratio_statistic, PAIR)

    released, interval = released_and_replayed(components, PAIR, signed, "sum")

    assert interval.estimate == released.estimate
    assert interval.low == released.low
    assert interval.high == released.high
    assert interval.confidence == released.confidence


def test_the_replayed_draws_reproduce_a_released_level_interval_exactly() -> None:
    rng = np.random.default_rng(5)
    components = rng.uniform(0.1, 3.0, size=(3, 25, 4))

    released, interval = released_and_replayed(components, LEVEL, ratio_statistic, "mean")

    assert (interval.estimate, interval.low, interval.high) == (
        released.estimate,
        released.low,
        released.high,
    )


def test_a_narrower_confidence_uses_the_same_replicates() -> None:
    components = integer_components(3, 30, 3)
    replay = draws.draw_two_stage(30, LEVEL, resamples=400, seed=3)
    replicates = draws.run_replicates(replay, components, ratio_statistic)
    estimate = ratio_statistic(np.sum(components, axis=1))

    wide = draws.interval_from_replicates(
        replicates, estimate, LEVEL, replay, combine="mean", confidence=1 - 0.05 / 3
    )
    narrow = draws.interval_from_replicates(
        replicates, estimate, LEVEL, replay, combine="mean", confidence=0.95
    )

    assert wide.estimate == narrow.estimate
    assert wide.low <= narrow.low <= narrow.high <= wide.high
    assert wide.confidence == pytest.approx(1 - 0.05 / 3)


def aurc(counts: np.ndarray, correct: np.ndarray) -> float:
    coverage, risk = selective_risk_from_histogram(counts, correct)
    return area_under_risk_coverage(coverage, risk)


def test_histogram_replicates_equal_the_dense_replicates() -> None:
    rng = np.random.default_rng(9)
    counts = rng.integers(0, 40, size=(12, 8))
    correct = np.minimum(counts, rng.integers(0, 40, size=(12, 8)))
    replay = draws.draw_two_stage(12, LEVEL, resamples=50, seed=1)

    fast = draws.histogram_run_replicates(replay, counts, correct, aurc, chunk=7)
    dense = draws.run_replicates(
        replay,
        np.concatenate([counts, correct], axis=1)[None, :, :].astype(np.float64),
        lambda summed: np.array(
            [aurc(summed[0, :8].astype(np.int64), summed[0, 8:].astype(np.int64))]
        ),
    )

    assert np.array_equal(fast, dense[:, 0])


def test_histograms_beyond_exact_float_range_are_refused() -> None:
    counts = np.full((2, 3), 2**52, dtype=np.int64)
    replay = draws.draw_two_stage(2, LEVEL, resamples=2, seed=0)

    with pytest.raises(ValueError, match="exactly"):
        draws.histogram_run_replicates(replay, counts, counts, aurc)


def test_histograms_that_do_not_pair_are_refused() -> None:
    replay = draws.draw_two_stage(2, LEVEL, resamples=2, seed=0)

    with pytest.raises(ValueError, match="shape"):
        draws.histogram_run_replicates(
            replay, np.ones((2, 3), dtype=np.int64), np.ones((2, 4), dtype=np.int64), aurc
        )
    with pytest.raises(ValueError, match="images"):
        draws.histogram_run_replicates(
            replay, np.ones((3, 3), dtype=np.int64), np.ones((3, 3), dtype=np.int64), aurc
        )


def test_components_for_other_images_are_refused() -> None:
    replay = draws.draw_two_stage(4, LEVEL, resamples=2, seed=0)

    with pytest.raises(ValueError, match="images"):
        draws.run_replicates(replay, integer_components(3, 5, 2), ratio_statistic)


def test_a_statistic_must_return_one_finite_value_per_run() -> None:
    replay = draws.draw_two_stage(4, LEVEL, resamples=2, seed=0)
    components = integer_components(3, 4, 2)

    with pytest.raises(ValueError, match="one finite value per run"):
        draws.run_replicates(replay, components, lambda summed: summed[:1, 0])


def test_an_unknown_combination_is_refused() -> None:
    replay = draws.draw_two_stage(4, LEVEL, resamples=2, seed=0)

    with pytest.raises(ValueError, match="combine"):
        draws.interval_from_replicates(
            np.zeros((2, 3)), np.zeros(3), LEVEL, replay, combine="median", confidence=0.95
        )


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_confidence_must_lie_strictly_between_zero_and_one(confidence: float) -> None:
    replay = draws.draw_two_stage(4, LEVEL, resamples=2, seed=0)

    with pytest.raises(ValueError, match="confidence"):
        draws.interval_from_replicates(
            np.zeros((2, 3)), np.zeros(3), LEVEL, replay, combine="mean", confidence=confidence
        )


def test_replicates_must_match_the_draws() -> None:
    replay = draws.draw_two_stage(4, LEVEL, resamples=2, seed=0)

    with pytest.raises(ValueError, match="replicates"):
        draws.interval_from_replicates(
            np.zeros((3, 3)), np.zeros(3), LEVEL, replay, combine="mean", confidence=0.95
        )


def test_invalid_draw_settings_are_refused() -> None:
    with pytest.raises(ValueError):
        draws.draw_two_stage(0, LEVEL)
    with pytest.raises(ValueError):
        draws.draw_two_stage(4, LEVEL, resamples=0)
