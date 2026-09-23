"""Contracts for top-label ECE rebuilt from the stored 65,536-level confidences.

The artifacts keep each pixel's top-1 confidence quantised to 65,536 levels and
whether its prediction was right. Fifteen equal-width bins line up exactly with
that grid, because 65,535 = 15 x 4,369, so every stored level falls in exactly
one bin with no rounding at the edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivemetrics.posthoc import allseed


def test_the_fifteen_bins_line_up_with_the_stored_confidence_grid() -> None:
    q16 = np.array([0, 4368, 4369, 8737, 8738, 61165, 61166, 65534, 65535], dtype=np.uint16)

    stats = allseed.toplabel_ece_components(q16, np.zeros(q16.size, dtype=bool))

    assert allseed.TOPLABEL_BINS == 15
    assert stats.shape == (15, 3)
    assert stats[:, 0].tolist() == [2, 2, 1] + [0] * 10 + [1, 3]


def test_a_perfectly_calibrated_toy_has_zero_error() -> None:
    q16 = np.array([0, 0, 65535, 65535], dtype=np.uint16)
    correct = np.array([False, False, True, True])

    assert allseed.toplabel_ece(allseed.toplabel_ece_components(q16, correct)) == 0.0


def test_the_error_is_the_pixel_weighted_gap_between_accuracy_and_confidence() -> None:
    q16 = np.array([32767, 32767, 65535, 65535], dtype=np.uint16)
    correct = np.array([True, False, False, True])

    stats = allseed.toplabel_ece_components(q16, correct)

    bin7_gap = abs(1 - 2 * 32767 / 65535)
    bin14_gap = abs(1 - 2 * 65535 / 65535)
    assert allseed.toplabel_ece(stats) == pytest.approx((bin7_gap + bin14_gap) / 4)
    assert stats[7].tolist() == [2, 2 * 32767 / 65535, 1]


def test_statistics_of_two_images_add_to_those_of_both_together() -> None:
    rng = np.random.default_rng(4)
    q16 = rng.integers(0, 65536, size=500).astype(np.uint16)
    correct = rng.random(500) < 0.7

    together = allseed.toplabel_ece_components(q16, correct)
    parts = allseed.toplabel_ece_components(q16[:200], correct[:200]) + (
        allseed.toplabel_ece_components(q16[200:], correct[200:])
    )

    assert np.allclose(together, parts)
    assert together[:, 0].sum() == 500
    assert together[:, 2].sum() == correct.sum()


def test_an_empty_histogram_has_no_error_to_report() -> None:
    with pytest.raises(ValueError, match="no pixel"):
        allseed.toplabel_ece(np.zeros((15, 3)))


def test_confidences_and_correctness_must_pair() -> None:
    with pytest.raises(ValueError, match="pair"):
        allseed.toplabel_ece_components(np.zeros(3, dtype=np.uint16), np.zeros(2, dtype=bool))
