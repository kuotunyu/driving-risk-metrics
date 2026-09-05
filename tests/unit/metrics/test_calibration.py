"""Behavior tests for calibration sufficient-statistic kernels."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st


def load_calibration() -> ModuleType:
    try:
        return importlib.import_module("drivemetrics.metrics.calibration")
    except ImportError:
        pytest.fail("drivemetrics.metrics.calibration is missing", pytrace=False)


def test_q16_confidence_preserves_endpoints() -> None:
    """An endpoint mapping bug would turn certainty or impossibility into another value."""

    module = load_calibration()
    probabilities = np.array([0.0, 1.0], dtype=np.float64)

    encoded = module.quantize_confidence(probabilities)

    np.testing.assert_array_equal(encoded, np.array([0, 65535], dtype=np.uint16))
    np.testing.assert_array_equal(module.dequantize_confidence(encoded), probabilities)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_q16_confidence_preserves_common_model_output_dtypes(
    dtype: npt.DTypeLike,
) -> None:
    """Scaling lower-precision model output must happen without dtype overflow."""

    module = load_calibration()
    probabilities = np.array([0.0, 0.5, 1.0], dtype=dtype)

    encoded = module.quantize_confidence(probabilities)
    restored = module.dequantize_confidence(encoded)

    np.testing.assert_array_equal(encoded, np.array([0, 32768, 65535], dtype=np.uint16))
    assert np.max(np.abs(restored - probabilities.astype(np.float64))) <= 1.0 / 65535.0


@given(
    values=st.lists(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=1,
        max_size=100,
    )
)
def test_q16_round_trip_error_never_exceeds_one_quantization_step(
    values: list[float],
) -> None:
    """Changing the declared q16 scale must be detected by the stored precision bound."""

    module = load_calibration()
    probabilities = np.asarray(values, dtype=np.float64)

    restored = module.dequantize_confidence(module.quantize_confidence(probabilities))

    assert np.max(np.abs(restored - probabilities)) <= 1.0 / 65535.0


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([-0.01], dtype=np.float64), r"^probability values must be finite and within"),
        (np.array([1.01], dtype=np.float64), r"^probability values must be finite and within"),
        (np.array([np.nan], dtype=np.float64), r"^probability values must be finite and within"),
        (np.array([np.inf], dtype=np.float64), r"^probability values must be finite and within"),
        (np.array([1], dtype=np.int64), r"^probability must be a floating-point array"),
    ],
)
def test_q16_confidence_rejects_invalid_probability(
    probabilities: np.ndarray,
    message: str,
) -> None:
    """Invalid confidence values must not silently wrap into plausible uint16 data.

    Each row names its own message. ``match="probability"`` matched both of
    this function's messages, so it could not tell a dtype rejection from a
    range rejection, and it survived either message being rewritten.
    """

    module = load_calibration()

    with pytest.raises(ValueError, match=message):
        module.quantize_confidence(probabilities)


def test_dequantization_requires_uint16_storage() -> None:
    """Interpreting a signed or wider array would violate the declared artifact scale."""

    module = load_calibration()

    with pytest.raises(ValueError, match=r"^value must be a"):
        module.dequantize_confidence(np.array([0, 65535], dtype=np.int64))


def test_correctness_bitset_round_trips_a_nonbyte_multiple() -> None:
    """Dropping tail bits would corrupt images whose valid-pixel count is not divisible by eight."""

    module = load_calibration()
    correct = np.array(
        [True, False, True, True, False, False, True, False, True, True], dtype=np.bool_
    )

    packed = module.pack_correctness(correct)

    assert len(packed) == 2
    np.testing.assert_array_equal(module.unpack_correctness(packed, 10), correct)


@pytest.mark.parametrize(
    ("data", "pixel_count", "message"),
    [
        (b"", 9, r"^correctness bitset byte length does not match pixel_count"),
        (b"\x00\x00", 8, r"^correctness bitset byte length does not match pixel_count"),
        (b"\x00\x80", 9, r"^correctness bitset has nonzero padding bits"),
        (b"\x02", 1, r"^correctness bitset has nonzero padding bits"),
        (b"", -1, r"^pixel_count must be nonnegative"),
    ],
)
def test_correctness_bitset_rejects_truncation_extra_bytes_or_nonzero_padding(
    data: bytes,
    pixel_count: int,
    message: str,
) -> None:
    """A malformed bitset must fail instead of silently changing correctness counts.

    The ``b"\\x02"`` row is the tightest padding case in the table. With one
    pixel the remainder is 1, so bit 0 is data and every higher bit is padding;
    the mask ``~((1 << 1) - 1)`` catches bit 1, while ``~((2 << 1) - 1)`` masks
    off bits 0 and 1 together and lets the corrupt byte through. The existing
    ``b"\\x00\\x80"`` row cannot separate them, because bit 7 is outside both
    masks.
    """

    module = load_calibration()

    with pytest.raises(ValueError, match=message):
        module.unpack_correctness(data, pixel_count)


def test_a_zero_pixel_bitset_unpacks_to_an_empty_vector() -> None:
    """Zero is a valid pixel count and must not be refused as malformed.

    An image whose every pixel is ignored contributes no valid pixel, and the
    artifact still records it. A guard written ``<= 0`` or ``< 1`` instead of
    ``< 0`` would reject that record, and the failure would read as a corrupt
    bitset rather than as a validator that cannot count to zero.
    """

    module = load_calibration()

    unpacked = module.unpack_correctness(b"", 0)

    assert unpacked.shape == (0,)
    assert unpacked.dtype == np.bool_


def test_pack_correctness_requires_boolean_array() -> None:
    """Implicit truthiness would hide corrupt correctness values such as two."""

    module = load_calibration()

    with pytest.raises(ValueError, match=r"^correct must be a boolean array"):
        module.pack_correctness(np.array([0, 2], dtype=np.uint8))


def test_multiclass_brier_sums_reconstruct_hand_computed_total() -> None:
    """Reducing across the wrong axis would change the exact multiclass Brier score."""

    module = load_calibration()
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)

    sums = module.multiclass_brier_sums(probabilities, targets, num_classes=3)

    expected_by_class = np.array([0.10, 0.53, 0.37], dtype=np.float64)
    np.testing.assert_allclose(sums, expected_by_class, rtol=0.0, atol=1e-12)
    assert float(sums.sum()) == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_multiclass_brier_sums_does_not_materialize_a_full_one_hot_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segmentation-scale Brier accumulation must not duplicate probabilities as one-hot."""

    module = load_calibration()

    def forbid_full_one_hot(_: np.ndarray) -> np.ndarray:
        raise AssertionError("full one-hot allocation attempted")

    monkeypatch.setattr(module.np, "zeros_like", forbid_full_one_hot)
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)

    np.testing.assert_allclose(
        module.multiclass_brier_sums(probabilities, targets, num_classes=3),
        np.array([0.10, 0.53, 0.37], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_multiclass_brier_sums_avoids_cancellation_for_near_perfect_large_batch() -> None:
    """Large near-perfect cohorts must retain tiny squared errors within 1e-12."""

    module = load_calibration()
    sample_count = 200_000
    probabilities = np.repeat(
        np.array([[1.0 - 1e-8, 1e-8]], dtype=np.float64),
        sample_count,
        axis=0,
    )
    targets = np.zeros(sample_count, dtype=np.int64)
    expected = np.array(
        [
            np.square(probabilities[:, 0] - 1.0).sum(dtype=np.float64),
            np.square(probabilities[:, 1]).sum(dtype=np.float64),
        ]
    )

    np.testing.assert_allclose(
        module.multiclass_brier_sums(probabilities, targets, num_classes=2),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("probabilities", "targets", "num_classes", "message"),
    [
        (
            np.ones((2, 2), dtype=np.float32) / 2,
            np.array([0, 1], dtype=np.int64),
            2,
            r"^probabilities must be a float64 array",
        ),
        (
            np.ones((2, 2), dtype=np.float64) / 2,
            np.array([0, 1], dtype=np.int32),
            2,
            r"^targets must be an int64 array",
        ),
        (
            np.ones((2, 2), dtype=np.float64) / 2,
            np.array([0], dtype=np.int64),
            2,
            r"^targets shape must match probabilities leading shape",
        ),
        (
            np.array([[np.nan, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            2,
            r"^probability values must be finite and within",
        ),
        (
            np.array([[0.2, 0.2]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            2,
            r"^probability rows must sum to one",
        ),
        (
            np.array([[0.5, 0.5]], dtype=np.float64),
            np.array([2], dtype=np.int64),
            2,
            r"^target is outside the probability class range",
        ),
        (
            np.array([[0.5, 0.5]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            3,
            r"^num_classes must match the probabilities final axis",
        ),
    ],
)
def test_brier_sums_reject_malformed_probability_contract(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    message: str,
) -> None:
    """Malformed probability tensors cannot produce trustworthy sufficient statistics."""

    module = load_calibration()

    with pytest.raises(ValueError, match=message):
        module.multiclass_brier_sums(probabilities, targets, num_classes)


def test_classwise_ece_uses_deterministic_left_closed_bin_boundaries() -> None:
    """Values exactly on a boundary must not move between bins across implementations."""

    module = load_calibration()
    probabilities = np.array(
        [[0.00, 1.00], [0.25, 0.75], [0.50, 0.50], [0.75, 0.25], [1.00, 0.00]],
        dtype=np.float64,
    )
    targets = np.array([1, 1, 0, 0, 0], dtype=np.int64)

    stats = module.classwise_ece_sufficient_statistics(
        probabilities,
        targets,
        num_classes=2,
        bin_count=4,
    )

    np.testing.assert_array_equal(
        stats.counts,
        np.array([[1, 1, 1, 2], [1, 1, 1, 2]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        stats.confidence_sums,
        np.array([[0.0, 0.25, 0.5, 1.75], [0.0, 0.25, 0.5, 1.75]]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        stats.positive_counts,
        np.array([[0, 0, 1, 2], [0, 0, 0, 2]], dtype=np.int64),
    )


@pytest.mark.parametrize("bin_count", [0, -1, 2.0, True])
def test_classwise_ece_requires_positive_integer_bin_count(bin_count: object) -> None:
    """A malformed bin count cannot define a trustworthy calibration table."""

    module = load_calibration()

    with pytest.raises(ValueError, match=r"^bin_count must be a positive integer"):
        module.classwise_ece_sufficient_statistics(
            np.array([[0.5, 0.5]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            num_classes=2,
            bin_count=bin_count,
        )


@pytest.mark.parametrize("num_classes", [0, -1, 2.0, True])
def test_probability_metrics_require_positive_integer_num_classes(
    num_classes: object,
) -> None:
    """Invalid class counts must produce one stable contract error, not NumPy internals."""

    module = load_calibration()

    with pytest.raises(ValueError, match=r"^num_classes must be a positive integer"):
        module.multiclass_brier_sums(
            np.array([[0.5, 0.5]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            num_classes,
        )


def test_probability_metrics_reject_empty_batches() -> None:
    """An empty batch has no metric evidence and must not yield zero-valued statistics."""

    module = load_calibration()

    with pytest.raises(ValueError, match=r"^probabilities must contain at least one sample"):
        module.multiclass_brier_sums(
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            num_classes=2,
        )


def test_the_correctness_bitset_has_a_pinned_wire_format() -> None:
    """The packed bytes are a published contract, not an internal detail.

    A round-trip through this module's own pack and unpack proves only that
    the two agree with each other; both could change together and every such
    test would still pass while every artifact already on disk became
    unreadable. The bytes are therefore asserted directly.

    Hand computed: with little bit order, element i is bit i of the byte, so
    [T, F, T, T, F, F, T, F] is 1 + 4 + 8 + 64 = 77 = 0x4D. Big bit order
    would put element 0 in the high bit and give 0xB2 instead.
    """

    module = load_calibration()
    correct = np.array([True, False, True, True, False, False, True, False], dtype=np.bool_)

    assert module.pack_correctness(correct) == b"\x4d"
    np.testing.assert_array_equal(module.unpack_correctness(b"\x4d", 8), correct)


def test_out_of_range_probabilities_are_refused_even_when_the_row_still_sums_to_one() -> None:
    """The range guard has to hold on its own, not lean on the row-sum check.

    `[1.5, -0.5]` sums to exactly 1.0, so the row-sum check is satisfied and
    the only thing standing between it and the metric is the range test. If
    that test were written `(p < 0.0) & (p > 1.0)` instead of `|`, no value
    could ever satisfy both halves, the guard would never fire, and a
    probability of 1.5 would flow into a published Brier score.

    The assertion names the range message specifically. A test that accepted
    any ValueError would pass on the row-sum message and prove nothing.
    """

    module = load_calibration()
    probabilities = np.array([[1.5, -0.5]], dtype=np.float64)
    targets = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match=r"^probability values must be finite and within"):
        module.multiclass_brier_sums(probabilities, targets, num_classes=2)


def test_a_value_above_one_is_refused_by_the_range_check_not_the_row_sum() -> None:
    """The upper bound must be one, and the range check must be what rejects it.

    The neighbouring `[1.5, -0.5]` case cannot prove this. Its negative entry
    satisfies the left half of `(p < 0.0) | (p > 1.0)`, so the guard fires
    whatever the right half says, and an upper bound loosened to 2.0 rejects
    that row just as the correct one does.

    `[1.5, 0.0]` has no negative entry, so only the upper bound can refuse it.
    With the bound at 1.0 the range message is raised. With the bound at 2.0
    the row passes the range check and is instead caught downstream by the
    row-sum check, which raises a different message — so naming the message is
    what makes this test able to tell the two apart.
    """

    module = load_calibration()
    probabilities = np.array([[1.5, 0.0]], dtype=np.float64)
    targets = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match=r"^probability values must be finite and within"):
        module.multiclass_brier_sums(probabilities, targets, num_classes=2)


@pytest.mark.parametrize(
    "excess",
    [5e-6, 5e-9],
    ids=["inside numpy's default rtol", "inside numpy's default atol"],
)
def test_row_sums_are_held_to_the_declared_absolute_tolerance_alone(excess: float) -> None:
    """`np.allclose` defaults would accept rows this project refuses.

    The call passes `rtol=0.0` and `atol=PROBABILITY_ROW_SUM_ATOLERANCE`
    (1e-12) deliberately: a relative tolerance on a quantity whose true value
    is exactly 1.0 is a disguised absolute tolerance of 1e-5, which is eight
    orders of magnitude looser than the contract.

    Each row here sits inside one of numpy's defaults and far outside 1e-12,
    and every value stays within [0, 1], so the row-sum check is the only one
    that can fire. Dropping either argument makes at least one row pass.
    """

    module = load_calibration()
    probabilities = np.array([[0.5 + excess / 2, 0.5 + excess / 2]], dtype=np.float64)
    targets = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match=r"^probability rows must sum to one"):
        module.multiclass_brier_sums(probabilities, targets, num_classes=2)


def test_leading_axes_are_flattened_and_the_class_axis_is_the_last_one() -> None:
    """Per-image probabilities arrive with image axes; only the final axis is classes.

    A `(2, 3, 4)` batch must give exactly what the same rows give as `(6, 4)`.
    Three separate contract lines are exercised at once, and each is invisible
    to a two-dimensional input because `shape[1]` and `shape[-1]` coincide
    there, as do `shape[:-1]` and `shape[:1]`:

    * the class count is compared against the FINAL axis, not the second one;
    * the targets shape is compared against ALL leading axes, not the first;
    * the targets are FLATTENED before use, so the boolean mask that selects
      the true class matches the flattened probability rows.
    """

    module = load_calibration()
    rng = np.random.default_rng(20260904)
    raw = rng.random((6, 4))
    flat_probabilities = raw / raw.sum(axis=1, keepdims=True)
    flat_targets = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)

    nested = module.multiclass_brier_sums(
        flat_probabilities.reshape(2, 3, 4),
        flat_targets.reshape(2, 3),
        num_classes=4,
    )
    flat = module.multiclass_brier_sums(flat_probabilities, flat_targets, num_classes=4)

    np.testing.assert_allclose(nested, flat, rtol=0.0, atol=0.0)


def test_a_single_class_problem_is_accepted() -> None:
    """One class is a degenerate but well-formed contract, and the guard says so.

    `num_classes` is refused when it is not positive. A guard written `<= 1`
    would additionally refuse the one-class case, whose only valid probability
    row is `[1.0]`, and the refusal would read as a malformed batch rather than
    as a validator that cannot count to one.
    """

    module = load_calibration()

    sums = module.multiclass_brier_sums(
        np.array([[1.0], [1.0]], dtype=np.float64),
        np.array([0, 0], dtype=np.int64),
        num_classes=1,
    )

    np.testing.assert_allclose(sums, np.array([0.0]), rtol=0.0, atol=0.0)


def test_the_default_bin_count_is_the_fifteen_the_protocol_declares() -> None:
    """The bin count decides the ECE value, so its default is part of the protocol.

    `bin_count` has a default so a caller may vary it, which means nothing
    pinned what the default IS. A run that quietly used sixteen bins would
    reproduce from its own record and not from the protocol, and the two ECE
    numbers are not comparable.
    """

    module = load_calibration()

    stats = module.classwise_ece_sufficient_statistics(
        np.array([[0.5, 0.5]], dtype=np.float64),
        np.array([0], dtype=np.int64),
        num_classes=2,
    )

    assert stats.counts.shape == (2, 15)
    assert stats.confidence_sums.shape == (2, 15)
    assert stats.positive_counts.shape == (2, 15)


def test_a_single_bin_is_a_valid_calibration_table() -> None:
    """One bin is the degenerate table whose reliability gap is the overall gap.

    A guard written `<= 1` instead of `<= 0` would refuse it, and the refusal
    would read as a malformed request rather than as an off-by-one bound.
    """

    module = load_calibration()

    stats = module.classwise_ece_sufficient_statistics(
        np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float64),
        np.array([1, 0], dtype=np.int64),
        num_classes=2,
        bin_count=1,
    )

    np.testing.assert_array_equal(stats.counts, np.array([[2], [2]], dtype=np.int64))
    np.testing.assert_allclose(stats.confidence_sums, np.array([[1.0], [1.0]]))
    np.testing.assert_array_equal(stats.positive_counts, np.array([[1], [1]], dtype=np.int64))


def test_the_brier_score_divides_the_summed_squared_error_by_the_pixels_it_covers() -> None:
    """The artifact stores sums so cohorts can be added; the score is the sum over its own N.

    Sums from two images add exactly, and so do their pixel counts, which is the
    only reason a cohort score can be recomputed years later from artifacts that
    never stored a probability. Dividing by anything else — the number of
    classes, or one image's count applied to two — silently rescales a published
    number in a direction no reader could detect.
    """

    module = load_calibration()
    first = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64)
    second = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    first_targets = np.array([0, 1], dtype=np.int64)
    second_targets = np.array([2], dtype=np.int64)

    cohort = module.multiclass_brier_sums(
        first, first_targets, num_classes=3
    ) + module.multiclass_brier_sums(second, second_targets, num_classes=3)
    whole = module.multiclass_brier_sums(
        np.concatenate([first, second]),
        np.concatenate([first_targets, second_targets]),
        num_classes=3,
    )

    np.testing.assert_allclose(cohort, whole, rtol=0.0, atol=1e-12)
    assert module.multiclass_brier_score(cohort, 3) == pytest.approx(
        float(whole.sum()) / 3.0, rel=0.0, abs=1e-12
    )


def test_a_brier_score_over_no_pixels_is_none_rather_than_zero() -> None:
    """Zero squared error is a perfect score, so an empty cohort must not report it."""

    module = load_calibration()

    assert module.multiclass_brier_score(np.zeros(3, dtype=np.float64), 0) is None


def test_the_classwise_ece_weights_each_bin_by_how_many_pixels_landed_in_it() -> None:
    """Hand-computed on two bins so the weighting, not just the gap, is pinned.

    One-vs-rest for class 0 over four pixels with confidences 0.1, 0.1, 0.9, 0.9
    and targets 0, 1, 0, 0. With fifteen bins the first pair lands in bin 1 and
    the second in bin 13. Bin 1 holds two pixels, mean confidence 0.1, one of
    them positive, so its gap is |0.5 - 0.1| = 0.4. Bin 13 holds two, mean
    confidence 0.9, both positive, so its gap is |1.0 - 0.9| = 0.1. Each bin
    carries half the pixels, so the class ECE is 0.5*0.4 + 0.5*0.1 = 0.25.
    Weighting the bins equally instead of by occupancy gives the same answer
    here only because the halves are equal, which is why the second case below
    makes them unequal.
    """

    module = load_calibration()
    probabilities = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]], dtype=np.float64)
    targets = np.array([0, 1, 0, 0], dtype=np.int64)

    statistics = module.classwise_ece_sufficient_statistics(probabilities, targets, 2)
    per_class = module.classwise_expected_calibration_error(statistics)

    assert per_class[0] == pytest.approx(0.25, rel=0.0, abs=1e-12)


def test_an_unequally_occupied_bin_pulls_the_classwise_ece_towards_it() -> None:
    """Three pixels in one bin and one in another must not count equally."""

    module = load_calibration()
    probabilities = np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
    targets = np.array([0, 1, 1, 0], dtype=np.int64)

    statistics = module.classwise_ece_sufficient_statistics(probabilities, targets, 2)
    per_class = module.classwise_expected_calibration_error(statistics)

    # Bin 1 holds three pixels, mean confidence 0.1, one positive: |1/3 - 0.1|.
    # Bin 13 holds one, confidence 0.9, positive: |1.0 - 0.9|.
    expected = 0.75 * abs(1.0 / 3.0 - 0.1) + 0.25 * 0.1
    assert per_class[0] == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_a_class_no_pixel_ever_saw_reports_none_rather_than_a_perfect_zero() -> None:
    """An unsupported class is unmeasured, and zero calibration error is the best score."""

    module = load_calibration()
    empty = module.ECEBinSufficientStatistics(
        counts=np.zeros((2, 15), dtype=np.int64),
        confidence_sums=np.zeros((2, 15), dtype=np.float64),
        positive_counts=np.zeros((2, 15), dtype=np.int64),
    )

    assert module.classwise_expected_calibration_error(empty) == (None, None)
    assert module.mean_classwise_expected_calibration_error(empty) is None


def test_the_mean_classwise_ece_averages_only_the_classes_that_have_support() -> None:
    """Averaging an unmeasured class in as zero would flatter every reported number."""

    module = load_calibration()
    probabilities = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
    targets = np.array([1, 0], dtype=np.int64)

    statistics = module.classwise_ece_sufficient_statistics(probabilities, targets, 2)
    per_class = module.classwise_expected_calibration_error(statistics)
    supported = [value for value in per_class if value is not None]

    assert module.mean_classwise_expected_calibration_error(statistics) == pytest.approx(
        sum(supported) / len(supported), rel=0.0, abs=1e-12
    )


def test_the_ece_statistics_of_two_images_add() -> None:
    """The cohort ECE is finalised from summed statistics, never from averaged ECEs."""

    module = load_calibration()
    first = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
    second = np.array([[0.3, 0.7]], dtype=np.float64)
    first_targets = np.array([1, 0], dtype=np.int64)
    second_targets = np.array([1], dtype=np.int64)

    a = module.classwise_ece_sufficient_statistics(first, first_targets, 2)
    b = module.classwise_ece_sufficient_statistics(second, second_targets, 2)
    summed = module.ECEBinSufficientStatistics(
        counts=a.counts + b.counts,
        confidence_sums=a.confidence_sums + b.confidence_sums,
        positive_counts=a.positive_counts + b.positive_counts,
    )
    whole = module.classwise_ece_sufficient_statistics(
        np.concatenate([first, second]),
        np.concatenate([first_targets, second_targets]),
        2,
    )

    assert module.classwise_expected_calibration_error(
        summed
    ) == module.classwise_expected_calibration_error(whole)


@pytest.mark.parametrize(
    ("sums", "pixels", "expected"),
    [
        (np.zeros((2, 2)), 4, r"^brier_sum_by_class must be a flat array, got shape "),
        (np.array([np.nan]), 4, r"^brier_sum_by_class must contain finite nonnegative values$"),
        (np.array([-1.0]), 4, r"^brier_sum_by_class must contain finite nonnegative values$"),
        (np.array([1.0]), 4.0, r"^valid_pixel_count must be an integer$"),
        (np.array([1.0]), True, r"^valid_pixel_count must be an integer$"),
        (np.array([1.0]), -1, r"^valid_pixel_count must be nonnegative$"),
    ],
)
def test_a_brier_score_refuses_input_it_cannot_be_computed_from(
    sums: npt.NDArray[np.float64], pixels: object, expected: str
) -> None:
    """Each refusal names its own subject, because they are different mistakes.

    A negative squared-error sum and a float pixel count arrive from different
    places — a corrupted artifact and a caller that divided somewhere it should
    not have — and a caller told only that "the input was bad" cannot act.
    """

    module = load_calibration()

    with pytest.raises(ValueError, match=expected):
        module.multiclass_brier_score(sums, pixels)


@pytest.mark.parametrize(
    ("counts", "sums", "positives", "expected"),
    [
        (
            np.zeros((2, 3), dtype=np.int64),
            np.zeros((2, 4)),
            np.zeros((2, 3), dtype=np.int64),
            r"^ECE statistics must share one class-by-bin shape$",
        ),
        (
            np.zeros(3, dtype=np.int64),
            np.zeros(3),
            np.zeros(3, dtype=np.int64),
            r"^ECE statistics must share one class-by-bin shape$",
        ),
        (
            np.full((1, 2), -1, dtype=np.int64),
            np.zeros((1, 2)),
            np.zeros((1, 2), dtype=np.int64),
            r"^ECE counts must be nonnegative$",
        ),
        (
            np.ones((1, 2), dtype=np.int64),
            np.zeros((1, 2)),
            np.full((1, 2), 2, dtype=np.int64),
            r"^ECE positive_counts must not exceed counts$",
        ),
        (
            np.ones((1, 2), dtype=np.int64),
            np.full((1, 2), np.nan),
            np.ones((1, 2), dtype=np.int64),
            r"^ECE confidence_sums must be finite$",
        ),
    ],
)
def test_the_ece_finaliser_refuses_statistics_that_cannot_be_a_histogram(
    counts: npt.NDArray[np.int64],
    sums: npt.NDArray[np.float64],
    positives: npt.NDArray[np.int64],
    expected: str,
) -> None:
    """These arrive from a file, so the finaliser checks them rather than trusting them.

    More positives than pixels in a bin is not a small error: it makes the gap
    negative and would report a model as better calibrated than perfect.
    """

    module = load_calibration()
    statistics = module.ECEBinSufficientStatistics(
        counts=counts, confidence_sums=sums, positive_counts=positives
    )

    with pytest.raises(ValueError, match=expected):
        module.classwise_expected_calibration_error(statistics)


def test_ece_statistics_are_normalised_to_int64_and_float64_whatever_they_arrive_as() -> None:
    """The validator's return types are its contract, not a restatement of its inputs.

    Counts may arrive as small unsigned integers and sums as float32 from a caller
    that never touched this project's artifacts. The validator promises int64 and
    float64 back, and every finaliser downstream divides on that promise.
    """

    from drivemetrics.metrics import calibration

    # Declared Any on purpose: the static type says int64, the runtime normalises
    # whatever arrives, and this test pins the runtime promise.
    counts: Any = np.array([[4, 0], [2, 1]], dtype=np.uint8)
    sums: Any = np.array([[2.5, 0.0], [1.25, 0.75]], dtype=np.float32)
    positives: Any = np.array([[3, 0], [1, 1]], dtype=np.uint8)

    got_counts, got_sums, got_positives = calibration._validated_ece(
        calibration.ECEBinSufficientStatistics(
            counts=counts, confidence_sums=sums, positive_counts=positives
        )
    )

    assert got_counts.dtype == np.int64
    assert got_sums.dtype == np.float64
    assert got_positives.dtype == np.int64
    assert got_counts.tolist() == counts.tolist()
    assert got_positives.tolist() == positives.tolist()


def test_negative_positive_counts_are_refused() -> None:
    """A negative count of correct pixels is not a statistic; it is corruption."""

    from drivemetrics.metrics import calibration

    with pytest.raises(ValueError, match=r"^ECE counts must be nonnegative$"):
        calibration._validated_ece(
            calibration.ECEBinSufficientStatistics(
                counts=np.array([[2]], dtype=np.int64),
                confidence_sums=np.array([[1.0]], dtype=np.float64),
                positive_counts=np.array([[-1]], dtype=np.int64),
            )
        )


def test_the_brier_score_is_summed_in_float64_whatever_the_sums_arrive_as() -> None:
    """Summing nineteen float32 values in float32 and in float64 gives different numbers.

    The sums are promoted before they are added, so the published score is the
    float64 sum of the values as given. The naive float32 sum is checked to differ
    first, so this test is known to distinguish the two paths rather than assumed to.
    """

    from drivemetrics.metrics import calibration

    sums: Any = np.full(19, 0.1, dtype=np.float32)  # float32 on purpose; see above
    promoted = float(np.asarray(sums, dtype=np.float64).sum()) / 7.0
    assert float(sums.sum()) / 7.0 != promoted

    assert calibration.multiclass_brier_score(sums, 7) == promoted
