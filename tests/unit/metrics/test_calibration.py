"""Behavior tests for calibration sufficient-statistic kernels."""

from __future__ import annotations

import importlib
from types import ModuleType

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
    "probabilities",
    [
        np.array([-0.01], dtype=np.float64),
        np.array([1.01], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
        np.array([np.inf], dtype=np.float64),
        np.array([1], dtype=np.int64),
    ],
)
def test_q16_confidence_rejects_invalid_probability(probabilities: np.ndarray) -> None:
    """Invalid confidence values must not silently wrap into plausible uint16 data."""

    module = load_calibration()

    with pytest.raises(ValueError, match="probability"):
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
        (b"", 9, "byte length"),
        (b"\x00\x00", 8, "byte length"),
        (b"\x00\x80", 9, "padding"),
        (b"", -1, "pixel_count"),
    ],
)
def test_correctness_bitset_rejects_truncation_extra_bytes_or_nonzero_padding(
    data: bytes,
    pixel_count: int,
    message: str,
) -> None:
    """A malformed bitset must fail instead of silently changing correctness counts."""

    module = load_calibration()

    with pytest.raises(ValueError, match=message):
        module.unpack_correctness(data, pixel_count)


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
        (np.ones((2, 2), dtype=np.float32) / 2, np.array([0, 1], dtype=np.int64), 2, "float64"),
        (np.ones((2, 2), dtype=np.float64) / 2, np.array([0, 1], dtype=np.int32), 2, "int64"),
        (np.ones((2, 2), dtype=np.float64) / 2, np.array([0], dtype=np.int64), 2, "shape"),
        (np.array([[np.nan, 0.0]], dtype=np.float64), np.array([0], dtype=np.int64), 2, "finite"),
        (np.array([[0.2, 0.2]], dtype=np.float64), np.array([0], dtype=np.int64), 2, "sum to one"),
        (np.array([[0.5, 0.5]], dtype=np.float64), np.array([2], dtype=np.int64), 2, "target"),
        (np.array([[0.5, 0.5]], dtype=np.float64), np.array([0], dtype=np.int64), 3, "num_classes"),
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
