"""Lossless calibration sufficient statistics and compact confidence codecs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]
UInt16Array = npt.NDArray[np.uint16]

_Q16_SCALE = 65535
PROBABILITY_ROW_SUM_ATOLERANCE = 1e-12


@dataclass(frozen=True)
class ECEBinSufficientStatistics:
    """Per-class fixed-bin counts and exact floating-point sums."""

    counts: Int64Array
    confidence_sums: Float64Array
    positive_counts: Int64Array


def quantize_confidence(probability: npt.NDArray[np.floating]) -> UInt16Array:
    """Encode probabilities on the declared reversible q16 scale."""

    if not isinstance(probability, np.ndarray) or not np.issubdtype(
        probability.dtype,
        np.floating,
    ):
        raise ValueError("probability must be a floating-point array")
    if not np.all(np.isfinite(probability)) or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probability values must be finite and within [0, 1]")
    scaled = probability.astype(np.float64) * _Q16_SCALE
    return np.rint(scaled).astype(np.uint16)


def dequantize_confidence(value: UInt16Array) -> Float64Array:
    """Decode q16 confidence values to float64 probabilities."""

    if not isinstance(value, np.ndarray) or value.dtype != np.uint16:
        raise ValueError("value must be a uint16 array")
    return value.astype(np.float64) / _Q16_SCALE


def pack_correctness(correct: BoolArray) -> bytes:
    """Pack a boolean correctness vector in little-bit order."""

    if not isinstance(correct, np.ndarray) or correct.dtype != np.bool_:
        raise ValueError("correct must be a boolean array")
    return np.packbits(correct.reshape(-1), bitorder="little").tobytes()


def unpack_correctness(data: bytes, pixel_count: int) -> BoolArray:
    """Unpack exactly one validated little-order correctness bitset."""

    if pixel_count < 0:
        raise ValueError("pixel_count must be nonnegative")
    expected_bytes = (pixel_count + 7) // 8
    if len(data) != expected_bytes:
        raise ValueError("correctness bitset byte length does not match pixel_count")
    remainder = pixel_count % 8
    if remainder and data[-1] & ~((1 << remainder) - 1):
        raise ValueError("correctness bitset has nonzero padding bits")
    packed = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little", count=pixel_count).astype(np.bool_)


def _validated_probabilities(
    probabilities: Float64Array,
    targets: Int64Array,
    num_classes: int,
) -> tuple[Float64Array, Int64Array]:
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.float64:
        raise ValueError("probabilities must be a float64 array")
    if not isinstance(targets, np.ndarray) or targets.dtype != np.int64:
        raise ValueError("targets must be an int64 array")
    if probabilities.ndim < 2 or probabilities.shape[-1] != num_classes:
        raise ValueError("num_classes must match the probabilities final axis")
    if probabilities.size == 0:
        raise ValueError("probabilities must contain at least one sample")
    if targets.shape != probabilities.shape[:-1]:
        raise ValueError("targets shape must match probabilities leading shape")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probability values must be finite and within [0, 1]")
    flattened = probabilities.reshape(-1, num_classes)
    if not np.allclose(
        flattened.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=PROBABILITY_ROW_SUM_ATOLERANCE,
    ):
        raise ValueError("probability rows must sum to one")
    flattened_targets = targets.reshape(-1)
    if np.any((flattened_targets < 0) | (flattened_targets >= num_classes)):
        raise ValueError("target is outside the probability class range")
    return flattened, flattened_targets


def multiclass_brier_sums(
    probabilities: Float64Array,
    targets: Int64Array,
    num_classes: int,
) -> Float64Array:
    """Return exact per-class squared-error sums for multiclass Brier score."""

    flattened, flattened_targets = _validated_probabilities(
        probabilities,
        targets,
        num_classes,
    )
    sums = np.empty(num_classes, dtype=np.float64)
    error_buffer = np.empty(flattened_targets.size, dtype=np.float64)
    for class_id in range(num_classes):
        np.copyto(error_buffer, flattened[:, class_id])
        error_buffer[flattened_targets == class_id] -= 1.0
        sums[class_id] = np.dot(error_buffer, error_buffer)
    return sums


def classwise_ece_sufficient_statistics(
    probabilities: Float64Array,
    targets: Int64Array,
    num_classes: int,
    bin_count: int = 15,
) -> ECEBinSufficientStatistics:
    """Accumulate one-vs-rest classwise ECE statistics in fixed equal-width bins."""

    if not isinstance(bin_count, int) or isinstance(bin_count, bool) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    flattened, flattened_targets = _validated_probabilities(
        probabilities,
        targets,
        num_classes,
    )
    counts = np.zeros((num_classes, bin_count), dtype=np.int64)
    confidence_sums = np.zeros((num_classes, bin_count), dtype=np.float64)
    positive_counts = np.zeros((num_classes, bin_count), dtype=np.int64)

    for class_id in range(num_classes):
        confidence = flattened[:, class_id]
        bins = np.minimum((confidence * bin_count).astype(np.int64), bin_count - 1)
        np.add.at(counts[class_id], bins, 1)
        np.add.at(confidence_sums[class_id], bins, confidence)
        np.add.at(positive_counts[class_id], bins, flattened_targets == class_id)

    return ECEBinSufficientStatistics(
        counts=counts,
        confidence_sums=confidence_sums,
        positive_counts=positive_counts,
    )
