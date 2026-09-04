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


def multiclass_brier_score(
    brier_sum_by_class: Float64Array,
    valid_pixel_count: int,
) -> float | None:
    """Return the multiclass Brier score from summed squared error over N pixels.

    The artifacts store per-class SUMS rather than a score, because sums from two
    images add exactly and a score does not. This is the only reason a cohort
    number can be recomputed years later from artifacts that never retained a
    probability: add the sums, add the counts, divide once.

    ``None`` for an empty cohort. Zero squared error is a perfect score, so
    returning zero for "nothing was measured" would make the best outcome
    indistinguishable from no outcome at all.
    """

    sums = np.asarray(brier_sum_by_class, dtype=np.float64)
    if sums.ndim != 1:
        raise ValueError(f"brier_sum_by_class must be a flat array, got shape {sums.shape}")
    if not np.all(np.isfinite(sums)) or np.any(sums < 0.0):
        raise ValueError("brier_sum_by_class must contain finite nonnegative values")
    if not isinstance(valid_pixel_count, (int, np.integer)) or isinstance(valid_pixel_count, bool):
        raise ValueError("valid_pixel_count must be an integer")
    if valid_pixel_count < 0:
        raise ValueError("valid_pixel_count must be nonnegative")
    if valid_pixel_count == 0:
        return None
    return float(sums.sum()) / float(valid_pixel_count)


def _validated_ece(
    statistics: ECEBinSufficientStatistics,
) -> tuple[Int64Array, Float64Array, Int64Array]:
    counts = np.asarray(statistics.counts, dtype=np.int64)
    sums = np.asarray(statistics.confidence_sums, dtype=np.float64)
    positives = np.asarray(statistics.positive_counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape != sums.shape or counts.shape != positives.shape:
        raise ValueError("ECE statistics must share one class-by-bin shape")
    if np.any(counts < 0) or np.any(positives < 0):
        raise ValueError("ECE counts must be nonnegative")
    if np.any(positives > counts):
        raise ValueError("ECE positive_counts must not exceed counts")
    if not np.all(np.isfinite(sums)):
        raise ValueError("ECE confidence_sums must be finite")
    return counts, sums, positives


def classwise_expected_calibration_error(
    statistics: ECEBinSufficientStatistics,
) -> tuple[float | None, ...]:
    """Return one-vs-rest expected calibration error per class, or ``None`` per class.

    Each bin contributes the gap between how often the class actually occurred in
    it and how confident the model was there, WEIGHTED by how many pixels landed
    in that bin. Weighting the bins equally instead would let a bin holding three
    pixels count as much as one holding three million, which is the difference
    between a calibration number and a decoration.

    A class no pixel ever saw is ``None``, not zero: zero calibration error is
    the best possible score and an unmeasured class has no score at all.
    """

    counts, sums, positives = _validated_ece(statistics)
    per_class: list[float | None] = []
    for class_id in range(counts.shape[0]):
        total = int(counts[class_id].sum())
        if total == 0:
            per_class.append(None)
            continue
        occupied = counts[class_id] > 0
        accuracy = positives[class_id][occupied] / counts[class_id][occupied]
        confidence = sums[class_id][occupied] / counts[class_id][occupied]
        weight = counts[class_id][occupied] / total
        per_class.append(float(np.sum(weight * np.abs(accuracy - confidence))))
    return tuple(per_class)


def mean_classwise_expected_calibration_error(
    statistics: ECEBinSufficientStatistics,
) -> float | None:
    """Average the per-class errors over the classes that have support.

    Counting an unmeasured class as zero would flatter every published number,
    and by an amount that grows with how many classes the cohort never contained.
    """

    supported = [
        value for value in classwise_expected_calibration_error(statistics) if value is not None
    ]
    if not supported:
        return None
    return float(np.mean(supported))
