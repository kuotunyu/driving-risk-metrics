"""Confusion accumulation and standard semantic-segmentation summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SegmentationMetrics:
    """Dataset-level metrics derived from one true-row/predicted-column matrix.

    A class value is ``None`` exactly when its denominator is zero. Mean IoU
    includes every class with a non-zero union and excludes only ``None``.
    """

    pixel_accuracy: float
    mean_iou: float
    class_iou: tuple[float | None, ...]
    class_precision: tuple[float | None, ...]
    class_recall: tuple[float | None, ...]


def _validate_confusion(confusion: Int64Array) -> Int64Array:
    if confusion.dtype != np.int64:
        raise TypeError("confusion matrix must have int64 dtype")
    if confusion.ndim != 2:
        raise ValueError("confusion matrix must be two-dimensional")
    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix must be square")
    if confusion.shape[0] == 0:
        raise ValueError("confusion matrix must contain at least one class")
    if np.any(confusion < 0):
        raise ValueError("confusion counts must be nonnegative")
    return confusion


def compute_confusion(
    y_true: Int64Array,
    y_pred: Int64Array,
    num_classes: int,
    ignore_index: int = 255,
) -> Int64Array:
    """Return int64 counts indexed as ``[true_class, predicted_class]``.

    The ignore rule is applied to ground truth first. Predictions at ignored
    pixels are deliberately not validated because they cannot affect a metric.
    """

    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if y_true.dtype != np.int64 or y_pred.dtype != np.int64:
        raise TypeError("y_true and y_pred must have int64 dtype")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    valid = y_true != ignore_index
    if np.any(valid & ((y_true < 0) | (y_true >= num_classes))):
        raise ValueError("true class ID is outside the declared class range")
    if np.any(valid & ((y_pred < 0) | (y_pred >= num_classes))):
        raise ValueError("predicted class ID is outside the declared class range")

    encoded = y_true[valid] * num_classes + y_pred[valid]
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).astype(np.int64, copy=False)


def _ratios(
    numerator: npt.NDArray[np.integer],
    denominator: npt.NDArray[np.integer],
) -> tuple[float | None, ...]:
    return tuple(
        None if int(bottom) == 0 else float(int(top) / int(bottom))
        for top, bottom in zip(numerator, denominator, strict=True)
    )


def summarize_confusion(confusion: Int64Array) -> SegmentationMetrics:
    """Compute dataset-level accuracy, IoU, precision, and recall.

    An all-zero matrix is rejected because neither pixel accuracy nor mean IoU
    has a meaningful denominator. Undefined per-class values remain ``None``.
    """

    matrix = _validate_confusion(confusion)
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("confusion matrix contains no valid pixel")

    true_positive = np.diag(matrix)
    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    union = true_support + predicted_support - true_positive
    class_iou = _ratios(true_positive, union)
    class_precision = _ratios(true_positive, predicted_support)
    class_recall = _ratios(true_positive, true_support)
    defined_iou = tuple(value for value in class_iou if value is not None)

    return SegmentationMetrics(
        pixel_accuracy=float(int(true_positive.sum()) / total),
        mean_iou=float(sum(defined_iou) / len(defined_iou)),
        class_iou=class_iou,
        class_precision=class_precision,
        class_recall=class_recall,
    )
