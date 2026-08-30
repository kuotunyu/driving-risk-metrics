"""Confusion-matrix accumulation and the IoU aggregations built on it.

This module exists because of a concrete finding in the source material this
repo grew out of. Four CamVid notebooks shared one ``iou()`` function and one
``num_class = 11`` declaration, yet reported mIoU values of 0.8157, 0.6780,
0.5657 and 0.3808 — numbers that were routinely compared to each other. They
were not comparable. Two distinct causes:

1. One run's stored output contains only three per-class values
   (``IoUs: [0.87150295 0.93709133 0.63858718]``), so it was an 11-class figure
   in name only.
2. The shared aggregation averaged per-image IoU with ``nanmean``, dropping
   images in which a class is absent. For rare classes that silently changes the
   quantity being measured.

So this module implements both aggregations and names them honestly:
:func:`dataset_iou` (accumulate one confusion matrix over the whole split, then
divide — the Cityscapes convention) and :func:`per_image_nanmean_iou` (the
notebook convention). ``scripts/compare_protocols.py`` reports both for every
model, which turns "these numbers were not comparable" from a claim into a
measurement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..taxonomy import IGNORE_INDEX, NUM_CLASSES

__all__ = [
    "ConfusionMatrix",
    "confusion_from_pair",
    "dataset_iou",
    "per_image_nanmean_iou",
    "IoUResult",
]


@dataclass
class IoUResult:
    """Per-class IoU plus the mean, with the aggregation used recorded."""

    per_class: np.ndarray  # shape (NUM_CLASSES,), NaN where undefined
    mean: float
    aggregation: str
    #: Number of classes that actually contributed to ``mean``. Reported because
    #: a mean over 3 classes and a mean over 11 are different quantities, and
    #: conflating them is the exact failure this repo was built to catch.
    n_classes_counted: int

    def as_dict(self) -> dict:
        return {
            "aggregation": self.aggregation,
            "mean_iou": None if np.isnan(self.mean) else float(self.mean),
            "n_classes_counted": int(self.n_classes_counted),
            "per_class_iou": [None if np.isnan(v) else float(v) for v in self.per_class],
        }


class ConfusionMatrix:
    """Accumulates a ``(num_classes, num_classes)`` matrix of ``[true][pred]``.

    Void pixels (``ignore_index``) are dropped before accumulation, so they can
    never inflate a score. Everything downstream — IoU, pixel accuracy, expected
    risk — is derived from this one matrix, which guarantees that the numbers a
    report puts side by side were computed from the same counts.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int | None = IGNORE_INDEX):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.n_images = 0

    def update(self, target: np.ndarray, pred: np.ndarray) -> ConfusionMatrix:
        """Accumulate one image (or a batch — shape is not interpreted)."""
        target = np.asarray(target).reshape(-1)
        pred = np.asarray(pred).reshape(-1)
        if target.shape != pred.shape:
            raise ValueError(
                f"target and pred must have the same number of pixels, "
                f"got {target.shape[0]} and {pred.shape[0]}"
            )

        valid = np.ones(target.shape, dtype=bool)
        if self.ignore_index is not None:
            valid &= target != self.ignore_index
        # Labels outside the class range are dropped rather than clipped: silently
        # folding an out-of-range label into class 0 would corrupt Sky's score.
        valid &= (target >= 0) & (target < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)

        t = target[valid].astype(np.int64)
        p = pred[valid].astype(np.int64)
        flat = np.bincount(
            t * self.num_classes + p, minlength=self.num_classes * self.num_classes
        )
        self.matrix += flat.reshape(self.num_classes, self.num_classes)
        self.n_images += 1
        return self

    # -- derived quantities -------------------------------------------------

    @property
    def support(self) -> np.ndarray:
        """Ground-truth pixel count per class."""
        return self.matrix.sum(axis=1)

    @property
    def predicted(self) -> np.ndarray:
        """Predicted pixel count per class."""
        return self.matrix.sum(axis=0)

    def pixel_accuracy(self) -> float:
        total = self.matrix.sum()
        if total == 0:
            return float("nan")
        return float(np.trace(self.matrix) / total)

    def per_class_iou(self) -> np.ndarray:
        """IoU per class; NaN for classes with no ground truth *and* no prediction."""
        tp = np.diag(self.matrix).astype(np.float64)
        union = self.support + self.predicted - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)
        return iou

    def per_class_recall(self) -> np.ndarray:
        """Fraction of each class's pixels that were recovered.

        For safety work this is often more informative than IoU: it isolates
        misses from false alarms, and a miss is the failure mode that hurts.
        """
        tp = np.diag(self.matrix).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.support > 0, tp / self.support, np.nan)

    def per_class_precision(self) -> np.ndarray:
        tp = np.diag(self.matrix).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.predicted > 0, tp / self.predicted, np.nan)

    def normalised(self) -> np.ndarray:
        """Row-normalised matrix — each row is P(pred | true)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.support[:, None] > 0, self.matrix / self.support[:, None], np.nan
            )

    def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
        if self.num_classes != other.num_classes:
            raise ValueError("cannot add confusion matrices with different class counts")
        merged = ConfusionMatrix(self.num_classes, self.ignore_index)
        merged.matrix = self.matrix + other.matrix
        merged.n_images = self.n_images + other.n_images
        return merged

    def as_dict(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "n_images": self.n_images,
            "matrix": self.matrix.tolist(),
        }


def confusion_from_pair(
    target: np.ndarray,
    pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = IGNORE_INDEX,
) -> ConfusionMatrix:
    """Convenience wrapper for a single ``(target, pred)`` pair."""
    return ConfusionMatrix(num_classes, ignore_index).update(target, pred)


def dataset_iou(cm: ConfusionMatrix) -> IoUResult:
    """Dataset-level mIoU: one confusion matrix over the split, then divide.

    This is the Cityscapes/SegNet convention and the one this repo reports as
    its headline. A class contributes exactly once regardless of how many images
    it appears in, so a rare class cannot be quietly excused.
    """
    iou = cm.per_class_iou()
    counted = int(np.sum(~np.isnan(iou)))
    mean = float(np.nanmean(iou)) if counted else float("nan")
    return IoUResult(per_class=iou, mean=mean, aggregation="dataset", n_classes_counted=counted)


def per_image_nanmean_iou(
    targets: Iterable[np.ndarray],
    preds: Iterable[np.ndarray],
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = IGNORE_INDEX,
) -> IoUResult:
    """The source notebooks' aggregation, reimplemented faithfully.

    For each image, IoU is computed per class, with NaN where that class is
    absent from *both* target and prediction in that image; those NaNs are then
    dropped by a ``nanmean`` across images, and the per-class results averaged.

    This is not a strawman — it is what the original code did, and it is a
    common pattern. It is reported next to :func:`dataset_iou` so the size of
    the resulting discrepancy is visible rather than argued about. It tends to
    flatter rare classes, because an image containing no pedestrians cannot
    record a pedestrian failure.
    """
    rows: list[np.ndarray] = []
    for target, pred in zip(targets, preds):
        cm = confusion_from_pair(target, pred, num_classes, ignore_index)
        rows.append(cm.per_class_iou())

    if not rows:
        empty = np.full(num_classes, np.nan)
        return IoUResult(empty, float("nan"), "per_image_nanmean", 0)

    stacked = np.vstack(rows)  # (n_images, num_classes)
    with np.errstate(invalid="ignore"):
        # A column that is all-NaN yields NaN here; numpy warns, and we mean it.
        per_class = np.array(
            [
                np.nanmean(stacked[:, c]) if np.any(~np.isnan(stacked[:, c])) else np.nan
                for c in range(num_classes)
            ]
        )
    counted = int(np.sum(~np.isnan(per_class)))
    mean = float(np.nanmean(per_class)) if counted else float("nan")
    return IoUResult(
        per_class=per_class,
        mean=mean,
        aggregation="per_image_nanmean",
        n_classes_counted=counted,
    )


def frequency_weighted_iou(cm: ConfusionMatrix) -> float:
    """IoU weighted by ground-truth frequency.

    Included for completeness and as a cautionary baseline: on CamVid this is
    dominated by Road, Building and Sky (72% of pixels between them), so it is
    almost blind to exactly the classes that matter for safety.
    """
    iou = cm.per_class_iou()
    support = cm.support.astype(np.float64)
    valid = ~np.isnan(iou) & (support > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(iou[valid] * support[valid]) / np.sum(support[valid]))
