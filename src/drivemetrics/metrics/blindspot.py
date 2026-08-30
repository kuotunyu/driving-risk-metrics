"""Blind-spot rate: how often a present hazard is missed entirely.

IoU and recall are pixel-level averages, and averages let a model trade a total
failure on one image against a good result on another. For driving that trade is
not admissible: a model that segments pedestrians perfectly in 90 images and
does not register a single pedestrian pixel in the other 10 is not "90% good",
it is a model that will occasionally not see anyone at all.

The blind-spot rate asks a per-image, per-class yes/no question instead — *was
this hazard, which was present, recovered at all?* — and reports the failure
fraction. It is deliberately crude, because crude is harder to game.

The choice of what counts as "present" and "recovered" is a threshold, and
thresholds invite cherry-picking. So the API computes a curve over thresholds
(:func:`blind_spot_curve`) and the report renders it, rather than publishing one
flattering operating point.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..taxonomy import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, VRU_CLASSES

__all__ = [
    "BlindSpotStats",
    "PerImageClassCounts",
    "collect_counts",
    "blind_spot_rate",
    "blind_spot_curve",
]


@dataclass
class PerImageClassCounts:
    """Ground-truth, predicted and intersection pixel counts for one image.

    Storing counts rather than masks keeps a whole split in a few kilobytes, so
    threshold sweeps re-run instantly without touching the images again.
    """

    image_id: str
    gt: np.ndarray  # (NUM_CLASSES,) ground-truth pixels per class
    pred: np.ndarray  # (NUM_CLASSES,) predicted pixels per class
    tp: np.ndarray  # (NUM_CLASSES,) correctly predicted pixels per class

    def as_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "gt": self.gt.astype(int).tolist(),
            "pred": self.pred.astype(int).tolist(),
            "tp": self.tp.astype(int).tolist(),
        }


def collect_counts(
    image_id: str,
    target: np.ndarray,
    pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = IGNORE_INDEX,
) -> PerImageClassCounts:
    """Reduce one ``(target, pred)`` pair to the three count vectors."""
    target = np.asarray(target).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if target.shape != pred.shape:
        raise ValueError("target and pred must have the same number of pixels")

    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index
    t = target[valid]
    p = pred[valid]

    gt = np.bincount(t, minlength=num_classes)[:num_classes].astype(np.int64)
    pr = np.bincount(p[(p >= 0) & (p < num_classes)], minlength=num_classes)[:num_classes].astype(
        np.int64
    )
    hit = t == p
    tp = np.bincount(t[hit], minlength=num_classes)[:num_classes].astype(np.int64)
    return PerImageClassCounts(image_id=image_id, gt=gt, pred=pr, tp=tp)


@dataclass
class BlindSpotStats:
    """Blind-spot outcome for one class at one operating point."""

    class_name: str
    present_images: int
    blind_images: int
    blind_rate: float
    min_gt_pixels: int
    recall_threshold: float
    #: Image ids that were blind — the gallery for the failure report.
    blind_image_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "class": self.class_name,
            "present_images": self.present_images,
            "blind_images": self.blind_images,
            "blind_rate": None if np.isnan(self.blind_rate) else float(self.blind_rate),
            "min_gt_pixels": self.min_gt_pixels,
            "recall_threshold": self.recall_threshold,
            "blind_image_ids": list(self.blind_image_ids),
        }


def blind_spot_rate(
    counts: Sequence[PerImageClassCounts],
    classes: Sequence[int] = VRU_CLASSES,
    min_gt_pixels: int = 50,
    recall_threshold: float = 0.10,
    keep_ids: bool = True,
) -> dict[str, BlindSpotStats]:
    """Fraction of images where a present class was effectively not recovered.

    An image *contains* class ``c`` when it has at least ``min_gt_pixels`` of it —
    a floor that excludes annotation slivers a few pixels wide, which no model
    can be expected to find and which would otherwise dominate the statistic.

    The class is *recovered* when per-image recall reaches ``recall_threshold``.
    The default of 0.10 is intentionally forgiving: it is not asking for a good
    segmentation, only for evidence that the model registered the hazard's
    existence. Failing this bar means the model produced almost nothing where a
    person was standing.
    """
    out: dict[str, BlindSpotStats] = {}
    for c in classes:
        present = 0
        blind = 0
        blind_ids: list[str] = []
        for item in counts:
            if item.gt[c] < min_gt_pixels:
                continue
            present += 1
            recall = item.tp[c] / item.gt[c] if item.gt[c] > 0 else 0.0
            if recall < recall_threshold:
                blind += 1
                if keep_ids:
                    blind_ids.append(item.image_id)
        rate = (blind / present) if present else float("nan")
        out[CLASS_NAMES[c]] = BlindSpotStats(
            class_name=CLASS_NAMES[c],
            present_images=present,
            blind_images=blind,
            blind_rate=rate,
            min_gt_pixels=min_gt_pixels,
            recall_threshold=recall_threshold,
            blind_image_ids=blind_ids,
        )
    return out


def blind_spot_curve(
    counts: Sequence[PerImageClassCounts],
    classes: Sequence[int] = VRU_CLASSES,
    min_gt_pixels: int = 50,
    thresholds: Iterable[float] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Blind-spot rate as a function of the recall threshold.

    Publishing the curve rather than a point makes the metric honest: a reader
    can see whether a low blind-spot rate holds across operating points or only
    at the one that was chosen. A model whose curve rises steeply from zero is
    producing a few token pixels rather than actually finding the hazard.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    curve: dict[str, list[tuple[float, float]]] = {CLASS_NAMES[c]: [] for c in classes}
    for thr in thresholds:
        stats = blind_spot_rate(
            counts, classes, min_gt_pixels, float(thr), keep_ids=False
        )
        for name, st in stats.items():
            curve[name].append((float(thr), st.blind_rate))
    return curve
