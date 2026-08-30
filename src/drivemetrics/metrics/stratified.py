"""Distance-stratified evaluation: the same metrics, sliced by how far away.

A single mIoU over a driving scene averages a pedestrian two metres from the
bumper with one at the end of the street. Those are different engineering
problems: the near miss is unrecoverable, the far miss is usually recoverable by
the next frame. Slicing the confusion matrix by ground distance separates them.

The slicing uses the flat-ground model in :mod:`drivemetrics.geometry.ipm`, whose
parameters are assumptions for CamVid. Every result produced here therefore
carries the camera model that produced it, so a number can never be quoted
without the assumptions attached.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry.ipm import DEFAULT_BANDS, CameraModel, band_labels, band_map
from ..taxonomy import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, HarmModel
from .confusion import ConfusionMatrix, dataset_iou
from .risk import expected_risk

__all__ = ["StratifiedEvaluator", "StratifiedResult"]


@dataclass
class StratifiedResult:
    """Per-band metrics, with the geometry that defined the bands."""

    camera: dict
    bands: list[str]
    band_edges: list[float]
    per_band: dict[str, dict]

    def as_dict(self) -> dict:
        return {
            "camera": self.camera,
            "bands": self.bands,
            "band_edges": [None if np.isinf(e) else float(e) for e in self.band_edges],
            "per_band": self.per_band,
        }


class StratifiedEvaluator:
    """Accumulates one confusion matrix per distance band.

    Usage mirrors :class:`~drivemetrics.metrics.confusion.ConfusionMatrix`::

        ev = StratifiedEvaluator(camera)
        for target, pred in loader:
            ev.update(target, pred)
        result = ev.result(harm)

    The band map is computed once from the camera model and reused, so the cost
    per image is a handful of boolean indexing operations.
    """

    def __init__(
        self,
        camera: CameraModel = None,
        bands: Sequence[float] = DEFAULT_BANDS,
        num_classes: int = NUM_CLASSES,
        ignore_index: int | None = IGNORE_INDEX,
    ):
        if camera is None:
            from ..geometry.ipm import CAMVID_DEFAULT_CAMERA

            camera = CAMVID_DEFAULT_CAMERA
        self.camera = camera
        self.bands = tuple(bands)
        self.labels = band_labels(self.bands)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self._band_map = band_map(camera, bands)
        self._masks = [self._band_map == i for i in range(len(self.labels))]
        self.matrices = [
            ConfusionMatrix(num_classes, ignore_index) for _ in range(len(self.labels))
        ]

    def update(self, target: np.ndarray, pred: np.ndarray) -> StratifiedEvaluator:
        target = np.asarray(target)
        pred = np.asarray(pred)
        if target.shape != pred.shape:
            raise ValueError("target and pred must have the same shape")
        if target.shape != self._band_map.shape:
            raise ValueError(
                f"mask shape {target.shape} does not match the camera model's "
                f"{self._band_map.shape}; rescale the camera with "
                "CameraModel.scaled_to() rather than resizing the masks silently"
            )
        for mask, cm in zip(self._masks, self.matrices):
            cm.update(target[mask], pred[mask])
        return self

    def result(self, harm: HarmModel | None = None) -> StratifiedResult:
        per_band: dict[str, dict] = {}
        for label, cm in zip(self.labels, self.matrices):
            iou = dataset_iou(cm)
            entry = {
                "pixels": int(cm.matrix.sum()),
                "pixel_accuracy": cm.pixel_accuracy(),
                "mean_iou": None if np.isnan(iou.mean) else float(iou.mean),
                "n_classes_present": int(np.sum(cm.support > 0)),
                "per_class_iou": {
                    CLASS_NAMES[i]: (None if np.isnan(v) else float(v))
                    for i, v in enumerate(iou.per_class)
                },
                "per_class_recall": {
                    CLASS_NAMES[i]: (None if np.isnan(v) else float(v))
                    for i, v in enumerate(cm.per_class_recall())
                },
                "support": {
                    CLASS_NAMES[i]: int(v) for i, v in enumerate(cm.support)
                },
            }
            if harm is not None:
                entry["risk"] = expected_risk(cm, harm).as_dict()
            per_band[label] = entry

        return StratifiedResult(
            camera=self.camera.as_dict(),
            bands=list(self.labels),
            band_edges=list(self.bands),
            per_band=per_band,
        )
