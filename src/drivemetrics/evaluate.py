"""One pass over a split, producing every metric this package reports.

The design rule here is that a model is evaluated **exactly once**, and every
headline number is derived from the same traversal. Nothing recomputes
predictions, and no metric can quietly be run on a different subset than
another. That is what makes the numbers in a report internally consistent, and
it is the failure this repository was built in response to: the source notebooks
produced mIoU values from four separate runs under three different protocols and
compared them as if they were one table.

A run produces an :class:`EvalBundle`, which serialises to a single JSON file
carrying the metrics, the harm model, the camera assumptions, the split
manifest digest, and the package version. A number that cannot be traced back to
those is not reportable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .geometry.ipm import CAMVID_DEFAULT_CAMERA, DEFAULT_BANDS, CameraModel
from .metrics.blindspot import (
    PerImageClassCounts,
    blind_spot_curve,
    blind_spot_rate,
    collect_counts,
)
from .metrics.confusion import (
    ConfusionMatrix,
    dataset_iou,
    frequency_weighted_iou,
    per_image_nanmean_iou,
)
from .metrics.risk import expected_risk, risk_contribution_by_confusion, safety_recall
from .metrics.stratified import StratifiedEvaluator
from .taxonomy import CLASS_NAMES, DEFAULT_HARM, IGNORE_INDEX, NUM_CLASSES, VRU_CLASSES, HarmModel

__all__ = ["EvalBundle", "evaluate_pairs", "PairSource"]

#: A source of ``(sample_id, target, prediction)`` triples.
PairSource = Iterable["tuple[str, np.ndarray, np.ndarray]"]


@dataclass
class EvalBundle:
    """Everything one model's evaluation produced, in one serialisable object."""

    model: str
    split: str
    n_samples: int
    #: Dataset-level protocol — the headline.
    dataset_iou: dict
    #: The source notebooks' protocol, reported for comparison, never as headline.
    per_image_iou: dict
    pixel_accuracy: float
    frequency_weighted_iou: float
    per_class_recall: dict
    per_class_precision: dict
    support: dict
    risk: dict
    risk_top_confusions: list
    safety_recall: dict
    blind_spot: dict
    blind_spot_curve: dict
    stratified: dict
    confusion_matrix: list
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "split": self.split,
            "n_samples": self.n_samples,
            "headline": {
                "mean_iou": self.dataset_iou.get("mean_iou"),
                "n_classes_counted": self.dataset_iou.get("n_classes_counted"),
                "pixel_accuracy": self.pixel_accuracy,
                "risk_skill": self.risk.get("risk_skill"),
                "expected_risk": self.risk.get("expected_risk"),
                "vru_recall": self.safety_recall.get("pooled"),
            },
            "iou": {
                "dataset_protocol": self.dataset_iou,
                "per_image_nanmean_protocol": self.per_image_iou,
                "frequency_weighted": self.frequency_weighted_iou,
            },
            "pixel_accuracy": self.pixel_accuracy,
            "per_class_recall": self.per_class_recall,
            "per_class_precision": self.per_class_precision,
            "support": self.support,
            "risk": self.risk,
            "risk_top_confusions": self.risk_top_confusions,
            "safety_recall": self.safety_recall,
            "blind_spot": self.blind_spot,
            "blind_spot_curve": self.blind_spot_curve,
            "stratified": self.stratified,
            "confusion_matrix": self.confusion_matrix,
            "provenance": self.provenance,
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    @property
    def protocol_gap(self) -> float | None:
        """How far the two IoU protocols disagree on this model.

        Reported because on the source material this gap reached 0.24 mIoU —
        larger than the spread between the architectures being compared.
        """
        a = self.dataset_iou.get("mean_iou")
        b = self.per_image_iou.get("mean_iou")
        if a is None or b is None:
            return None
        return float(b - a)


def evaluate_pairs(
    pairs: PairSource,
    model: str,
    split: str = "val",
    harm: HarmModel = DEFAULT_HARM,
    camera: CameraModel | None = CAMVID_DEFAULT_CAMERA,
    bands: Sequence[float] = DEFAULT_BANDS,
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = IGNORE_INDEX,
    vru_classes: Sequence[int] = VRU_CLASSES,
    min_gt_pixels: int = 50,
    recall_threshold: float = 0.10,
    provenance: dict | None = None,
) -> EvalBundle:
    """Run every metric over one traversal of ``pairs``.

    ``pairs`` yields ``(sample_id, target, prediction)``. Targets and
    predictions are 2-D integer class-index arrays of identical shape.

    The per-image IoU protocol needs each image separately while the dataset
    protocol needs them pooled, so both are accumulated in the same loop rather
    than by iterating twice — iterating twice is how two "identical" evaluations
    come to disagree.
    """
    cm = ConfusionMatrix(num_classes, ignore_index)
    counts: list[PerImageClassCounts] = []
    per_image_targets: list[np.ndarray] = []
    per_image_preds: list[np.ndarray] = []
    strat: StratifiedEvaluator | None = None
    n = 0

    for sample_id, target, pred in pairs:
        target = np.asarray(target)
        pred = np.asarray(pred)
        if target.shape != pred.shape:
            raise ValueError(
                f"{sample_id}: target shape {target.shape} != prediction shape {pred.shape}"
            )

        cm.update(target, pred)
        counts.append(collect_counts(sample_id, target, pred, num_classes, ignore_index))
        per_image_targets.append(target)
        per_image_preds.append(pred)

        if camera is not None:
            if strat is None:
                cam = camera
                if (cam.image_height, cam.image_width) != target.shape[:2]:
                    try:
                        cam = cam.scaled_to(target.shape[0], target.shape[1])
                    except ValueError as exc:
                        raise ValueError(
                            f"cannot apply the camera model to {target.shape[0]}x"
                            f"{target.shape[1]} masks: it is defined for "
                            f"{cam.image_height}x{cam.image_width}, a different "
                            f"aspect ratio. Supply a CameraModel matching your "
                            f"masks, or pass camera=None to skip distance "
                            f"stratification. ({exc})"
                        ) from exc
                strat = StratifiedEvaluator(cam, bands, num_classes, ignore_index)
            strat.update(target, pred)
        n += 1

    if n == 0:
        raise ValueError(f"no samples supplied for model {model!r}")

    iou = dataset_iou(cm)
    per_image = per_image_nanmean_iou(
        per_image_targets, per_image_preds, num_classes, ignore_index
    )
    risk = expected_risk(cm, harm)
    bs = blind_spot_rate(counts, vru_classes, min_gt_pixels, recall_threshold)

    prov = {
        "package_version": _version(),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harm_model": {"name": harm.name, "notes": harm.notes},
        "camera": camera.as_dict() if camera is not None else None,
        "band_edges": [None if np.isinf(b) else float(b) for b in bands],
        "blind_spot_thresholds": {
            "min_gt_pixels": min_gt_pixels,
            "recall_threshold": recall_threshold,
        },
        "ignore_index": ignore_index,
    }
    if provenance:
        prov.update(provenance)

    return EvalBundle(
        model=model,
        split=split,
        n_samples=n,
        dataset_iou=iou.as_dict(),
        per_image_iou=per_image.as_dict(),
        pixel_accuracy=cm.pixel_accuracy(),
        frequency_weighted_iou=frequency_weighted_iou(cm),
        per_class_recall=_named(cm.per_class_recall()),
        per_class_precision=_named(cm.per_class_precision()),
        support={CLASS_NAMES[i]: int(v) for i, v in enumerate(cm.support)},
        risk=risk.as_dict(),
        risk_top_confusions=risk_contribution_by_confusion(cm, harm, top_k=10),
        safety_recall=safety_recall(cm, vru_classes),
        blind_spot={k: v.as_dict() for k, v in bs.items()},
        blind_spot_curve={
            k: [[float(t), None if np.isnan(r) else float(r)] for t, r in v]
            for k, v in blind_spot_curve(counts, vru_classes, min_gt_pixels).items()
        },
        stratified=strat.result(harm).as_dict() if strat is not None else {},
        confusion_matrix=cm.matrix.tolist(),
        provenance=prov,
    )


def _named(values: np.ndarray) -> dict:
    return {
        CLASS_NAMES[i]: (None if np.isnan(v) else float(v)) for i, v in enumerate(values)
    }


def _version() -> str:
    from . import __version__

    return __version__


# ---------------------------------------------------------------------------
# Loading predictions from disk
# ---------------------------------------------------------------------------


def pairs_from_directory(
    root: Path,
    split_name: str,
    prediction_dir: Path,
    suffix: str = ".png",
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(sample_id, target, prediction)`` by matching filenames.

    Predictions are index-mask images named ``<sample_id><suffix>``. A missing
    prediction is an error rather than a skip: silently evaluating a model on
    the subset it happened to emit is how a model comes to look better than it
    is, and it is precisely the kind of drift this package exists to prevent.
    """
    from .data.camvid import load_split, read_mask

    prediction_dir = Path(prediction_dir)
    split = load_split(root, split_name)
    missing = []

    for sample in split:
        pred_path = prediction_dir / f"{sample.sample_id}{suffix}"
        if not pred_path.exists():
            missing.append(sample.sample_id)
            continue
        yield sample.sample_id, read_mask(sample.label), read_mask(pred_path)

    if missing:
        shown = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} of {len(split)} predictions missing from "
            f"{prediction_dir}: {shown}{more}"
        )
