"""Risk-weighted metrics: what the errors would cost, not how many there were.

mIoU answers "how much of each class did the model get right, averaged over
classes". For driving that is the wrong question in two ways. It treats a missed
pedestrian and a mislabelled patch of sky as the same unit of error, and it
treats a miss and a false alarm as the same unit of error. The metrics here fix
both, using the cost tables declared in :mod:`drivemetrics.taxonomy`.

Every function takes an explicit :class:`~drivemetrics.taxonomy.HarmModel`. None
of them has a default harm model baked in — the caller always states which one
it used, so a reported number can never be silently attributed to the wrong
assumptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..taxonomy import (
    CLASS_NAMES,
    NUM_CLASSES,
    VRU_CLASSES,
    HarmModel,
)
from .confusion import ConfusionMatrix

__all__ = [
    "RiskResult",
    "expected_risk",
    "risk_weighted_iou",
    "safety_recall",
    "risk_contribution_by_confusion",
]


@dataclass
class RiskResult:
    """Outcome of a risk evaluation under one named harm model."""

    harm_model: str
    #: Mean cost per valid pixel. Lower is better. Unitless — only comparable
    #: across models evaluated under the same harm model and split.
    expected_risk: float
    #: Cost of predicting the most frequent class everywhere: the "learned
    #: nothing" reference, and the denominator of :attr:`risk_skill`.
    majority_risk: float
    #: Which class that was. On CamVid, Road.
    majority_class: str
    #: Cost of the *cheapest* single-class constant. Under a steep harm model
    #: this is not the majority class but a mid-severity hedge — on CamVid,
    #: predicting "Pole" everywhere, which under-calls nothing except VRUs and
    #: cars. It is a strictly harder bar than :attr:`majority_risk` and is
    #: reported so that a model which fails to beat even a blanket hedge cannot
    #: hide behind a flattering skill score.
    best_constant_risk: float
    best_constant_class: str
    #: ``1 - expected_risk / majority_risk``. 1.0 is perfect; 0.0 means no more
    #: useful than predicting the majority class everywhere; negative means
    #: actively worse than that.
    risk_skill: float
    #: The same, against the cheapest constant. Always ≤ :attr:`risk_skill`.
    risk_skill_strict: float
    #: Share of total risk attributable to each true class. Sums to 1.
    risk_share_by_true_class: np.ndarray
    risk_weighted_iou: float

    def as_dict(self) -> dict:
        return {
            "harm_model": self.harm_model,
            "expected_risk": float(self.expected_risk),
            "majority_risk": float(self.majority_risk),
            "majority_class": self.majority_class,
            "best_constant_risk": float(self.best_constant_risk),
            "best_constant_class": self.best_constant_class,
            "risk_skill": float(self.risk_skill),
            "risk_skill_strict": float(self.risk_skill_strict),
            "risk_weighted_iou": float(self.risk_weighted_iou),
            "risk_share_by_true_class": {
                CLASS_NAMES[i]: float(v) for i, v in enumerate(self.risk_share_by_true_class)
            },
        }


def _cost_array(harm: HarmModel) -> np.ndarray:
    return np.asarray(harm.cost_matrix(), dtype=np.float64)


def expected_risk(cm: ConfusionMatrix, harm: HarmModel) -> RiskResult:
    """Mean cost per pixel under ``harm``, plus the context needed to read it.

    The raw expected risk is not interpretable on its own — its scale depends
    entirely on the arbitrary units of the harm model. So it is always returned
    alongside ``trivial_risk``, the cost of the best constant prediction, and
    ``risk_skill``, the normalised improvement over that. ``risk_skill`` is the
    number to compare across models; ``expected_risk`` is the number to compare
    across harm models for one fixed model.
    """
    if cm.num_classes != NUM_CLASSES:
        raise ValueError(
            f"harm model covers {NUM_CLASSES} classes but confusion matrix has {cm.num_classes}"
        )

    cost = _cost_array(harm)
    counts = cm.matrix.astype(np.float64)
    total = counts.sum()
    if total == 0:
        nan_share = np.full(NUM_CLASSES, np.nan)
        return RiskResult(
            harm_model=harm.name,
            expected_risk=float("nan"),
            majority_risk=float("nan"),
            majority_class="",
            best_constant_risk=float("nan"),
            best_constant_class="",
            risk_skill=float("nan"),
            risk_skill_strict=float("nan"),
            risk_share_by_true_class=nan_share,
            risk_weighted_iou=float("nan"),
        )

    weighted = cost * counts
    risk = float(weighted.sum() / total)

    # Two constant-prediction references, both computed against this split's own
    # class prior. The majority-class one is the conventional "no skill"
    # reference and gives the headline skill score. The cheapest-constant one is
    # a strictly harder bar: under a steep harm model it is a mid-severity hedge
    # rather than the majority class, because over-calling is cheap and
    # under-calling is not. Reporting both keeps the normalisation visible
    # instead of buried in a single constant.
    support = cm.support.astype(np.float64)
    per_constant = np.array(
        [float((cost[:, p] * support).sum() / total) for p in range(NUM_CLASSES)]
    )
    best_p = int(np.argmin(per_constant))
    majority_p = int(np.argmax(support))
    best_risk = float(per_constant[best_p])
    majority = float(per_constant[majority_p])

    skill = float("nan") if majority == 0 else float(1.0 - risk / majority)
    skill_strict = float("nan") if best_risk == 0 else float(1.0 - risk / best_risk)

    row_risk = weighted.sum(axis=1)
    total_risk = row_risk.sum()
    share = row_risk / total_risk if total_risk > 0 else np.full(NUM_CLASSES, np.nan)

    return RiskResult(
        harm_model=harm.name,
        expected_risk=risk,
        majority_risk=majority,
        majority_class=CLASS_NAMES[majority_p],
        best_constant_risk=best_risk,
        best_constant_class=CLASS_NAMES[best_p],
        risk_skill=skill,
        risk_skill_strict=skill_strict,
        risk_share_by_true_class=share,
        risk_weighted_iou=risk_weighted_iou(cm, harm),
    )


def risk_weighted_iou(cm: ConfusionMatrix, harm: HarmModel) -> float:
    """Per-class IoU averaged with harm weights instead of uniformly.

    Simpler and more familiar than :func:`expected_risk`, and useful precisely
    because it is a minimal edit to the metric people already report: identical
    to mIoU when the harm model is uniform. It cannot express the miss/false-alarm
    asymmetry, so it is reported as a secondary number rather than a headline.
    """
    iou = cm.per_class_iou()
    weights = np.asarray(harm.class_weights(), dtype=np.float64)
    valid = ~np.isnan(iou)
    if not np.any(valid) or weights[valid].sum() == 0:
        return float("nan")
    return float(np.sum(iou[valid] * weights[valid]) / np.sum(weights[valid]))


def safety_recall(
    cm: ConfusionMatrix, classes: Sequence[int] = VRU_CLASSES
) -> dict[str, float]:
    """Recall restricted to the classes that get people hurt.

    Reported per class and pooled. Recall rather than IoU because a false alarm
    on a pedestrian costs a jolt of unnecessary braking, while a miss costs a
    person — and IoU blends the two into one number that hides which is
    happening.
    """
    recall = cm.per_class_recall()
    out: dict[str, float] = {}
    tp_total = 0.0
    support_total = 0.0
    for c in classes:
        out[CLASS_NAMES[c]] = float(recall[c]) if not np.isnan(recall[c]) else float("nan")
        tp_total += float(cm.matrix[c, c])
        support_total += float(cm.support[c])
    out["pooled"] = float(tp_total / support_total) if support_total > 0 else float("nan")
    return out


def risk_contribution_by_confusion(
    cm: ConfusionMatrix, harm: HarmModel, top_k: int = 10
) -> list[dict]:
    """The specific confusions carrying the most total risk.

    Answers "where is the danger actually coming from" with a ranked list of
    ``true → predicted`` pairs. This is what turns a low score into an
    actionable finding: on CamVid it typically surfaces Pedestrian→Building and
    Pole→Building long before anything involving Road or Sky.
    """
    cost = _cost_array(harm)
    counts = cm.matrix.astype(np.float64)
    weighted = cost * counts
    total_risk = weighted.sum()
    if total_risk == 0:
        return []

    entries = []
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            if t == p or weighted[t, p] == 0:
                continue
            entries.append(
                {
                    "true": CLASS_NAMES[t],
                    "pred": CLASS_NAMES[p],
                    "pixels": int(counts[t, p]),
                    "unit_cost": float(cost[t, p]),
                    "total_risk": float(weighted[t, p]),
                    "risk_share": float(weighted[t, p] / total_risk),
                }
            )
    entries.sort(key=lambda e: e["total_risk"], reverse=True)
    return entries[:top_k]
