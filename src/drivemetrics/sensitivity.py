"""Does the conclusion survive a different harm model, or a different camera?

Both of this package's headline instruments rest on choices that cannot be
derived from data: how much worse it is to miss a pedestrian than a patch of
sky, and where CamVid's horizon sits. Neither is knowable exactly. A ranking
that only holds at one setting of them is not a finding, it is an artefact of
the setting.

So every comparative claim this repo makes is accompanied by a sweep, and the
sweep reports the fraction of settings under which the claim holds. The honest
outcome is often "this ranking is stable" or "this ranking flips once the VRU
weight drops below 8x" — both are useful; only an unqualified ranking is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry.ipm import CameraModel
from .metrics.confusion import dataset_iou
from .metrics.risk import expected_risk
from .taxonomy import DEFAULT_HARM, VRU_CLASSES, HarmModel

__all__ = [
    "HarmSweepResult",
    "sweep_harm_model",
    "rank_stability",
    "camera_variants",
]


@dataclass
class HarmSweepResult:
    """Model rankings under a range of harm models."""

    #: Multipliers applied to the VRU miss cost.
    factors: list
    #: ``{model_name: [risk_skill at each factor]}``
    skill_by_model: dict
    #: ``{model_name: [rank at each factor]}``, 1 = best.
    rank_by_model: dict
    #: mIoU ranking, which does not depend on the harm model at all.
    miou_rank: dict
    #: Fraction of swept settings at which the risk ranking differs from mIoU's.
    disagreement_fraction: float
    #: Ranking that holds at the largest number of settings.
    modal_risk_ranking: list

    def as_dict(self) -> dict:
        return {
            "vru_weight_factors": [float(f) for f in self.factors],
            "risk_skill_by_model": {
                k: [None if v is None or np.isnan(v) else float(v) for v in vals]
                for k, vals in self.skill_by_model.items()
            },
            "risk_rank_by_model": {k: list(map(int, v)) for k, v in self.rank_by_model.items()},
            "miou_rank": {k: int(v) for k, v in self.miou_rank.items()},
            "disagreement_fraction": float(self.disagreement_fraction),
            "modal_risk_ranking": list(self.modal_risk_ranking),
        }


def _rank(scores: dict, higher_is_better: bool = True) -> dict:
    """Rank models by score; 1 is best. NaN sorts last."""
    def key(item):
        v = item[1]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return (1, 0.0)
        return (0, -v if higher_is_better else v)

    ordered = sorted(scores.items(), key=key)
    return {name: i + 1 for i, (name, _) in enumerate(ordered)}


def sweep_harm_model(
    matrices: dict,
    factors: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0),
    base: HarmModel = DEFAULT_HARM,
    classes: Sequence[int] = VRU_CLASSES,
) -> HarmSweepResult:
    """Re-rank models while scaling the VRU miss cost across orders of magnitude.

    ``matrices`` maps model name to its :class:`ConfusionMatrix`. Because every
    metric is derived from that matrix, the whole sweep runs without touching a
    single image again — a hundred harm models cost no more than one traversal.

    ``factors`` spans 0.1x to 10x deliberately: 0.1x makes a pedestrian barely
    more important than a kerb, 10x makes one missed person outweigh everything
    else in the frame. A ranking that holds across that whole range is a
    genuinely robust ranking.
    """
    if not matrices:
        raise ValueError("no models supplied")

    miou = {name: dataset_iou(cm).mean for name, cm in matrices.items()}
    miou_rank = _rank(miou)

    skill_by_model: dict = {name: [] for name in matrices}
    rank_by_model: dict = {name: [] for name in matrices}
    rankings: list = []

    for f in factors:
        harm = base.scaled(f, classes)
        skills = {name: expected_risk(cm, harm).risk_skill for name, cm in matrices.items()}
        ranks = _rank(skills)
        for name in matrices:
            skill_by_model[name].append(skills[name])
            rank_by_model[name].append(ranks[name])
        rankings.append(tuple(sorted(matrices, key=lambda n: ranks[n])))

    miou_order = tuple(sorted(matrices, key=lambda n: miou_rank[n]))
    disagreements = sum(1 for r in rankings if r != miou_order)

    modal = max(set(rankings), key=rankings.count) if rankings else ()

    return HarmSweepResult(
        factors=list(factors),
        skill_by_model=skill_by_model,
        rank_by_model=rank_by_model,
        miou_rank=miou_rank,
        disagreement_fraction=disagreements / len(rankings) if rankings else 0.0,
        modal_risk_ranking=list(modal),
    )


def rank_stability(sweep: HarmSweepResult) -> dict:
    """Summarise how much each model's rank moved across the sweep."""
    out = {}
    for name, ranks in sweep.rank_by_model.items():
        arr = np.asarray(ranks)
        out[name] = {
            "best_rank": int(arr.min()),
            "worst_rank": int(arr.max()),
            "stable": bool(arr.min() == arr.max()),
            "miou_rank": int(sweep.miou_rank[name]),
            "rank_changed_vs_miou": bool(
                arr.min() != sweep.miou_rank[name] or arr.max() != sweep.miou_rank[name]
            ),
        }
    return out


def camera_variants(
    base: CameraModel,
    horizon_delta: Sequence[float] = (-20.0, -10.0, 0.0, 10.0, 20.0),
    focal_scale: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.2),
    height_scale: Sequence[float] = (0.85, 1.0, 1.15),
) -> list:
    """Perturbed camera models for the distance-stratification sweep.

    CamVid publishes no calibration, so the band boundaries are a guess. This
    generates the neighbourhood of that guess: a horizon 20 rows out, a focal
    length 20% off, a camera 15% higher or lower than assumed. If a
    distance-stratified conclusion holds across all of these, the missing
    calibration did not manufacture it.
    """
    out = []
    for dh in horizon_delta:
        for fs in focal_scale:
            for hs in height_scale:
                row = base.horizon_row + dh
                if not (0 <= row < base.image_height):
                    continue
                out.append(
                    CameraModel(
                        horizon_row=row,
                        focal_px=base.focal_px * fs,
                        height_m=base.height_m * hs,
                        image_height=base.image_height,
                        image_width=base.image_width,
                        source=(
                            f"{base.source} [swept: horizon{dh:+g}, "
                            f"focal x{fs:g}, height x{hs:g}]"
                        ),
                    )
                )
    return out
