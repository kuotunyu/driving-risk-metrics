"""Risk-weighted, distance-stratified evaluation for driving segmentation.

The premise: on CamVid, Road, Building and Sky are 72% of labelled pixels while
pedestrians, cyclists and poles are 1.9%. A metric that averages over pixels, or
averages classes uniformly, is therefore dominated by the parts of the scene
that cannot hurt anyone. This package measures the parts that can.
"""

from .evaluate import EvalBundle, evaluate_pairs, pairs_from_directory
from .geometry.ipm import (
    CAMVID_DEFAULT_CAMERA,
    DEFAULT_BANDS,
    CameraModel,
)
from .metrics.blindspot import (
    PerImageClassCounts,
    blind_spot_curve,
    blind_spot_rate,
    collect_counts,
)
from .metrics.confusion import (
    ConfusionMatrix,
    IoUResult,
    confusion_from_pair,
    dataset_iou,
    frequency_weighted_iou,
    per_image_nanmean_iou,
)
from .metrics.risk import (
    RiskResult,
    expected_risk,
    risk_contribution_by_confusion,
    risk_weighted_iou,
    safety_recall,
)
from .metrics.stratified import StratifiedEvaluator, StratifiedResult
from .sensitivity import (
    HarmSweepResult,
    camera_variants,
    rank_stability,
    sweep_harm_model,
)
from .synthetic import PROFILES, SYNTHETIC_WARNING, DegradationProfile, degrade
from .taxonomy import (
    CAMVID_11,
    CLASS_NAMES,
    DEFAULT_HARM,
    DYNAMIC_CLASSES,
    IGNORE_INDEX,
    NUM_CLASSES,
    UNIFORM_HARM,
    VRU_CLASSES,
    ClassSpec,
    HarmModel,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # taxonomy
    "CAMVID_11", "CLASS_NAMES", "NUM_CLASSES", "IGNORE_INDEX",
    "VRU_CLASSES", "DYNAMIC_CLASSES", "ClassSpec", "HarmModel",
    "DEFAULT_HARM", "UNIFORM_HARM",
    # geometry
    "CameraModel", "CAMVID_DEFAULT_CAMERA", "DEFAULT_BANDS",
    # confusion / iou
    "ConfusionMatrix", "IoUResult", "confusion_from_pair",
    "dataset_iou", "per_image_nanmean_iou", "frequency_weighted_iou",
    # risk
    "RiskResult", "expected_risk", "risk_weighted_iou",
    "safety_recall", "risk_contribution_by_confusion",
    # stratified
    "StratifiedEvaluator", "StratifiedResult",
    # blind spot
    "PerImageClassCounts", "collect_counts", "blind_spot_rate", "blind_spot_curve",
    # evaluation pipeline
    "EvalBundle", "evaluate_pairs", "pairs_from_directory",
    # sensitivity
    "HarmSweepResult", "sweep_harm_model", "rank_stability", "camera_variants",
    # synthetic fixtures
    "PROFILES", "DegradationProfile", "degrade", "SYNTHETIC_WARNING",
]
