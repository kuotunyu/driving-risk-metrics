"""Pure NumPy segmentation metrics with explicit undefined-value policies."""

from .calibration import (
    ECEBinSufficientStatistics,
    classwise_ece_sufficient_statistics,
    classwise_expected_calibration_error,
    dequantize_confidence,
    mean_classwise_expected_calibration_error,
    multiclass_brier_score,
    multiclass_brier_sums,
    pack_correctness,
    quantize_confidence,
    unpack_correctness,
)
from .confusion import SegmentationMetrics, compute_confusion, summarize_confusion
from .instances import InstanceCoverage, instance_coverages, learn_area_tertiles
from .risk import compute_cost_risk, critical_false_negative_rate
from .selective import area_under_risk_coverage, selective_risk_curve
from .spatial import normalized_image_bands

__all__ = [
    "ECEBinSufficientStatistics",
    "InstanceCoverage",
    "SegmentationMetrics",
    "area_under_risk_coverage",
    "classwise_ece_sufficient_statistics",
    "classwise_expected_calibration_error",
    "compute_confusion",
    "compute_cost_risk",
    "critical_false_negative_rate",
    "dequantize_confidence",
    "instance_coverages",
    "learn_area_tertiles",
    "mean_classwise_expected_calibration_error",
    "multiclass_brier_score",
    "multiclass_brier_sums",
    "normalized_image_bands",
    "pack_correctness",
    "quantize_confidence",
    "selective_risk_curve",
    "summarize_confusion",
    "unpack_correctness",
]
