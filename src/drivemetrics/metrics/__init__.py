"""Pure NumPy segmentation metrics with explicit undefined-value policies."""

from .confusion import SegmentationMetrics, compute_confusion, summarize_confusion
from .instances import InstanceCoverage, instance_coverages, learn_area_tertiles
from .risk import compute_cost_risk, critical_false_negative_rate
from .spatial import normalized_image_bands

__all__ = [
    "InstanceCoverage",
    "SegmentationMetrics",
    "compute_confusion",
    "compute_cost_risk",
    "critical_false_negative_rate",
    "instance_coverages",
    "learn_area_tertiles",
    "normalized_image_bands",
    "summarize_confusion",
]
