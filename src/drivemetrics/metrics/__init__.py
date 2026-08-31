"""Pure NumPy segmentation metrics with explicit undefined-value policies."""

from .confusion import SegmentationMetrics, compute_confusion, summarize_confusion
from .risk import compute_cost_risk, critical_false_negative_rate

__all__ = [
    "SegmentationMetrics",
    "compute_confusion",
    "compute_cost_risk",
    "critical_false_negative_rate",
    "summarize_confusion",
]
