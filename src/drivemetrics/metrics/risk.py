"""Safety-aware rates and class-cost risk derived from confusion counts."""

from __future__ import annotations

import numpy as np

from drivemetrics.metrics.confusion import Int64Array, _validate_confusion
from drivemetrics.protocol.risk_profiles import RiskProfile


def _validate_critical_class_ids(critical_class_ids: tuple[int, ...], num_classes: int) -> None:
    if len(set(critical_class_ids)) != len(critical_class_ids):
        raise ValueError("critical class IDs must be unique")
    for class_id in critical_class_ids:
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise TypeError("critical class IDs must be integers")
        if class_id < 0 or class_id >= num_classes:
            raise ValueError("critical class ID is outside the confusion taxonomy")


def critical_false_negative_rate(
    confusion: Int64Array,
    critical_class_ids: tuple[int, ...],
) -> float | None:
    """Return pooled false negatives divided by pooled true critical pixels.

    ``None`` means the requested classes have no ground-truth support, including
    the explicit empty-class request. It is never silently converted to zero.
    """

    matrix = _validate_confusion(confusion)
    _validate_critical_class_ids(critical_class_ids, matrix.shape[0])
    support = matrix.sum(axis=1)
    denominator = int(support[list(critical_class_ids)].sum())
    if denominator == 0:
        return None
    true_positive = np.diag(matrix)
    false_negative = support - true_positive
    return float(int(false_negative[list(critical_class_ids)].sum()) / denominator)


def compute_cost_risk(confusion: Int64Array, profile: RiskProfile) -> float:
    """Return normalized false-negative cost per valid true pixel.

    For each true class ``c``, its off-diagonal row count is multiplied by its
    declared base cost. Critical-class costs are additionally multiplied by the
    profile sensitivity. The sum is divided by all valid true pixels, so the
    result is comparable only for the same normalized profile and sensitivity.
    """

    matrix = _validate_confusion(confusion)
    num_classes = matrix.shape[0]
    if tuple(profile.class_cost) != tuple(range(num_classes)):
        raise ValueError("risk profile must declare exactly one cost for every confusion class")

    total = int(matrix.sum())
    if total == 0:
        raise ValueError("confusion matrix contains no valid pixel")

    costs = np.array([profile.class_cost[class_id] for class_id in range(num_classes)])
    if profile.critical_class_ids:
        costs[list(profile.critical_class_ids)] *= profile.sensitivity
    false_negative = matrix.sum(axis=1) - np.diag(matrix)
    return float(np.dot(false_negative.astype(np.float64), costs) / total)
