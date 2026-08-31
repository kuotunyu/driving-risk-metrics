"""Instance-balanced semantic coverage derived from explicit instance IDs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from drivemetrics.metrics.confusion import Int64Array

AreaTertile = Literal["small", "medium", "large"]


@dataclass(frozen=True)
class InstanceCoverage:
    """Semantic correctness for one annotated object instance."""

    instance_id: int
    class_id: int
    area_pixels: int
    correct_fraction: float
    is_critical_miss: bool
    area_tertile: AreaTertile


def learn_area_tertiles(
    training_instances: Sequence[tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    """Learn deterministic classwise area edges from training instances only.

    Sorted observations are divided into three rank groups. Remainders are
    assigned to the earlier groups, and ties remain on the same threshold.
    """

    if not training_instances:
        raise ValueError("training instances must not be empty")

    areas_by_class: defaultdict[int, list[int]] = defaultdict(list)
    for class_id, area_pixels in training_instances:
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise TypeError("training class IDs must be integers")
        if class_id < 0:
            raise ValueError("training class IDs must be nonnegative")
        if isinstance(area_pixels, bool) or not isinstance(area_pixels, int):
            raise TypeError("training instance area must be an integer")
        if area_pixels <= 0:
            raise ValueError("training instance areas must be positive")
        areas_by_class[class_id].append(area_pixels)

    edges: dict[int, tuple[int, int]] = {}
    for class_id in sorted(areas_by_class):
        areas = sorted(areas_by_class[class_id])
        count = len(areas)
        edges[class_id] = (areas[(count - 1) // 3], areas[(2 * count - 1) // 3])
    return edges


def _area_tertile(area_pixels: int, edges: tuple[int, int]) -> AreaTertile:
    low, high = edges
    if area_pixels <= low:
        return "small"
    if area_pixels <= high:
        return "medium"
    return "large"


def _validate_arrays(y_true: Int64Array, y_pred: Int64Array, instance_ids: Int64Array) -> None:
    for name, array in (
        ("y_true", y_true),
        ("y_pred", y_pred),
        ("instance_ids", instance_ids),
    ):
        if array.dtype != np.int64:
            raise TypeError(f"{name} must have int64 dtype")
    if y_true.shape != y_pred.shape or y_true.shape != instance_ids.shape:
        raise ValueError("y_true, y_pred, and instance_ids must have the same shape")


def _validated_instance_contract(
    instance_id: int,
    class_id: int,
    tertile_edges: Mapping[int, tuple[int, int]],
) -> tuple[int, int]:
    if isinstance(instance_id, bool) or not isinstance(instance_id, int):
        raise TypeError("instance IDs must be integers")
    if instance_id <= 0:
        raise ValueError("instance IDs must be positive")
    if isinstance(class_id, bool) or not isinstance(class_id, int):
        raise TypeError("class IDs must be integers")
    if class_id < 0:
        raise ValueError("class IDs must be nonnegative")
    if class_id not in tertile_edges:
        raise ValueError(f"tertile edges are missing for class {class_id}")

    edges = tertile_edges[class_id]
    if not isinstance(edges, tuple) or len(edges) != 2:
        raise TypeError("tertile edges must be integer pairs")
    low, high = edges
    if any(isinstance(edge, bool) or not isinstance(edge, int) for edge in edges):
        raise TypeError("tertile edges must be integer pairs")
    if low <= 0 or high <= 0:
        raise ValueError("tertile edges must be positive")
    if low > high:
        raise ValueError("tertile edges must be ordered")
    return edges


def instance_coverages(
    y_true: Int64Array,
    y_pred: Int64Array,
    instance_ids: Int64Array,
    valid_instance_classes: Mapping[int, int],
    tertile_edges: Mapping[int, tuple[int, int]],
) -> tuple[InstanceCoverage, ...]:
    """Return one equally weighted semantic-coverage record per instance ID."""

    _validate_arrays(y_true, y_pred, instance_ids)
    records: list[InstanceCoverage] = []
    for instance_id in sorted(valid_instance_classes):
        class_id = valid_instance_classes[instance_id]
        edges = _validated_instance_contract(instance_id, class_id, tertile_edges)
        instance_mask = instance_ids == instance_id
        if not np.any(instance_mask):
            raise ValueError(f"declared instance ID {instance_id} is absent")
        mask = instance_mask & (y_true != 255)
        area_pixels = int(mask.sum())
        if area_pixels == 0:
            raise ValueError(f"instance ID {instance_id} has no non-ignored semantic pixels")
        if np.any(y_true[mask] != class_id):
            raise ValueError(f"instance ID {instance_id} does not match its semantic class")
        correct_fraction = float(np.count_nonzero(y_true[mask] == y_pred[mask]) / area_pixels)
        records.append(
            InstanceCoverage(
                instance_id=instance_id,
                class_id=class_id,
                area_pixels=area_pixels,
                correct_fraction=correct_fraction,
                is_critical_miss=correct_fraction < 0.5,
                area_tertile=_area_tertile(area_pixels, edges),
            )
        )
    return tuple(records)
