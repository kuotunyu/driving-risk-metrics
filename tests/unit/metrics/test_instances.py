"""Hand-computed contracts for instance-balanced semantic coverage."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import ModuleType

import numpy as np
import pytest


def load_instances_module() -> ModuleType:
    try:
        from drivemetrics.metrics import instances
    except ImportError:
        pytest.fail("drivemetrics.metrics.instances is missing", pytrace=False)
    return instances


def test_unequal_instance_sizes_remain_equally_weighted_records() -> None:
    instances = load_instances_module()
    y_true = np.full((2, 4), 2, dtype=np.int64)
    y_pred = np.array([[0, 0, 2, 2], [2, 2, 2, 2]], dtype=np.int64)
    instance_ids = np.array([[10, 10, 20, 20], [20, 20, 20, 20]], dtype=np.int64)

    result = instances.instance_coverages(
        y_true,
        y_pred,
        instance_ids,
        valid_instance_classes={10: 2, 20: 2},
        tertile_edges={2: (2, 4)},
    )

    assert tuple(item.instance_id for item in result) == (10, 20)
    assert tuple(item.area_pixels for item in result) == (2, 6)
    assert tuple(item.correct_fraction for item in result) == (0.0, 1.0)
    assert tuple(item.area_tertile for item in result) == ("small", "large")
    assert tuple(item.is_critical_miss for item in result) == (True, False)
    assert np.mean([item.correct_fraction for item in result]) == 0.5
    assert np.mean(y_true == y_pred) == 0.75


def test_exactly_half_correct_is_not_a_critical_instance_miss() -> None:
    instances = load_instances_module()
    y_true = np.full((2, 2), 5, dtype=np.int64)
    y_pred = np.array([[5, 0], [5, 0]], dtype=np.int64)
    instance_ids = np.full((2, 2), 7, dtype=np.int64)

    (result,) = instances.instance_coverages(
        y_true,
        y_pred,
        instance_ids,
        valid_instance_classes={7: 5},
        tertile_edges={5: (2, 4)},
    )

    assert result.correct_fraction == 0.5
    assert result.is_critical_miss is False
    assert result.area_tertile == "medium"


def test_ignored_semantic_pixels_do_not_affect_instance_area_or_coverage() -> None:
    instances = load_instances_module()
    y_true = np.array([[3, 255], [3, 3]], dtype=np.int64)
    y_pred = np.array([[3, 999], [0, 3]], dtype=np.int64)
    instance_ids = np.full((2, 2), 9, dtype=np.int64)

    (result,) = instances.instance_coverages(
        y_true,
        y_pred,
        instance_ids,
        valid_instance_classes={9: 3},
        tertile_edges={3: (3, 8)},
    )

    assert result.area_pixels == 3
    assert result.correct_fraction == pytest.approx(2 / 3)
    assert result.area_tertile == "small"


def test_area_tertiles_use_only_supplied_training_intersection_by_class() -> None:
    instances = load_instances_module()
    training_instances = [(2, 30), (3, 100), (2, 10), (3, 5), (2, 20), (3, 5)]

    edges = instances.learn_area_tertiles(training_instances)

    assert edges == {2: (10, 20), 3: (5, 5)}

    # This validation-only area was never supplied while learning the edges.
    y_true = np.full((1, 60), 2, dtype=np.int64)
    y_pred = y_true.copy()
    instance_ids = np.full((1, 60), 99, dtype=np.int64)
    (validation_record,) = instances.instance_coverages(
        y_true,
        y_pred,
        instance_ids,
        valid_instance_classes={99: 2},
        tertile_edges=edges,
    )
    assert validation_record.area_tertile == "large"


def test_disconnected_instance_ids_remain_distinct_and_sorted() -> None:
    instances = load_instances_module()
    y_true = np.full((2, 3), 4, dtype=np.int64)
    y_pred = y_true.copy()
    instance_ids = np.array([[20, 0, 10], [10, 0, 20]], dtype=np.int64)

    result = instances.instance_coverages(
        y_true,
        y_pred,
        instance_ids,
        valid_instance_classes={20: 4, 10: 4},
        tertile_edges={4: (2, 3)},
    )

    assert tuple(item.instance_id for item in result) == (10, 20)
    assert tuple(item.area_pixels for item in result) == (2, 2)


@pytest.mark.parametrize("array_name", ["y_true", "y_pred", "instance_ids"])
def test_instance_coverages_requires_int64_arrays(array_name: str) -> None:
    instances = load_instances_module()
    arrays = {
        "y_true": np.full((1, 2), 2, dtype=np.int64),
        "y_pred": np.full((1, 2), 2, dtype=np.int64),
        "instance_ids": np.full((1, 2), 5, dtype=np.int64),
    }
    arrays[array_name] = arrays[array_name].astype(np.int32)

    with pytest.raises(TypeError, match="int64"):
        instances.instance_coverages(
            arrays["y_true"],
            arrays["y_pred"],
            arrays["instance_ids"],
            valid_instance_classes={5: 2},
            tertile_edges={2: (1, 2)},
        )


@pytest.mark.parametrize("array_name", ["y_pred", "instance_ids"])
def test_instance_coverages_requires_matching_shapes(array_name: str) -> None:
    instances = load_instances_module()
    arrays = {
        "y_true": np.full((1, 2), 2, dtype=np.int64),
        "y_pred": np.full((1, 2), 2, dtype=np.int64),
        "instance_ids": np.full((1, 2), 5, dtype=np.int64),
    }
    arrays[array_name] = np.full((2, 1), 2, dtype=np.int64)

    with pytest.raises(ValueError, match=r"^y_true, y_pred, and instance_ids must have the same"):
        instances.instance_coverages(
            arrays["y_true"],
            arrays["y_pred"],
            arrays["instance_ids"],
            valid_instance_classes={5: 2},
            tertile_edges={2: (1, 2)},
        )


@pytest.mark.parametrize(
    ("valid_instance_classes", "tertile_edges", "expected"),
    [
        ({0: 2}, {2: (1, 2)}, "positive"),
        ({True: 2}, {2: (1, 2)}, "integer"),
        ({5: True}, {1: (1, 2)}, "class"),
        ({5: -1}, {-1: (1, 2)}, "nonnegative"),
        ({5: 2}, {}, "tertile"),
        ({5: 2}, {2: (0, 2)}, "positive"),
        ({5: 2}, {2: (3, 2)}, "ordered"),
        ({5: 2}, {2: [1, 2]}, "pairs"),
        ({5: 2}, {2: (True, 2)}, "pairs"),
    ],
)
def test_instance_coverages_rejects_invalid_mappings_and_edges(
    valid_instance_classes: dict[int, int],
    tertile_edges: dict[int, object],
    expected: str,
) -> None:
    instances = load_instances_module()
    y_true = np.full((1, 2), 2, dtype=np.int64)
    y_pred = y_true.copy()
    instance_ids = np.full((1, 2), 5, dtype=np.int64)

    with pytest.raises((TypeError, ValueError), match=expected):
        instances.instance_coverages(
            y_true,
            y_pred,
            instance_ids,
            valid_instance_classes=valid_instance_classes,
            tertile_edges=tertile_edges,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("y_true", "instance_ids", "expected"),
    [
        (np.full((1, 2), 2, dtype=np.int64), np.full((1, 2), 8, dtype=np.int64), "absent"),
        (np.full((1, 2), 255, dtype=np.int64), np.full((1, 2), 5, dtype=np.int64), "non-ignored"),
        (np.full((1, 2), 3, dtype=np.int64), np.full((1, 2), 5, dtype=np.int64), "semantic"),
    ],
)
def test_instance_coverages_rejects_missing_ignored_or_misclassified_instances(
    y_true: np.ndarray, instance_ids: np.ndarray, expected: str
) -> None:
    instances = load_instances_module()

    with pytest.raises(ValueError, match=expected):
        instances.instance_coverages(
            y_true,
            y_true.copy(),
            instance_ids,
            valid_instance_classes={5: 2},
            tertile_edges={2: (1, 2)},
        )


@pytest.mark.parametrize(
    ("training_instances", "expected"),
    [
        ([], "training"),
        ([(True, 10)], "class"),
        ([(-1, 10)], "nonnegative"),
        ([(2, True)], "area"),
        ([(2, 0)], "positive"),
    ],
)
def test_learn_area_tertiles_rejects_invalid_training_observations(
    training_instances: list[tuple[int, int]], expected: str
) -> None:
    instances = load_instances_module()

    with pytest.raises((TypeError, ValueError), match=expected):
        instances.learn_area_tertiles(training_instances)


def test_instance_coverage_record_is_frozen() -> None:
    instances = load_instances_module()
    record = instances.InstanceCoverage(1, 2, 3, 1.0, False, "small")

    with pytest.raises(FrozenInstanceError):
        record.area_pixels = 99  # type: ignore[misc]


def test_metrics_package_exports_p1_07_public_interfaces() -> None:
    import drivemetrics.metrics as metrics

    assert metrics.InstanceCoverage is load_instances_module().InstanceCoverage
    assert metrics.instance_coverages is load_instances_module().instance_coverages
    assert metrics.learn_area_tertiles is load_instances_module().learn_area_tertiles


def test_the_tertile_edges_are_the_hand_computed_ranks() -> None:
    """The edge indices decide which instances count as small, medium and large.

    Six observations is chosen because the two indices are sensitive there:
    `(6 - 1) // 3` is 1 and `(2 * 6 - 1) // 3` is 3, so any arithmetic slip in
    either expression lands on a different observation and moves the size
    boundary for every instance in the class.
    """

    instances = load_instances_module()

    edges = instances.learn_area_tertiles([(0, 60), (0, 10), (0, 40), (0, 20), (0, 50), (0, 30)])

    # Sorted areas are [10, 20, 30, 40, 50, 60]; the edges are ranks 1 and 3.
    assert edges == {0: (20, 40)}
