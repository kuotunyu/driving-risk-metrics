"""Hand-computed contracts for segmentation confusion and standard metrics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st


def load_confusion_module() -> ModuleType:
    try:
        from drivemetrics.metrics import confusion
    except ImportError:
        pytest.fail("drivemetrics.metrics.confusion is missing", pytrace=False)
    return confusion


def load_risk_module() -> ModuleType:
    try:
        from drivemetrics.metrics import risk
    except ImportError:
        pytest.fail("drivemetrics.metrics.risk is missing", pytrace=False)
    return risk


def test_perfect_three_class_confusion_and_summary_are_exact() -> None:
    confusion = load_confusion_module()
    y_true = np.array([[0, 0], [1, 2]], dtype=np.int64)

    matrix = confusion.compute_confusion(y_true, y_true.copy(), num_classes=3)
    metrics = confusion.summarize_confusion(matrix)

    np.testing.assert_array_equal(matrix, np.diag([2, 1, 1]))
    assert matrix.dtype == np.int64
    assert metrics.pixel_accuracy == 1.0
    assert metrics.mean_iou == 1.0
    assert metrics.class_iou == (1.0, 1.0, 1.0)
    assert metrics.class_precision == (1.0, 1.0, 1.0)
    assert metrics.class_recall == (1.0, 1.0, 1.0)

    with pytest.raises(FrozenInstanceError):
        metrics.pixel_accuracy = 0.0  # type: ignore[misc]


def test_all_wrong_three_class_summary_matches_hand_calculation() -> None:
    confusion = load_confusion_module()
    y_true = np.array([0, 1, 2], dtype=np.int64)
    y_pred = np.array([1, 2, 0], dtype=np.int64)

    matrix = confusion.compute_confusion(y_true, y_pred, num_classes=3)
    metrics = confusion.summarize_confusion(matrix)

    np.testing.assert_array_equal(matrix, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert metrics.pixel_accuracy == 0.0
    assert metrics.mean_iou == 0.0
    assert metrics.class_iou == (0.0, 0.0, 0.0)
    assert metrics.class_precision == (0.0, 0.0, 0.0)
    assert metrics.class_recall == (0.0, 0.0, 0.0)


def test_absent_class_is_none_but_predicted_only_class_has_zero_iou() -> None:
    confusion = load_confusion_module()
    absent = np.array([[2, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.int64)
    predicted_only = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int64)

    absent_metrics = confusion.summarize_confusion(absent)
    predicted_only_metrics = confusion.summarize_confusion(predicted_only)

    assert absent_metrics.class_iou == (pytest.approx(2 / 3), pytest.approx(1 / 2), None)
    assert absent_metrics.class_precision == (pytest.approx(2 / 3), 1.0, None)
    assert absent_metrics.class_recall == (1.0, 0.5, None)
    assert absent_metrics.mean_iou == pytest.approx(7 / 12)
    assert predicted_only_metrics.class_iou == (0.5, 1.0, 0.0)
    assert predicted_only_metrics.class_precision == (1.0, 1.0, 0.0)
    assert predicted_only_metrics.class_recall == (0.5, 1.0, None)


def test_ignore_pixels_are_dropped_before_prediction_range_validation() -> None:
    confusion = load_confusion_module()
    y_true = np.array([[0, 255], [1, 2]], dtype=np.int64)
    y_pred = np.array([[0, 999], [2, 2]], dtype=np.int64)

    matrix = confusion.compute_confusion(y_true, y_pred, num_classes=3)

    np.testing.assert_array_equal(matrix, np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]]))
    assert int(matrix.sum()) == 3


@pytest.mark.parametrize(
    ("y_true", "y_pred", "num_classes", "expected"),
    [
        (
            np.zeros((2, 2), dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            2,
            r"^y_true and y_pred must have the same shape$",
        ),
        (
            np.array([0, 2], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            2,
            r"^true class ID is outside the declared class range$",
        ),
        (
            np.array([0, 1], dtype=np.int64),
            np.array([0, -1], dtype=np.int64),
            2,
            r"^predicted class ID is outside the declared class range$",
        ),
        (
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int64),
            2,
            r"^y_true and y_pred must have int64 dtype$",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int32),
            2,
            r"^y_true and y_pred must have int64 dtype$",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            0,
            r"^num_classes must be a positive integer$",
        ),
        (
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            True,
            r"^num_classes must be a positive integer$",
        ),
    ],
)
def test_compute_confusion_rejects_shape_dtype_class_and_label_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    expected: str,
) -> None:
    confusion = load_confusion_module()

    with pytest.raises((TypeError, ValueError), match=expected):
        confusion.compute_confusion(y_true, y_pred, num_classes=num_classes)


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (np.zeros((2, 3), dtype=np.int64), r"^confusion matrix must be square$"),
        (np.zeros(3, dtype=np.int64), r"^confusion matrix must be two-dimensional$"),
        (np.array([[1, -1], [0, 1]], dtype=np.int64), r"^confusion counts must be nonnegative$"),
        (np.eye(2, dtype=np.int32), r"^confusion matrix must have int64 dtype$"),
        (
            np.zeros((0, 0), dtype=np.int64),
            r"^confusion matrix must contain at least one class$",
        ),
    ],
)
def test_summarize_confusion_rejects_invalid_matrices(matrix: np.ndarray, expected: str) -> None:
    confusion = load_confusion_module()

    with pytest.raises((TypeError, ValueError), match=expected):
        confusion.summarize_confusion(matrix)


def test_summarize_confusion_rejects_zero_valid_pixel_denominator() -> None:
    confusion = load_confusion_module()

    with pytest.raises(ValueError, match=r"^confusion matrix contains no valid pixel"):
        confusion.summarize_confusion(np.zeros((3, 3), dtype=np.int64))


def test_critical_false_negative_rate_is_pooled_and_absence_is_none() -> None:
    risk = load_risk_module()
    matrix = np.array([[5, 1, 0], [0, 3, 1], [2, 0, 2]], dtype=np.int64)

    assert risk.critical_false_negative_rate(matrix, (1, 2)) == pytest.approx(3 / 8)
    assert risk.critical_false_negative_rate(matrix, ()) is None
    assert risk.critical_false_negative_rate(np.diag([2, 0, 0]), (1, 2)) is None


@pytest.mark.parametrize("class_ids", [(0, 0), (-1,), (3,), (True,)])
def test_critical_false_negative_rate_rejects_invalid_class_ids(
    class_ids: tuple[int, ...],
) -> None:
    risk = load_risk_module()
    matrix = np.eye(3, dtype=np.int64)

    with pytest.raises((TypeError, ValueError), match=r"^critical class ID"):
        risk.critical_false_negative_rate(matrix, class_ids)


@given(st.integers(min_value=1, max_value=100))
def test_positive_count_scaling_preserves_standard_and_rate_metrics(scale: int) -> None:
    confusion = load_confusion_module()
    risk = load_risk_module()
    matrix = np.array([[5, 1, 0], [0, 3, 1], [2, 0, 2]], dtype=np.int64)

    assert confusion.summarize_confusion(matrix * scale) == confusion.summarize_confusion(matrix)
    assert risk.critical_false_negative_rate(
        matrix * scale, (1, 2)
    ) == risk.critical_false_negative_rate(matrix, (1, 2))


@given(
    height=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=1, max_value=8),
    num_classes=st.integers(min_value=1, max_value=6),
)
def test_confusion_shape_and_total_follow_arbitrary_valid_input_shape(
    height: int,
    width: int,
    num_classes: int,
) -> None:
    confusion = load_confusion_module()
    y_true = (np.arange(height * width, dtype=np.int64) % num_classes).reshape(height, width)
    y_pred = np.flip(y_true, axis=1).copy()

    matrix = confusion.compute_confusion(y_true, y_pred, num_classes)

    assert matrix.shape == (num_classes, num_classes)
    assert matrix.dtype == np.int64
    assert int(matrix.sum()) == height * width


@given(
    num_classes=st.integers(min_value=1, max_value=20),
    invalid_side=st.sampled_from(("negative", "upper")),
    invalid_array=st.sampled_from(("true", "predicted")),
)
def test_out_of_range_label_errors_hold_across_class_counts(
    num_classes: int,
    invalid_side: str,
    invalid_array: str,
) -> None:
    confusion = load_confusion_module()
    invalid_id = -1 if invalid_side == "negative" else num_classes
    y_true = np.array([invalid_id if invalid_array == "true" else 0], dtype=np.int64)
    y_pred = np.array([invalid_id if invalid_array == "predicted" else 0], dtype=np.int64)

    with pytest.raises(ValueError, match=invalid_array):
        confusion.compute_confusion(y_true, y_pred, num_classes)


def test_class_zero_is_a_valid_critical_class() -> None:
    """Zero is a class ID, not a sentinel.

    A guard written `class_id <= 0` instead of `< 0` would reject the first
    class in the taxonomy, and BDD100K's class 0 is `road` — the largest class
    in the dataset. The rejection would look like a configuration error rather
    than a bug in the validator.
    """

    risk = pytest.importorskip("drivemetrics.metrics.risk")
    confusion = np.zeros((3, 3), dtype=np.int64)
    confusion[0, 0] = 8
    confusion[0, 1] = 2

    assert risk.critical_false_negative_rate(confusion, (0,)) == pytest.approx(0.2)
