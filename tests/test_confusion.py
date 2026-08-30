"""Tests for confusion accumulation and the two IoU aggregations.

The aggregation tests are the important ones. They pin down the exact
discrepancy between the dataset-level protocol and the per-image ``nanmean``
protocol used by the source notebooks, so that the claim "these numbers were not
comparable" is enforced by CI rather than asserted in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivemetrics import (
    IGNORE_INDEX,
    NUM_CLASSES,
    ConfusionMatrix,
    confusion_from_pair,
    dataset_iou,
    frequency_weighted_iou,
    per_image_nanmean_iou,
)


def test_perfect_prediction_gives_unit_iou():
    target = np.array([[0, 1], [2, 3]])
    cm = confusion_from_pair(target, target.copy())
    iou = dataset_iou(cm)
    assert cm.pixel_accuracy() == pytest.approx(1.0)
    assert iou.mean == pytest.approx(1.0)
    assert iou.n_classes_counted == 4  # only the 4 classes present are defined


def test_absent_class_is_nan_not_zero():
    """A class with no ground truth and no prediction must not count as 0.0.

    Scoring it as zero would drag mIoU down for a class the model was never
    asked about; scoring it as one would inflate it. NaN, excluded from the
    mean, is the only defensible option.
    """
    target = np.zeros((4, 4), dtype=np.int64)
    cm = confusion_from_pair(target, target.copy())
    iou = cm.per_class_iou()
    assert iou[0] == pytest.approx(1.0)
    assert np.all(np.isnan(iou[1:]))
    assert dataset_iou(cm).n_classes_counted == 1


def test_ignore_index_excluded_from_every_statistic():
    target = np.array([[IGNORE_INDEX, IGNORE_INDEX], [3, 3]])
    pred = np.array([[0, 0], [3, 3]])  # both void pixels predicted wrong
    cm = confusion_from_pair(target, pred)
    # Void pixels must not appear anywhere: not in the totals, not as errors.
    assert cm.matrix.sum() == 2
    assert cm.pixel_accuracy() == pytest.approx(1.0)
    assert dataset_iou(cm).mean == pytest.approx(1.0)


def test_out_of_range_labels_dropped_not_clipped():
    """A stray label of 99 must not be folded into class 0."""
    target = np.array([[99, 3]])
    pred = np.array([[0, 3]])
    cm = confusion_from_pair(target, pred)
    assert cm.matrix.sum() == 1
    assert cm.support[0] == 0


def test_iou_matches_hand_computation():
    # 4 pixels of class 3, of which 2 predicted as 3 and 2 as 0.
    # 2 pixels of class 0, both predicted 0.
    target = np.array([3, 3, 3, 3, 0, 0])
    pred = np.array([3, 3, 0, 0, 0, 0])
    cm = confusion_from_pair(target, pred)
    iou = cm.per_class_iou()
    # class 3: TP=2, union = 4 (gt) + 2 (pred) - 2 = 4 -> 0.5
    assert iou[3] == pytest.approx(0.5)
    # class 0: TP=2, union = 2 (gt) + 4 (pred) - 2 = 4 -> 0.5
    assert iou[0] == pytest.approx(0.5)
    assert cm.per_class_recall()[3] == pytest.approx(0.5)
    assert cm.per_class_precision()[3] == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same number of pixels"):
        confusion_from_pair(np.zeros((2, 2)), np.zeros((3, 3)))


def test_matrices_add():
    a = confusion_from_pair(np.array([0, 0]), np.array([0, 0]))
    b = confusion_from_pair(np.array([3, 3]), np.array([3, 0]))
    merged = a + b
    assert merged.matrix.sum() == 4
    assert merged.n_images == 2
    assert merged.matrix[3, 3] == 1
    assert merged.matrix[3, 0] == 1


# ---------------------------------------------------------------------------
# The aggregation discrepancy — the reason this package exists
# ---------------------------------------------------------------------------


def test_per_image_nanmean_flatters_rare_classes():
    """Reproduce the structural bias in the notebooks' aggregation.

    A rare class appears in two images: as a large object that the model misses
    entirely, and as a tiny object that the model happens to nail.

    Pooling pixels (dataset protocol) is dominated by the large failure, because
    that is where nearly all the pedestrian pixels are: IoU near zero.
    Averaging per image gives the tiny success equal say with the large failure,
    lifting the class to roughly one half.

    This is the mechanism that makes the two protocols non-comparable on exactly
    the classes this repo cares about — rare ones, whose per-image sample is
    small enough for one lucky image to move the number a long way.
    """
    # Image A: a large pedestrian region, entirely missed.
    big_t = np.full((20, 20), 3, dtype=np.int64)
    big_t[:10, :10] = 9  # 100 pedestrian pixels
    big_p = np.full((20, 20), 3, dtype=np.int64)  # all predicted road

    # Image B: a single pedestrian pixel, predicted exactly right.
    small_t = np.full((20, 20), 3, dtype=np.int64)
    small_t[0, 0] = 9
    small_p = np.full((20, 20), 3, dtype=np.int64)
    small_p[0, 0] = 9

    targets = [big_t, small_t]
    preds = [big_p, small_p]

    cm = ConfusionMatrix()
    for tt, pp in zip(targets, preds):
        cm.update(tt, pp)
    ds = dataset_iou(cm)
    pi = per_image_nanmean_iou(targets, preds)

    # Pooled over pixels the class is a near-total failure...
    assert ds.per_class[9] < 0.02
    # ...but averaged per image it looks like a coin flip.
    assert pi.per_class[9] == pytest.approx(0.5)
    # A greater-than-20x gap on the class that matters most.
    assert pi.per_class[9] > 20 * ds.per_class[9]
    assert ds.aggregation == "dataset"
    assert pi.aggregation == "per_image_nanmean"


def test_per_image_nanmean_can_exceed_dataset_iou():
    """Per-image averaging can report a higher score on identical predictions.

    Image A is large and segmented poorly; image B is tiny and perfect. Weighted
    by pixels (dataset protocol) the poor result dominates. Averaged per image,
    the tiny perfect image gets equal say and lifts the score.
    """
    big_t = np.full((100, 100), 3, dtype=np.int64)
    big_p = np.full((100, 100), 3, dtype=np.int64)
    big_p[:50] = 0  # half of a large image is wrong

    small_t = np.full((2, 2), 3, dtype=np.int64)
    small_p = small_t.copy()  # a tiny perfect image

    targets = [big_t, small_t]
    preds = [big_p, small_p]

    cm = ConfusionMatrix()
    for t, p in zip(targets, preds):
        cm.update(t, p)

    ds = dataset_iou(cm)
    pi = per_image_nanmean_iou(targets, preds)
    assert pi.mean > ds.mean


def test_n_classes_counted_exposes_the_three_class_bug():
    """The specific failure that made four notebook results incomparable.

    A run that only ever sees 3 classes reports a mean over 3 classes. Nothing
    in the number itself says so, which is how a 3-class 0.82 came to be
    compared against an 11-class 0.57. ``n_classes_counted`` makes it visible.
    """
    three_class = np.array([[0, 1], [2, 2]])
    cm3 = confusion_from_pair(three_class, three_class.copy())
    assert dataset_iou(cm3).n_classes_counted == 3

    eleven = np.arange(NUM_CLASSES).reshape(1, -1)
    cm11 = confusion_from_pair(eleven, eleven.copy())
    assert dataset_iou(cm11).n_classes_counted == NUM_CLASSES


def test_frequency_weighted_iou_is_dominated_by_common_classes():
    """Demonstrates why frequency weighting is the wrong direction for safety."""
    # 99 road pixels segmented perfectly, 1 pedestrian pixel missed.
    target = np.array([3] * 99 + [9])
    pred = np.array([3] * 100)
    cm = confusion_from_pair(target, pred)
    fw = frequency_weighted_iou(cm)
    ds = dataset_iou(cm).mean
    # Frequency weighting reports near-perfect; the unweighted class mean does not.
    assert fw > 0.95
    assert ds < 0.55


def test_empty_input_is_nan_not_crash():
    cm = ConfusionMatrix()
    assert np.isnan(cm.pixel_accuracy())
    assert np.isnan(dataset_iou(cm).mean)
    result = per_image_nanmean_iou([], [])
    assert np.isnan(result.mean)
    assert result.n_classes_counted == 0
