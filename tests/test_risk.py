"""Tests for the harm model and the risk-weighted metrics.

The properties pinned here are the ones that justify the metric existing at all:
that missing a pedestrian costs more than missing sky, that missing costs more
than hallucinating, and that a uniform harm model collapses back to something
mIoU-like. If any of these stop holding, the metric is no longer measuring what
the README says it measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivemetrics import (
    CLASS_NAMES,
    DEFAULT_HARM,
    UNIFORM_HARM,
    ConfusionMatrix,
    HarmModel,
    confusion_from_pair,
    dataset_iou,
    expected_risk,
    risk_contribution_by_confusion,
    risk_weighted_iou,
    safety_recall,
)
from drivemetrics.taxonomy import class_index

PED = class_index("Pedestrian")
ROAD = class_index("Road")
SKY = class_index("Sky")
BUILDING = class_index("Building")


# ---------------------------------------------------------------------------
# Harm model structure
# ---------------------------------------------------------------------------


def test_cost_matrix_diagonal_is_free():
    cost = np.array(DEFAULT_HARM.cost_matrix())
    assert np.all(np.diag(cost) == 0.0)


def test_missing_a_pedestrian_costs_more_than_missing_sky():
    cost = np.array(DEFAULT_HARM.cost_matrix())
    assert cost[PED, ROAD] > cost[SKY, ROAD]


def test_miss_costs_more_than_phantom():
    """Asymmetry is the whole reason for a cost matrix rather than class weights."""
    cost = np.array(DEFAULT_HARM.cost_matrix())
    missed_pedestrian = cost[PED, ROAD]   # a person we did not see
    phantom_pedestrian = cost[ROAD, PED]  # a brake for nobody
    assert missed_pedestrian > phantom_pedestrian
    assert missed_pedestrian / phantom_pedestrian > 5


def test_confusing_pedestrian_for_cyclist_is_free():
    """Both are VRUs; a planner brakes for either, so the substitution is free.

    This is the property that forced the subtractive cost decomposition. Under a
    naive ``miss(t) + phantom(p)`` rule this confusion would cost *more* than
    calling the pedestrian "road", which is exactly backwards.
    """
    cost = np.array(DEFAULT_HARM.cost_matrix())
    cyc = class_index("Bicyclist")
    assert cost[PED, cyc] == pytest.approx(0.0)
    assert cost[PED, ROAD] > 90


def test_partial_credit_for_seeing_some_obstacle():
    """Pedestrian mistaken for a car is bad, but not as bad as for empty road."""
    cost = np.array(DEFAULT_HARM.cost_matrix())
    car = class_index("Car")
    assert 0 < cost[PED, car] < cost[PED, ROAD]


def test_cost_matrix_is_asymmetric():
    """A missed person and an unnecessary brake are different events."""
    cost = np.array(DEFAULT_HARM.cost_matrix())
    assert cost[PED, ROAD] != cost[ROAD, PED]
    assert cost[PED, ROAD] > cost[ROAD, PED]


def test_harm_model_rejects_incomplete_tables():
    with pytest.raises(ValueError, match="no entry for"):
        HarmModel(name="broken", miss_cost={0: 1.0}, phantom_cost={0: 1.0})


def test_harm_model_rejects_negative_costs():
    miss = dict.fromkeys(range(len(CLASS_NAMES)), 1.0)
    phantom = dict(miss)
    miss[PED] = -1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        HarmModel(name="negative", miss_cost=miss, phantom_cost=phantom)


def test_scaled_produces_an_independent_model():
    scaled = DEFAULT_HARM.scaled(2.0, [PED])
    assert scaled.miss_cost[PED] == 2 * DEFAULT_HARM.miss_cost[PED]
    # The original must not be mutated — the sweep reuses it many times.
    assert DEFAULT_HARM.miss_cost[PED] == 100.0
    assert scaled.miss_cost[ROAD] == DEFAULT_HARM.miss_cost[ROAD]


# ---------------------------------------------------------------------------
# expected_risk
# ---------------------------------------------------------------------------


def test_perfect_prediction_has_zero_risk_and_unit_skill():
    target = np.array([[ROAD, ROAD], [PED, SKY]])
    res = expected_risk(confusion_from_pair(target, target.copy()), DEFAULT_HARM)
    assert res.expected_risk == pytest.approx(0.0)
    assert res.risk_skill == pytest.approx(1.0)


def test_expected_risk_matches_hand_computation():
    # Four pixels: three road (correct), one pedestrian predicted as road.
    target = np.array([ROAD, ROAD, ROAD, PED])
    pred = np.array([ROAD, ROAD, ROAD, ROAD])
    res = expected_risk(confusion_from_pair(target, pred), DEFAULT_HARM)
    # Severity not signalled = miss(Pedestrian) 100 - miss(Road) 3 = 97.
    # No over-call charge, because Road is less severe than Pedestrian.
    # One such pixel out of four.
    assert res.expected_risk == pytest.approx(97.0 / 4.0)


def test_risk_is_dominated_by_the_rare_dangerous_class():
    """A model can be 99% correct by pixels and still have almost all its risk
    concentrated in the 1% it got wrong."""
    target = np.array([ROAD] * 99 + [PED])
    pred = np.array([ROAD] * 100)
    cm = confusion_from_pair(target, pred)
    res = expected_risk(cm, DEFAULT_HARM)
    assert cm.pixel_accuracy() == pytest.approx(0.99)
    assert res.risk_share_by_true_class[PED] == pytest.approx(1.0)


def test_two_models_can_tie_on_iou_and_differ_on_risk():
    """The headline claim of the repo, as an executable statement.

    Model A misses a pedestrian and segments a patch of sky perfectly.
    Model B segments the pedestrian perfectly and mislabels the same-sized
    patch of sky. Both make exactly the same number of wrong pixels, so
    accuracy is identical -- but only one of them drove into someone.
    """
    target = np.array([PED] * 10 + [SKY] * 10 + [ROAD] * 80)

    pred_a = np.array([ROAD] * 10 + [SKY] * 10 + [ROAD] * 80)  # missed the person
    pred_b = np.array([PED] * 10 + [ROAD] * 10 + [ROAD] * 80)  # mislabelled sky

    cm_a = confusion_from_pair(target, pred_a)
    cm_b = confusion_from_pair(target, pred_b)

    # Identical pixel accuracy: 10 wrong pixels each.
    assert cm_a.pixel_accuracy() == pytest.approx(cm_b.pixel_accuracy())

    risk_a = expected_risk(cm_a, DEFAULT_HARM).expected_risk
    risk_b = expected_risk(cm_b, DEFAULT_HARM).expected_risk
    assert risk_a > risk_b
    assert risk_a / risk_b > 10

    # And under a uniform harm model the distinction vanishes -- confirming the
    # difference comes from the harm model, not from some artefact of the maths.
    uni_a = expected_risk(cm_a, UNIFORM_HARM).expected_risk
    uni_b = expected_risk(cm_b, UNIFORM_HARM).expected_risk
    assert uni_a == pytest.approx(uni_b)


def test_cheapest_constant_never_costs_more_than_the_majority_one():
    """Invariant: the minimum over constants is a minimum."""
    target = np.array([ROAD] * 90 + [BUILDING] * 9 + [PED])
    pred = np.array([ROAD] * 100)
    res = expected_risk(confusion_from_pair(target, pred), DEFAULT_HARM)
    assert res.majority_class == "Road"
    assert res.best_constant_risk <= res.majority_risk


def test_on_the_real_camvid_prior_the_hedge_beats_the_majority_class():
    """A property of the harm model that a reader deserves to be told about.

    Under CamVid's actual class distribution and the default harm model, the
    cheapest constant prediction is *not* "everything is road" but a
    mid-severity hedge — because over-calling danger is cheap and under-calling
    it is not. That is a defensible consequence of the cost asymmetry, but it
    makes the cheapest constant a strange yardstick, which is exactly why the
    headline skill score is normalised against the majority class instead.
    """
    # CamVid training-set pixel shares, measured from the 367 annotation masks.
    shares = {
        0: 16.845, 1: 23.259, 2: 0.983, 3: 31.658, 4: 4.486, 5: 9.724,
        6: 1.173, 7: 1.127, 8: 5.866, 9: 0.639, 10: 0.292,
    }
    target = np.concatenate(
        [np.full(int(round(v * 100)), k, dtype=np.int64) for k, v in shares.items()]
    )
    pred = np.full_like(target, ROAD)
    res = expected_risk(confusion_from_pair(target, pred), DEFAULT_HARM)

    assert res.majority_class == "Road"
    assert res.best_constant_class != "Road"
    assert res.best_constant_risk < res.majority_risk
    # And therefore the strict skill score is the harsher of the two.
    assert res.risk_skill_strict < res.risk_skill


def test_strict_skill_is_never_above_headline_skill():
    """`risk_skill_strict` uses a harder denominator, so it must not flatter."""
    rng = np.random.default_rng(7)
    target = rng.choice([ROAD, BUILDING, SKY, PED], size=500, p=[0.6, 0.2, 0.15, 0.05])
    pred = np.where(rng.random(500) < 0.8, target, ROAD)
    res = expected_risk(confusion_from_pair(target, pred), DEFAULT_HARM)
    assert res.risk_skill_strict <= res.risk_skill + 1e-9


def test_empty_matrix_is_nan_not_crash():
    res = expected_risk(ConfusionMatrix(), DEFAULT_HARM)
    assert np.isnan(res.expected_risk)
    assert np.isnan(res.risk_skill)
    assert np.isnan(res.risk_skill_strict)


# ---------------------------------------------------------------------------
# risk_weighted_iou and friends
# ---------------------------------------------------------------------------


def test_uniform_harm_makes_risk_weighted_iou_equal_miou():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 11, size=(32, 32))
    pred = rng.integers(0, 11, size=(32, 32))
    cm = confusion_from_pair(target, pred)
    assert risk_weighted_iou(cm, UNIFORM_HARM) == pytest.approx(dataset_iou(cm).mean)


def test_risk_weighted_iou_penalises_the_dangerous_failure_harder():
    target = np.array([PED] * 50 + [SKY] * 50)
    miss_ped = np.array([ROAD] * 50 + [SKY] * 50)   # lost the pedestrian
    miss_sky = np.array([PED] * 50 + [ROAD] * 50)   # lost the sky

    cm_ped = confusion_from_pair(target, miss_ped)
    cm_sky = confusion_from_pair(target, miss_sky)

    assert dataset_iou(cm_ped).mean == pytest.approx(dataset_iou(cm_sky).mean)
    assert risk_weighted_iou(cm_ped, DEFAULT_HARM) < risk_weighted_iou(cm_sky, DEFAULT_HARM)


def test_safety_recall_reports_per_class_and_pooled():
    target = np.array([PED] * 10 + [ROAD] * 10)
    pred = np.array([PED] * 4 + [ROAD] * 16)
    out = safety_recall(confusion_from_pair(target, pred))
    assert out["Pedestrian"] == pytest.approx(0.4)
    assert np.isnan(out["Bicyclist"])  # absent from this image
    assert out["pooled"] == pytest.approx(0.4)


def test_risk_contribution_ranks_the_dangerous_confusion_first():
    # Many cheap sky/building confusions, one expensive pedestrian miss.
    target = np.array([SKY] * 200 + [PED] * 5)
    pred = np.array([BUILDING] * 200 + [ROAD] * 5)
    top = risk_contribution_by_confusion(confusion_from_pair(target, pred), DEFAULT_HARM)
    assert top[0]["true"] == "Pedestrian"
    assert top[0]["pred"] == "Road"
    assert top[0]["risk_share"] > 0.5
    assert sum(e["risk_share"] for e in top) <= 1.0 + 1e-9


def test_risk_contribution_empty_when_perfect():
    target = np.array([ROAD, PED])
    assert risk_contribution_by_confusion(
        confusion_from_pair(target, target.copy()), DEFAULT_HARM
    ) == []
