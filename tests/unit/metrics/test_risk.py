"""Hand-computed contracts for class-cost segmentation risk."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from types import ModuleType

import numpy as np
import pytest


def load_risk_module() -> ModuleType:
    try:
        from drivemetrics.metrics import risk
    except ImportError:
        pytest.fail("drivemetrics.metrics.risk is missing", pytrace=False)
    return risk


def load_profiles_module() -> ModuleType:
    try:
        from drivemetrics.protocol import risk_profiles
    except ImportError:
        pytest.fail("drivemetrics.protocol.risk_profiles is missing", pytrace=False)
    return risk_profiles


def make_profile(*, sensitivity: float = 1.0) -> object:
    profiles = load_profiles_module()
    return profiles.RiskProfile(
        name="three-class-test",
        class_cost={0: 0.5, 1: 1.0, 2: 1.5},
        critical_class_ids=(2,),
        sensitivity=sensitivity,
    )


def test_cost_risk_matches_hand_computed_false_negative_formula() -> None:
    risk = load_risk_module()
    matrix = np.array([[5, 1, 0], [0, 3, 1], [2, 0, 2]], dtype=np.int64)

    result = risk.compute_cost_risk(matrix, make_profile(sensitivity=2.0))

    # FN counts are (1, 1, 2); class 2's declared cost is doubled by sensitivity.
    assert result == pytest.approx((1 * 0.5 + 1 * 1.0 + 2 * 1.5 * 2.0) / 14)


def test_perfect_confusion_has_zero_cost_risk() -> None:
    risk = load_risk_module()
    profiles = load_profiles_module()
    no_critical = profiles.RiskProfile("three-class-balanced", {0: 1.0, 1: 1.0, 2: 1.0}, (), 1.0)

    assert risk.compute_cost_risk(np.diag([4, 3, 2]), make_profile()) == 0.0
    assert risk.compute_cost_risk(np.diag([4, 3, 2]), no_critical) == 0.0


def test_sensitivity_changes_only_critical_false_negative_cost() -> None:
    risk = load_risk_module()
    critical_miss = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int64)
    ordinary_miss = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int64)

    critical_risks = tuple(
        risk.compute_cost_risk(critical_miss, make_profile(sensitivity=value))
        for value in (0.5, 1.0, 2.0)
    )
    ordinary_risks = tuple(
        risk.compute_cost_risk(ordinary_miss, make_profile(sensitivity=value))
        for value in (0.5, 1.0, 2.0)
    )

    assert critical_risks == pytest.approx((0.75, 1.5, 3.0))
    assert ordinary_risks == pytest.approx((0.5, 0.5, 0.5))


def test_cost_risk_rejects_zero_denominator_and_profile_dimension_mismatch() -> None:
    risk = load_risk_module()

    with pytest.raises(ValueError, match="valid pixel"):
        risk.compute_cost_risk(np.zeros((3, 3), dtype=np.int64), make_profile())
    with pytest.raises(ValueError, match="exactly one cost"):
        risk.compute_cost_risk(np.eye(2, dtype=np.int64), make_profile())


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"name": ""}, "name"),
        ({"class_cost": {0: 0.0, 1: 0.0, 2: 0.0}}, "positive"),
        ({"class_cost": {0: 1.0, 1: -1.0, 2: 3.0}}, "nonnegative"),
        ({"class_cost": {0: 1.0, 1: float("inf"), 2: 1.0}}, "finite"),
        ({"class_cost": {0: 0.5, 1: 0.5, 2: 0.5}}, "mean non-zero"),
        ({"class_cost": {0: 1.0, 2: 1.0, 19: 1.0}}, "taxonomy"),
        ({"class_cost": {False: 1.0, 1: 1.0, 2: 1.0}}, "integer"),
        ({"critical_class_ids": (2, 2)}, "unique"),
        ({"critical_class_ids": (3,)}, "declared"),
        ({"critical_class_ids": (19,)}, "taxonomy"),
        ({"critical_class_ids": (True,)}, "integer"),
        ({"sensitivity": 1.5}, "sensitivity"),
        ({"sensitivity": Decimal("1.0")}, "built-in float"),
    ],
)
def test_risk_profile_rejects_invalid_programmatic_values(
    changes: dict[str, object], expected: str
) -> None:
    profiles = load_profiles_module()
    values: dict[str, object] = {
        "name": "three-class-test",
        "class_cost": {0: 0.5, 1: 1.0, 2: 1.5},
        "critical_class_ids": (2,),
        "sensitivity": 1.0,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=expected):
        profiles.RiskProfile(**values)


def test_risk_profile_is_frozen_and_copies_cost_mapping() -> None:
    profiles = load_profiles_module()
    source = {2: 1.5, 0: 0.5, 1: 1.0}
    critical_source = [2]
    profile = profiles.RiskProfile("three-class-test", source, critical_source, 1.0)

    source[2] = 99.0
    critical_source.append(0)
    assert tuple(profile.class_cost) == (0, 1, 2)
    assert profile.critical_class_ids == (2,)
    assert profile.class_cost[2] == 1.5
    with pytest.raises(TypeError):
        profile.class_cost[2] = 99.0
    with pytest.raises(FrozenInstanceError):
        profile.sensitivity = 2.0  # type: ignore[misc]
