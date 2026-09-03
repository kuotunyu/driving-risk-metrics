"""Hand-computed contracts for the locked learning-rate schedule and optimizer table."""

from __future__ import annotations

from types import ModuleType

import pytest
from hypothesis import given
from hypothesis import strategies as st


def load_schedule_module() -> ModuleType:
    try:
        from drivemetrics.training import schedule
    except ImportError:
        pytest.fail("drivemetrics.training.schedule is missing", pytrace=False)
    return schedule


@pytest.mark.parametrize(
    ("step", "expected"),
    [(1, 0.5), (2, 1.0), (3, 0.5), (4, 0.0)],
)
def test_the_schedule_warms_up_linearly_then_decays_to_zero(step: int, expected: float) -> None:
    """A shifted step index would change every learning rate in the locked protocol.

    With base 1.0, two warmup steps, four total steps and power 1.0 the schedule is
    exactly 0.5, 1.0, 0.5, 0.0, so warmup peaks on its last step and decay reaches
    zero on the final step.
    """

    schedule = load_schedule_module()

    value = schedule.polynomial_learning_rate(
        step,
        base_lr=1.0,
        warmup_steps=2,
        total_steps=4,
        power=1.0,
    )

    assert value == expected


def test_the_first_protocol_step_uses_exactly_one_warmup_fraction() -> None:
    """Starting warmup at full learning rate would destabilize the first updates."""

    schedule = load_schedule_module()

    assert schedule.polynomial_learning_rate(1, base_lr=0.01) == 0.01 / 1000


def test_the_last_warmup_step_reaches_the_base_learning_rate() -> None:
    """An off-by-one warmup boundary would silently shorten or extend the ramp."""

    schedule = load_schedule_module()

    assert schedule.polynomial_learning_rate(1000, base_lr=0.01) == 0.01


def test_the_first_step_after_warmup_has_already_started_decaying() -> None:
    """Restarting the ramp after warmup would apply the base rate twice."""

    schedule = load_schedule_module()

    peak = schedule.polynomial_learning_rate(1000, base_lr=0.01)
    first_decayed = schedule.polynomial_learning_rate(1001, base_lr=0.01)

    assert 0.0 < first_decayed < peak


def test_the_final_protocol_step_reaches_zero() -> None:
    """A nonzero final rate would mean the decay never completed within 30,000 steps."""

    schedule = load_schedule_module()

    assert schedule.polynomial_learning_rate(30000, base_lr=0.01) == 0.0


@pytest.mark.parametrize("step", [0, -1, 30001, True, 2.5])
def test_a_step_outside_the_locked_range_fails_closed(step: object) -> None:
    """Silently clamping would let a misconfigured loop train past the locked horizon."""

    schedule = load_schedule_module()

    with pytest.raises(ValueError, match=r"^step must be"):
        schedule.polynomial_learning_rate(step, base_lr=0.01)  # type: ignore[arg-type]


@given(step=st.integers(min_value=1, max_value=30000))
def test_every_locked_step_stays_within_zero_and_the_base_rate(step: int) -> None:
    """A rate above base or below zero would break the recorded optimizer contract."""

    schedule = load_schedule_module()

    value = schedule.polynomial_learning_rate(step, base_lr=0.01)

    assert 0.0 <= value <= 0.01


@given(step=st.integers(min_value=1001, max_value=29999))
def test_the_rate_never_increases_after_warmup(step: int) -> None:
    """Any rebound after warmup would be a cyclic schedule, not the approved decay."""

    schedule = load_schedule_module()

    assert schedule.polynomial_learning_rate(
        step, base_lr=0.01
    ) >= schedule.polynomial_learning_rate(
        step + 1,
        base_lr=0.01,
    )


@pytest.mark.parametrize("base_lr", [0.0, -0.01, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_base_rate_fails_closed(base_lr: float) -> None:
    """A zero or undefined base rate would silently train a model that never learns."""

    schedule = load_schedule_module()

    with pytest.raises(ValueError, match=r"^base_lr must be a finite positive number"):
        schedule.polynomial_learning_rate(1, base_lr=base_lr)


def test_warmup_may_not_consume_the_whole_schedule() -> None:
    """Warmup at or past the horizon would divide by zero inside the decay branch."""

    schedule = load_schedule_module()

    with pytest.raises(ValueError, match=r"^warmup_steps must be smaller than total_steps"):
        schedule.polynomial_learning_rate(1, base_lr=0.01, warmup_steps=10, total_steps=10)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        (
            "upernet_convnextv2_tiny",
            {
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "weight_decay": 0.05,
            },
        ),
        (
            "upernet_dinov2_small",
            {
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "weight_decay": 0.05,
            },
        ),
        (
            "segformer_b2",
            {"optimizer": "adamw", "learning_rate": 0.00006, "weight_decay": 0.01},
        ),
    ],
)
def test_the_optimizer_table_pins_the_approved_values(
    model_name: str,
    expected: dict[str, object],
) -> None:
    """Drifting from the approved optimizer would invalidate every cross-model comparison."""

    schedule = load_schedule_module()

    assert schedule.optimizer_spec(model_name) == expected


def test_an_unapproved_model_has_no_optimizer_specification() -> None:
    """Falling back to a default optimizer would train an unlabelled configuration."""

    schedule = load_schedule_module()

    with pytest.raises(
        ValueError, match=r"^'setr' is not an approved model with a pinned optimizer"
    ):
        schedule.optimizer_spec("setr")


def test_each_call_returns_an_independent_specification() -> None:
    """A shared mutable table would let one caller silently retune every later run."""

    schedule = load_schedule_module()

    first = schedule.optimizer_spec("upernet_convnextv2_tiny")
    first["learning_rate"] = 999.0

    assert schedule.optimizer_spec("upernet_convnextv2_tiny")["learning_rate"] == 0.0001


def test_training_package_exports_the_schedule_entry_points() -> None:
    """The engine and the CLI consume these through the package entry point."""

    import drivemetrics.training as training

    schedule = load_schedule_module()
    assert training.polynomial_learning_rate is schedule.polynomial_learning_rate
    assert training.optimizer_spec is schedule.optimizer_spec


@pytest.mark.parametrize("total_steps", [0, -1, True, 2.5])
def test_a_non_positive_or_non_integer_horizon_fails_closed(total_steps: object) -> None:
    """A zero or fractional horizon would make the decay denominator meaningless."""

    schedule = load_schedule_module()

    with pytest.raises(ValueError, match=r"^total_steps must be an integer of at least"):
        schedule.polynomial_learning_rate(
            1,
            base_lr=0.01,
            total_steps=total_steps,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("warmup_steps", [-1, True, 2.5])
def test_a_negative_or_non_integer_warmup_fails_closed(warmup_steps: object) -> None:
    """A fractional or boolean warmup boundary would split the schedule ambiguously."""

    schedule = load_schedule_module()

    with pytest.raises(ValueError, match=r"^warmup_steps must be an integer of at least"):
        schedule.polynomial_learning_rate(
            1,
            base_lr=0.01,
            warmup_steps=warmup_steps,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("base_lr", ["0.01", None, True])
def test_a_non_numeric_base_rate_fails_closed(base_lr: object) -> None:
    """A string rate would multiply into a repeated string instead of failing."""

    schedule = load_schedule_module()

    with pytest.raises(ValueError, match=r"^base_lr must be a finite positive number"):
        schedule.polynomial_learning_rate(1, base_lr=base_lr)  # type: ignore[arg-type]


@pytest.mark.parametrize("power", [0.0, -0.9, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_power_fails_closed(power: float) -> None:
    """A zero or negative power would turn the decay into a constant or a rising curve."""

    schedule = load_schedule_module()

    with pytest.raises(ValueError, match=r"^power must be a finite positive number"):
        schedule.polynomial_learning_rate(1, base_lr=0.01, power=power)
