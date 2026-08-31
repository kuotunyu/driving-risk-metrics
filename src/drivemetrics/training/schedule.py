"""Locked polynomial learning-rate schedule and the approved optimizer table."""

from __future__ import annotations

import math

from drivemetrics.models.registry import ModelName

WARMUP_STEPS = 1000
TOTAL_STEPS = 30000
DECAY_POWER = 0.9

_OPTIMIZER_TABLE: dict[str, dict[str, float | str]] = {
    "fcn_resnet50": {
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    },
    "deeplabv3_resnet50": {
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    },
    "segformer_b0": {
        "optimizer": "adamw",
        "learning_rate": 0.00006,
        "weight_decay": 0.01,
    },
}

OPTIMIZER_TABLE_MODELS: tuple[str, ...] = tuple(_OPTIMIZER_TABLE)


def _validate_integer_at_least(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def polynomial_learning_rate(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int = WARMUP_STEPS,
    total_steps: int = TOTAL_STEPS,
    power: float = DECAY_POWER,
) -> float:
    """Return the locked learning rate for one one-based optimizer step.

    Steps ``1..warmup_steps`` ramp linearly to ``base_lr``, so the last warmup
    step is exactly the base rate. Later steps decay as
    ``base_lr * (1 - progress) ** power`` over the remaining horizon, so the
    final step is exactly zero. Steps are one-based and outside the locked range
    the schedule fails closed rather than clamping.
    """

    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("step must be an integer within the locked range")
    _validate_integer_at_least("total_steps", total_steps, 1)
    _validate_integer_at_least("warmup_steps", warmup_steps, 0)
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")
    if isinstance(base_lr, bool) or not isinstance(base_lr, int | float):
        raise ValueError("base_lr must be a finite positive number")
    if not math.isfinite(base_lr) or base_lr <= 0.0:
        raise ValueError("base_lr must be a finite positive number")
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("power must be a finite positive number")
    if step < 1 or step > total_steps:
        raise ValueError("step must be within the locked range")

    if step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * (1.0 - progress) ** power


def optimizer_spec(model_name: ModelName) -> dict[str, float | str]:
    """Return an independent copy of the approved optimizer settings for one model.

    The copy prevents a caller from retuning the locked table for every later
    run through a shared mutable mapping.
    """

    if model_name not in _OPTIMIZER_TABLE:
        raise ValueError(f"{model_name!r} is not an approved model with a pinned optimizer")
    return dict(_OPTIMIZER_TABLE[model_name])
