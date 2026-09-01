"""CPU-only scalar temperature fitting for multiclass logits."""

from __future__ import annotations

import math
import re
import secrets
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

_LOG_TEMPERATURE_BOUNDS = (-5.0, 5.0)
_MINIMUM_TEMPERATURE = math.exp(_LOG_TEMPERATURE_BOUNDS[0])
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CalibrationProvenance:
    """Identity of the frozen cohort used for formal temperature fitting."""

    split_name: str
    sample_ids: tuple[str, ...]
    dataset_manifest_sha256: str


def _validated_calibration_inputs(
    logits: Float64Array,
    targets: Int64Array,
    ignore_index: int,
) -> tuple[Float64Array, Int64Array]:
    if not isinstance(logits, np.ndarray) or logits.dtype != np.float64:
        raise ValueError("logits must be a float64 array")
    if not isinstance(targets, np.ndarray) or targets.dtype != np.int64:
        raise ValueError("targets must be an int64 array")
    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("logits must have at least two classes on the final axis")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets shape must match logits leading shape")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite")
    if np.any(np.abs(logits) > np.finfo(np.float64).max * _MINIMUM_TEMPERATURE / 2.0):
        raise ValueError("logit range is unsafe for bounded temperature scaling and cross-entropy")

    flattened_logits = logits.reshape(-1, logits.shape[-1])
    flattened_targets = targets.reshape(-1)
    valid = flattened_targets != ignore_index
    if not np.any(valid):
        raise ValueError("no valid calibration targets")
    valid_targets = flattened_targets[valid]
    if np.any((valid_targets < 0) | (valid_targets >= logits.shape[-1])):
        raise ValueError("target is outside the logits class range")
    return flattened_logits[valid], valid_targets


def apply_temperature(logits: Float64Array, temperature: float) -> Float64Array:
    """Divide float64 logits by one finite positive scalar temperature."""

    if not isinstance(logits, np.ndarray) or logits.dtype != np.float64:
        raise ValueError("logits must be a float64 array")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        calibrated = logits / temperature
    if not np.all(np.isfinite(calibrated)):
        raise ValueError("temperature scaling overflowed float64")
    return calibrated


def fit_scalar_temperature(
    logits: Float64Array,
    targets: Int64Array,
    ignore_index: int = 255,
) -> float:
    """Fit one bounded log-temperature by minimizing multiclass NLL."""

    valid_logits, valid_targets = _validated_calibration_inputs(logits, targets, ignore_index)
    row_indices = np.arange(valid_targets.size)

    def negative_log_likelihood(log_temperature: float) -> float:
        scaled = valid_logits / math.exp(log_temperature)
        losses = logsumexp(scaled, axis=1) - scaled[row_indices, valid_targets]
        return float(np.sum(losses / losses.size))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=_LOG_TEMPERATURE_BOUNDS,
        method="bounded",
        options={"xatol": 1e-12},
    )
    if (
        not result.success
        or not math.isfinite(float(result.fun))
        or not math.isfinite(float(result.x))
    ):
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return math.exp(float(result.x))


def fit_provenance_checked_temperature(
    logits: Float64Array,
    targets: Int64Array,
    provenance: CalibrationProvenance,
    *,
    expected_dataset_manifest_sha256: str,
    expected_sample_ids: tuple[str, ...],
    ignore_index: int = 255,
) -> float:
    """Fit only when inputs match an expected frozen calibration cohort exactly."""

    if provenance.split_name != "calibration":
        raise ValueError("temperature fitting requires the calibration split")
    _validate_sample_ids(provenance.sample_ids, "calibration provenance")
    _validate_sample_ids(expected_sample_ids, "expected calibration")
    if _SHA256_PATTERN.fullmatch(provenance.dataset_manifest_sha256) is None:
        raise ValueError("dataset_manifest_sha256 must be 64 lowercase hexadecimal characters")
    if _SHA256_PATTERN.fullmatch(expected_dataset_manifest_sha256) is None:
        raise ValueError(
            "expected_dataset_manifest_sha256 must be 64 lowercase hexadecimal characters"
        )
    if not secrets.compare_digest(
        provenance.dataset_manifest_sha256,
        expected_dataset_manifest_sha256,
    ):
        raise ValueError("calibration provenance manifest hash does not match expected manifest")
    if provenance.sample_ids != expected_sample_ids:
        raise ValueError("calibration provenance membership does not match expected sample order")
    if (
        isinstance(logits, np.ndarray)
        and logits.ndim >= 1
        and logits.shape[0] != len(provenance.sample_ids)
    ):
        raise ValueError("logits batch dimension must match calibration sample_ids")
    return fit_scalar_temperature(logits, targets, ignore_index)


def _validate_sample_ids(sample_ids: tuple[str, ...], label: str) -> None:
    if not sample_ids:
        raise ValueError(f"{label} sample_ids must not be empty")
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError(f"{label} sample_ids must contain nonempty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{label} sample_ids must be unique")


def softmax_probabilities(logits: Float64Array) -> Float64Array:
    """Return numerically stable probabilities with classes on the final axis.

    The row maximum is subtracted before exponentiation, so saturated logits
    cannot overflow into a non-finite probability row.
    """

    if not isinstance(logits, np.ndarray) or logits.dtype != np.float64:
        raise ValueError("logits must be a float64 array")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have at least two classes on the final axis")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite")
    exponentiated = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)
