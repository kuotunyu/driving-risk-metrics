"""Behavior tests for scalar temperature calibration."""

from __future__ import annotations

import importlib
import math
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st


def load_temperature() -> ModuleType:
    try:
        return importlib.import_module("drivemetrics.calibration.temperature")
    except ImportError:
        pytest.fail("drivemetrics.calibration.temperature is missing", pytrace=False)


def multiclass_nll(logits: np.ndarray, targets: np.ndarray) -> float:
    """Hand-independent stable NLL used only to assert the optimizer outcome."""

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return float(-np.mean(log_probabilities[np.arange(targets.size), targets]))


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_apply_temperature_rejects_nonpositive_or_nonfinite_temperature(
    temperature: float,
) -> None:
    """Accepting an invalid divisor would produce unusable confidence values."""

    module = load_temperature()
    logits = np.array([[2.0, -1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match=r"^temperature must be finite and positive"):
        module.apply_temperature(logits, temperature)


def test_temperature_one_is_an_exact_identity() -> None:
    """A no-op calibration must not introduce numerical drift."""

    module = load_temperature()
    logits = np.array([[2.0, -1.0], [-3.0, 4.0]], dtype=np.float64)

    calibrated = module.apply_temperature(logits, 1.0)

    np.testing.assert_array_equal(calibrated, logits)
    assert calibrated.dtype == np.float64


def test_fitted_temperature_improves_synthetic_overconfident_nll() -> None:
    """Returning a constant one would leave deliberately overconfident errors uncalibrated."""

    module = load_temperature()
    logits = np.array(
        [[8.0, 0.0], [8.0, 0.0], [8.0, 0.0], [8.0, 0.0]],
        dtype=np.float64,
    )
    targets = np.array([0, 0, 1, 1], dtype=np.int64)

    temperature = module.fit_scalar_temperature(logits, targets)
    calibrated = module.apply_temperature(logits, temperature)

    assert math.isfinite(temperature)
    assert math.exp(-5.0) <= temperature <= math.exp(5.0)
    assert multiclass_nll(calibrated, targets) < multiclass_nll(logits, targets)


@given(
    temperature=st.floats(
        min_value=0.01,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_positive_scalar_temperature_never_changes_hard_argmax(temperature: float) -> None:
    """Temperature calibration is invalid if it changes the segmentation mask."""

    module = load_temperature()
    logits = np.array(
        [[3.0, 1.0, -2.0], [-4.0, 5.0, 2.0], [0.5, -1.0, 4.0]],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        np.argmax(module.apply_temperature(logits, temperature), axis=-1),
        np.array([0, 1, 2]),
    )


def test_temperature_fit_ignores_declared_ignore_targets() -> None:
    """Padding pixels must not influence the fitted calibration parameter."""

    module = load_temperature()
    logits = np.array([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    with_ignore = np.array([0, 1, 255], dtype=np.int64)

    observed = module.fit_scalar_temperature(logits, with_ignore)
    expected = module.fit_scalar_temperature(logits[:2], with_ignore[:2])

    assert observed == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("logits", "targets", "message"),
    [
        (np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int64), "no valid"),
        (np.ones((2, 2), dtype=np.float64), np.array([255, 255], dtype=np.int64), "no valid"),
        (np.ones((2, 2), dtype=np.float64), np.array([0], dtype=np.int64), "shape"),
        (np.ones((2, 2), dtype=np.float32), np.array([0, 1], dtype=np.int64), "float64"),
        (np.ones((2, 2), dtype=np.float64), np.array([0, 1], dtype=np.int32), "int64"),
        (np.ones(2, dtype=np.float64), np.array(0, dtype=np.int64), "two classes"),
        (
            np.array([[np.nan, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            "finite",
        ),
        (np.ones((2, 2), dtype=np.float64), np.array([0, 2], dtype=np.int64), "target"),
    ],
)
def test_temperature_fit_rejects_empty_or_malformed_inputs(
    logits: np.ndarray,
    targets: np.ndarray,
    message: str,
) -> None:
    """Malformed calibration tensors must fail before optimization."""

    module = load_temperature()

    with pytest.raises(ValueError, match=message):
        module.fit_scalar_temperature(logits, targets)


def test_formal_calibration_requires_nonempty_calibration_split_provenance() -> None:
    """An empty provenance record cannot prove calibration-only fitting."""

    module = load_temperature()
    logits = np.array([[2.0, 0.0]], dtype=np.float64)
    targets = np.array([0], dtype=np.int64)
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=(),
        dataset_manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match=r"^calibration provenance sample_ids must not be empty"):
        module.fit_provenance_checked_temperature(
            logits,
            targets,
            provenance,
            expected_dataset_manifest_sha256="a" * 64,
            expected_sample_ids=("sample-1",),
        )


@pytest.mark.parametrize("split_name", ["validation", "locked_validation", "train"])
def test_formal_calibration_refuses_noncalibration_or_locked_validation_provenance(
    split_name: str,
) -> None:
    """Locked validation must never become a temperature-fitting input."""

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name=split_name,
        sample_ids=("sample-1",),
        dataset_manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match=r"^temperature fitting requires the calibration split"):
        module.fit_provenance_checked_temperature(
            np.array([[2.0, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            provenance,
            expected_dataset_manifest_sha256="a" * 64,
            expected_sample_ids=("sample-1",),
        )


def test_formal_calibration_rejects_invalid_manifest_hash() -> None:
    """A provenance record with no verifiable manifest identity must fail closed."""

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=("sample-1",),
        dataset_manifest_sha256="not-a-hash",
    )

    with pytest.raises(ValueError, match=r"^dataset_manifest_sha256 must be"):
        module.fit_provenance_checked_temperature(
            np.array([[2.0, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            provenance,
            expected_dataset_manifest_sha256="a" * 64,
            expected_sample_ids=("sample-1",),
        )


def test_formal_calibration_rejects_invalid_expected_manifest_hash() -> None:
    """The caller's frozen-manifest identity must itself be a valid SHA-256."""

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=("sample-1",),
        dataset_manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match=r"^expected_dataset_manifest_sha256 must be"):
        module.fit_provenance_checked_temperature(
            np.array([[2.0, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            provenance,
            expected_dataset_manifest_sha256="invalid",
            expected_sample_ids=("sample-1",),
        )


@pytest.mark.parametrize(
    "logits",
    [
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[np.inf, 0.0]], dtype=np.float64),
    ],
)
def test_apply_temperature_rejects_malformed_logits(logits: np.ndarray) -> None:
    """Scaling a wrong-dtype or nonfinite tensor must fail before division."""

    module = load_temperature()

    with pytest.raises(ValueError, match=r"float64|finite"):
        module.apply_temperature(logits, 1.0)


def test_formal_calibration_accepts_valid_calibration_provenance() -> None:
    """The checked formal entry point must remain usable for the frozen calibration cohort."""

    module = load_temperature()
    logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=("sample-1", "sample-2"),
        dataset_manifest_sha256="a" * 64,
    )

    assert module.fit_provenance_checked_temperature(
        logits,
        targets,
        provenance,
        expected_dataset_manifest_sha256="a" * 64,
        expected_sample_ids=("sample-1", "sample-2"),
    ) == pytest.approx(module.fit_scalar_temperature(logits, targets), rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("sample_ids", "message"),
    [
        (("",), "nonempty"),
        (("sample-1", "sample-1"), "unique"),
    ],
)
def test_formal_calibration_rejects_malformed_sample_membership(
    sample_ids: tuple[str, ...],
    message: str,
) -> None:
    """Blank or duplicate IDs cannot identify an exact frozen cohort."""

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=sample_ids,
        dataset_manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match=message):
        module.fit_provenance_checked_temperature(
            np.ones((len(sample_ids), 2), dtype=np.float64),
            np.zeros(len(sample_ids), dtype=np.int64),
            provenance,
            expected_dataset_manifest_sha256="a" * 64,
            expected_sample_ids=sample_ids,
        )


@pytest.mark.parametrize(
    ("expected_hash", "expected_ids", "message"),
    [
        ("b" * 64, ("sample-1", "sample-2"), "manifest"),
        ("a" * 64, ("sample-2", "sample-1"), "membership"),
    ],
)
def test_formal_calibration_binds_expected_manifest_and_ordered_membership(
    expected_hash: str,
    expected_ids: tuple[str, ...],
    message: str,
) -> None:
    """Relabeling another cohort as calibration must not satisfy the formal gate."""

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=("sample-1", "sample-2"),
        dataset_manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match=message):
        module.fit_provenance_checked_temperature(
            np.ones((2, 2), dtype=np.float64),
            np.zeros(2, dtype=np.int64),
            provenance,
            expected_dataset_manifest_sha256=expected_hash,
            expected_sample_ids=expected_ids,
        )


def test_formal_calibration_binds_sample_ids_to_logits_batch_dimension() -> None:
    """A cohort identity with a different batch size cannot describe the supplied logits."""

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=("sample-1", "sample-2"),
        dataset_manifest_sha256="a" * 64,
    )

    with pytest.raises(
        ValueError, match=r"^logits batch dimension must match calibration sample_ids"
    ):
        module.fit_provenance_checked_temperature(
            np.ones((1, 2), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            provenance,
            expected_dataset_manifest_sha256="a" * 64,
            expected_sample_ids=("sample-1", "sample-2"),
        )


def test_temperature_scaling_rejects_finite_values_that_overflow_after_division() -> None:
    """Finite input must not silently become infinite calibrated logits."""

    module = load_temperature()
    logits = np.array([[np.finfo(np.float64).max, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match=r"^temperature scaling overflowed"):
        module.apply_temperature(logits, 0.01)


def test_temperature_fit_rejects_logits_unsafe_for_bounded_scaling() -> None:
    """The optimizer must reject a finite range that overflows inside its declared bounds."""

    module = load_temperature()
    logits = np.array([[np.finfo(np.float64).max, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match=r"^logit range is unsafe for bounded temperature scaling"):
        module.fit_scalar_temperature(logits, np.array([0], dtype=np.int64))


def test_temperature_fit_rejects_row_range_that_can_overflow_cross_entropy() -> None:
    """Opposite finite logits must not overflow the target-class NLL after scaling."""

    module = load_temperature()
    magnitude = np.finfo(np.float64).max * math.exp(-5.0) * 0.75
    logits = np.array([[magnitude, -magnitude]], dtype=np.float64)

    with pytest.raises(ValueError, match=r"^logit range is unsafe for bounded temperature scaling"):
        module.fit_scalar_temperature(logits, np.array([1], dtype=np.int64))


def test_temperature_objective_averages_multiple_extreme_losses_without_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite mathematical mean must not overflow while summing its loss rows."""

    module = load_temperature()
    magnitude = np.finfo(np.float64).max * math.exp(-5.0) * 0.49
    logits = np.array(
        [[magnitude, -magnitude], [magnitude, -magnitude]],
        dtype=np.float64,
    )

    def evaluate_lower_bound(objective: object, **_: object) -> SimpleNamespace:
        value = objective(-5.0)  # type: ignore[operator]
        assert math.isfinite(value)
        return SimpleNamespace(success=True, fun=value, x=-5.0, message="synthetic success")

    monkeypatch.setattr(module, "minimize_scalar", evaluate_lower_bound)

    assert module.fit_scalar_temperature(logits, np.array([1, 1], dtype=np.int64)) == pytest.approx(
        math.exp(-5.0)
    )


def test_temperature_fit_rejects_failed_or_nonfinite_optimizer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed numerical optimization cannot be published as a fitted temperature."""

    module = load_temperature()
    monkeypatch.setattr(
        module,
        "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            fun=math.nan,
            x=0.0,
            message="synthetic failure",
        ),
    )

    with pytest.raises(RuntimeError, match=r"^temperature optimization failed: synthetic failure"):
        module.fit_scalar_temperature(
            np.array([[2.0, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
        )


def test_softmax_turns_logits_into_a_hand_computed_probability_row() -> None:
    """A missing normalization would break every Brier, ECE and confidence contract.

    With logits 0 and log 3 the exact softmax is 0.25 and 0.75.
    """

    module = load_temperature()
    logits = np.array([[0.0, np.log(3.0)]], dtype=np.float64)

    probabilities = module.softmax_probabilities(logits)

    np.testing.assert_allclose(probabilities, np.array([[0.25, 0.75]]))


def test_softmax_rows_sum_to_one_for_extreme_logits() -> None:
    """Subtracting the row maximum is what keeps large logits from overflowing."""

    module = load_temperature()
    logits = np.array([[900.0, 899.0, -900.0]], dtype=np.float64)

    probabilities = module.softmax_probabilities(logits)

    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=-1), np.ones(1))
    assert probabilities[0, 0] > probabilities[0, 1] > probabilities[0, 2]


def test_softmax_rejects_inputs_that_are_not_finite_float64_logits() -> None:
    """Silent dtype promotion or a NaN logit would poison every downstream metric."""

    module = load_temperature()

    with pytest.raises(ValueError, match=r"^logits must be a"):
        module.softmax_probabilities(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"^logits must be finite"):
        module.softmax_probabilities(np.array([[0.0, np.nan]], dtype=np.float64))
    with pytest.raises(ValueError, match=r"^logits must have at least two classes on the final"):
        module.softmax_probabilities(np.zeros((1, 1), dtype=np.float64))
