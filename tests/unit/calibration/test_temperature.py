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
        (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            r"^no valid calibration targets",
        ),
        (
            np.ones((2, 2), dtype=np.float64),
            np.array([255, 255], dtype=np.int64),
            r"^no valid calibration targets",
        ),
        (
            np.ones((2, 2), dtype=np.float64),
            np.array([0], dtype=np.int64),
            r"^targets shape must match logits leading shape",
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            r"^logits must be a float64 array",
        ),
        (
            np.ones((2, 2), dtype=np.float64),
            np.array([0, 1], dtype=np.int32),
            r"^targets must be an int64 array",
        ),
        (
            np.ones(2, dtype=np.float64),
            np.array(0, dtype=np.int64),
            r"^logits must have at least two classes on the final axis",
        ),
        (
            np.array([[np.nan, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            r"^logits must be finite",
        ),
        (
            np.ones((2, 2), dtype=np.float64),
            np.array([0, 2], dtype=np.int64),
            r"^target is outside the logits class range",
        ),
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
    ("logits", "message"),
    [
        (np.array([[1.0, 0.0]], dtype=np.float32), r"^logits must be a float64 array"),
        (np.array([[np.inf, 0.0]], dtype=np.float64), r"^logits must be finite"),
    ],
)
def test_apply_temperature_rejects_malformed_logits(
    logits: np.ndarray,
    message: str,
) -> None:
    """Scaling a wrong-dtype or nonfinite tensor must fail before division.

    Each row names its own message. ``match=r"float64|finite"`` accepted either
    message for either input, so it could not tell the two rejections apart.
    """

    module = load_temperature()

    with pytest.raises(ValueError, match=message):
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
        (("",), r"^calibration provenance sample_ids must contain nonempty strings"),
        (("sample-1", "sample-1"), r"^calibration provenance sample_ids must be unique"),
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
        (
            "b" * 64,
            ("sample-1", "sample-2"),
            r"^calibration provenance manifest hash does not match expected manifest",
        ),
        (
            "a" * 64,
            ("sample-2", "sample-1"),
            r"^calibration provenance membership does not match expected sample order",
        ),
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


def test_the_fit_recovers_a_planted_temperature_exactly() -> None:
    """Cross-entropy is minimised only where the predicted distribution matches the data.

    Every row carries the same logits `2 * ln(p)` for `p = (0.6, 0.3, 0.1)`, and
    the targets occur in exactly those proportions: six, three and one of ten.
    Dividing those logits by `T = 2` returns exactly `p`, and cross-entropy
    `H(p, q) >= H(p)` with equality only at `q = p`, so the objective over
    these rows has its unique minimum at `T = 2.0`.

    The existing tests assert only that the fit lowers the NLL, which a flipped
    sign in the objective, a reduction over the wrong axis, or a loosened
    optimiser tolerance can all satisfy. This one names the answer.

    The tolerance comes from measurement, not from taste. The correct objective
    lands within `1.6e-08` of 2.0; nulling, dropping or misspelling the `xatol`
    option lands at `6.8e-07`; widening the tolerance to 1.0 lands at `2.33`.
    An absolute bound of `1e-7` therefore clears the true value by more than six
    times and rejects the nearest defect by more than six times.
    """

    module = load_temperature()
    reference = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    logits = np.tile(2.0 * np.log(reference), (10, 1))
    targets = np.array([0] * 6 + [1] * 3 + [2], dtype=np.int64)

    temperature = module.fit_scalar_temperature(logits, targets)

    assert temperature == pytest.approx(2.0, rel=0.0, abs=1e-7)
    np.testing.assert_allclose(
        module.softmax_probabilities(module.apply_temperature(logits, temperature))[0],
        reference,
        rtol=0.0,
        atol=1e-7,
    )


def test_softmax_accepts_a_one_dimensional_logit_vector() -> None:
    """One row of logits is a valid input, and the class axis is its only axis.

    The guard is `ndim < 1`, so a vector is admissible. Written `<= 1` or `< 2`
    it would refuse every single-row call, and reading the class count from
    `shape[1]` rather than `shape[-1]` would raise an index error here instead
    of a contract error.
    """

    module = load_temperature()

    probabilities = module.softmax_probabilities(np.array([0.0, np.log(3.0)], dtype=np.float64))

    np.testing.assert_allclose(probabilities, np.array([0.25, 0.75]))


def test_softmax_normalizes_the_final_axis_of_a_multi_axis_batch() -> None:
    """Per-image logits arrive with image axes, and only the last axis is classes.

    A `(2, 3, 4)` batch must give exactly what the same rows give as `(6, 4)`.
    Both the maximum that is subtracted and the sum that normalises are taken
    over the final axis; on a two-dimensional input `axis=-1` and `axis=1`
    coincide, so neither can be distinguished there.
    """

    module = load_temperature()
    rng = np.random.default_rng(20260904)
    flat = rng.normal(size=(6, 4))

    nested = module.softmax_probabilities(flat.reshape(2, 3, 4))

    np.testing.assert_allclose(
        nested.reshape(6, 4),
        module.softmax_probabilities(flat),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(nested.sum(axis=-1), np.ones((2, 3)), rtol=0.0, atol=1e-12)


def test_softmax_subtracts_each_row_maximum_not_the_batch_maximum() -> None:
    """The shift is per row, and a batch-wide shift is what overflow looks like.

    Row zero is enormous and row one is ordinary. Subtracting the batch maximum
    would drive row one's logits to about -900, whose exponentials underflow to
    zero, and the row would normalise to zeros or to NaN rather than to a
    probability distribution.
    """

    module = load_temperature()
    logits = np.array([[900.0, 899.0], [1.0, 0.0]], dtype=np.float64)

    probabilities = module.softmax_probabilities(logits)

    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=-1), np.ones(2), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        probabilities[1],
        module.softmax_probabilities(np.array([[1.0, 0.0]], dtype=np.float64))[0],
        rtol=0.0,
        atol=1e-12,
    )


def test_the_fit_flattens_leading_axes_and_reads_classes_from_the_final_one() -> None:
    """Calibration logits arrive per image, and every leading axis is a batch axis.

    The shape `(2, 1, 4)` separates three contract lines that a two-dimensional
    input cannot: the class count is read from the final axis, which is 4, and
    not from `shape[1]`, which is 1 and would fail the two-class requirement;
    and the target range is compared against that same final axis, so targets 2
    and 3 are valid here while `shape[1]` or `shape[-2]` would reject them.
    """

    module = load_temperature()
    rng = np.random.default_rng(902)
    flat = rng.normal(size=(2, 4))
    targets = np.array([2, 3], dtype=np.int64)

    nested = module.fit_scalar_temperature(flat.reshape(2, 1, 4), targets.reshape(2, 1))

    assert nested == pytest.approx(module.fit_scalar_temperature(flat, targets), rel=0.0, abs=1e-12)


def test_a_logit_exactly_at_the_overflow_bound_is_accepted() -> None:
    """The overflow guard is strict, so the bound itself is still a usable input.

    The guard refuses a magnitude ABOVE `float64 max * exp(-5) / 2`, which is
    the largest value that survives division by the smallest temperature the
    optimiser may choose. Written `>=` it would refuse the bound itself, and
    the refusal would read as unsafe data rather than as an off-by-one
    comparison.
    """

    module = load_temperature()
    bound = np.finfo(np.float64).max * math.exp(-5.0) / 2.0
    logits = np.array([[bound, 0.0]], dtype=np.float64)

    temperature = module.fit_scalar_temperature(logits, np.array([0], dtype=np.int64))

    assert math.isfinite(temperature)
    assert temperature > 0.0


@pytest.mark.parametrize(
    ("success", "fun", "x"),
    [
        (False, 1.0, 0.0),
        (True, math.nan, 0.0),
        (True, 1.0, math.nan),
    ],
    ids=["optimizer reported failure", "objective is not finite", "minimiser is not finite"],
)
def test_each_optimizer_failure_condition_is_refused_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
    success: bool,
    fun: float,
    x: float,
) -> None:
    """Three independent reasons to distrust a result, each sufficient by itself.

    The neighbouring test sets all three at once, so it passes even when the
    `or` chain is rewritten as an `and` and two of the three checks stop firing.
    Each row here trips exactly one condition and leaves the other two clean.
    """

    module = load_temperature()
    monkeypatch.setattr(
        module,
        "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(
            success=success,
            fun=fun,
            x=x,
            message="synthetic failure",
        ),
    )

    with pytest.raises(RuntimeError, match=r"^temperature optimization failed: synthetic failure"):
        module.fit_scalar_temperature(
            np.array([[2.0, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
        )


def test_the_expected_cohort_is_named_in_its_own_validation_message() -> None:
    """Two sample-ID lists are validated here, and the message says which one failed.

    The provenance list and the expected list are checked by the same helper
    with different labels. If the expected list's label were blank or reworded,
    an operator debugging a frozen cohort would be told that "calibration
    provenance" is empty when the provenance is fine and the expectation is not.
    """

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=("sample-1",),
        dataset_manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match=r"^expected calibration sample_ids must not be empty"):
        module.fit_provenance_checked_temperature(
            np.array([[2.0, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            provenance,
            expected_dataset_manifest_sha256="a" * 64,
            expected_sample_ids=(),
        )


def test_a_one_dimensional_logits_array_still_has_its_batch_dimension_checked() -> None:
    """The batch check runs for any array with an axis, not only for a matrix.

    Written `ndim > 1` or `ndim >= 2` the check would skip a vector entirely,
    and the caller would be told the logits lack a class axis instead of being
    told the batch does not match the frozen cohort. The first message sends an
    operator to the tensor shape; the second sends them to the cohort, which is
    where the disagreement actually is.
    """

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=("sample-1",),
        dataset_manifest_sha256="a" * 64,
    )

    with pytest.raises(
        ValueError, match=r"^logits batch dimension must match calibration sample_ids"
    ):
        module.fit_provenance_checked_temperature(
            np.array([2.0, 0.0], dtype=np.float64),
            np.array([0], dtype=np.int64),
            provenance,
            expected_dataset_manifest_sha256="a" * 64,
            expected_sample_ids=("sample-1",),
        )


def test_the_provenance_entry_point_forwards_its_ignore_index() -> None:
    """A caller-chosen ignore index must reach the fit, not be replaced by the default.

    Target 5 is outside the two declared classes, so it is only admissible
    because this call declares 5 to be the ignore value. Were the argument
    dropped on the way through, the default 255 would apply, target 5 would be
    scored as a real class, and the fit would fail as out of range instead of
    ignoring the row.
    """

    module = load_temperature()
    provenance = module.CalibrationProvenance(
        split_name="calibration",
        sample_ids=("sample-1", "sample-2"),
        dataset_manifest_sha256="a" * 64,
    )

    temperature = module.fit_provenance_checked_temperature(
        np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64),
        np.array([0, 5], dtype=np.int64),
        provenance,
        expected_dataset_manifest_sha256="a" * 64,
        expected_sample_ids=("sample-1", "sample-2"),
        ignore_index=5,
    )

    assert math.isfinite(temperature)
    assert temperature > 0.0


def test_the_fit_minimises_an_independently_computed_cross_entropy() -> None:
    """The objective is the mean of PER-ROW cross-entropies, not a batch-wide quantity.

    `logsumexp` normalises each row's logits into that row's log partition
    function. Reducing over the whole array instead produces the log partition
    of the flattened batch, which is a different function of the temperature.

    The neighbouring planted-temperature test cannot see this. Every one of its
    rows is identical, so the batch-wide reduction differs from the per-row one
    by exactly `log(row count)` — a constant in T, which shifts the objective
    without moving its minimum. These four rows carry different distributions
    and different magnitudes, and there the two reductions disagree sharply:
    2.57 against 6.96.

    Rather than pin the optimiser's own output, this asserts the property the
    function promises. The returned temperature must minimise the multiclass
    NLL of the calibrated logits, and that NLL is recomputed here without
    `logsumexp` at all, then compared against a 401-point scan of the entire
    admissible range `[exp(-5), exp(5)]`. A continuous minimum can never exceed
    any point of a grid inside its own domain, so the real fit clears this by
    construction; the batch-wide reduction misses it by 0.07, which is four
    orders of magnitude above the tolerance needed.
    """

    module = load_temperature()
    logits = np.array([[8.0, 0.0], [1.0, 0.0], [0.0, 6.0], [0.0, 0.5]], dtype=np.float64)
    targets = np.array([0, 1, 1, 0], dtype=np.int64)

    temperature = module.fit_scalar_temperature(logits, targets)

    fitted_nll = multiclass_nll(module.apply_temperature(logits, temperature), targets)
    scanned = [
        multiclass_nll(module.apply_temperature(logits, float(candidate)), targets)
        for candidate in np.exp(np.linspace(-5.0, 5.0, 401))
    ]

    assert fitted_nll <= min(scanned)
