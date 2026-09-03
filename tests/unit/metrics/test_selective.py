"""Hand-computed contracts for selective risk and risk-coverage area."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st


def load_selective_module() -> ModuleType:
    try:
        from drivemetrics.metrics import selective
    except ImportError:
        pytest.fail("drivemetrics.metrics.selective is missing", pytrace=False)
    return selective


def test_fully_correct_predictions_have_zero_risk_at_every_coverage() -> None:
    """Any positive risk here would invent failures a correct model never made."""

    selective = load_selective_module()
    confidence = np.array([0.9, 0.1, 0.5], dtype=np.float64)
    correctness = np.ones(3, dtype=np.bool_)

    coverage, risk = selective.selective_risk_curve(confidence, correctness)

    np.testing.assert_allclose(coverage, np.array([1 / 3, 2 / 3, 1.0]))
    np.testing.assert_array_equal(risk, np.zeros(3, dtype=np.float64))
    assert selective.area_under_risk_coverage(coverage, risk) == 0.0


def test_perfect_confidence_ordering_defers_risk_until_correct_samples_are_exhausted() -> None:
    """Ascending or unsorted selection would move errors into the most confident coverage."""

    selective = load_selective_module()
    confidence = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    correctness = np.array([True, True, False, False], dtype=np.bool_)

    coverage, risk = selective.selective_risk_curve(confidence, correctness)

    np.testing.assert_allclose(coverage, np.array([0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(risk, np.array([0.0, 0.0, 1 / 3, 0.5]))
    assert selective.area_under_risk_coverage(coverage, risk) == pytest.approx(7 / 36)


def test_tied_confidence_keeps_the_original_index_order() -> None:
    """An unstable sort would silently reorder equally confident samples and move the risk."""

    selective = load_selective_module()
    confidence = np.array([0.5, 0.5], dtype=np.float64)
    correctness = np.array([False, True], dtype=np.bool_)

    coverage, risk = selective.selective_risk_curve(confidence, correctness)

    np.testing.assert_allclose(coverage, np.array([0.5, 1.0]))
    np.testing.assert_allclose(risk, np.array([1.0, 0.5]))


def test_fully_incorrect_predictions_have_full_risk_at_every_coverage() -> None:
    """A curve that decays toward zero would hide a model that is always wrong."""

    selective = load_selective_module()
    confidence = np.array([0.2, 0.9, 0.4], dtype=np.float64)
    correctness = np.zeros(3, dtype=np.bool_)

    coverage, risk = selective.selective_risk_curve(confidence, correctness)

    np.testing.assert_allclose(coverage, np.array([1 / 3, 2 / 3, 1.0]))
    np.testing.assert_array_equal(risk, np.ones(3, dtype=np.float64))
    assert selective.area_under_risk_coverage(coverage, risk) == 1.0


def test_risk_coverage_area_is_the_span_normalized_trapezoid() -> None:
    """Dropping the documented span normalization would make the score depend on sample count."""

    selective = load_selective_module()
    coverage = np.array([0.25, 0.5, 1.0], dtype=np.float64)
    risk = np.array([0.0, 0.4, 0.6], dtype=np.float64)

    # trapezoid = 0.25 * (0.0 + 0.4) / 2 + 0.5 * (0.4 + 0.6) / 2 = 0.05 + 0.25 = 0.30
    # span      = 1.0 - 0.25 = 0.75
    assert selective.area_under_risk_coverage(coverage, risk) == pytest.approx(0.30 / 0.75)


def test_metrics_package_exports_selective_risk_entry_points() -> None:
    """Analysis and report stages consume these through the package entry point."""

    import drivemetrics.metrics as metrics

    selective = load_selective_module()
    assert metrics.selective_risk_curve is selective.selective_risk_curve
    assert metrics.area_under_risk_coverage is selective.area_under_risk_coverage


def test_selective_risk_curve_rejects_a_non_float64_confidence_array() -> None:
    """Silent dtype promotion would change the tie order that fixes the published curve."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="float64"):
        selective.selective_risk_curve(
            np.array([0.5, 0.4], dtype=np.float32),
            np.ones(2, dtype=np.bool_),
        )


def test_selective_risk_curve_rejects_a_non_boolean_correctness_array() -> None:
    """Integer correctness would let a count of 2 be treated as a single error."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="boolean"):
        selective.selective_risk_curve(
            np.array([0.5, 0.4], dtype=np.float64),
            np.array([1, 0], dtype=np.int64),
        )


def test_selective_risk_curve_rejects_multidimensional_input() -> None:
    """A per-image matrix would flatten into a curve that no coverage grid describes."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="one-dimensional"):
        selective.selective_risk_curve(
            np.zeros((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.bool_),
        )


def test_selective_risk_curve_rejects_mismatched_lengths() -> None:
    """Broadcasting a shorter correctness vector would score the wrong samples."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="same length"):
        selective.selective_risk_curve(
            np.array([0.5, 0.4], dtype=np.float64),
            np.ones(1, dtype=np.bool_),
        )


def test_selective_risk_curve_rejects_an_empty_cohort() -> None:
    """An empty curve would publish a coverage grid with no evidence behind it."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="at least one sample"):
        selective.selective_risk_curve(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.bool_),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_selective_risk_curve_rejects_non_finite_confidence(bad_value: float) -> None:
    """NaN sorts unpredictably, so it would silently reorder the accepted set."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="finite"):
        selective.selective_risk_curve(
            np.array([0.5, bad_value], dtype=np.float64),
            np.ones(2, dtype=np.bool_),
        )


def test_risk_coverage_area_rejects_a_single_point() -> None:
    """A zero-width span has no defined area and must never be divided by."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="at least two points"):
        selective.area_under_risk_coverage(
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        )


def test_risk_coverage_area_rejects_a_non_increasing_coverage_grid() -> None:
    """An unsorted grid would make the trapezoid rule subtract area from itself."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="strictly increasing"):
        selective.area_under_risk_coverage(
            np.array([0.5, 0.5, 1.0], dtype=np.float64),
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )


@pytest.mark.parametrize("grid", [[0.0, 1.0], [0.5, 1.5]])
def test_risk_coverage_area_rejects_coverage_outside_the_unit_interval(
    grid: list[float],
) -> None:
    """Coverage is a retained fraction; zero or above one is not a reportable operating point."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        selective.area_under_risk_coverage(
            np.array(grid, dtype=np.float64),
            np.array([0.1, 0.2], dtype=np.float64),
        )


@pytest.mark.parametrize("bad_risk", [[-0.1, 0.2], [0.1, 1.2], [0.1, np.nan]])
def test_risk_coverage_area_rejects_risk_outside_the_unit_interval(
    bad_risk: list[float],
) -> None:
    """A risk outside [0, 1] is not an error rate and would produce an uninterpretable area."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        selective.area_under_risk_coverage(
            np.array([0.5, 1.0], dtype=np.float64),
            np.array(bad_risk, dtype=np.float64),
        )


def test_risk_coverage_area_rejects_mismatched_lengths() -> None:
    """Integrating unequal arrays would pair a risk with the wrong coverage."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="same length"):
        selective.area_under_risk_coverage(
            np.array([0.5, 1.0], dtype=np.float64),
            np.array([0.1], dtype=np.float64),
        )


@pytest.mark.parametrize("array", ["not-an-array", [0.5, 1.0]])
def test_risk_coverage_area_rejects_non_float64_arrays(array: object) -> None:
    """Accepting a Python list would bypass every finiteness and ordering check."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match="float64"):
        selective.area_under_risk_coverage(
            array,  # type: ignore[arg-type]
            np.array([0.1, 0.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="float64"):
        selective.area_under_risk_coverage(
            np.array([0.5, 1.0], dtype=np.float64),
            array,  # type: ignore[arg-type]
        )


@given(
    samples=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.booleans(),
        ),
        min_size=2,
        max_size=50,
    )
)
def test_selective_risk_ends_at_the_overall_error_rate_and_stays_bounded(
    samples: list[tuple[float, bool]],
) -> None:
    """Full coverage must reproduce the ordinary error rate for any confidence ordering."""

    selective = load_selective_module()
    confidence = np.array([value for value, _ in samples], dtype=np.float64)
    correctness = np.array([flag for _, flag in samples], dtype=np.bool_)

    coverage, risk = selective.selective_risk_curve(confidence, correctness)

    assert np.all(np.diff(coverage) > 0.0)
    assert coverage[-1] == 1.0
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    assert risk[-1] == pytest.approx(float(np.mean(~correctness)))
    assert 0.0 <= selective.area_under_risk_coverage(coverage, risk) <= 1.0


def test_risk_coverage_area_rejects_multidimensional_arrays() -> None:
    """A per-model matrix would integrate along an axis the coverage grid does not describe."""

    selective = load_selective_module()
    valid = np.array([0.5, 1.0], dtype=np.float64)
    matrix = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="coverage must be one-dimensional"):
        selective.area_under_risk_coverage(matrix, valid)
    with pytest.raises(ValueError, match="risk must be one-dimensional"):
        selective.area_under_risk_coverage(valid, matrix)


def test_the_curve_follows_confidence_order_rather_than_array_order() -> None:
    """Scoring in array order would report a different study's risk curve.

    The whole point of a selective-risk curve is that samples are accepted in
    descending confidence. If the ordering step were dropped, the curve would
    still be finite, monotone-looking and plausible, which is exactly why it
    needs a case where the two orders disagree.
    """

    selective = load_selective_module()
    # Least confident first, and it is the only wrong one.
    confidence = np.array([0.1, 0.8, 0.9], dtype=np.float64)
    correctness = np.array([False, True, True], dtype=np.bool_)

    _, risk = selective.selective_risk_curve(confidence, correctness)

    # Confidence order accepts the two correct samples first, so risk stays at
    # zero until the last sample; array order would put the error first.
    assert risk.tolist() == [0.0, 0.0, pytest.approx(1.0 / 3.0)]
