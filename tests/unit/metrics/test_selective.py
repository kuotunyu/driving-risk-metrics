"""Hand-computed contracts for selective risk and risk-coverage area."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import numpy.typing as npt
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


def test_the_tie_ordering_matches_an_independently_computed_stable_order() -> None:
    """The two-sample test above names the right property but cannot detect its loss.

    NumPy's introsort falls back to insertion sort below sixteen elements, and
    insertion sort is stable, so a two-element case gives the same answer with
    or without ``kind="stable"``. The property was documented, asserted, and
    unprotected.

    The reference here is computed independently, with Python's own ``sorted``,
    whose stability is a language guarantee rather than a NumPy implementation
    detail. That makes the test fail for ANY reordering of tied samples instead
    of only for the inputs where this NumPy version happens to reorder them.

    It matters on real data rather than in principle: confidence is stored
    quantized to uint16, so exact ties are expected, not unusual. Without a
    stable sort the selective-risk curve — and therefore every reported AURC —
    would depend on NumPy's partitioning rather than on the model.
    """

    selective = load_selective_module()
    generator = np.random.default_rng(20260903)
    count = 200
    # Five distinct confidence levels over two hundred samples, shuffled: every
    # level is a heavily tied block, which is the shape quantized confidence
    # actually produces.
    confidence = generator.integers(0, 5, size=count).astype(np.float64) / 4.0
    correctness = generator.random(count) < 0.6

    _, risk = selective.selective_risk_curve(confidence, correctness)

    expected_order = sorted(range(count), key=lambda index: (-confidence[index], index))
    accepted = np.arange(1, count + 1, dtype=np.float64)
    expected = np.cumsum(~correctness[np.asarray(expected_order)], dtype=np.float64) / accepted

    np.testing.assert_allclose(risk, expected)


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

    with pytest.raises(ValueError, match=r"^confidence must be a"):
        selective.selective_risk_curve(
            np.array([0.5, 0.4], dtype=np.float32),
            np.ones(2, dtype=np.bool_),
        )


def test_selective_risk_curve_rejects_a_non_boolean_correctness_array() -> None:
    """Integer correctness would let a count of 2 be treated as a single error."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^correctness must be a boolean array"):
        selective.selective_risk_curve(
            np.array([0.5, 0.4], dtype=np.float64),
            np.array([1, 0], dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("confidence_shape", "correctness_shape"),
    [((2, 2), (2, 2)), ((2, 2), (4,)), ((4,), (2, 2))],
    ids=["both are matrices", "only confidence is a matrix", "only correctness is a matrix"],
)
def test_selective_risk_curve_rejects_multidimensional_input(
    confidence_shape: tuple[int, ...],
    correctness_shape: tuple[int, ...],
) -> None:
    """A per-image matrix would flatten into a curve that no coverage grid describes.

    Either array on its own is enough to refuse the call. The both-matrices row
    cannot show that: written `and` instead of `or` the check still fires there,
    and only a mixed pair reveals that one bad array would pass. The mixed rows
    keep the total sizes equal, so the length check downstream cannot stand in
    for the dimensionality check.
    """

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^confidence and correctness must be one-dimensional$"):
        selective.selective_risk_curve(
            np.zeros(confidence_shape, dtype=np.float64),
            np.ones(correctness_shape, dtype=np.bool_),
        )


def test_selective_risk_curve_rejects_mismatched_lengths() -> None:
    """Broadcasting a shorter correctness vector would score the wrong samples."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^confidence and correctness must have the same length"):
        selective.selective_risk_curve(
            np.array([0.5, 0.4], dtype=np.float64),
            np.ones(1, dtype=np.bool_),
        )


def test_selective_risk_curve_rejects_an_empty_cohort() -> None:
    """An empty curve would publish a coverage grid with no evidence behind it."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^confidence must contain at least one sample"):
        selective.selective_risk_curve(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.bool_),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_selective_risk_curve_rejects_non_finite_confidence(bad_value: float) -> None:
    """NaN sorts unpredictably, so it would silently reorder the accepted set."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^confidence values must be finite"):
        selective.selective_risk_curve(
            np.array([0.5, bad_value], dtype=np.float64),
            np.ones(2, dtype=np.bool_),
        )


def test_risk_coverage_area_rejects_a_single_point() -> None:
    """A zero-width span has no defined area and must never be divided by."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^coverage must contain at least two points"):
        selective.area_under_risk_coverage(
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        )


def test_risk_coverage_area_rejects_a_non_increasing_coverage_grid() -> None:
    """An unsorted grid would make the trapezoid rule subtract area from itself."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^coverage must be strictly increasing"):
        selective.area_under_risk_coverage(
            np.array([0.5, 0.5, 1.0], dtype=np.float64),
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )


@pytest.mark.parametrize(
    "grid",
    [[0.0, 1.0], [0.5, 1.5], [0.25, 0.5, 1.5]],
    ids=["zero coverage", "final point above one", "final point above one on a longer grid"],
)
def test_risk_coverage_area_rejects_coverage_outside_the_unit_interval(
    grid: list[float],
) -> None:
    """Coverage is a retained fraction; zero or above one is not a reportable operating point.

    The upper bound is checked on the LAST point, because the grid is already
    known to be strictly increasing. On a two-point grid `coverage[-1]` and
    `coverage[1]` are the same element, so the third row is what separates
    them: there the offending value is at the end and position 1 is innocent.
    """

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^coverage values must lie in \(0, 1\]$"):
        selective.area_under_risk_coverage(
            np.array(grid, dtype=np.float64),
            np.array([0.1, 0.2, 0.3][: len(grid)], dtype=np.float64),
        )


@pytest.mark.parametrize("bad_risk", [[-0.1, 0.2], [0.1, 1.2], [0.1, np.nan]])
def test_risk_coverage_area_rejects_risk_outside_the_unit_interval(
    bad_risk: list[float],
) -> None:
    """A risk outside [0, 1] is not an error rate and would produce an uninterpretable area."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^risk values must lie in \[0, 1\]$"):
        selective.area_under_risk_coverage(
            np.array([0.5, 1.0], dtype=np.float64),
            np.array(bad_risk, dtype=np.float64),
        )


def test_risk_coverage_area_rejects_mismatched_lengths() -> None:
    """Integrating unequal arrays would pair a risk with the wrong coverage."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^coverage and risk must have the same length"):
        selective.area_under_risk_coverage(
            np.array([0.5, 1.0], dtype=np.float64),
            np.array([0.1], dtype=np.float64),
        )


@pytest.mark.parametrize("array", ["not-an-array", [0.5, 1.0]])
def test_risk_coverage_area_rejects_non_float64_arrays(array: object) -> None:
    """Accepting a Python list would bypass every finiteness and ordering check."""

    selective = load_selective_module()

    with pytest.raises(ValueError, match=r"^coverage must be a"):
        selective.area_under_risk_coverage(
            array,  # type: ignore[arg-type]
            np.array([0.1, 0.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match=r"^risk must be a"):
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

    with pytest.raises(ValueError, match=r"^coverage must be one-dimensional"):
        selective.area_under_risk_coverage(matrix, valid)
    with pytest.raises(ValueError, match=r"^risk must be one-dimensional"):
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


def test_a_histogram_curve_matches_the_per_pixel_curve_at_every_bin_boundary() -> None:
    """A cohort has a billion pixels, so the curve is rebuilt from counts, not from pixels.

    Nine hundred and twenty million per-pixel confidences cannot be retained, and
    the artifacts store a quantized histogram instead. Inside one bin the
    per-pixel curve is not recoverable — the pixels there are indistinguishable
    by construction — so the published curve is DEFINED at bin boundaries, and
    this test pins that it agrees with the per-pixel curve exactly wherever both
    are defined.
    """

    module = load_selective_module()
    bins = 8
    counts = np.zeros(bins, dtype=np.int64)
    correct = np.zeros(bins, dtype=np.int64)
    # Three pixels at bin 6, two at bin 3, one at bin 1: high confidence is mostly
    # right and low confidence is mostly wrong, which is the shape a curve reads.
    counts[6], correct[6] = 3, 3
    counts[3], correct[3] = 2, 1
    counts[1], correct[1] = 1, 0

    coverage, risk = module.selective_risk_from_histogram(counts, correct)

    # The equivalent per-pixel population, accepted in descending bin order.
    confidence = np.array([6, 6, 6, 3, 3, 1], dtype=np.float64)
    correctness = np.array([True, True, True, True, False, False])
    exact_coverage, exact_risk = module.selective_risk_curve(confidence, correctness)

    assert coverage.tolist() == [3 / 6, 5 / 6, 6 / 6]
    for position, point in enumerate(coverage):
        index = int(np.argmin(np.abs(exact_coverage - point)))
        assert exact_coverage[index] == pytest.approx(point, rel=0.0, abs=1e-12)
        assert risk[position] == pytest.approx(exact_risk[index], rel=0.0, abs=1e-12)


def test_an_empty_bin_adds_no_point_to_the_curve() -> None:
    """A bin no pixel reached is not a coverage level anything can be evaluated at."""

    module = load_selective_module()
    counts = np.array([0, 2, 0, 2, 0], dtype=np.int64)
    correct = np.array([0, 2, 0, 1, 0], dtype=np.int64)

    coverage, _ = module.selective_risk_from_histogram(counts, correct)

    assert coverage.tolist() == [0.5, 1.0]


def test_the_histogram_curve_reads_the_bins_from_most_confident_to_least() -> None:
    """Reversing the order would report the model's worst pixels as its most certain."""

    module = load_selective_module()
    counts = np.array([2, 2], dtype=np.int64)
    correct = np.array([0, 2], dtype=np.int64)

    _, risk = module.selective_risk_from_histogram(counts, correct)

    # Bin 1 is the confident one and is entirely correct, so the first point is 0.
    assert risk.tolist() == [0.0, 0.5]


def test_a_histogram_with_no_pixels_at_all_is_refused() -> None:
    """An empty cohort has no curve, and a curve of one point has no area."""

    module = load_selective_module()

    with pytest.raises(ValueError, match=r"^the histogram contains no pixel"):
        module.selective_risk_from_histogram(
            np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)
        )


def test_a_histogram_claiming_more_correct_than_it_counted_is_refused() -> None:
    """Correct pixels are a subset of the pixels in a bin, and the artifact may be wrong."""

    module = load_selective_module()

    with pytest.raises(ValueError, match=r"^correct counts must not exceed the bin counts"):
        module.selective_risk_from_histogram(
            np.array([1, 2], dtype=np.int64), np.array([2, 2], dtype=np.int64)
        )


@pytest.mark.parametrize(
    ("counts", "correct", "expected"),
    [
        (
            np.zeros((2, 2), dtype=np.int64),
            np.zeros((2, 2), dtype=np.int64),
            r"^counts and correct counts must be flat arrays of the same length$",
        ),
        (
            np.zeros(3, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            r"^counts and correct counts must be flat arrays of the same length$",
        ),
        (
            np.array([-1, 2], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            r"^histogram counts must be nonnegative$",
        ),
        (
            np.array([1, 2], dtype=np.int64),
            np.array([-1, 2], dtype=np.int64),
            r"^histogram counts must be nonnegative$",
        ),
    ],
)
def test_a_malformed_histogram_is_refused_rather_than_curved(
    counts: npt.NDArray[np.int64], correct: npt.NDArray[np.int64], expected: str
) -> None:
    """The histogram comes from a file, and a negative count would run the cumulative sum backwards."""

    module = load_selective_module()

    with pytest.raises(ValueError, match=expected):
        module.selective_risk_from_histogram(counts, correct)
