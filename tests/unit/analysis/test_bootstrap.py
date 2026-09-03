"""Contracts for the two-stage paired image-and-seed bootstrap."""

from __future__ import annotations

import math
from types import ModuleType

import numpy as np
import pytest


def load_bootstrap_module() -> ModuleType:
    try:
        from drivemetrics.analysis import bootstrap
    except ImportError:
        pytest.fail("drivemetrics.analysis.bootstrap is missing", pytrace=False)
    return bootstrap


def test_one_shared_image_draw_preserves_paired_run_differences() -> None:
    """Resampling images independently per run would break pairing and widen this interval.

    The two runs are exactly anti-correlated across the two images, so a single
    shared image draw always cancels: every replicate must equal 5.0.
    """

    bootstrap = load_bootstrap_module()
    values = np.array([[0.0, 10.0], [10.0, 0.0]], dtype=np.float64)

    interval = bootstrap.two_stage_paired_bootstrap(
        values,
        (0, 1),
        resamples=128,
        seed=20260831,
    )

    assert interval.estimate == 5.0
    assert interval.low == 5.0
    assert interval.high == 5.0


def test_the_same_seed_reproduces_the_same_interval() -> None:
    """A nondeterministic draw would make every published confidence interval unreplayable."""

    bootstrap = load_bootstrap_module()
    values = np.array([[0.0, 1.0, 0.0, 1.0], [0.2, 0.9, 0.1, 0.8]], dtype=np.float64)

    first = bootstrap.two_stage_paired_bootstrap(values, (0, 0), resamples=64, seed=20260831)
    second = bootstrap.two_stage_paired_bootstrap(values, (0, 0), resamples=64, seed=20260831)

    assert first == second
    assert first.low < first.estimate < first.high


def test_identical_values_collapse_to_a_degenerate_interval() -> None:
    """Inventing spread for a constant metric would report uncertainty that does not exist."""

    bootstrap = load_bootstrap_module()
    values = np.full((2, 3), 0.25, dtype=np.float64)

    interval = bootstrap.two_stage_paired_bootstrap(values, (0, 0), resamples=16, seed=7)

    assert interval.estimate == 0.25
    assert interval.low == 0.25
    assert interval.high == 0.25
    assert interval.confidence == 0.95
    assert interval.resamples == 16
    assert interval.seed == 7


def test_seed_axis_length_mismatch_fails_closed() -> None:
    """Silently recycling model labels would attribute a run to the wrong model."""

    bootstrap = load_bootstrap_module()
    values = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match=r"^model_seed_ids must label exactly one model per run"):
        bootstrap.two_stage_paired_bootstrap(values, (0,), resamples=8, seed=1)


def test_runs_without_a_common_image_axis_fail_closed() -> None:
    """Accepting per-run summaries would silently drop the paired-image estimand."""

    bootstrap = load_bootstrap_module()
    values = np.array([0.1, 0.2], dtype=np.float64)

    with pytest.raises(ValueError, match=r"^values must be a two-dimensional run-by-image array"):
        bootstrap.two_stage_paired_bootstrap(values, (0, 0), resamples=8, seed=1)


def test_analysis_package_exports_bootstrap_entry_points() -> None:
    """The statistics stage consumes these through the package entry point."""

    import drivemetrics.analysis as analysis

    bootstrap = load_bootstrap_module()
    assert analysis.two_stage_paired_bootstrap is bootstrap.two_stage_paired_bootstrap
    assert analysis.BootstrapInterval is bootstrap.BootstrapInterval


def test_the_estimand_is_the_unweighted_mean_over_models() -> None:
    """Weighting by run count would let a model with more seeds dominate the estimate.

    Run means are 2.0 and 5.0 for model 0 and 0.0 for model 1, so the model means
    are 3.5 and 0.0 and the estimate is 1.75. A run-weighted mean would be
    (2.0 + 5.0 + 0.0) / 3 = 2.333..., which this test rejects.
    """

    bootstrap = load_bootstrap_module()
    values = np.array([[1.0, 3.0], [5.0, 5.0], [0.0, 0.0]], dtype=np.float64)

    interval = bootstrap.two_stage_paired_bootstrap(values, (0, 0, 1), resamples=8, seed=3)

    assert interval.estimate == 1.75


def test_seed_resampling_matches_the_hand_enumerated_distribution() -> None:
    """Skipping the seed stage would report a degenerate interval for a two-seed model.

    Every image is identical, so only the seed stage varies. Model 0 draws two of
    its two runs with replacement, giving run-mean averages 2.0, 4.0, 4.0 and 6.0
    with probabilities 1/4, 1/2 and 1/4. Model 1 is always 0.0, so the statistic
    is 1.0, 2.0 or 3.0 with the same probabilities, and the 2.5th and 97.5th
    percentiles are exactly 1.0 and 3.0.
    """

    bootstrap = load_bootstrap_module()
    values = np.array([[2.0, 2.0], [6.0, 6.0], [0.0, 0.0]], dtype=np.float64)

    interval = bootstrap.two_stage_paired_bootstrap(
        values,
        (0, 0, 1),
        resamples=1000,
        seed=20260831,
    )

    assert interval.estimate == 2.0
    assert interval.low == 1.0
    assert interval.high == 3.0


def test_bootstrap_rejects_a_non_float64_value_array() -> None:
    """A float32 metric table would silently lose precision before percentile selection."""

    bootstrap = load_bootstrap_module()

    with pytest.raises(ValueError, match=r"^values must be a"):
        bootstrap.two_stage_paired_bootstrap(
            np.zeros((2, 2), dtype=np.float32),
            (0, 0),
            resamples=8,
            seed=1,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_bootstrap_rejects_non_finite_values(bad_value: float) -> None:
    """One undefined per-image metric would silently poison every replicate."""

    bootstrap = load_bootstrap_module()
    values = np.array([[0.1, bad_value]], dtype=np.float64)

    with pytest.raises(ValueError, match=r"^values must be finite"):
        bootstrap.two_stage_paired_bootstrap(values, (0,), resamples=8, seed=1)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_bootstrap_rejects_an_empty_run_or_image_axis(shape: tuple[int, int]) -> None:
    """An empty axis has no estimand and would divide by zero inside the replicate loop."""

    bootstrap = load_bootstrap_module()
    values = np.zeros(shape, dtype=np.float64)

    with pytest.raises(ValueError, match=r"^values must contain at least one run and one image"):
        bootstrap.two_stage_paired_bootstrap(
            values,
            (0,) * shape[0],
            resamples=8,
            seed=1,
        )


@pytest.mark.parametrize("resamples", [0, -1, True, 2.5])
def test_bootstrap_rejects_a_non_positive_resample_count(resamples: object) -> None:
    """A zero-resample interval would be reported as a measured uncertainty."""

    bootstrap = load_bootstrap_module()
    values = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"^resamples must be a positive integer"):
        bootstrap.two_stage_paired_bootstrap(
            values,
            (0,),
            resamples=resamples,  # type: ignore[arg-type]
            seed=1,
        )


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_bootstrap_rejects_an_unreproducible_seed(seed: object) -> None:
    """A seed the run record cannot store exactly would make the interval unreplayable."""

    bootstrap = load_bootstrap_module()
    values = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"^seed must be a nonnegative integer"):
        bootstrap.two_stage_paired_bootstrap(
            values,
            (0,),
            resamples=8,
            seed=seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("model_seed_ids", [(True, False), ("a", "b")])
def test_bootstrap_rejects_non_integer_model_labels(model_seed_ids: tuple[object, ...]) -> None:
    """Non-integer labels would group runs by accident instead of by declared model."""

    bootstrap = load_bootstrap_module()
    values = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(TypeError, match=r"^model_seed_ids must contain integers"):
        bootstrap.two_stage_paired_bootstrap(
            values,
            model_seed_ids,  # type: ignore[arg-type]
            resamples=8,
            seed=1,
        )


def test_the_protocol_defaults_are_recorded_in_the_interval() -> None:
    """5,000 resamples and seed 20260831 are cited by every published interval.

    They are arguments with defaults rather than constants precisely so a
    caller can vary them, which means nothing else pins what the default IS.
    A run that quietly used 5,001 resamples or seed 20260832 would still
    reproduce from its own record and would not reproduce from the protocol.
    """

    bootstrap = load_bootstrap_module()
    values = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=np.float64)

    interval = bootstrap.two_stage_paired_bootstrap(values, (0, 1))

    assert interval.resamples == 5000
    assert interval.seed == 20260831


def test_the_component_estimator_carries_the_same_defaults() -> None:
    """The two estimators must not drift apart in what they default to.

    They are documented as generating their draws in the same order, which is
    only meaningful if they start from the same seed and take the same number
    of resamples.
    """

    bootstrap = load_bootstrap_module()
    components = np.ones((2, 4, 3), dtype=np.float64)

    interval = bootstrap.two_stage_paired_bootstrap_statistic(
        components,
        (0, 1),
        lambda summed: summed[:, 0],
    )

    assert interval.resamples == 5000
    assert interval.seed == 20260831


def test_seed_zero_is_a_usable_seed() -> None:
    """Zero is a seed like any other, and it is the one a careless caller passes.

    A guard written `seed <= 0` instead of `< 0` would refuse it, which turns a
    perfectly reproducible run into an error the operator cannot explain from
    the message.
    """

    bootstrap = load_bootstrap_module()
    values = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=np.float64)

    interval = bootstrap.two_stage_paired_bootstrap(values, (0, 1), resamples=32, seed=0)

    assert interval.seed == 0


@pytest.mark.parametrize("shape", [(0, 2, 2), (2, 0, 2), (2, 2, 0)])
def test_an_empty_dimension_is_refused(shape: tuple[int, int, int]) -> None:
    """Each of the three axes must be checked, not just whichever one is first.

    The guard is a chain of `or`, and every link matters: a run-by-image-by-
    component array with no images resamples an empty cohort, and one with no
    components has nothing to sum. Written with `and` instead, only an array
    empty on ALL THREE axes would be refused, and every partially empty array
    would reach the estimator.
    """

    bootstrap = load_bootstrap_module()

    with pytest.raises(ValueError, match=r"^components must contain at least one run"):
        bootstrap.two_stage_paired_bootstrap_statistic(
            np.ones(shape, dtype=np.float64),
            tuple(range(shape[0])),
            lambda summed: summed[:, 0],
        )


def test_a_single_resample_is_a_usable_draw_count() -> None:
    """One resample is a degenerate but well-formed request, and the guard says so.

    The bound is strict positivity, so one must pass it. Written `<= 1` the
    validator would refuse the smallest interval anybody can ask for, and the
    refusal would read as a malformed call rather than as an off-by-one bound.
    The interval it returns is degenerate, which is the caller's problem, not
    the validator's.
    """

    bootstrap = load_bootstrap_module()
    values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

    interval = bootstrap.two_stage_paired_bootstrap(values, (0, 1), resamples=1, seed=1)

    assert interval.resamples == 1
    assert math.isfinite(interval.estimate)
    assert interval.low <= interval.estimate <= interval.high
