"""Contracts for the two-stage paired image-and-seed bootstrap."""

from __future__ import annotations

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

    with pytest.raises(ValueError, match="model_seed_ids"):
        bootstrap.two_stage_paired_bootstrap(values, (0,), resamples=8, seed=1)


def test_runs_without_a_common_image_axis_fail_closed() -> None:
    """Accepting per-run summaries would silently drop the paired-image estimand."""

    bootstrap = load_bootstrap_module()
    values = np.array([0.1, 0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="two-dimensional"):
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

    with pytest.raises(ValueError, match="float64"):
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

    with pytest.raises(ValueError, match="finite"):
        bootstrap.two_stage_paired_bootstrap(values, (0,), resamples=8, seed=1)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_bootstrap_rejects_an_empty_run_or_image_axis(shape: tuple[int, int]) -> None:
    """An empty axis has no estimand and would divide by zero inside the replicate loop."""

    bootstrap = load_bootstrap_module()
    values = np.zeros(shape, dtype=np.float64)

    with pytest.raises(ValueError, match="at least one run and one image"):
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

    with pytest.raises(ValueError, match="positive integer"):
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

    with pytest.raises(ValueError, match="nonnegative integer"):
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

    with pytest.raises(TypeError, match="integers"):
        bootstrap.two_stage_paired_bootstrap(
            values,
            model_seed_ids,  # type: ignore[arg-type]
            resamples=8,
            seed=1,
        )
