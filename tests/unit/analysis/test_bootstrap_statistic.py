"""Contracts for bootstrapping a ratio metric rather than a per-image mean.

Cohort mIoU is a ratio of summed confusions: sum the intersections and unions
over the cohort, divide, then average over classes. The mean of per-image mIoU
is a different number, and reporting one while calling it the other is the kind
of silent conflation this project exists to prevent.

Both estimators must be available, they must share the identical two-stage draw
so they can be compared, and the difference between them must be demonstrable.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest


def load_bootstrap() -> ModuleType:
    from drivemetrics.analysis import bootstrap

    return bootstrap


def mean_statistic(image_count: int):
    """The linear statistic the existing per-image-mean estimator computes."""

    def statistic(summed: np.ndarray) -> np.ndarray:
        return summed[:, 0] / image_count

    return statistic


def iou_statistic(summed: np.ndarray) -> np.ndarray:
    """Intersection over union from summed components, the genuinely non-linear case."""

    return summed[:, 0] / summed[:, 1]


def test_a_linear_statistic_reproduces_the_per_image_mean_estimator_exactly() -> None:
    """Both estimators must share one draw, or their intervals cannot be compared."""

    bootstrap = load_bootstrap()
    rng = np.random.default_rng(3)
    values = rng.uniform(0.2, 0.9, (6, 40))
    model_ids = (0, 0, 0, 1, 1, 1)

    plain = bootstrap.two_stage_paired_bootstrap(values, model_ids, resamples=200, seed=99)
    general = bootstrap.two_stage_paired_bootstrap_statistic(
        values[:, :, None],
        model_ids,
        mean_statistic(values.shape[1]),
        resamples=200,
        seed=99,
    )

    assert general.estimate == pytest.approx(plain.estimate)
    assert general.low == pytest.approx(plain.low)
    assert general.high == pytest.approx(plain.high)


def test_a_ratio_of_sums_differs_from_the_mean_of_per_image_ratios() -> None:
    """This is the conflation the ratio estimator exists to prevent, demonstrated."""

    bootstrap = load_bootstrap()
    rng = np.random.default_rng(5)
    # Deliberately heterogeneous image sizes: a large image should carry more
    # weight in a cohort ratio, and none at all in a mean of per-image ratios.
    unions = rng.uniform(10.0, 4000.0, (4, 30))
    intersections = unions * rng.uniform(0.3, 0.95, (4, 30))
    components = np.stack([intersections, unions], axis=-1)
    model_ids = (0, 0, 1, 1)

    per_image_ratio = intersections / unions
    means = bootstrap.two_stage_paired_bootstrap(per_image_ratio, model_ids, resamples=200, seed=7)
    ratios = bootstrap.two_stage_paired_bootstrap_statistic(
        components, model_ids, iou_statistic, resamples=200, seed=7
    )

    assert ratios.estimate != pytest.approx(means.estimate, rel=1e-6)


def test_the_ratio_estimator_is_deterministic_for_a_fixed_seed() -> None:
    """An interval that moved between runs could not appear in a published claim."""

    bootstrap = load_bootstrap()
    rng = np.random.default_rng(11)
    components = np.abs(rng.normal(50.0, 10.0, (4, 25, 2))) + 1.0
    components[:, :, 1] += components[:, :, 0]
    model_ids = (0, 0, 1, 1)

    first = bootstrap.two_stage_paired_bootstrap_statistic(
        components, model_ids, iou_statistic, resamples=150, seed=20260831
    )
    second = bootstrap.two_stage_paired_bootstrap_statistic(
        components, model_ids, iou_statistic, resamples=150, seed=20260831
    )

    assert first == second


def test_identical_runs_collapse_to_a_degenerate_interval() -> None:
    """Zero variation must produce zero width, not a spurious spread."""

    bootstrap = load_bootstrap()
    components = np.ones((3, 12, 2), dtype=np.float64)
    components[:, :, 1] = 4.0

    interval = bootstrap.two_stage_paired_bootstrap_statistic(
        components, (0, 0, 0), iou_statistic, resamples=50, seed=1
    )

    assert interval.estimate == pytest.approx(0.25)
    assert interval.low == pytest.approx(0.25)
    assert interval.high == pytest.approx(0.25)


def test_a_component_array_that_is_not_three_dimensional_is_refused() -> None:
    """A two-dimensional array is the per-image-mean shape and means something else."""

    bootstrap = load_bootstrap()

    with pytest.raises(
        ValueError, match=r"^components must be a three-dimensional run-by-image-by-component"
    ):
        bootstrap.two_stage_paired_bootstrap_statistic(
            np.ones((3, 4), dtype=np.float64), (0, 0, 0), iou_statistic, resamples=10, seed=1
        )


def test_a_statistic_returning_the_wrong_shape_is_refused() -> None:
    """A statistic that collapses the runs would silently unpair the comparison."""

    bootstrap = load_bootstrap()
    components = np.ones((3, 5, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"^statistic must return one value per run, expected"):
        bootstrap.two_stage_paired_bootstrap_statistic(
            components,
            (0, 0, 0),
            lambda summed: np.array([summed.sum()]),
            resamples=10,
            seed=1,
        )


def test_a_statistic_returning_a_non_finite_value_is_refused() -> None:
    """A statistic that overflows or divides by zero must stop, not publish a NaN.

    The value is returned directly rather than produced by a division, because
    the suite already turns a NumPy divide-by-zero RuntimeWarning into an error.
    That is a second, earlier line of defence; this test drives the guard itself.
    """

    bootstrap = load_bootstrap()
    components = np.ones((3, 5, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"^statistic must return finite values"):
        bootstrap.two_stage_paired_bootstrap_statistic(
            components,
            (0, 0, 0),
            lambda summed: np.full(summed.shape[0], np.inf),
            resamples=10,
            seed=1,
        )


def test_non_finite_components_are_refused() -> None:
    """A NaN sufficient statistic would poison every replicate silently."""

    bootstrap = load_bootstrap()
    components = np.ones((3, 5, 2), dtype=np.float64)
    components[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=r"^components must be finite"):
        bootstrap.two_stage_paired_bootstrap_statistic(
            components, (0, 0, 0), iou_statistic, resamples=10, seed=1
        )


def test_model_labels_must_name_one_model_per_run() -> None:
    """A mislabelled run would be averaged into the wrong model."""

    bootstrap = load_bootstrap()
    components = np.ones((3, 5, 2), dtype=np.float64)
    components[:, :, 1] = 2.0

    with pytest.raises(ValueError, match=r"^model_seed_ids must label exactly one model per run"):
        bootstrap.two_stage_paired_bootstrap_statistic(
            components, (0, 0), iou_statistic, resamples=10, seed=1
        )


def test_a_non_float64_component_array_is_refused() -> None:
    """Integer confusion counts must be widened deliberately, not silently."""

    bootstrap = load_bootstrap()

    with pytest.raises(ValueError, match=r"^components must be a"):
        bootstrap.two_stage_paired_bootstrap_statistic(
            np.ones((3, 5, 2), dtype=np.int64), (0, 0, 0), iou_statistic, resamples=10, seed=1
        )


def test_an_empty_component_axis_is_refused() -> None:
    """Zero components would sum to nothing and make every statistic meaningless."""

    bootstrap = load_bootstrap()

    with pytest.raises(ValueError, match=r"^components must contain at least one run, image,"):
        bootstrap.two_stage_paired_bootstrap_statistic(
            np.ones((3, 5, 0), dtype=np.float64), (0, 0, 0), iou_statistic, resamples=10, seed=1
        )
