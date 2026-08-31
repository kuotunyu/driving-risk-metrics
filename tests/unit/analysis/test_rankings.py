"""Contracts for conventional-versus-safety ranking comparison."""

from __future__ import annotations

from types import ModuleType

import pytest


def load_rankings_module() -> ModuleType:
    try:
        from drivemetrics.analysis import rankings
    except ImportError:
        pytest.fail("drivemetrics.analysis.rankings is missing", pytrace=False)
    return rankings


def test_identical_model_order_reports_no_reversal() -> None:
    """Reporting a reversal for a stable ranking would manufacture the headline result."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.65, "segformer_b0": 0.60},
        "critical_recall": {
            "fcn_resnet50": 0.80,
            "deeplabv3_resnet50": 0.75,
            "segformer_b0": 0.70,
        },
    }

    comparisons = rankings.compare_rankings(metric_table, "miou")

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.metric_name == "critical_recall"
    assert comparison.baseline_order == ("fcn_resnet50", "deeplabv3_resnet50", "segformer_b0")
    assert comparison.comparison_order == ("fcn_resnet50", "deeplabv3_resnet50", "segformer_b0")
    assert comparison.reversal_observed is False


def test_strictly_reversed_model_order_reports_a_reversal() -> None:
    """Missing a genuine pair flip would hide the safety failure the project exists to find."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.60},
        "critical_recall": {"fcn_resnet50": 0.40, "deeplabv3_resnet50": 0.90},
    }

    comparisons = rankings.compare_rankings(metric_table, "miou")

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.baseline_order == ("fcn_resnet50", "deeplabv3_resnet50")
    assert comparison.comparison_order == ("deeplabv3_resnet50", "fcn_resnet50")
    assert comparison.reversal_observed is True


def test_comparisons_start_without_pairwise_intervals() -> None:
    """A metric table of scalars cannot support an interval; the statistics stage attaches them."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.60},
        "critical_recall": {"fcn_resnet50": 0.40, "deeplabv3_resnet50": 0.90},
    }

    comparisons = rankings.compare_rankings(metric_table, "miou")

    assert comparisons[0].pairwise_intervals == {}


def test_analysis_package_exports_ranking_entry_points() -> None:
    """The statistics and report stages consume these through the package entry point."""

    import drivemetrics.analysis as analysis

    rankings = load_rankings_module()
    assert analysis.compare_rankings is rankings.compare_rankings
    assert analysis.RankingComparison is rankings.RankingComparison


def test_a_baseline_tie_is_never_reported_as_a_reversal() -> None:
    """Tuple inequality alone would turn an arbitrary tie-break into a safety finding."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.65, "deeplabv3_resnet50": 0.65},
        "critical_recall": {"fcn_resnet50": 0.10, "deeplabv3_resnet50": 0.90},
    }

    comparisons = rankings.compare_rankings(metric_table, "miou")

    comparison = comparisons[0]
    assert comparison.baseline_order == ("deeplabv3_resnet50", "fcn_resnet50")
    assert comparison.comparison_order == ("deeplabv3_resnet50", "fcn_resnet50")
    assert comparison.reversal_observed is False


def test_one_flipped_pair_among_three_models_reports_a_reversal() -> None:
    """Comparing only the top-ranked model would miss a reversal deeper in the ranking."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.65, "segformer_b0": 0.60},
        "critical_recall": {
            "fcn_resnet50": 0.90,
            "deeplabv3_resnet50": 0.50,
            "segformer_b0": 0.60,
        },
    }

    comparisons = rankings.compare_rankings(metric_table, "miou")

    comparison = comparisons[0]
    assert comparison.comparison_order == ("fcn_resnet50", "segformer_b0", "deeplabv3_resnet50")
    assert comparison.reversal_observed is True


def test_comparisons_are_returned_in_sorted_metric_name_order() -> None:
    """Iteration order of a caller dict must never change a published report."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.60},
        "vru_recall": {"fcn_resnet50": 0.50, "deeplabv3_resnet50": 0.40},
        "aurc_complement": {"fcn_resnet50": 0.30, "deeplabv3_resnet50": 0.20},
    }

    comparisons = rankings.compare_rankings(metric_table, "miou")

    assert tuple(item.metric_name for item in comparisons) == ("aurc_complement", "vru_recall")


def test_a_table_holding_only_the_baseline_metric_has_nothing_to_compare() -> None:
    """Inventing a self-comparison would report a guaranteed non-reversal as evidence."""

    rankings = load_rankings_module()
    metric_table = {"miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.60}}

    assert rankings.compare_rankings(metric_table, "miou") == ()


def test_an_unknown_baseline_metric_fails_closed() -> None:
    """Falling back to an arbitrary metric would silently change the reported estimand."""

    rankings = load_rankings_module()
    metric_table = {"miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.60}}

    with pytest.raises(ValueError, match="baseline_metric"):
        rankings.compare_rankings(metric_table, "critical_recall")


def test_an_empty_metric_table_fails_closed() -> None:
    """An empty ranking report would look like a completed comparison."""

    rankings = load_rankings_module()

    with pytest.raises(ValueError, match="at least one metric"):
        rankings.compare_rankings({}, "miou")


def test_metrics_scoring_different_model_sets_fail_closed() -> None:
    """A missing model would be silently dropped from one half of the comparison."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.60},
        "critical_recall": {"fcn_resnet50": 0.40, "segformer_b0": 0.90},
    }

    with pytest.raises(ValueError, match="exactly the same models"):
        rankings.compare_rankings(metric_table, "miou")


def test_a_single_model_cannot_be_ranked() -> None:
    """A one-model ranking has no pair and would always report a stable result."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.70},
        "critical_recall": {"fcn_resnet50": 0.40},
    }

    with pytest.raises(ValueError, match="at least two models"):
        rankings.compare_rankings(metric_table, "miou")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True, "0.5"])
def test_non_finite_or_non_numeric_scores_fail_closed(bad_value: object) -> None:
    """NaN compares false against everything and would hide a real pair flip."""

    rankings = load_rankings_module()
    metric_table = {
        "miou": {"fcn_resnet50": 0.70, "deeplabv3_resnet50": 0.60},
        "critical_recall": {"fcn_resnet50": bad_value, "deeplabv3_resnet50": 0.90},
    }

    with pytest.raises(ValueError, match="finite numbers"):
        rankings.compare_rankings(metric_table, "miou")  # type: ignore[arg-type]


def test_pairwise_intervals_are_copied_into_the_frozen_comparison() -> None:
    """Aliasing a caller dict would let a published frozen comparison change afterwards."""

    from drivemetrics.analysis import BootstrapInterval

    rankings = load_rankings_module()
    interval = BootstrapInterval(
        estimate=0.1,
        low=0.0,
        high=0.2,
        confidence=0.95,
        resamples=5000,
        seed=20260831,
    )
    supplied = {"fcn_resnet50|deeplabv3_resnet50": interval}

    comparison = rankings.RankingComparison(
        metric_name="critical_recall",
        baseline_order=("fcn_resnet50", "deeplabv3_resnet50"),
        comparison_order=("deeplabv3_resnet50", "fcn_resnet50"),
        reversal_observed=True,
        pairwise_intervals=supplied,
    )
    supplied.clear()

    assert comparison.pairwise_intervals == {"fcn_resnet50|deeplabv3_resnet50": interval}
