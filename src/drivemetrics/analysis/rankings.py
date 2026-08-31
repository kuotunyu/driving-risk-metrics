"""Conventional-versus-safety ranking comparison with strict pair-flip detection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from drivemetrics.analysis.bootstrap import BootstrapInterval


@dataclass(frozen=True)
class RankingComparison:
    """One safety metric compared against the conventional baseline ranking."""

    metric_name: str
    baseline_order: tuple[str, ...]
    comparison_order: tuple[str, ...]
    reversal_observed: bool
    pairwise_intervals: dict[str, BootstrapInterval]

    def __post_init__(self) -> None:
        """Copy the supplied intervals so a frozen comparison cannot change later."""

        object.__setattr__(self, "pairwise_intervals", dict(self.pairwise_intervals))


def _validate_metric_table(
    metric_table: Mapping[str, Mapping[str, float]],
    baseline_metric: str,
) -> None:
    if not metric_table:
        raise ValueError("metric_table must contain at least one metric")
    if baseline_metric not in metric_table:
        raise ValueError("baseline_metric must be present in metric_table")
    models = tuple(sorted(metric_table[baseline_metric]))
    if len(models) < 2:
        raise ValueError("ranking comparison requires at least two models")
    for scores in metric_table.values():
        if tuple(sorted(scores)) != models:
            raise ValueError("every metric must score exactly the same models")
        for value in scores.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError("metric values must be finite numbers")


def _ordered_models(scores: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(sorted(scores, key=lambda model: (-scores[model], model)))


def _has_strict_pair_flip(
    baseline_scores: Mapping[str, float],
    comparison_scores: Mapping[str, float],
) -> bool:
    models = sorted(baseline_scores)
    for position, first in enumerate(models):
        for second in models[position + 1 :]:
            baseline_delta = baseline_scores[first] - baseline_scores[second]
            comparison_delta = comparison_scores[first] - comparison_scores[second]
            if baseline_delta * comparison_delta < 0.0:
                return True
    return False


def compare_rankings(
    metric_table: Mapping[str, Mapping[str, float]],
    baseline_metric: str,
) -> tuple[RankingComparison, ...]:
    """Compare every non-baseline metric ranking against the baseline metric ranking.

    Models are ordered by descending metric value with ties broken by ascending
    model name, so callers must orient every metric so that larger is better. A
    reversal is reported only when a model pair strictly flips order, which means
    a tie is never counted as a reversal. ``reversal_observed`` is a derived
    observation and is never a release requirement. ``pairwise_intervals`` starts
    empty because a table of scalars cannot support interval estimation; the
    statistics stage attaches intervals produced by ``two_stage_paired_bootstrap``.
    """

    _validate_metric_table(metric_table, baseline_metric)
    baseline_scores = metric_table[baseline_metric]
    baseline_order = _ordered_models(baseline_scores)
    return tuple(
        RankingComparison(
            metric_name=metric_name,
            baseline_order=baseline_order,
            comparison_order=_ordered_models(metric_table[metric_name]),
            reversal_observed=_has_strict_pair_flip(
                baseline_scores,
                metric_table[metric_name],
            ),
            pairwise_intervals={},
        )
        for metric_name in sorted(metric_table)
        if metric_name != baseline_metric
    )
