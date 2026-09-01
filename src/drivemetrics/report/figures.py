"""Deterministic Plotly figure documents built entirely in Python.

Category order is always sorted rather than inherited from dictionary order, so
the same inputs always produce the same published chart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def bar_figure(title: str, value_title: str, values: Mapping[str, float]) -> dict[str, Any]:
    """Return one bar figure whose category order never depends on dictionary order."""

    categories = sorted(values)
    return {
        "data": [
            {
                "type": "bar",
                "name": value_title,
                "x": list(categories),
                "y": [float(values[name]) for name in categories],
            }
        ],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": {"text": "model"}},
            "yaxis": {"title": {"text": value_title}},
        },
    }


def interval_figure(
    title: str,
    intervals: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Return one effect-size figure that always shows the interval around the estimate.

    Plotting an estimate without its interval, or an interval without its
    estimate, would misrepresent the uncertainty the study actually measured.
    """

    labels = sorted(intervals)
    estimates = [float(intervals[label]["estimate"]) for label in labels]
    upper = [
        float(intervals[label]["high"]) - estimate
        for label, estimate in zip(labels, estimates, strict=True)
    ]
    lower = [
        estimate - float(intervals[label]["low"])
        for label, estimate in zip(labels, estimates, strict=True)
    ]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "orientation": "h",
                "x": estimates,
                "y": list(labels),
                "error_x": {
                    "type": "data",
                    "array": upper,
                    "arrayminus": lower,
                    "visible": True,
                },
            }
        ],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": {"text": "effect size"}},
        },
    }
