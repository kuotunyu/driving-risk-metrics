"""Deterministic static evidence report generated from verified claims."""

from .builder import LIMITATIONS, ReportResult, build_report
from .figures import bar_figure, interval_figure

__all__ = [
    "LIMITATIONS",
    "ReportResult",
    "bar_figure",
    "build_report",
    "interval_figure",
]
