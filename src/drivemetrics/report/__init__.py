"""Deterministic static evidence report generated from verified claims."""

from .builder import LIMITATIONS, ReportResult, build_report, load_json_object
from .figures import bar_figure, interval_figure
from .svg import FiguresResult, write_figures

__all__ = [
    "LIMITATIONS",
    "FiguresResult",
    "ReportResult",
    "bar_figure",
    "build_report",
    "interval_figure",
    "load_json_object",
    "write_figures",
]
