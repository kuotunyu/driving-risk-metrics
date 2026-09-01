"""Statistics aggregation command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.analysis.aggregate import aggregate_runs
from drivemetrics.cli._output import run

AGGREGATE_SERVICE = aggregate_runs


def aggregate_command(
    index: Annotated[
        Path,
        typer.Option("--index", exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Turn the validated formal run index into the three published documents."""

    def operation() -> dict[str, Any]:
        result = AGGREGATE_SERVICE(index, output_dir)
        return {
            "command": "aggregate",
            "metrics_path": str(result.metrics_path),
            "intervals_path": str(result.intervals_path),
            "rankings_path": str(result.rankings_path),
            "models": list(result.models),
            "sample_count": result.sample_count,
        }

    run(operation)
