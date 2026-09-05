"""Evidence figures command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.cli._output import run
from drivemetrics.report.svg import write_figures

FIGURES_SERVICE = write_figures


def figures_command(
    artifacts_dir: Annotated[
        Path,
        typer.Option("--artifacts-dir", exists=True, file_okay=False),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Draw the two evidence figures as deterministic SVG from the committed documents."""

    def operation() -> dict[str, Any]:
        result = FIGURES_SERVICE(artifacts_dir, output_dir)
        return {
            "command": "figures",
            "figure_paths": [str(path) for path in result.figure_paths],
            "figure_count": len(result.figure_paths),
        }

    run(operation)
