"""Failure-gallery selection command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.analysis.gallery import DEFAULT_PER_MODEL, select_gallery
from drivemetrics.cli._output import run

GALLERY_SERVICE = select_gallery


def gallery_command(
    index: Annotated[
        Path,
        typer.Option("--index", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    per_model: Annotated[
        int,
        typer.Option(
            "--per-model",
            help="How many worst and how many best images to name for each model.",
        ),
    ] = DEFAULT_PER_MODEL,
) -> None:
    """Name each model's hardest and easiest images by a rule fixed in advance."""

    def operation() -> dict[str, Any]:
        result = GALLERY_SERVICE(index, output, per_model=per_model)
        return {
            "command": "gallery",
            "manifest_path": str(result.manifest_path),
            "per_model": result.per_model,
            "models": list(result.models),
        }

    run(operation)
