"""Extended metrics command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.analysis.extended import extended_metrics
from drivemetrics.cli._output import run

EXTENDED_SERVICE = extended_metrics


def extended_metrics_command(
    index: Annotated[
        Path,
        typer.Option("--index", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    labels_root: Annotated[
        Path | None,
        typer.Option(
            "--labels-root",
            exists=True,
            file_okay=False,
            help="Dataset root holding labels/sem_seg/masks/val. Bands need it.",
        ),
    ] = None,
    instance_root: Annotated[
        Path | None,
        typer.Option(
            "--instance-root",
            exists=True,
            file_okay=False,
            help="Dataset root holding labels/ins_seg/bitmasks/val. Instances need it.",
        ),
    ] = None,
    tertiles: Annotated[
        Path | None,
        typer.Option(
            "--tertiles",
            exists=True,
            dir_okay=False,
            help="The FROZEN area tertiles. They are never re-learned here.",
        ),
    ] = None,
) -> None:
    """Publish selective risk, image bands and instance coverage where the inputs allow."""

    def operation() -> dict[str, Any]:
        result = EXTENDED_SERVICE(
            index,
            output,
            labels_root=labels_root,
            instance_root=instance_root,
            tertiles_path=tertiles,
        )
        return {
            "command": "extended-metrics",
            "document_path": str(result.document_path),
            "models": list(result.models),
            "computed": list(result.computed),
        }

    run(operation)
