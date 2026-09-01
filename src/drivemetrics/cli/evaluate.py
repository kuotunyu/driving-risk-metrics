"""Locked-cohort evaluation command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.cli._output import run
from drivemetrics.evaluation.backends import TorchEvaluationBackend
from drivemetrics.evaluation.engine import evaluate_checkpoint

EVALUATE_SERVICE = evaluate_checkpoint
BACKEND_FACTORY = TorchEvaluationBackend


def evaluate_command(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ],
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", exists=True, dir_okay=False, readable=True),
    ],
    data_root: Annotated[Path, typer.Option("--data-root", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Score one checkpoint over one frozen cohort and publish per-image evidence."""

    def operation() -> dict[str, Any]:
        result = EVALUATE_SERVICE(
            config,
            manifest,
            checkpoint,
            data_root,
            output_dir,
            backend=BACKEND_FACTORY(),
        )
        return {
            "command": "evaluate",
            "evaluated_samples": result.evaluated_samples,
            "artifact_count": len(result.artifact_paths),
            "run_record_path": str(result.run_record_path),
        }

    run(operation)
