"""Scalar temperature fitting command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.calibration.service import calibrate_checkpoint
from drivemetrics.cli._output import run
from drivemetrics.evaluation.backends import TorchEvaluationBackend

CALIBRATE_SERVICE = calibrate_checkpoint
BACKEND_FACTORY = TorchEvaluationBackend


def calibrate_command(
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
    """Fit one scalar temperature on the frozen calibration cohort."""

    def operation() -> dict[str, Any]:
        result = CALIBRATE_SERVICE(
            config,
            manifest,
            checkpoint,
            data_root,
            output_dir,
            backend=BACKEND_FACTORY(),
        )
        return {
            "command": "calibrate",
            "temperature": result.temperature,
            "artifact_path": str(result.artifact_path),
            "run_record_path": str(result.run_record_path),
            "sampled_images": result.sampled_images,
            "pixels_per_image": result.pixels_per_image,
        }

    run(operation)
