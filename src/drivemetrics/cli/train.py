"""Locked training command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.cli._output import run
from drivemetrics.training.backends import build_training_backend
from drivemetrics.training.engine import train

TRAIN_SERVICE = train
BACKEND_FACTORY = build_training_backend


def train_command(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ],
    data_root: Annotated[Path, typer.Option("--data-root", exists=True, file_okay=False)],
    seed: Annotated[int, typer.Option("--seed")],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Run one locked training job and write its single final-step checkpoint."""

    def operation() -> dict[str, Any]:
        backend = BACKEND_FACTORY(config, manifest, data_root)
        result = TRAIN_SERVICE(config, manifest, output_dir, seed, backend=backend)
        return {
            "command": "train",
            "final_step": result.final_step,
            "checkpoint_path": str(result.checkpoint_path),
            "checkpoint_sha256": result.checkpoint_sha256,
            "run_record_path": str(result.run_record_path),
        }

    run(operation)
