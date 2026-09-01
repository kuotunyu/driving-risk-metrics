"""Locked training command."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TextIO

import typer

from drivemetrics.cli._output import run
from drivemetrics.training.backends import build_training_backend
from drivemetrics.training.engine import train

TRAIN_SERVICE = train
BACKEND_FACTORY = build_training_backend

# A formal run is eight to twelve hours inside one notebook cell. One line every
# 500 optimizer steps (sixty lines per run) is enough to tell a healthy run from
# a hung one, and to see how far a killed one got, without burying the result.
PROGRESS_EVERY = 500


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h"


def format_progress(step: int, total: int, loss: float, *, elapsed_seconds: float) -> str:
    """One heartbeat line: where the run is, how it is doing, and when it should end."""

    remaining = (total - step) * elapsed_seconds / step
    return (
        f"step {step}/{total}  loss {loss:.4f}  "
        f"elapsed {_hours(elapsed_seconds)}  remaining {_hours(remaining)}"
    )


def progress_printer(
    every: int = PROGRESS_EVERY,
    stream: TextIO | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Callable[[int, int, float], None]:
    """Build a step observer that writes a heartbeat on the interval and at the end."""

    started = clock()

    def observe(step: int, total: int, loss: float) -> None:
        if step % every == 0 or step == total:
            line = format_progress(step, total, loss, elapsed_seconds=clock() - started)
            print(line, file=sys.stderr if stream is None else stream, flush=True)

    return observe


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
        result = TRAIN_SERVICE(
            config, manifest, output_dir, seed, backend=backend, on_step=progress_printer()
        )
        return {
            "command": "train",
            "final_step": result.final_step,
            "checkpoint_path": str(result.checkpoint_path),
            "checkpoint_sha256": result.checkpoint_sha256,
            "run_record_path": str(result.run_record_path),
        }

    run(operation)
