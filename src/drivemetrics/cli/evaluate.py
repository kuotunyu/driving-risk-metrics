"""Locked-cohort evaluation command."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TextIO

import typer

from drivemetrics.cli._output import run
from drivemetrics.evaluation.backends import TorchEvaluationBackend
from drivemetrics.evaluation.engine import evaluate_checkpoint

EVALUATE_SERVICE = evaluate_checkpoint
BACKEND_FACTORY = TorchEvaluationBackend

# Scoring the locked cohort and copying its artifacts takes most of an hour, and
# the first formal session was interrupted by hand because that whole time was
# silent: a working run and a hung one looked identical. One line every fifty
# images is enough to tell them apart.
PROGRESS_EVERY = 50


def format_sample_progress(done: int, total: int, *, elapsed_seconds: float) -> str:
    """One heartbeat line: how far the cohort has been scored and what remains."""

    remaining = (total - done) * elapsed_seconds / done
    return (
        f"scored {done}/{total}  elapsed {elapsed_seconds / 60:.1f}m  "
        f"remaining {remaining / 60:.1f}m"
    )


def sample_progress_printer(
    every: int = PROGRESS_EVERY,
    stream: TextIO | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Callable[[int, int], None]:
    """Build a sample observer that writes a heartbeat on the interval and at the end."""

    started = clock()

    def observe(done: int, total: int) -> None:
        if done % every == 0 or done == total:
            line = format_sample_progress(done, total, elapsed_seconds=clock() - started)
            print(line, file=sys.stderr if stream is None else stream, flush=True)

    return observe


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
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Torch device the model runs on, such as cuda or cpu. There is no default.",
        ),
    ],
    temperature: Annotated[
        Path | None,
        typer.Option("--temperature", exists=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Score one checkpoint over one frozen cohort and publish per-image evidence."""

    def operation() -> dict[str, Any]:
        result = EVALUATE_SERVICE(
            config,
            manifest,
            checkpoint,
            data_root,
            output_dir,
            backend=BACKEND_FACTORY(device=device),
            temperature_path=temperature,
            on_sample=sample_progress_printer(),
        )
        return {
            "command": "evaluate",
            "calibrated": temperature is not None,
            "evaluated_samples": result.evaluated_samples,
            "artifact_count": len(result.artifact_paths),
            "run_record_path": str(result.run_record_path),
        }

    run(operation)
