"""Dataset preflight command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.cli._output import run
from drivemetrics.data.preflight import run_preflight

app = typer.Typer(help="Dataset preflight and manifest commands.", no_args_is_help=True)

PREFLIGHT_SERVICE = run_preflight


@app.command("preflight")
def preflight_command(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
    data_root: Annotated[Path, typer.Option("--data-root", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", file_okay=False)],
) -> None:
    """Verify the dataset against the protocol and freeze the formal cohorts."""

    def operation() -> dict[str, Any]:
        result = PREFLIGHT_SERVICE(config, data_root, output)
        return {
            "command": "data preflight",
            "protocol_sha256": result.protocol_sha256,
            "counts": dict(result.counts),
            "manifest_sha256": dict(result.manifest_sha256),
            "manifests": {name: str(path) for name, path in result.manifest_paths.items()},
        }

    run(operation)
