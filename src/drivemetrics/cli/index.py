"""Formal run index command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.artifacts.formal_index import build_formal_run_index
from drivemetrics.cli._output import run
from drivemetrics.protocol.risk_profiles import load_risk_profile

INDEX_SERVICE = build_formal_run_index
PROFILE_LOADER = load_risk_profile


def index_command(
    runs_root: Annotated[Path, typer.Option("--runs-root", exists=True, file_okay=False)],
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ],
    risk_profile: Annotated[
        Path,
        typer.Option("--risk-profile", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Assemble and gate the immutable index of the nine formal runs."""

    def operation() -> dict[str, Any]:
        profile = PROFILE_LOADER(risk_profile)
        result = INDEX_SERVICE(
            runs_root,
            config,
            manifest,
            output,
            critical_class_ids=profile.critical_class_ids,
        )
        return {
            "command": "index",
            "index_path": str(result.index_path),
            "run_count": result.run_count,
            "protocol_sha256": result.protocol_sha256,
            "dataset_manifest_sha256": result.dataset_manifest_sha256,
        }

    run(operation)
