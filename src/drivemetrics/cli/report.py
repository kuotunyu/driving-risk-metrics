"""Static report and claim-audit commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.analysis.claims import audit_claims
from drivemetrics.cli._output import run
from drivemetrics.report.builder import build_report

REPORT_SERVICE = build_report
AUDIT_SERVICE = audit_claims


def report_command(
    claims: Annotated[
        Path,
        typer.Option("--claims", exists=True, dir_okay=False, readable=True),
    ],
    artifacts_dir: Annotated[
        Path,
        typer.Option("--artifacts-dir", exists=True, file_okay=False),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", exists=True, file_okay=False),
    ] = Path(),
) -> None:
    """Render the static evidence report from verified claims and frozen artifacts."""

    def operation() -> dict[str, Any]:
        result = REPORT_SERVICE(claims, artifacts_dir, output_dir, repository_root=repo_root)
        return {
            "command": "report",
            "index_path": str(result.index_path),
            "figure_count": len(result.figure_paths),
            "claim_count": result.claim_count,
        }

    run(operation)


def audit_claims_command(
    claims: Annotated[
        Path,
        typer.Option("--claims", exists=True, dir_okay=False, readable=True),
    ],
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", exists=True, file_okay=False),
    ] = Path(),
) -> None:
    """Fail unless every claim can be reproduced from its own artifact."""

    def operation() -> dict[str, Any]:
        violations = AUDIT_SERVICE(claims, repo_root)
        if violations:
            raise ValueError("; ".join(violations))
        return {"command": "audit-claims", "violations": 0}

    run(operation)
