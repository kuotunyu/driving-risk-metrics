"""Post-release all-seed analysis commands: one verified extraction, then the statistics.

The analysis is pre-registered in ``docs/posthoc/allseed-v1/analysis-plan.md``.
The commands hold no logic of their own; they validate options and call one
service each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.cli._output import run
from drivemetrics.cli.evaluate import sample_progress_printer
from drivemetrics.posthoc.allseed import extract_all_seeds
from drivemetrics.posthoc.allseed_evidence import write_allseed_evidence
from drivemetrics.posthoc.allseed_statistics import analyse_all_seeds

EXTRACT_SERVICE = extract_all_seeds
ANALYSE_SERVICE = analyse_all_seeds
EVIDENCE_SERVICE = write_allseed_evidence

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Post-release all-seed analysis of the released predictions: one verified "
        "extraction, then the pre-registered statistics behind a reproduction gate."
    ),
)


@app.command("extract")
def extract_command(
    index: Annotated[
        Path,
        typer.Option(
            "--index",
            exists=True,
            dir_okay=False,
            help="formal_run_index.json at the root of the nine runs' artifact directories.",
        ),
    ],
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            help="The frozen locked-cohort manifest the masks are verified against.",
        ),
    ],
    labels_root: Annotated[Path, typer.Option("--labels-root", exists=True, file_okay=False)],
    instance_root: Annotated[
        Path,
        typer.Option(
            "--instance-root",
            exists=True,
            file_okay=False,
            help="Directory holding labels/ins_seg/bitmasks/val/<sample_id>.png.",
        ),
    ],
    tertiles: Annotated[Path, typer.Option("--tertiles", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Read and verify every artifact once, and keep what the statistics need."""

    def operation() -> dict[str, Any]:
        result = EXTRACT_SERVICE(
            index,
            manifest,
            labels_root,
            instance_root,
            tertiles,
            output_dir,
            on_image=sample_progress_printer(),
        )
        return {
            "command": "allseed extract",
            "output_dir": str(result.output_dir),
            "runs": list(result.runs),
            "images": result.images,
        }

    run(operation)


@app.command("analyse")
def analyse_command(
    extract_dir: Annotated[Path, typer.Option("--extract-dir", exists=True, file_okay=False)],
    evidence_dir: Annotated[
        Path,
        typer.Option(
            "--evidence-dir",
            exists=True,
            file_okay=False,
            help="The released evidence the reproduction gate compares against.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Run the reproduction gate, then the pre-registered statistics (CPU only)."""

    def operation() -> dict[str, Any]:
        result = ANALYSE_SERVICE(extract_dir, evidence_dir, output_dir)
        return {
            "command": "allseed analyse",
            "summary_path": str(result.summary_path),
            "categories": result.categories,
            "separable": result.separable,
        }

    run(operation)


@app.command("evidence")
def evidence_command(
    summary: Annotated[
        Path,
        typer.Option(
            "--summary",
            exists=True,
            dir_okay=False,
            readable=True,
            help="The summary.json written by `allseed analyse`.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Derive the flat, claim-auditable evidence the published report cites (no recomputation)."""

    def operation() -> dict[str, Any]:
        result = EVIDENCE_SERVICE(summary, output)
        return {"command": "allseed evidence", "evidence_path": str(result.evidence_path)}

    run(operation)
