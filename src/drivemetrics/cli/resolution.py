"""Post-release resolution sweep commands: parity gate, sweep, analysis and report evidence.

The analysis is pre-registered in ``docs/posthoc/resolution-v1/analysis-plan.md``
and runs on the calibration split only. The commands hold no logic of their
own; they validate options and call one service each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from drivemetrics.cli._output import run
from drivemetrics.cli.evaluate import sample_progress_printer
from drivemetrics.evaluation.backends import TorchEvaluationBackend
from drivemetrics.posthoc.evidence import write_resolution_evidence
from drivemetrics.posthoc.resolution import (
    DEFAULT_ARMS,
    DEFAULT_PARITY_IMAGES,
    formal_parity,
    resolution_sweep,
)
from drivemetrics.posthoc.statistics import analyse_sweep

PARITY_SERVICE = formal_parity
SWEEP_SERVICE = resolution_sweep
ANALYSE_SERVICE = analyse_sweep
EVIDENCE_SERVICE = write_resolution_evidence
BACKEND_FACTORY = TorchEvaluationBackend

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Post-release resolution sweep on the calibration split: parity gate, "
        "resumable sweep, pre-registered analysis, claim-auditable report evidence."
    ),
)

ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)]
ManifestOption = Annotated[
    Path,
    typer.Option(
        "--manifest",
        exists=True,
        dir_okay=False,
        readable=True,
        help="The frozen CALIBRATION manifest. Any other split is refused.",
    ),
]
CheckpointsOption = Annotated[
    Path,
    typer.Option(
        "--checkpoints",
        exists=True,
        dir_okay=False,
        readable=True,
        help='JSON object {"<run_id>": ["<checkpoint path>", "<expected sha256>"]}.',
    ),
]
DataRootOption = Annotated[Path, typer.Option("--data-root", exists=True, file_okay=False)]
BitmasksOption = Annotated[
    Path,
    typer.Option(
        "--instance-bitmasks",
        exists=True,
        file_okay=False,
        help="Directory holding <sample_id>.png BDD100K instance bitmasks.",
    ),
]
TertilesOption = Annotated[
    Path,
    typer.Option(
        "--tertiles",
        exists=True,
        dir_okay=False,
        help="The FROZEN area tertiles. They are never re-learned here.",
    ),
]
DeviceOption = Annotated[
    str,
    typer.Option("--device", help="Torch device, such as cuda or cpu. There is no default."),
]


@app.command("parity")
def parity_command(
    config: ConfigOption,
    manifest: ManifestOption,
    checkpoints: CheckpointsOption,
    data_root: DataRootOption,
    instance_bitmasks: BitmasksOption,
    tertiles: TertilesOption,
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    device: DeviceOption,
    images: Annotated[int, typer.Option("--images")] = DEFAULT_PARITY_IMAGES,
    criterion: Annotated[
        str,
        typer.Option(
            "--criterion",
            help="exact (zero mismatched pixels) or decision (L4 only: zero changed decisions).",
        ),
    ] = "exact",
) -> None:
    """Prove the formal arm reproduces the released evaluation path on this machine."""

    def operation() -> dict[str, Any]:
        result = PARITY_SERVICE(
            config,
            manifest,
            checkpoints,
            data_root,
            instance_bitmasks,
            tertiles,
            output,
            backend=BACKEND_FACTORY(device=device),
            images=images,
            criterion=criterion,
        )
        return {
            "command": "resolution-sweep parity",
            "passed": True,
            "criterion": result.criterion,
            "report_path": str(result.report_path),
            "mismatched_pixels": result.mismatched_pixels,
            "changed_critical_miss_decisions": result.changed_decisions,
        }

    run(operation)


@app.command("run")
def run_command(
    config: ConfigOption,
    manifest: ManifestOption,
    checkpoints: CheckpointsOption,
    data_root: DataRootOption,
    instance_bitmasks: BitmasksOption,
    tertiles: TertilesOption,
    parity_report: Annotated[
        Path,
        typer.Option(
            "--parity-report",
            exists=True,
            dir_okay=False,
            help="A PASSED parity report for the same protocol, cohort and checkpoints.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    device: DeviceOption,
    arms: Annotated[
        str,
        typer.Option(
            "--arms",
            help="Comma-separated arms; must include formal and native. Optional: s060, "
            "formal_unpadded (outside the pre-registration).",
        ),
    ] = ",".join(DEFAULT_ARMS),
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Score at most this many not-yet-scored images."),
    ] = None,
) -> None:
    """Score calibration images at every arm; resumes where a previous call stopped."""

    def operation() -> dict[str, Any]:
        result = SWEEP_SERVICE(
            config,
            manifest,
            checkpoints,
            data_root,
            instance_bitmasks,
            tertiles,
            parity_report,
            output_dir,
            backend=BACKEND_FACTORY(device=device),
            arms=arms.split(","),
            limit=limit,
            on_sample=sample_progress_printer(),
        )
        return {
            "command": "resolution-sweep run",
            "results_path": str(result.results_path),
            "scored_samples": result.scored_samples,
            "completed_samples": result.completed_samples,
            "total_samples": result.total_samples,
            "complete": result.completed_samples == result.total_samples,
        }

    run(operation)


@app.command("analyse")
def analyse_command(
    output_dir: Annotated[Path, typer.Option("--output-dir", exists=True, file_okay=False)],
) -> None:
    """Compute the pre-registered statistics from a finished sweep (CPU only)."""

    def operation() -> dict[str, Any]:
        result = ANALYSE_SERVICE(output_dir)
        return {
            "command": "resolution-sweep analyse",
            "summary_path": str(result.summary_path),
            "categories": result.categories,
            "overall": result.overall,
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
            help="The summary.json written by `resolution-sweep analyse`.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Derive the flat, claim-auditable evidence the published report cites (no recomputation)."""

    def operation() -> dict[str, Any]:
        result = EVIDENCE_SERVICE(summary, output)
        return {
            "command": "resolution-sweep evidence",
            "evidence_path": str(result.evidence_path),
            "overall": result.overall,
        }

    run(operation)
