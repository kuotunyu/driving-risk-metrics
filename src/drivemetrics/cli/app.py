"""Top-level command-line application."""

from __future__ import annotations

import typer

from drivemetrics.cli import (
    aggregate,
    calibrate,
    data,
    evaluate,
    extended,
    figures,
    gallery,
    index,
    report,
    train,
)

app = typer.Typer(
    add_completion=False,
    help="Safety-aware evaluation tools for road-scene semantic segmentation.",
    no_args_is_help=True,
)

app.add_typer(data.app, name="data")
app.command("train")(train.train_command)
app.command("calibrate")(calibrate.calibrate_command)
app.command("evaluate")(evaluate.evaluate_command)
app.command("index")(index.index_command)
app.command("aggregate")(aggregate.aggregate_command)
app.command("gallery")(gallery.gallery_command)
app.command("extended-metrics")(extended.extended_metrics_command)
app.command("report")(report.report_command)
app.command("figures")(figures.figures_command)
app.command("audit-claims")(report.audit_claims_command)
