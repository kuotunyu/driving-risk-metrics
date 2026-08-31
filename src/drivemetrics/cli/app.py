"""Top-level command-line application."""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="Safety-aware evaluation tools for road-scene semantic segmentation.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Expose the command group while feature commands are added incrementally."""
