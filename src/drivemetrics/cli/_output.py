"""Shared status and diagnostic conventions for every command."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import typer


def run(operation: Callable[[], Mapping[str, Any]]) -> None:
    """Run one service call, printing a JSON status or a stderr diagnostic.

    Commands hold no metric, training, or reporting logic. They validate options,
    call one injected service, and translate its outcome into a machine-readable
    status on stdout, or a diagnostic on stderr with a nonzero exit. Notebooks and
    release workflows parse that status, so it is never mixed with prose.
    """

    try:
        payload = operation()
    except (OSError, TypeError, ValueError, KeyError) as error:
        typer.echo(f"{type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(dict(payload), sort_keys=True))
