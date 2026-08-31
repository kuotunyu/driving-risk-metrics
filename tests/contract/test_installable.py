"""Installation and repository-layout contracts for the v1 package foundation."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_console_entrypoint_imports() -> None:
    """Removing ``drivemetrics.cli.app`` must break the public CLI contract."""

    try:
        module = importlib.import_module("drivemetrics.cli.app")
    except ModuleNotFoundError:
        pytest.fail("the declared drivemetrics.cli.app module is missing", pytrace=False)

    assert callable(module.app)


def test_data_package_is_not_ignored() -> None:
    """A broad dataset ignore must never hide package code under ``src``."""

    result = subprocess.run(
        ["git", "check-ignore", "-q", "src/drivemetrics/data/bdd100k.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, (
        "src/drivemetrics/data/bdd100k.py is ignored; package data adapters "
        f"would be lost (git stderr: {result.stderr.strip()!r})"
    )
