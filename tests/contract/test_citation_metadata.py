"""The citation metadata GitHub shows must describe the version the package declares."""

import datetime as dt
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_citation_version_matches_the_package_version() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert str(citation["version"]) == project["version"]
    released = citation["date-released"]
    assert isinstance(released, str)
    assert dt.date.fromisoformat(released).isoformat() == released
