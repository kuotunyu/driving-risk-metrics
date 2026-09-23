"""Every workflow runs on a fixed runner image and fixed action commits."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parents[2] / ".github/workflows").glob("*.yml"))
PINNED_ACTION = re.compile(r"[\w.-]+/[\w.-]+(/[\w.-]+)*@[0-9a-f]{40}")


def jobs(path: Path) -> dict[str, Any]:
    found: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
    return found


def test_every_known_workflow_is_checked() -> None:
    assert {"ci.yml", "pages.yml", "release.yml"} <= {path.name for path in WORKFLOWS}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_job_runs_on_the_pinned_ubuntu_image(path: Path) -> None:
    for name, job in jobs(path).items():
        # A job that calls a reusable workflow takes its runner from that workflow.
        if "uses" not in job:
            assert job.get("runs-on") == "ubuntu-24.04", name


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_action_is_pinned_to_a_full_commit_sha(path: Path) -> None:
    for name, job in jobs(path).items():
        references = [job["uses"]] if "uses" in job else []
        references += [step["uses"] for step in job.get("steps", []) if "uses" in step]
        for reference in references:
            assert PINNED_ACTION.fullmatch(reference), f"{name}: {reference}"
