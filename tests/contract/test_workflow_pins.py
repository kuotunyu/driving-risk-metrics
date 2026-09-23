"""Every workflow runs on a fixed runner image and fixed action commits."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github/workflows"
PINNED_ACTION = re.compile(r"[\w.-]+/[\w.-]+(/[\w.-]+)*@[0-9a-f]{40}")


def workflow_files(directory: Path) -> list[Path]:
    # GitHub runs workflows saved with either YAML extension.
    return sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])


WORKFLOWS = workflow_files(WORKFLOW_DIR)


def jobs(path: Path) -> dict[str, Any]:
    found: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
    return found


def test_every_known_workflow_is_checked() -> None:
    assert {"ci.yml", "pages.yml", "release.yml"} <= {path.name for path in WORKFLOWS}


def test_workflow_files_include_both_yaml_extensions(tmp_path: Path) -> None:
    for name in ("a.yml", "b.yaml", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert [path.name for path in workflow_files(tmp_path)] == ["a.yml", "b.yaml"]


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


def test_pages_build_checkout_does_not_persist_credentials() -> None:
    steps = jobs(WORKFLOW_DIR / "pages.yml")["build"]["steps"]
    checkouts = [step for step in steps if "actions/checkout@" in step.get("uses", "")]
    assert checkouts
    for checkout in checkouts:
        assert checkout.get("with", {}).get("persist-credentials") is False
