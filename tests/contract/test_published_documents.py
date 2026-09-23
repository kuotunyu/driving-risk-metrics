"""The published documents must pass the claims validator before they can merge.

README.md and README.en.md say that a number nobody can trace fails the build.
The validator that enforces this used to run only in the Pages workflow, after a
merge to main, so a pull request could put an untraced number on the public
README and still pass CI. Running the same command here puts it inside the
ordinary test suite, where every pull request meets it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / ".agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py"
DOCUMENTS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "README.en.md",
    *sorted((REPO_ROOT / "docs/release-notes").glob("*.md")),
]
PAGES_WORKFLOW = REPO_ROOT / ".github/workflows/pages.yml"
#: Mirrors the validator: a claim comment anywhere outside a fenced block is traced.
CLAIM_COMMENT = re.compile(r"<!--\s*claim:")
FENCE_PREFIXES = ("```", "~~~")


def claim_markers_outside_fences(document: Path) -> int:
    """Count claim markers the way the validator reads them, skipping fenced code."""

    count = 0
    in_fence = False
    for line in document.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(FENCE_PREFIXES):
            in_fence = not in_fence
            continue
        if not in_fence:
            count += len(CLAIM_COMMENT.findall(line))
    return count


def test_every_published_document_passes_the_claims_validator() -> None:
    """Every marked number traces to its claim, and no unmarked number sits beside a metric."""

    arguments = [
        sys.executable,
        str(VALIDATOR),
        "--claims",
        "docs/claims.yaml",
        "--repo-root",
        str(REPO_ROOT),
    ]
    for document in DOCUMENTS:
        arguments += ["--document", str(document)]
    result = subprocess.run(
        arguments,
        cwd=REPO_ROOT,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
        check=False,
    )
    output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    assert result.returncode == 0, output
    try:
        trace = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(f"validator trace is not JSON: {error}\n{output}") from error
    markers = sum(claim_markers_outside_fences(document) for document in DOCUMENTS)
    assert markers > 0
    assert len(trace["statements"]) >= markers, output


def test_the_pages_workflow_validates_every_published_document() -> None:
    """The post-merge Pages check must not audit fewer documents than the test suite."""

    workflow = yaml.safe_load(PAGES_WORKFLOW.read_text(encoding="utf-8"))
    runs = [
        str(step.get("run", ""))
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "validate_claims.py" in str(step.get("run", ""))
    ]

    assert len(runs) == 1
    for document in DOCUMENTS:
        relative = document.relative_to(REPO_ROOT).as_posix()
        assert f"--document {relative}" in runs[0], relative
