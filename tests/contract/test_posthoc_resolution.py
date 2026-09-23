"""The published resolution analysis must trace to the run it reports.

docs/posthoc/resolution-v1/ holds the post-release resolution analysis on the
calibration split. Its summary is the unedited output of `resolution-sweep
analyse` from the Colab run; its evidence is derived from that summary by
package code; and every number in its results document is traced to a claim in
its own registry, kept apart from the released docs/claims.yaml so the v1.0.x
claims are untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from drivemetrics.posthoc.evidence import write_resolution_evidence

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTHOC = REPO_ROOT / "docs/posthoc/resolution-v1"
SUMMARY = POSTHOC / "evidence/summary.json"
EVIDENCE = POSTHOC / "evidence/resolution-evidence.json"
CLAIMS = POSTHOC / "claims.yaml"
RESULTS = POSTHOC / "results.md"
VALIDATOR = REPO_ROOT / ".agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py"
#: SHA-256 of summary.json as the Colab A100 run wrote it (2026-09-23), re-derived
#: byte for byte on a second machine before it was committed.
COLAB_SUMMARY_SHA256 = "d73e3bef84d239ec03d387b54bf190b60e5ca2bd2606c5f6d8228f042d2113d5"
#: BDD100K sample IDs look like 0a1b2c3d-4e5f6a7b; none may be published here.
SAMPLE_ID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{8}\b")


def test_the_committed_summary_is_the_colab_run_output() -> None:
    assert hashlib.sha256(SUMMARY.read_bytes()).hexdigest() == COLAB_SUMMARY_SHA256


def test_the_committed_evidence_is_derived_byte_for_byte_from_the_summary(
    tmp_path: Path,
) -> None:
    result = write_resolution_evidence(SUMMARY, tmp_path / "resolution-evidence.json")

    assert result.evidence_path.read_bytes() == EVIDENCE.read_bytes()


def test_no_published_file_names_an_image_or_an_instance() -> None:
    """Only aggregate numbers are published: no sample IDs, no per-instance records."""

    for path in (SUMMARY, EVIDENCE):
        text = path.read_text(encoding="utf-8")
        assert "sample_id" not in text, path.name
        assert "instance_id" not in text, path.name
        assert SAMPLE_ID.search(text) is None, path.name


def test_the_results_document_passes_the_claims_validator() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--claims",
            str(CLAIMS.relative_to(REPO_ROOT)),
            "--repo-root",
            str(REPO_ROOT),
            "--document",
            str(RESULTS),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
        check=False,
    )
    output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    assert result.returncode == 0, output
    trace = json.loads(result.stdout)
    markers = len(re.findall(r"<!--\s*claim:", RESULTS.read_text(encoding="utf-8")))
    assert markers > 0
    assert len(trace["statements"]) == markers, output
