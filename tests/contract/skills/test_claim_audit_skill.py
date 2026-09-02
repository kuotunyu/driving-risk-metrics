"""Contracts for the `auditing-driving-risk-claims` skill and its validator.

The registry audit (`driving-risk audit-claims`) proves that every claim in
`docs/claims.yaml` reproduces from its artifact. It says nothing about the
sentence that ends up in a README, a slide or a report. The gap between the two
is where a number gets retyped, rounded, carried over from a stale run, or
simply made up, and it is what this skill's validator closes: every statement
proposed for publication is traced to a claim ID, an artifact path and a JSON
pointer, and the numbers it states are compared with the numbers at that
pointer.

Every rejection test asserts on the validator's own message, never on a bare
non-zero exit, so a missing validator cannot pass a test by failing to run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "auditing-driving-risk-claims"
SKILL_DOC = SKILL_DIR / "SKILL.md"
VALIDATOR = SKILL_DIR / "scripts" / "validate_claims.py"

PROTOCOL_HASH = "1" * 64
MANIFEST_HASH = "2" * 64
ARTIFACT = "artifacts/analysis/bdd100k_semseg_v1/metrics.json"
MIOU_POINTER = "/metrics/segformer_b2/miou"
ACCURACY_POINTER = "/metrics/upernet_dinov2_small/pixel_accuracy"


def registry_document(claims: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "allowed_evidence_types": ["observed", "derived", "synthetic", "illustrative"],
        "claim_required_fields": [
            "claim_id",
            "text",
            "evidence_type",
            "protocol_hash",
            "dataset_manifest_hash",
            "artifact_path",
            "metric_path",
            "status",
        ],
        "allowed_statuses": ["draft", "verified", "rejected", "superseded"],
        "claims": claims,
    }


def claim(
    claim_id: str,
    text: str,
    metric_path: str,
    *,
    status: str = "verified",
    evidence_type: str = "observed",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "text": text,
        "evidence_type": evidence_type,
        "protocol_hash": PROTOCOL_HASH,
        "dataset_manifest_hash": MANIFEST_HASH,
        "artifact_path": ARTIFACT,
        "metric_path": metric_path,
        "status": status,
    }


DEFAULT_CLAIMS = [
    claim("segformer-b2-miou", "segformer_b2 reaches 0.712 mIoU.", MIOU_POINTER),
    claim(
        "upernet-dinov2-small-pixel-accuracy",
        "upernet_dinov2_small has the highest pixel accuracy, 0.947.",
        ACCURACY_POINTER,
    ),
]


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Any]:
    """A repository-shaped directory whose registry audit passes."""

    artifact = tmp_path / ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "protocol_hash": PROTOCOL_HASH,
                "dataset_manifest_hash": MANIFEST_HASH,
                "metrics": {
                    "segformer_b2": {"miou": 0.712, "pixel_accuracy": 0.941},
                    "upernet_dinov2_small": {"miou": 0.729, "pixel_accuracy": 0.947},
                },
            }
        ),
        encoding="utf-8",
    )
    claims_path = tmp_path / "docs" / "claims.yaml"
    claims_path.parent.mkdir()
    write_registry(claims_path, DEFAULT_CLAIMS)
    return {"root": tmp_path, "claims": claims_path, "artifact": artifact}


def write_registry(path: Path, claims: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump(registry_document(claims)), encoding="utf-8")


def write_proposal(root: Path, proposals: list[dict[str, str]]) -> Path:
    path = root / "proposal.yaml"
    path.write_text(yaml.safe_dump({"proposals": proposals}), encoding="utf-8")
    return path


def write_document(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def run_validator(
    workspace: dict[str, Any],
    *,
    proposal: Path | None = None,
    documents: tuple[Path, ...] = (),
) -> subprocess.CompletedProcess[str]:
    arguments = [
        sys.executable,
        str(VALIDATOR),
        "--claims",
        str(workspace["claims"]),
        "--repo-root",
        str(workspace["root"]),
    ]
    if proposal is not None:
        arguments += ["--proposal", str(proposal)]
    for document in documents:
        arguments += ["--document", str(document)]
    return subprocess.run(arguments, cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def test_the_skill_declares_its_exact_name_and_triggering_description() -> None:
    """A skill the agent cannot find by description never runs at all."""

    front_matter = yaml.safe_load(SKILL_DOC.read_text(encoding="utf-8").split("---")[1])

    assert front_matter["name"] == "auditing-driving-risk-claims"
    description = front_matter["description"].lower()
    assert description.startswith("use when")
    assert "claim" in description
    assert "readme" in description


def test_the_skill_excludes_pure_wording_changes() -> None:
    """Rewriting prose that keeps every number is not a claim audit."""

    front_matter = yaml.safe_load(SKILL_DOC.read_text(encoding="utf-8").split("---")[1])
    description = front_matter["description"].lower()

    assert "wording" in description or "style" in description


def test_the_skill_routes_through_the_validator_and_names_the_marker() -> None:
    """A trace an agent keeps in its head is exactly what the baseline lost."""

    text = SKILL_DOC.read_text(encoding="utf-8")

    assert "validate_claims.py" in text
    assert "<!-- claim:" in text


def test_a_proposal_that_matches_its_artifact_is_accepted_with_a_trace(
    workspace: dict[str, Any],
) -> None:
    """The success path must hand back the trace, or nobody can cite it."""

    proposal = write_proposal(
        workspace["root"],
        [{"claim_id": "segformer-b2-miou", "text": "segformer_b2 reaches 0.712 mIoU."}],
    )

    result = run_validator(workspace, proposal=proposal)

    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    [statement] = status["statements"]
    assert statement["claim_id"] == "segformer-b2-miou"
    assert statement["artifact_path"] == ARTIFACT
    assert statement["metric_path"] == MIOU_POINTER
    assert statement["verdict"] == "pass"


def test_a_number_the_artifact_does_not_hold_is_rejected_with_its_pointer(
    workspace: dict[str, Any],
) -> None:
    """The retyped digit is the failure this validator exists for."""

    proposal = write_proposal(
        workspace["root"],
        [
            {
                "claim_id": "upernet-dinov2-small-pixel-accuracy",
                "text": "upernet_dinov2_small has the highest pixel accuracy at 0.974.",
            }
        ],
    )

    result = run_validator(workspace, proposal=proposal)

    assert result.returncode == 1
    assert "upernet-dinov2-small-pixel-accuracy" in result.stderr
    assert ARTIFACT in result.stderr
    assert ACCURACY_POINTER in result.stderr
    assert "0.974" in result.stderr
    assert "0.947" in result.stderr


def test_a_proposal_citing_no_registry_claim_is_rejected(workspace: dict[str, Any]) -> None:
    """A number with no claim behind it cannot be traced, so it cannot be published."""

    proposal = write_proposal(
        workspace["root"],
        [{"claim_id": "segformer-b2-critical-recall", "text": "critical recall is 0.913."}],
    )

    result = run_validator(workspace, proposal=proposal)

    assert result.returncode == 1
    assert "segformer-b2-critical-recall" in result.stderr
    assert "no registry claim" in result.stderr


def test_a_proposal_citing_an_unverified_claim_is_rejected(workspace: dict[str, Any]) -> None:
    """Draft, rejected and superseded claims are not evidence yet, or no longer."""

    write_registry(
        workspace["claims"],
        [
            *DEFAULT_CLAIMS,
            claim(
                "draft-miou",
                "draft says 0.729.",
                "/metrics/upernet_dinov2_small/miou",
                status="draft",
            ),
        ],
    )
    proposal = write_proposal(
        workspace["root"], [{"claim_id": "draft-miou", "text": "the draft mIoU is 0.729."}]
    )

    result = run_validator(workspace, proposal=proposal)

    assert result.returncode == 1
    assert "draft-miou" in result.stderr
    assert "'draft'" in result.stderr
    assert "verified" in result.stderr


def test_a_registry_that_fails_its_own_audit_blocks_every_statement(
    workspace: dict[str, Any],
) -> None:
    """A matching sentence over a broken registry is still an unbacked number."""

    workspace["artifact"].unlink()
    proposal = write_proposal(
        workspace["root"],
        [{"claim_id": "segformer-b2-miou", "text": "segformer_b2 reaches 0.712 mIoU."}],
    )

    result = run_validator(workspace, proposal=proposal)

    assert result.returncode == 1
    assert "registry" in result.stderr
    assert "segformer-b2-miou" in result.stderr


def test_a_document_line_with_a_metric_number_and_no_marker_is_rejected(
    workspace: dict[str, Any],
) -> None:
    """An unmarked result sentence is untraceable even when its number is right."""

    document = write_document(
        workspace["root"],
        "README.md",
        "# Results\n\nsegformer_b2 reaches 0.712 mIoU on the locked cohort.\n",
    )

    result = run_validator(workspace, documents=(document,))

    assert result.returncode == 1
    assert "README.md:3" in result.stderr
    assert "marker" in result.stderr
    assert "0.712" in result.stderr


def test_a_marked_document_line_is_traced_and_checked(workspace: dict[str, Any]) -> None:
    """The marker binds the sentence to its claim; the numbers still have to agree."""

    document = write_document(
        workspace["root"],
        "README.md",
        "\n".join(
            [
                "# Results",
                "",
                "segformer_b2 reaches 0.712 mIoU. <!-- claim: segformer-b2-miou -->",
                "upernet_dinov2_small has the highest pixel accuracy at 0.974."
                " <!-- claim: upernet-dinov2-small-pixel-accuracy -->",
                "",
            ]
        ),
    )

    result = run_validator(workspace, documents=(document,))

    assert result.returncode == 1
    assert "README.md:4" in result.stderr
    assert ACCURACY_POINTER in result.stderr
    assert "0.974" in result.stderr
    assert "README.md:3" not in result.stderr


def test_a_fully_marked_document_is_accepted(workspace: dict[str, Any]) -> None:
    """The document the release produces must pass, or the gate is ignored."""

    document = write_document(
        workspace["root"],
        "README.md",
        "segformer_b2 reaches 0.712 mIoU. <!-- claim: segformer-b2-miou -->\n",
    )

    result = run_validator(workspace, documents=(document,))

    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    [statement] = status["statements"]
    assert statement["source"] == "README.md:1"
    assert statement["metric_path"] == MIOU_POINTER


def test_fenced_code_blocks_are_not_audited(workspace: dict[str, Any]) -> None:
    """Commands and JSON excerpts mention metrics and numbers without claiming anything."""

    document = write_document(
        workspace["root"],
        "README.md",
        '```json\n{"miou": 0.5}\n```\n',
    )

    result = run_validator(workspace, documents=(document,))

    assert result.returncode == 0, result.stderr


def test_a_synthetic_claim_must_be_labelled_where_it_is_stated(
    workspace: dict[str, Any],
) -> None:
    """A synthetic number printed like a measurement is what the vocabulary exists to stop."""

    write_registry(
        workspace["claims"],
        [
            *DEFAULT_CLAIMS,
            claim(
                "synthetic-miou",
                "The synthetic chain reports 0.712 mIoU.",
                MIOU_POINTER,
                evidence_type="synthetic",
            ),
        ],
    )
    document = write_document(
        workspace["root"],
        "README.md",
        "The chain reports 0.712 mIoU. <!-- claim: synthetic-miou -->\n",
    )

    result = run_validator(workspace, documents=(document,))

    assert result.returncode == 1
    assert "synthetic-miou" in result.stderr
    assert "synthetic" in result.stderr
    assert "README.md:1" in result.stderr


def test_nothing_to_audit_is_an_error_not_a_pass(workspace: dict[str, Any]) -> None:
    """Exit zero with no input would let an empty invocation stand as evidence."""

    result = run_validator(workspace)

    assert result.returncode == 2
    assert "nothing to audit" in result.stderr


def test_every_option_is_declared() -> None:
    """The trace must be reproducible from the command line alone."""

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for option in ("--claims", "--repo-root", "--proposal", "--document"):
        assert option in result.stdout
