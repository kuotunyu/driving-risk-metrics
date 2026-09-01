"""Contracts for the `running-locked-segmentation-evals` skill and its validator.

The skill is guidance, so the deterministic part of it is the validator. These
tests pin the validator's fail-closed behaviour and the parts of SKILL.md that a
reader depends on: its identity, its triggering description, and the fact that it
routes through the validator rather than asking an agent to eyeball a run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from drivemetrics.data.manifest import DatasetManifest, build_paired_manifest
from drivemetrics.protocol.config import load_protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "running-locked-segmentation-evals"
SKILL_DOC = SKILL_DIR / "SKILL.md"
VALIDATOR = SKILL_DIR / "scripts" / "validate_locked_eval.py"
PROTOCOL = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

LOCKED_VALIDATION_COUNT = 1000
APPROVED_SEED = 17
COMMIT = "a" * 40
LOCK_HASH = "b" * 64
CONFIG_HASH = "c" * 64


def build_cohort(tmp_path: Path, sample_count: int, split_name: str) -> DatasetManifest:
    """Freeze a synthetic cohort of the requested size.

    The manifest hashes file bytes and never decodes them, so one byte per file
    is enough to exercise the real pairing, ordering and hashing path quickly.
    """

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(sample_count):
        sample_id = f"sample{index:05d}"
        (images / f"{sample_id}.jpg").write_bytes(b"i")
        (labels / f"{sample_id}_train_id.png").write_bytes(b"l")
    return build_paired_manifest(images, labels, split_name)


def write_manifest(manifest: DatasetManifest, path: Path) -> Path:
    from dataclasses import asdict

    path.write_text(json.dumps(asdict(manifest), sort_keys=True), encoding="utf-8")
    return path


def evaluation_record(manifest: DatasetManifest, protocol_hash: str, **overrides: Any) -> Any:
    """Build the run record a locked-cohort evaluation writes."""

    document: dict[str, Any] = {
        "schema_version": "driving-risk-run/v1",
        "run_id": "fcn_resnet50-seed-17-eval",
        "commit": COMMIT,
        "config_sha256": CONFIG_HASH,
        "protocol_sha256": protocol_hash,
        "dataset_manifest_sha256": manifest.manifest_sha256,
        "lock_sha256": LOCK_HASH,
        "hardware": {"gpu": "NVIDIA A100-SXM4-40GB", "runtime": "colab"},
        "seed": APPROVED_SEED,
        "started_at_utc": "2026-09-01T00:00:00Z",
        "finished_at_utc": "2026-09-01T01:00:00Z",
        "status": "succeeded",
        "artifacts": {f"sample{index:05d}": "d" * 64 for index in range(3)},
    }
    document.update(overrides)
    return document


def write_claims(path: Path, claims: list[dict[str, Any]]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
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
        ),
        encoding="utf-8",
    )
    return path


def run_validator(
    manifest_path: Path,
    record_path: Path,
    claims_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(PROTOCOL),
            "--manifest",
            str(manifest_path),
            "--run-record",
            str(record_path),
            "--claims",
            str(claims_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def protocol_hash() -> str:
    return load_protocol(PROTOCOL).protocol_sha256


@pytest.fixture(scope="module")
def locked_cohort(tmp_path_factory: pytest.TempPathFactory) -> DatasetManifest:
    """One correctly sized locked-validation cohort, shared by every case."""

    return build_cohort(
        tmp_path_factory.mktemp("locked"), LOCKED_VALIDATION_COUNT, "locked_validation"
    )


@pytest.fixture
def workspace(
    tmp_path: Path,
    locked_cohort: DatasetManifest,
    protocol_hash: str,
) -> dict[str, Any]:
    manifest_path = write_manifest(locked_cohort, tmp_path / "locked_validation.json")
    claims_path = write_claims(tmp_path / "claims.yaml", [])
    return {
        "dir": tmp_path,
        "manifest": manifest_path,
        "claims": claims_path,
        "cohort": locked_cohort,
        "protocol_hash": protocol_hash,
    }


def check(workspace: dict[str, Any], **overrides: Any) -> subprocess.CompletedProcess[str]:
    record = evaluation_record(workspace["cohort"], workspace["protocol_hash"], **overrides)
    record_path = workspace["dir"] / "run_record.json"
    record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return run_validator(workspace["manifest"], record_path, workspace["claims"])


def test_the_skill_declares_its_exact_name_and_triggering_description() -> None:
    """A skill the agent cannot find by description never runs at all."""

    text = SKILL_DOC.read_text(encoding="utf-8")
    front_matter = yaml.safe_load(text.split("---")[1])

    assert front_matter["name"] == "running-locked-segmentation-evals"
    description = front_matter["description"].lower()
    assert description.startswith("use when")
    assert "bdd100k" in description
    assert "segmentation" in description


def test_the_skill_excludes_generic_image_classification() -> None:
    """Without an exclusion the description competes with unrelated vision work."""

    front_matter = yaml.safe_load(SKILL_DOC.read_text(encoding="utf-8").split("---")[1])

    assert "classification" in front_matter["description"].lower()


def test_the_skill_routes_through_the_validator_rather_than_inspection() -> None:
    """Guidance an agent must apply by eye is exactly what the baseline got wrong."""

    text = SKILL_DOC.read_text(encoding="utf-8")

    assert "validate_locked_eval.py" in text


def test_a_consistent_locked_evaluation_is_accepted(workspace: dict[str, Any]) -> None:
    """A validator that rejects a correct run would be abandoned on first use."""

    result = check(workspace)

    assert result.returncode == 0, result.stderr
    assert "locked_validation" in result.stdout


def test_a_wrong_split_count_is_rejected(
    tmp_path: Path,
    protocol_hash: str,
) -> None:
    """A short cohort silently changes what every published number means."""

    cohort = build_cohort(tmp_path / "short", 3, "locked_validation")
    manifest_path = write_manifest(cohort, tmp_path / "locked_validation.json")
    record_path = tmp_path / "run_record.json"
    record_path.write_text(
        json.dumps(evaluation_record(cohort, protocol_hash), sort_keys=True),
        encoding="utf-8",
    )
    claims_path = write_claims(tmp_path / "claims.yaml", [])

    result = run_validator(manifest_path, record_path, claims_path)

    assert result.returncode != 0
    assert "1000" in result.stderr


def test_training_on_the_locked_cohort_is_rejected(workspace: dict[str, Any]) -> None:
    """Fitting anything on the locked cohort destroys the only held-out evidence."""

    result = check(workspace, artifacts={"final_checkpoint": "e" * 64})

    assert result.returncode != 0
    # Not just "locked": the validator's own path contains that word, so a
    # missing-file error would otherwise satisfy this test without the check
    # ever existing.
    assert "locked validation cohort" in result.stderr.lower()


def test_an_unapproved_seed_is_rejected(workspace: dict[str, Any]) -> None:
    """A run outside seeds 17, 42 and 73 cannot enter the nine-job matrix."""

    result = check(workspace, seed=5)

    assert result.returncode != 0
    assert "seed" in result.stderr.lower()


def test_a_disagreeing_protocol_hash_is_rejected(workspace: dict[str, Any]) -> None:
    """A run scored under a different protocol is not comparable to the others."""

    result = check(workspace, protocol_sha256="f" * 64)

    assert result.returncode != 0
    assert "protocol" in result.stderr.lower()


def test_a_non_final_checkpoint_is_rejected(
    tmp_path: Path,
    protocol_hash: str,
) -> None:
    """Selecting a mid-training checkpoint is selection on the evaluation cohort."""

    cohort = build_cohort(tmp_path / "train", 3, "train")
    manifest_path = write_manifest(cohort, tmp_path / "train.json")
    record_path = tmp_path / "run_record.json"
    record_path.write_text(
        json.dumps(
            evaluation_record(
                cohort,
                protocol_hash,
                artifacts={"checkpoint_step_15000": "e" * 64},
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    claims_path = write_claims(tmp_path / "claims.yaml", [])

    result = run_validator(manifest_path, record_path, claims_path)

    assert result.returncode != 0
    assert "final" in result.stderr.lower()


def test_a_run_record_for_a_different_manifest_is_rejected(workspace: dict[str, Any]) -> None:
    """Pairing a record with the wrong cohort would validate an unrelated run."""

    result = check(workspace, dataset_manifest_sha256="0" * 64)

    assert result.returncode != 0
    assert "manifest" in result.stderr.lower()


def test_a_verified_claim_about_the_locked_cohort_must_be_measured(
    workspace: dict[str, Any],
) -> None:
    """Publishing an illustrative number as a verified result is the core failure."""

    write_claims(
        workspace["claims"],
        [
            {
                "claim_id": "miou-fcn",
                "text": "FCN reaches 0.61 mIoU on the locked cohort.",
                "evidence_type": "illustrative",
                "protocol_hash": workspace["protocol_hash"],
                "dataset_manifest_hash": workspace["cohort"].manifest_sha256,
                "artifact_path": "artifacts/metrics.json",
                "metric_path": "/metrics/fcn_resnet50/miou",
                "status": "verified",
            }
        ],
    )

    result = check(workspace)

    assert result.returncode != 0
    assert "evidence" in result.stderr.lower()


def test_a_claim_citing_another_protocol_is_rejected(workspace: dict[str, Any]) -> None:
    """A claim bound to a different protocol is evidence for a different study."""

    write_claims(
        workspace["claims"],
        [
            {
                "claim_id": "miou-fcn",
                "text": "FCN reaches 0.61 mIoU on the locked cohort.",
                "evidence_type": "observed",
                "protocol_hash": "9" * 64,
                "dataset_manifest_hash": workspace["cohort"].manifest_sha256,
                "artifact_path": "artifacts/metrics.json",
                "metric_path": "/metrics/fcn_resnet50/miou",
                "status": "verified",
            }
        ],
    )

    result = check(workspace)

    assert result.returncode != 0
    assert "protocol" in result.stderr.lower()


def test_a_measured_verified_claim_is_accepted(workspace: dict[str, Any]) -> None:
    """The validator must pass the case the whole pipeline is built to produce."""

    write_claims(
        workspace["claims"],
        [
            {
                "claim_id": "miou-fcn",
                "text": "FCN reaches 0.61 mIoU on the locked cohort.",
                "evidence_type": "observed",
                "protocol_hash": workspace["protocol_hash"],
                "dataset_manifest_hash": workspace["cohort"].manifest_sha256,
                "artifact_path": "artifacts/metrics.json",
                "metric_path": "/metrics/fcn_resnet50/miou",
                "status": "verified",
            }
        ],
    )

    result = check(workspace)

    assert result.returncode == 0, result.stderr


def test_a_malformed_run_record_is_rejected(workspace: dict[str, Any]) -> None:
    """A record that fails its own schema must stop the run, not warn about it."""

    record_path = workspace["dir"] / "run_record.json"
    record_path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    result = run_validator(workspace["manifest"], record_path, workspace["claims"])

    assert result.returncode != 0
    assert "run record" in result.stderr.lower()


def test_every_required_option_is_declared() -> None:
    """A validator that runs with defaults would silently check the wrong cohort."""

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for option in ("--config", "--manifest", "--run-record", "--claims"):
        assert option in result.stdout
