"""Fail-closed validator for one locked-cohort segmentation evaluation.

This exists because the things that invalidate a locked evaluation are not
visible by reading output: a cohort that is the wrong size, a run scored under a
different protocol, a checkpoint chosen part-way through training, a seed
outside the approved matrix, or a published claim whose evidence type does not
match how the number was actually obtained. Each one produces a plausible
report.

Every violation is collected and reported rather than stopping at the first, so
one run tells you everything that is wrong instead of one thing at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from drivemetrics.analysis.claims import verified_claims
from drivemetrics.artifacts.run_record import RunRecordV1
from drivemetrics.data.manifest import load_manifest
from drivemetrics.protocol.config import BDD100KSemanticProtocolV1, load_protocol

APPROVED_SEEDS: tuple[int, ...] = (17, 42, 73)
FINAL_CHECKPOINT_KEY = "final_checkpoint"
MEASURED_EVIDENCE_TYPES: tuple[str, ...] = ("observed", "derived")
LOCKED_SPLIT = "locked_validation"


def _expected_counts(protocol: BDD100KSemanticProtocolV1) -> dict[str, int]:
    splits = protocol.splits
    return {
        "source_train": splits.source_train,
        "train": splits.train,
        "calibration": splits.calibration,
        LOCKED_SPLIT: splits.locked_validation,
    }


def _checkpoint_keys(artifacts: dict[str, str]) -> tuple[str, ...]:
    return tuple(sorted(key for key in artifacts if "checkpoint" in key))


def validate(
    config_path: Path,
    manifest_path: Path,
    run_record_path: Path,
    claims_path: Path,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Return every violation, plus the status of an evaluation that has none."""

    violations: list[str] = []
    loaded = load_protocol(config_path)
    protocol_hash = loaded.protocol_sha256
    manifest = load_manifest(manifest_path)

    expected = _expected_counts(loaded.protocol)
    actual_count = len(manifest.sample_ids)
    if manifest.split_name not in expected:
        violations.append(
            f"manifest split name {manifest.split_name!r} is not one of {sorted(expected)}"
        )
    elif actual_count != expected[manifest.split_name]:
        violations.append(
            f"{manifest.split_name} split must contain exactly "
            f"{expected[manifest.split_name]} paired samples, found {actual_count}"
        )

    try:
        record = RunRecordV1.model_validate(json.loads(run_record_path.read_text(encoding="utf-8")))
    except (ValidationError, ValueError) as error:
        violations.append(f"run record failed its own schema: {error}")
        return tuple(violations), {}

    if record.protocol_sha256 != protocol_hash:
        violations.append(
            f"run record protocol hash {record.protocol_sha256} does not match the "
            f"protocol at {config_path} ({protocol_hash})"
        )
    if record.dataset_manifest_sha256 != manifest.manifest_sha256:
        violations.append(
            "run record describes a different dataset manifest: "
            f"{record.dataset_manifest_sha256} rather than {manifest.manifest_sha256}"
        )
    if record.seed not in APPROVED_SEEDS:
        violations.append(f"seed {record.seed} is not one of the approved seeds {APPROVED_SEEDS}")

    checkpoints = _checkpoint_keys(record.artifacts)
    unexpected = tuple(key for key in checkpoints if key != FINAL_CHECKPOINT_KEY)
    if unexpected:
        violations.append(
            f"checkpoint artifact {unexpected[0]!r} is not the final step; the protocol "
            f"selects {FINAL_CHECKPOINT_KEY} only"
        )
    if manifest.split_name == LOCKED_SPLIT and FINAL_CHECKPOINT_KEY in record.artifacts:
        violations.append(
            "the locked validation cohort was used to fit a checkpoint; it may only be "
            "scored, never trained or tuned on"
        )

    violations.extend(_claim_violations(claims_path, manifest.manifest_sha256, protocol_hash))

    status: dict[str, object] = {
        "validator": "validate_locked_eval",
        "split_name": manifest.split_name,
        "sample_count": actual_count,
        "protocol_sha256": protocol_hash,
        "dataset_manifest_sha256": manifest.manifest_sha256,
        "seed": record.seed,
        "run_id": record.run_id,
    }
    return tuple(violations), status


def _claim_violations(
    claims_path: Path,
    manifest_hash: str,
    protocol_hash: str,
) -> tuple[str, ...]:
    """Check only claims that cite this cohort, and only ones cleared to publish."""

    try:
        claims = verified_claims(claims_path)
    except (ValidationError, ValueError) as error:
        return (f"claims registry failed its own schema: {error}",)

    violations: list[str] = []
    for claim in claims:
        if claim.dataset_manifest_hash != manifest_hash:
            continue
        if claim.protocol_hash != protocol_hash:
            violations.append(
                f"claim {claim.claim_id!r} cites protocol hash {claim.protocol_hash}, "
                f"but this cohort was scored under {protocol_hash}"
            )
        if claim.evidence_type not in MEASURED_EVIDENCE_TYPES:
            violations.append(
                f"claim {claim.claim_id!r} is verified but its evidence type is "
                f"{claim.evidence_type!r}; a published locked-cohort result must be "
                f"one of {MEASURED_EVIDENCE_TYPES}"
            )
    return tuple(violations)


def main(argv: list[str] | None = None) -> int:
    """Print a JSON status and exit zero, or print every violation and exit one."""

    parser = argparse.ArgumentParser(
        description="Validate one locked-cohort segmentation evaluation before it is trusted."
    )
    parser.add_argument("--config", type=Path, required=True, help="Locked protocol YAML.")
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen cohort manifest.")
    parser.add_argument("--run-record", type=Path, required=True, help="Run record JSON.")
    parser.add_argument("--claims", type=Path, required=True, help="Claims registry YAML.")
    arguments = parser.parse_args(argv)

    try:
        violations, status = validate(
            arguments.config,
            arguments.manifest,
            arguments.run_record,
            arguments.claims,
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"validation could not run: {error}", file=sys.stderr)
        return 2

    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
