"""Contracts for the claim-auditable evidence derived from a resolution summary.

The claims audit reads an artifact's top-level protocol and manifest hashes and
the numbers under one JSON pointer. The pre-registered summary keeps those hashes
under ``sweep`` and its intervals in lists, so the published report cites a flat
evidence file derived from the summary. Nothing is recomputed: every number in
the evidence is copied from the summary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from drivemetrics.analysis.claims import (
    ALLOWED_EVIDENCE_TYPES,
    ALLOWED_STATUSES,
    CLAIM_REQUIRED_FIELDS,
    audit_claims,
)
from drivemetrics.posthoc import evidence, statistics

PRIMARY = statistics.PRIMARY_CONFIDENCE
DESCRIPTIVE = statistics.DESCRIPTIVE_CONFIDENCE


def contrast(
    estimate: float,
    primary: tuple[float, float],
    descriptive: tuple[float, float],
    per_seed: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    return {
        "estimate": estimate,
        "instances": 355,
        "intervals": [
            {"confidence": PRIMARY, "low": primary[0], "high": primary[1]},
            {"confidence": DESCRIPTIVE, "low": descriptive[0], "high": descriptive[1]},
        ],
        "per_seed": list(per_seed),
        "resamples": 5000,
        "seed": 20260831,
        "undefined_resamples": 0,
    }


def summary_document(arms: tuple[str, ...] = ("formal", "s085", "native")) -> dict[str, Any]:
    """The fields of a real summary that the evidence reads, with distinct values."""

    rates = {"formal": 0.75, "s085": 0.63, "native": 0.66}
    mious = {"formal": 0.555, "s085": 0.518, "native": 0.528}
    block: dict[str, Any] = {
        "category": "B",
        "downgraded_from": None,
        "formal_minus_native": contrast(
            0.09, (0.035, 0.146), (0.045, 0.135), per_seed=(0.1, 0.08, 0.09)
        ),
        "runs": ["segformer_b2-seed-17", "segformer_b2-seed-42", "segformer_b2-seed-73"],
        "large_person_formal_minus_native": contrast(-0.044, (-0.088, -0.007), (-0.079, -0.013)),
        "scale_mismatch": {
            "confounded": True,
            "large_person_native_worse": True,
            "miou_drop_exceeds_limit": True,
            "miou_drop_formal_minus_native": 0.027,
        },
    }
    if "s085" in arms:
        block["formal_minus_s085"] = contrast(0.115, (0.047, 0.183), (0.06, 0.172))
    return {
        "schema_version": statistics.SUMMARY_SCHEMA_VERSION,
        "analysis_plan": "docs/posthoc/resolution-v1/analysis-plan.md",
        "status": "post-release analysis on the calibration split, not the locked cohort",
        "overall": "B",
        "counts": {"images": 700, "images_with_small_person": 127, "small_person_instances": 355},
        "settings": {"primary_confidence": PRIMARY, "descriptive_confidence": DESCRIPTIVE},
        "sweep": {
            "arms": list(arms),
            "commit": "c" * 40,
            "protocol_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "tertiles_sha256": "d" * 64,
            "hardware": [{"gpu": "NVIDIA A100-SXM4-40GB, 40960 MiB", "runtime": "colab"}],
            "parity_criteria": ["exact"],
        },
        "primary": {"segformer_b2": block},
        "secondary": {
            "dose_response": {
                "segformer_b2": {
                    "monotone_non_increasing": False,
                    "steps": [{"arm": arm, "miss_rate": rates[arm]} for arm in arms],
                }
            },
            "iou": {"segformer_b2": {arm: {"seed_mean_miou": mious[arm]} for arm in arms}},
        },
    }


def test_the_evidence_carries_the_hashes_the_claims_audit_reads() -> None:
    document = evidence.resolution_evidence(summary_document(), "e" * 64)

    assert document["schema_version"] == evidence.EVIDENCE_SCHEMA_VERSION
    assert document["protocol_hash"] == "a" * 64
    assert document["dataset_manifest_hash"] == "b" * 64
    assert document["source_summary_sha256"] == "e" * 64
    assert document["tertiles_sha256"] == "d" * 64
    assert document["commit"] == "c" * 40
    assert document["overall"] == "B"
    assert document["counts"] == {
        "images": 700,
        "images_with_small_person": 127,
        "small_person_instances": 355,
    }
    assert document["gpus"] == ["NVIDIA A100-SXM4-40GB, 40960 MiB"]
    assert document["parity_criteria"] == ["exact"]
    assert document["status"].startswith("post-release analysis on the calibration split")


def test_each_model_block_copies_rates_contrast_and_scale_diagnostic() -> None:
    document = evidence.resolution_evidence(summary_document(), "e" * 64)

    assert document["models"]["segformer_b2"] == {
        "category": "B",
        "formal_miss_rate": 0.75,
        "s085_miss_rate": 0.63,
        "native_miss_rate": 0.66,
        "delta": 0.09,
        "delta_low": 0.035,
        "delta_high": 0.146,
        "delta_low_95": 0.045,
        "delta_high_95": 0.135,
        "delta_seed_17": 0.1,
        "delta_seed_42": 0.08,
        "delta_seed_73": 0.09,
        "formal_minus_s085": 0.115,
        "formal_minus_s085_low": 0.047,
        "formal_minus_s085_high": 0.183,
        "large_person_delta": -0.044,
        "large_person_delta_low_95": -0.079,
        "large_person_delta_high_95": -0.013,
        "formal_miou": 0.555,
        "s085_miou": 0.518,
        "native_miou": 0.528,
        "miou_drop": 0.027,
        "confounded": True,
    }


def test_the_s085_fields_are_absent_when_that_arm_was_not_run() -> None:
    document = evidence.resolution_evidence(summary_document(("formal", "native")), "e" * 64)

    block = document["models"]["segformer_b2"]
    assert not [key for key in block if key.startswith("s085") or "s085" in key]
    assert block["formal_miss_rate"] == 0.75
    assert block["native_miss_rate"] == 0.66


def test_a_document_that_is_not_a_resolution_summary_is_refused() -> None:
    document = summary_document()
    document["schema_version"] = "something-else/v1"

    with pytest.raises(ValueError, match="not a resolution summary"):
        evidence.resolution_evidence(document, "e" * 64)


def test_a_summary_without_the_primary_interval_is_refused() -> None:
    document = summary_document()
    document["primary"]["segformer_b2"]["formal_minus_native"]["intervals"].pop(0)

    with pytest.raises(ValueError, match="no interval at confidence"):
        evidence.resolution_evidence(document, "e" * 64)


def test_the_evidence_file_is_written_from_the_summary_bytes(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    raw = (json.dumps(summary_document(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    summary_path.write_bytes(raw)
    output = tmp_path / "evidence" / "resolution-evidence.json"

    result = evidence.write_resolution_evidence(summary_path, output)

    assert result.evidence_path == output
    assert result.overall == "B"
    written = output.read_bytes()
    assert written.endswith(b"}\n")
    assert b"\r\n" not in written
    assert json.loads(written)["source_summary_sha256"] == hashlib.sha256(raw).hexdigest()
    evidence.write_resolution_evidence(summary_path, output)
    assert output.read_bytes() == written


def test_the_claims_audit_accepts_a_claim_citing_the_evidence(tmp_path: Path) -> None:
    """The point of the derived file: a registry claim can trace its numbers to it."""

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary_document()), encoding="utf-8")
    evidence.write_resolution_evidence(summary_path, tmp_path / "resolution-evidence.json")
    claims = tmp_path / "claims.yaml"
    claims.write_text(
        yaml.safe_dump(
            {
                "allowed_evidence_types": list(ALLOWED_EVIDENCE_TYPES),
                "claim_required_fields": list(CLAIM_REQUIRED_FIELDS),
                "allowed_statuses": list(ALLOWED_STATUSES),
                "claims": [
                    {
                        "claim_id": "p1.posthoc.resolution.segformer",
                        "text": "Small-person miss rate 0.75 at formal and 0.66 at native.",
                        "evidence_type": "observed",
                        "protocol_hash": "a" * 64,
                        "dataset_manifest_hash": "b" * 64,
                        "artifact_path": "resolution-evidence.json",
                        "metric_path": "/models/segformer_b2",
                        "status": "verified",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert audit_claims(claims, tmp_path) == ()
