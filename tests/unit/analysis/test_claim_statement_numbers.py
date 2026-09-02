"""Behavior tests for comparing a published statement against its claim's metric."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

ARTIFACT_PATH = "evidence/metrics.json"


def load_claims_module() -> ModuleType:
    """Import inside tests so a missing module is an intentional RED failure."""

    try:
        from drivemetrics.analysis import claims
    except ImportError:
        pytest.fail("drivemetrics.analysis.claims is missing", pytrace=False)
    return claims


def verified_claim(claims: ModuleType) -> object:
    return claims.ClaimV1.model_validate(
        {
            "claim_id": "p1-miou-example",
            "text": "Measured mIoU is 0.8125.",
            "evidence_type": "observed",
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "artifact_path": ARTIFACT_PATH,
            "metric_path": "/metrics/miou",
            "status": "verified",
        }
    )


def write_artifact(repository_root: Path, value: object) -> None:
    artifact = repository_root / ARTIFACT_PATH
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(value), encoding="utf-8")


def test_text_numbers_extracts_every_number_in_document_order() -> None:
    """The README audit compares the numbers a sentence states, not the sentence."""

    claims = load_claims_module()

    numbers = claims.text_numbers("0.712 mIoU, 91.3% critical recall, 5000 resamples")

    assert numbers == (Decimal("0.712"), Decimal("91.3"), Decimal("5000"))


def test_metric_numbers_returns_the_numbers_at_the_claim_pointer(tmp_path: Path) -> None:
    """A published statement is checked against the exact value its claim cites."""

    claims = load_claims_module()
    write_artifact(tmp_path, {"metrics": {"miou": 0.8125, "other": 0.5}})

    assert claims.metric_numbers(verified_claim(claims), tmp_path) == {Decimal("0.8125")}


def test_metric_numbers_fails_closed_on_a_missing_pointer(tmp_path: Path) -> None:
    """A pointer that resolves to nothing must never read as an empty number set."""

    claims = load_claims_module()
    write_artifact(tmp_path, {"metrics": {"other": 0.5}})

    with pytest.raises(LookupError):
        claims.metric_numbers(verified_claim(claims), tmp_path)


def test_metric_numbers_fails_closed_on_a_missing_artifact(tmp_path: Path) -> None:
    """An absent artifact is the clearest case of an unbacked number."""

    claims = load_claims_module()

    with pytest.raises(FileNotFoundError):
        claims.metric_numbers(verified_claim(claims), tmp_path)
