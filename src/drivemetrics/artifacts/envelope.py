"""Strict cross-repository portfolio artifact envelope."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PortfolioArtifactEnvelopeV1(BaseModel):
    """Immutable provenance envelope shared independently by all three projects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["portfolio-artifact-envelope/v1"]
    producer_repository: Literal[
        "driving-risk-metrics",
        "bev-calibration-lab",
        "perception-error-to-aeb",
    ]
    producer_release: str = Field(pattern=r"^v[0-9]+\.[0-9]+\.[0-9]+$")
    producer_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    artifact_type: str = Field(min_length=1)
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, Any]

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_at_utc(cls, value: str) -> str:
        """Require an unambiguous, parseable UTC timestamp in canonical Z form."""

        if not value.endswith("Z"):
            raise ValueError("created_at_utc must end in Z")
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise ValueError("created_at_utc must be a valid ISO 8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware UTC")
        return value


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value to stable compact UTF-8 bytes."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def verify_envelope(
    path: Path,
    expected_artifact_type: str,
) -> PortfolioArtifactEnvelopeV1:
    """Load one envelope and fail closed on type or canonical payload drift."""

    raw_value = json.loads(path.read_text(encoding="utf-8"))
    envelope = PortfolioArtifactEnvelopeV1.model_validate(raw_value)
    if envelope.artifact_type != expected_artifact_type:
        raise ValueError(
            "unexpected artifact type: "
            f"expected {expected_artifact_type!r}, got {envelope.artifact_type!r}"
        )

    actual_sha256 = hashlib.sha256(canonical_json_bytes(envelope.payload)).hexdigest()
    if not secrets.compare_digest(actual_sha256, envelope.payload_sha256):
        raise ValueError("payload SHA-256 mismatch")
    return envelope
