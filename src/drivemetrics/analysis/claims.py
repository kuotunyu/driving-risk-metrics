"""Strict public-claim vocabulary, filtering, and evidence audit."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ALLOWED_EVIDENCE_TYPES = ("observed", "derived", "synthetic", "illustrative")
CLAIM_REQUIRED_FIELDS = (
    "claim_id",
    "text",
    "evidence_type",
    "protocol_hash",
    "dataset_manifest_hash",
    "artifact_path",
    "metric_path",
    "status",
)
ALLOWED_STATUSES = ("draft", "verified", "rejected", "superseded")

_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


class ClaimV1(BaseModel):
    """One immutable public statement linked to an exact metric artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    text: str = Field(min_length=1)
    evidence_type: Literal["observed", "derived", "synthetic", "illustrative"]
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_path: str = Field(min_length=1)
    metric_path: str = Field(pattern=r"^/")
    status: Literal["draft", "verified", "rejected", "superseded"]


class ClaimsRegistryV1(BaseModel):
    """Repository-local copy of the approved cross-repository claim vocabulary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_evidence_types: tuple[str, ...]
    claim_required_fields: tuple[str, ...]
    allowed_statuses: tuple[str, ...]
    claims: tuple[ClaimV1, ...]

    @model_validator(mode="after")
    def validate_shared_vocabulary(self) -> Self:
        if self.allowed_evidence_types != ALLOWED_EVIDENCE_TYPES:
            raise ValueError("allowed_evidence_types must match the approved vocabulary")
        if self.claim_required_fields != CLAIM_REQUIRED_FIELDS:
            raise ValueError("claim_required_fields must match the approved vocabulary")
        if self.allowed_statuses != ALLOWED_STATUSES:
            raise ValueError("allowed_statuses must match the approved vocabulary")
        return self


def _load_registry(claims_path: Path) -> ClaimsRegistryV1:
    raw_value = yaml.safe_load(claims_path.read_text(encoding="utf-8"))
    return ClaimsRegistryV1.model_validate(raw_value)


def verified_claims(claims_path: Path) -> tuple[ClaimV1, ...]:
    """Return only claims permitted to reach a public README or report."""

    registry = _load_registry(claims_path)
    return tuple(claim for claim in registry.claims if claim.status == "verified")


def _resolve_json_pointer(document: object, pointer: str) -> object:
    current = document
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and (
            token == "0" or (not token.startswith("0") and token.isdecimal())
        ):
            index = int(token)
            if index >= len(current):
                raise LookupError(pointer)
            current = current[index]
        else:
            raise LookupError(pointer)
    return current


def text_numbers(text: str) -> tuple[Decimal, ...]:
    """Return every number a statement states, in order, with any percent sign dropped."""

    return tuple(
        Decimal(match.group().removesuffix("%")) for match in _NUMBER_PATTERN.finditer(text)
    )


def _json_numbers(value: object) -> set[Decimal]:
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float, Decimal)):
        return {Decimal(str(value))}
    if isinstance(value, dict):
        numbers: set[Decimal] = set()
        for child in value.values():
            numbers.update(_json_numbers(child))
        return numbers
    if isinstance(value, list):
        numbers = set()
        for child in value:
            numbers.update(_json_numbers(child))
        return numbers
    return set()


def metric_numbers(claim: ClaimV1, repository_root: Path) -> set[Decimal]:
    """Return every number at the claim's metric pointer.

    A missing artifact raises ``FileNotFoundError`` and a pointer that resolves to
    nothing raises ``LookupError``; neither may read as an empty set, because an
    empty set would make a statement with no backing look like one with none to
    check.
    """

    artifact: Any = json.loads(
        (repository_root / claim.artifact_path).read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    return _json_numbers(_resolve_json_pointer(artifact, claim.metric_path))


def audit_claims(claims_path: Path, repository_root: Path) -> tuple[str, ...]:
    """Report every claim whose exact evidence cannot be reproduced locally."""

    try:
        registry = _load_registry(claims_path)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
        return (f"claims registry is invalid: {exc}",)

    root = repository_root.resolve()
    violations: list[str] = []
    for claim in registry.claims:
        artifact_path = (root / claim.artifact_path).resolve()
        if not artifact_path.is_relative_to(root):
            violations.append(
                f"{claim.claim_id}: artifact path escapes repository: {claim.artifact_path}"
            )
            continue
        if not artifact_path.is_file():
            violations.append(f"{claim.claim_id}: artifact does not exist: {claim.artifact_path}")
            continue

        try:
            artifact: Any = json.loads(
                artifact_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeDecodeError, ValueError):
            violations.append(f"{claim.claim_id}: artifact is not valid UTF-8 JSON")
            continue

        if not isinstance(artifact, dict):
            violations.append(f"{claim.claim_id}: artifact root must be an object")
            continue
        if artifact.get("protocol_hash") != claim.protocol_hash:
            violations.append(f"{claim.claim_id}: protocol hash mismatch")
        if artifact.get("dataset_manifest_hash") != claim.dataset_manifest_hash:
            violations.append(f"{claim.claim_id}: dataset manifest hash mismatch")

        try:
            metric = _resolve_json_pointer(artifact, claim.metric_path)
        except LookupError:
            violations.append(
                f"{claim.claim_id}: metric JSON pointer does not exist: {claim.metric_path}"
            )
            continue

        metric_numbers = _json_numbers(metric)
        for number in text_numbers(claim.text):
            if number not in metric_numbers:
                violations.append(f"{claim.claim_id}: claim number is absent from metric: {number}")

    return tuple(violations)
