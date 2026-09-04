"""Behavior tests for the public claim registry and evidence audit."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from pydantic import ValidationError

EXPECTED_EVIDENCE_TYPES = ("observed", "derived", "synthetic", "illustrative")
EXPECTED_REQUIRED_FIELDS = (
    "claim_id",
    "text",
    "evidence_type",
    "protocol_hash",
    "dataset_manifest_hash",
    "artifact_path",
    "metric_path",
    "status",
)
EXPECTED_STATUSES = ("draft", "verified", "rejected", "superseded")
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_claims_module() -> ModuleType:
    """Import inside tests so a missing module is an intentional RED failure."""

    try:
        from drivemetrics.analysis import claims
    except ImportError:
        pytest.fail("drivemetrics.analysis.claims is missing", pytrace=False)
    return claims


def valid_claim_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "claim_id": "p1-miou-example",
        "text": "Measured mIoU is 0.8125.",
        "evidence_type": "observed",
        "protocol_hash": "a" * 64,
        "dataset_manifest_hash": "b" * 64,
        "artifact_path": "evidence/metrics.json",
        "metric_path": "/metrics/miou",
        "status": "verified",
    }
    values.update(overrides)
    return values


def valid_registry_values(claims: list[dict[str, object]]) -> dict[str, object]:
    return {
        "allowed_evidence_types": list(EXPECTED_EVIDENCE_TYPES),
        "claim_required_fields": list(EXPECTED_REQUIRED_FIELDS),
        "allowed_statuses": list(EXPECTED_STATUSES),
        "claims": claims,
    }


def write_claims(path: Path, claims: list[dict[str, object]]) -> None:
    path.write_text(
        yaml.safe_dump(valid_registry_values(claims), sort_keys=False),
        encoding="utf-8",
    )


def write_artifact(repository_root: Path, value: object) -> None:
    artifact = repository_root / "evidence" / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("evidence_type", EXPECTED_EVIDENCE_TYPES)
def test_claim_accepts_every_approved_evidence_type(evidence_type: str) -> None:
    """Dropping one shared evidence type would break cross-repository vocabulary."""

    claims = load_claims_module()
    claim = claims.ClaimV1.model_validate(valid_claim_values(evidence_type=evidence_type))

    assert claim.evidence_type == evidence_type


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("evidence_type", "estimated"),
        ("status", "published"),
        ("protocol_hash", "a" * 63),
        ("dataset_manifest_hash", "b" * 63),
        ("metric_path", "metrics/miou"),
    ],
)
def test_claim_rejects_unsupported_vocabulary_or_hash(
    field: str,
    invalid_value: str,
) -> None:
    """Unsupported vocabulary or malformed provenance must not enter the registry."""

    claims = load_claims_module()
    values = valid_claim_values()
    values[field] = invalid_value

    with pytest.raises(ValidationError, match=rf"\n{field}\n"):
        claims.ClaimV1.model_validate(values)


def test_claim_rejects_extra_fields() -> None:
    """An unversioned field must not silently change the public claim contract."""

    claims = load_claims_module()
    values = valid_claim_values(confidence="high")

    with pytest.raises(ValidationError):
        claims.ClaimV1.model_validate(values)


@pytest.mark.parametrize(
    ("vocabulary_field", "invalid_value"),
    [
        ("allowed_evidence_types", ["observed", "derived"]),
        ("claim_required_fields", ["claim_id", "text"]),
        ("allowed_statuses", ["draft", "verified"]),
    ],
)
def test_registry_requires_exact_shared_vocabulary(
    vocabulary_field: str,
    invalid_value: list[str],
) -> None:
    """A repository-specific vocabulary drift would make claim labels incomparable."""

    claims = load_claims_module()
    values = valid_registry_values([])
    values[vocabulary_field] = invalid_value

    expected = (
        rf"^1 validation error for ClaimsRegistryV1\n  Value error, {vocabulary_field} "
        r"must match the approved vocabulary"
    )
    with pytest.raises(ValidationError, match=expected):
        claims.ClaimsRegistryV1.model_validate(values)


def test_audit_claims_accepts_matching_artifact_hashes_pointer_and_number(
    tmp_path: Path,
) -> None:
    """A fully traceable verified number is the successful public-evidence path."""

    claims = load_claims_module()
    write_artifact(
        tmp_path,
        {
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "metrics": {"miou": 0.8125},
        },
    )
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values()])

    assert claims.audit_claims(claims_path, tmp_path) == ()


def test_audit_claims_reports_missing_artifact(tmp_path: Path) -> None:
    """A registry entry cannot be verified when its evidence file is absent."""

    claims = load_claims_module()
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values()])

    assert claims.audit_claims(claims_path, tmp_path) == (
        "p1-miou-example: artifact does not exist: evidence/metrics.json",
    )


def test_audit_claims_reports_invalid_registry(tmp_path: Path) -> None:
    """Malformed YAML must fail closed before any evidence is trusted."""

    claims = load_claims_module()
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text("claims: [", encoding="utf-8")

    violations = claims.audit_claims(claims_path, tmp_path)

    assert len(violations) == 1
    assert violations[0].startswith("claims registry is invalid:")


def test_audit_claims_rejects_artifact_path_outside_repository(tmp_path: Path) -> None:
    """Repository claims must not follow traversal paths into private workspace state."""

    claims = load_claims_module()
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values(artifact_path="../private.json")])

    assert claims.audit_claims(claims_path, tmp_path) == (
        "p1-miou-example: artifact path escapes repository: ../private.json",
    )


def test_audit_claims_reports_missing_metric_json_pointer(tmp_path: Path) -> None:
    """A real artifact is insufficient when it does not contain the cited metric."""

    claims = load_claims_module()
    write_artifact(
        tmp_path,
        {
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "metrics": {},
        },
    )
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values()])

    assert claims.audit_claims(claims_path, tmp_path) == (
        "p1-miou-example: metric JSON pointer does not exist: /metrics/miou",
    )


@pytest.mark.parametrize(
    ("artifact_text", "message"),
    [
        ("not JSON", "artifact is not valid UTF-8 JSON"),
        ("[]", "artifact root must be an object"),
    ],
)
def test_audit_claims_rejects_invalid_artifact_root(
    tmp_path: Path,
    artifact_text: str,
    message: str,
) -> None:
    """Evidence must be valid JSON with an object root before hash lookup."""

    claims = load_claims_module()
    artifact = tmp_path / "evidence" / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(artifact_text, encoding="utf-8")
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values()])

    assert claims.audit_claims(claims_path, tmp_path) == (f"p1-miou-example: {message}",)


def test_audit_claims_rejects_nonfinite_json_number(tmp_path: Path) -> None:
    """Non-standard JSON NaN must not become reproducible public evidence."""

    claims = load_claims_module()
    artifact = tmp_path / "evidence" / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        '{"protocol_hash":"'
        + "a" * 64
        + '","dataset_manifest_hash":"'
        + "b" * 64
        + '","metrics":{"miou":NaN}}',
        encoding="utf-8",
    )
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values()])

    assert claims.audit_claims(claims_path, tmp_path) == (
        "p1-miou-example: artifact is not valid UTF-8 JSON",
    )


def test_audit_claims_resolves_array_and_escaped_json_pointer_tokens(tmp_path: Path) -> None:
    """RFC 6901 array indices and escaped object keys must resolve to real numbers."""

    claims = load_claims_module()
    write_artifact(
        tmp_path,
        {
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "metrics": {
                "series": [True, None, "skip", {"score": 0.8125}],
                "a/b": {"~value": 2},
            },
        },
    )
    claims_path = tmp_path / "claims.yaml"
    write_claims(
        claims_path,
        [
            valid_claim_values(metric_path="/metrics/series", text="Score is 0.8125."),
            valid_claim_values(
                claim_id="p1-escaped-pointer",
                metric_path="/metrics/a~1b/~0value",
                text="Count is 2.",
            ),
            valid_claim_values(
                claim_id="p1-zero-index",
                metric_path="/metrics/series/0",
                text="Boolean diagnostic is recorded.",
            ),
        ],
    )

    assert claims.audit_claims(claims_path, tmp_path) == ()


@pytest.mark.parametrize(
    "invalid_pointer",
    ["/metrics/series/2", "/metrics/series/01", "/metrics/series/not-an-index"],
)
def test_audit_claims_rejects_invalid_array_json_pointer(
    tmp_path: Path,
    invalid_pointer: str,
) -> None:
    """Out-of-range and leading-zero array indices are not valid evidence pointers."""

    claims = load_claims_module()
    write_artifact(
        tmp_path,
        {
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "metrics": {"series": [0.0, 0.8125]},
        },
    )
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values(metric_path=invalid_pointer)])

    assert claims.audit_claims(claims_path, tmp_path) == (
        f"p1-miou-example: metric JSON pointer does not exist: {invalid_pointer}",
    )


@pytest.mark.parametrize(
    ("artifact_field", "message"),
    [
        ("protocol_hash", "protocol hash mismatch"),
        ("dataset_manifest_hash", "dataset manifest hash mismatch"),
    ],
)
def test_audit_claims_reports_artifact_hash_mismatch(
    tmp_path: Path,
    artifact_field: str,
    message: str,
) -> None:
    """A metric from another protocol or cohort cannot support the claim."""

    claims = load_claims_module()
    artifact = {
        "protocol_hash": "a" * 64,
        "dataset_manifest_hash": "b" * 64,
        "metrics": {"miou": 0.8125},
    }
    artifact[artifact_field] = "f" * 64
    write_artifact(tmp_path, artifact)
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values()])

    assert claims.audit_claims(claims_path, tmp_path) == (f"p1-miou-example: {message}",)


def test_audit_claims_rejects_number_absent_from_referenced_metric(tmp_path: Path) -> None:
    """A prose number must be present in the exact cited metric value."""

    claims = load_claims_module()
    write_artifact(
        tmp_path,
        {
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "metrics": {"miou": 0.75},
        },
    )
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, [valid_claim_values()])

    assert claims.audit_claims(claims_path, tmp_path) == (
        "p1-miou-example: claim number is absent from metric: 0.8125",
    )


def test_verified_claims_excludes_every_unverified_status(tmp_path: Path) -> None:
    """Draft, rejected, and superseded prose must never reach public rendering."""

    claims = load_claims_module()
    entries = [valid_claim_values()]
    for status in ("draft", "rejected", "superseded"):
        entries.append(
            valid_claim_values(
                claim_id=f"p1-{status}",
                status=status,
            )
        )
    claims_path = tmp_path / "claims.yaml"
    write_claims(claims_path, entries)

    verified = claims.verified_claims(claims_path)

    assert tuple(claim.claim_id for claim in verified) == ("p1-miou-example",)


def test_checked_in_claim_registry_has_exact_vocabulary_and_no_claims() -> None:
    """P1-03 must establish vocabulary without publishing invented results."""

    claims = load_claims_module()
    claims_path = REPO_ROOT / "docs" / "claims.yaml"
    raw_value = yaml.safe_load(claims_path.read_text(encoding="utf-8"))

    assert raw_value == valid_registry_values([])
    assert claims.verified_claims(claims_path) == ()


def test_the_audit_examines_every_claim_after_the_first_violation(tmp_path: Path) -> None:
    """One bad claim must not stop the audit, and this is the reason it matters.

    The auditor exists to stop a published number that does not match its
    artifact. If the loop broke on the first problem instead of continuing,
    a registry whose first claim had a path error and whose second claim
    carried a WRONG NUMBER would report only the path error, and the wrong
    number would ship. Every violation the audit can see, it must report.

    Five claims below trigger the five different skip paths in turn, so any one
    of them turning into an early exit loses at least one violation.
    """

    claims = load_claims_module()
    claims_path = tmp_path / "claims.yaml"

    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "notjson.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "evidence" / "notobject.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "evidence" / "metrics.json").write_text(
        json.dumps({"metrics": {"miou": 0.8125}}), encoding="utf-8"
    )

    write_claims(
        claims_path,
        [
            valid_claim_values(claim_id="c1-escapes", artifact_path="../private.json"),
            valid_claim_values(claim_id="c2-absent", artifact_path="evidence/gone.json"),
            valid_claim_values(claim_id="c3-notjson", artifact_path="evidence/notjson.json"),
            valid_claim_values(claim_id="c4-notobject", artifact_path="evidence/notobject.json"),
            valid_claim_values(claim_id="c5-nopointer", metric_path="/metrics/absent"),
            # Last on purpose, and the most important one: a claim whose stated
            # number is not the number in its artifact. If any earlier skip
            # became an early exit, THIS is the violation that would be lost,
            # and a wrong published figure is exactly what the audit exists to
            # stop.
            valid_claim_values(
                claim_id="c6-wrongnumber",
                text="Measured mIoU is 0.9500.",
            ),
        ],
    )

    violations = claims.audit_claims(claims_path, tmp_path)
    reported = {violation.split(":")[0] for violation in violations}

    assert reported == {
        "c1-escapes",
        "c2-absent",
        "c3-notjson",
        "c4-notobject",
        "c5-nopointer",
        "c6-wrongnumber",
    }
