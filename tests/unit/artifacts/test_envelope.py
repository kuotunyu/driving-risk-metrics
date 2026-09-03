"""Behavior tests for the shared portfolio artifact envelope contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import ModuleType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "portfolio_artifact_envelope_v1.json"


JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**31), max_value=2**31 - 1)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5)
    ),
    max_leaves=15,
)


def load_envelope_module() -> ModuleType:
    """Import inside tests so a missing module is an intentional RED failure."""

    try:
        from drivemetrics.artifacts import envelope
    except ImportError:
        pytest.fail("drivemetrics.artifacts.envelope is missing", pytrace=False)
    return envelope


def load_schemas_module() -> ModuleType:
    try:
        from drivemetrics.artifacts import schemas
    except ImportError:
        pytest.fail("drivemetrics.artifacts.schemas is missing", pytrace=False)
    return schemas


def valid_envelope_values() -> dict[str, object]:
    """Return a complete hand-checked envelope without using production builders."""

    canonical_payload = '{"count":3,"label":"行人"}'.encode()
    return {
        "schema_version": "portfolio-artifact-envelope/v1",
        "producer_repository": "driving-risk-metrics",
        "producer_release": "v1.2.3",
        "producer_commit": "a" * 40,
        "artifact_type": "risk-summary/v1",
        "protocol_hash": "b" * 64,
        "dataset_manifest_hash": "c" * 64,
        "created_at_utc": "2026-08-31T00:00:00Z",
        "payload_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        "payload": {"label": "行人", "count": 3},
    }


def write_envelope(path: Path, values: dict[str, object]) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")


def test_canonical_json_bytes_sorts_keys_compacts_and_preserves_utf8() -> None:
    """Losing key order, compact separators, or UTF-8 changes every payload hash."""

    envelope = load_envelope_module()

    assert envelope.canonical_json_bytes({"z": "行人", "a": [2, 1]}) == (
        '{"a":[2,1],"z":"行人"}'.encode()
    )


def test_canonical_json_bytes_rejects_nonfinite_numbers() -> None:
    """JSON NaN has no portable canonical representation and must fail closed."""

    envelope = load_envelope_module()

    with pytest.raises(ValueError):
        envelope.canonical_json_bytes({"value": math.nan})


@given(JSON_VALUES)
@settings(deadline=None)
def test_canonical_json_bytes_round_trip_json_values(value: object) -> None:
    """Nested JSON payloads must survive canonical serialization without drift."""

    envelope = load_envelope_module()

    assert json.loads(envelope.canonical_json_bytes(value).decode("utf-8")) == value


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("producer_release", "1.2.3"),
        ("producer_commit", "A" * 40),
        ("producer_commit", "a" * 39),
        ("protocol_hash", "b" * 63),
        ("dataset_manifest_hash", "c" * 63),
        ("payload_sha256", "d" * 63),
    ],
)
def test_envelope_rejects_invalid_release_commit_or_hash(
    field: str,
    invalid_value: str,
) -> None:
    """Relaxing a provenance pattern would accept an ambiguous artifact identity."""

    envelope = load_envelope_module()
    values = valid_envelope_values()
    values[field] = invalid_value

    with pytest.raises(ValidationError):
        envelope.PortfolioArtifactEnvelopeV1.model_validate(values)


def test_envelope_rejects_extra_fields() -> None:
    """Silently ignored future fields could change meaning without changing validation."""

    envelope = load_envelope_module()
    values = valid_envelope_values()
    values["undocumented"] = True

    with pytest.raises(ValidationError):
        envelope.PortfolioArtifactEnvelopeV1.model_validate(values)


@pytest.mark.parametrize(
    "invalid_created_at",
    [
        "2026-08-31T00:00:00",
        "2026-08-31T08:00:00+08:00",
        "2026-08-31Z",
        "not-a-timestamp",
        "not-a-timestampZ",
    ],
)
def test_envelope_requires_parseable_utc_timestamp_ending_in_z(
    invalid_created_at: str,
) -> None:
    """A naive or offset timestamp would make cross-repository ordering ambiguous."""

    envelope = load_envelope_module()
    values = valid_envelope_values()
    values["created_at_utc"] = invalid_created_at

    with pytest.raises(ValidationError):
        envelope.PortfolioArtifactEnvelopeV1.model_validate(values)


def test_verify_envelope_returns_frozen_validated_model(tmp_path: Path) -> None:
    """The consumer boundary must validate type and payload before exposing fields."""

    envelope = load_envelope_module()
    path = tmp_path / "artifact.json"
    write_envelope(path, valid_envelope_values())

    verified = envelope.verify_envelope(path, "risk-summary/v1")

    assert verified.payload == {"label": "行人", "count": 3}
    with pytest.raises(ValidationError):
        verified.artifact_type = "changed/v1"


def test_verify_envelope_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    """Changing payload bytes without changing the hash must never pass validation."""

    envelope = load_envelope_module()
    values = valid_envelope_values()
    values["payload"] = {"label": "車輛", "count": 3}
    path = tmp_path / "artifact.json"
    write_envelope(path, values)

    with pytest.raises(ValueError, match=r"^payload SHA-256 mismatch"):
        envelope.verify_envelope(path, "risk-summary/v1")


def test_verify_envelope_rejects_unexpected_artifact_type(tmp_path: Path) -> None:
    """A valid envelope for another artifact type is not interchangeable evidence."""

    envelope = load_envelope_module()
    path = tmp_path / "artifact.json"
    write_envelope(path, valid_envelope_values())

    with pytest.raises(ValueError, match=r"^unexpected artifact type: expected"):
        envelope.verify_envelope(path, "calibration-summary/v1")


def test_checked_in_envelope_fixture_is_canonical_and_valid() -> None:
    """The future cross-repository conformance fixture must exercise the real reader."""

    envelope = load_envelope_module()
    raw_value = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert FIXTURE_PATH.read_bytes() == envelope.canonical_json_bytes(raw_value) + b"\n"
    verified = envelope.verify_envelope(FIXTURE_PATH, "portfolio-contract-fixture/v1")
    assert verified.payload == {"fixture": "synthetic", "values": [0, 1]}


def test_contract_schema_writer_is_deterministic(tmp_path: Path) -> None:
    """Regenerating schemas twice must produce the same exact model-derived bytes."""

    schemas = load_schemas_module()

    first_paths = schemas.write_contract_schemas(tmp_path)
    first_bytes = {path.name: path.read_bytes() for path in first_paths}
    second_paths = schemas.write_contract_schemas(tmp_path)

    assert tuple(path.name for path in first_paths) == (
        "portfolio_artifact_envelope_v1.json",
        "prediction_artifact_v1.json",
        "run_record_v1.json",
    )
    assert {path.name: path.read_bytes() for path in second_paths} == first_bytes


def test_checked_in_contract_schemas_match_generated_bytes() -> None:
    """A hand-edited JSON schema must fail against its Pydantic source of truth."""

    schemas = load_schemas_module()

    for filename, expected_bytes in schemas.contract_schema_documents().items():
        assert (REPO_ROOT / "schemas" / filename).read_bytes() == expected_bytes
