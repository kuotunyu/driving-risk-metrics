"""Behavior tests for immutable experiment run records."""

from __future__ import annotations

from types import ModuleType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError


def load_run_record_module() -> ModuleType:
    """Import inside tests so a missing module is an intentional RED failure."""

    try:
        from drivemetrics.artifacts import run_record
    except ImportError:
        pytest.fail("drivemetrics.artifacts.run_record is missing", pytrace=False)
    return run_record


def valid_run_values() -> dict[str, object]:
    return {
        "schema_version": "driving-risk-run/v1",
        "run_id": "segformer-b0-seed17",
        "commit": "a" * 40,
        "config_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "dataset_manifest_sha256": "d" * 64,
        "lock_sha256": "e" * 64,
        "hardware": {"accelerator": "Colab A100", "runtime": "Python 3.11"},
        "seed": 17,
        "started_at_utc": "2026-08-31T01:00:00Z",
        "finished_at_utc": "2026-08-31T02:00:00Z",
        "status": "succeeded",
        "artifacts": {"metrics": "metrics.json"},
    }


def test_run_record_accepts_complete_terminal_run_and_is_frozen() -> None:
    """A valid terminal record must retain exact immutable provenance."""

    run_record = load_run_record_module()
    record = run_record.RunRecordV1.model_validate(valid_run_values())

    assert record.seed == 17
    assert record.finished_at_utc == "2026-08-31T02:00:00Z"
    with pytest.raises(ValidationError):
        record.status = "failed"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema_version", "driving-risk-run/v2"),
        ("run_id", ""),
        ("commit", "A" * 40),
        ("commit", "a" * 39),
        ("config_sha256", "b" * 63),
        ("protocol_sha256", "c" * 63),
        ("dataset_manifest_sha256", "d" * 63),
        ("lock_sha256", "e" * 63),
    ],
)
def test_run_record_rejects_invalid_identity_or_hash(
    field: str,
    invalid_value: str,
) -> None:
    """Relaxed provenance fields would make two runs impossible to distinguish safely."""

    run_record = load_run_record_module()
    values = valid_run_values()
    values[field] = invalid_value

    with pytest.raises(ValidationError, match=rf"\n{field}\n"):
        run_record.RunRecordV1.model_validate(values)


def test_run_record_rejects_extra_fields() -> None:
    """Unexpected fields must not silently alter the meaning of a versioned record."""

    run_record = load_run_record_module()
    values = valid_run_values()
    values["best_validation_score"] = 0.99

    with pytest.raises(ValidationError):
        run_record.RunRecordV1.model_validate(values)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("started_at_utc", "2026-08-31T01:00:00"),
        ("started_at_utc", "2026-08-31T09:00:00+08:00"),
        ("started_at_utc", "2026-08-31Z"),
        ("finished_at_utc", "not-a-timestampZ"),
    ],
)
def test_run_record_requires_parseable_utc_timestamps_ending_in_z(
    field: str,
    invalid_value: str,
) -> None:
    """Naive, offset, or malformed timestamps make duration comparisons unreliable."""

    run_record = load_run_record_module()
    values = valid_run_values()
    values[field] = invalid_value

    with pytest.raises(ValidationError, match=rf"\n{field}\n"):
        run_record.RunRecordV1.model_validate(values)


def test_running_record_requires_unfinished_timestamp() -> None:
    """A running record with a finish time contradicts its state."""

    run_record = load_run_record_module()
    values = valid_run_values()
    values["status"] = "running"

    with pytest.raises(
        ValidationError,
        match=r"^1 validation error for RunRecordV1\n  Value error, running run must not have finished_at_utc",
    ):
        run_record.RunRecordV1.model_validate(values)

    values["finished_at_utc"] = None
    record = run_record.RunRecordV1.model_validate(values)
    assert record.status == "running"


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "aborted"])
def test_terminal_record_requires_finished_timestamp(terminal_status: str) -> None:
    """Every terminal outcome needs a finish time for reproducible duration accounting."""

    run_record = load_run_record_module()
    values = valid_run_values()
    values["status"] = terminal_status
    values["finished_at_utc"] = None

    with pytest.raises(
        ValidationError,
        match=r"^1 validation error for RunRecordV1\n  Value error, terminal run requires finished_at_utc",
    ):
        run_record.RunRecordV1.model_validate(values)


def test_run_record_rejects_finish_before_start() -> None:
    """A negative run duration indicates corrupt or mismatched provenance."""

    run_record = load_run_record_module()
    values = valid_run_values()
    values["finished_at_utc"] = "2026-08-31T00:59:59Z"

    with pytest.raises(
        ValidationError,
        match=r"^1 validation error for RunRecordV1\n  Value error, finished_at_utc must not precede started_at_utc",
    ):
        run_record.RunRecordV1.model_validate(values)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(deadline=None)
def test_run_record_model_dump_round_trip(seed: int) -> None:
    """Serialization must preserve every validated run-record field exactly."""

    run_record = load_run_record_module()
    values = valid_run_values()
    values["seed"] = seed
    original = run_record.RunRecordV1.model_validate(values)

    assert run_record.RunRecordV1.model_validate(original.model_dump()) == original
