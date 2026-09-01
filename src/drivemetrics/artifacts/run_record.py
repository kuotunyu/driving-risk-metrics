"""Immutable provenance record for one driving-risk experiment run."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROVENANCE_ENV_VAR = "DRIVEMETRICS_RUN_PROVENANCE"


def _parse_utc_z(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"{field_name} must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return parsed


class RunRecordV1(BaseModel):
    """Frozen identity, environment, lifecycle, and artifact references for a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["driving-risk-run/v1"]
    run_id: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware: dict[str, str]
    seed: int
    started_at_utc: str
    finished_at_utc: str | None
    status: Literal["running", "succeeded", "failed", "aborted"]
    artifacts: dict[str, str]

    @field_validator("started_at_utc", "finished_at_utc")
    @classmethod
    def validate_utc_timestamp(cls, value: str | None, info: object) -> str | None:
        """Require every present lifecycle timestamp to use canonical UTC Z form."""

        if value is None:
            return None
        field_name = getattr(info, "field_name", "timestamp")
        _parse_utc_z(value, field_name)
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        """Keep running and terminal timestamps logically consistent."""

        if self.status == "running":
            if self.finished_at_utc is not None:
                raise ValueError("running run must not have finished_at_utc")
            return self

        if self.finished_at_utc is None:
            raise ValueError("terminal run requires finished_at_utc")
        if _parse_utc_z(self.finished_at_utc, "finished_at_utc") < _parse_utc_z(
            self.started_at_utc,
            "started_at_utc",
        ):
            raise ValueError("finished_at_utc must not precede started_at_utc")
        return self


class RunProvenance(BaseModel):
    """Environment facts a run cannot derive and must never invent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware: dict[str, str] = Field(min_length=1)


def load_run_provenance() -> RunProvenance:
    """Read the commit, lock hash, and hardware of the current run from the environment.

    Making this an explicit environment contract keeps the Colab hardware record
    a deliberate act rather than a guess, and fails closed when it is missing.
    """

    raw_value = os.environ.get(PROVENANCE_ENV_VAR)
    if raw_value is None:
        raise ValueError(
            f"{PROVENANCE_ENV_VAR} must supply the commit, lock hash, and hardware of this run"
        )
    try:
        document = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{PROVENANCE_ENV_VAR} must contain a JSON object") from error
    if not isinstance(document, dict):
        raise ValueError(f"{PROVENANCE_ENV_VAR} must contain a JSON object")
    return RunProvenance.model_validate(document)
