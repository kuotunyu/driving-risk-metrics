"""Validate the immutable index of the nine formal runs before it is trusted.

This gate runs before any paid GPU job and again before analysis. Every failure
it names produces a result set that looks complete: a missing seed narrows every
interval, a duplicate double-counts, a drifted cohort hash compares two studies
as if they were one, a mid-training checkpoint is selection on the evaluation
cohort, and a missing calibrated artifact turns a calibration comparison into a
comparison against nothing.

Every violation is collected rather than raised at the first, because fixing
nine paid runs one defect at a time is not affordable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "drivemetrics-formal-set/v1"
APPROVED_MODELS: tuple[str, ...] = (
    "segformer_b2",
    "upernet_convnextv2_tiny",
    "upernet_dinov2_small",
)
APPROVED_SEEDS: tuple[int, ...] = (17, 42, 73)
REQUIRED_STATUS = "succeeded"


def _run_label(entry: Mapping[str, Any], position: int) -> str:
    model = entry.get("model", "<no model>")
    seed = entry.get("seed", "<no seed>")
    return f"run {position} ({model}, seed {seed})"


def _check_entry(
    entry: Mapping[str, Any],
    position: int,
    *,
    protocol_sha256: str,
    dataset_manifest_sha256: str,
    expected_steps: int,
) -> list[str]:
    label = _run_label(entry, position)
    problems: list[str] = []

    if entry.get("model") not in APPROVED_MODELS:
        problems.append(f"{label}: model is not one of {APPROVED_MODELS}")
    if entry.get("seed") not in APPROVED_SEEDS:
        problems.append(f"{label}: seed is not one of the approved seeds {APPROVED_SEEDS}")
    if entry.get("protocol_sha256") != protocol_sha256:
        problems.append(f"{label}: protocol hash differs from the index protocol hash")
    if entry.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        problems.append(f"{label}: dataset manifest hash differs from the index manifest hash")
    if entry.get("final_step") != expected_steps:
        problems.append(
            f"{label}: checkpoint is at step {entry.get('final_step')}, "
            f"not the final step {expected_steps}"
        )
    if entry.get("status") != REQUIRED_STATUS:
        problems.append(f"{label}: status is {entry.get('status')!r}, not {REQUIRED_STATUS!r}")

    temperature = entry.get("temperature")
    if not isinstance(temperature, int | float) or isinstance(temperature, bool):
        problems.append(f"{label}: temperature is missing or not a number")
    elif temperature <= 0.0:
        problems.append(f"{label}: temperature must be positive")

    uncalibrated = tuple(entry.get("uncalibrated_sample_ids") or ())
    calibrated = tuple(entry.get("calibrated_sample_ids") or ())
    if not uncalibrated:
        problems.append(f"{label}: no uncalibrated prediction artifacts")
    if not calibrated:
        problems.append(f"{label}: no calibrated prediction artifacts")
    for name, ids in (("uncalibrated", uncalibrated), ("calibrated", calibrated)):
        if len(set(ids)) != len(ids):
            problems.append(f"{label}: duplicate sample ID in the {name} artifacts")
    if uncalibrated and calibrated and set(uncalibrated) != set(calibrated):
        problems.append(
            f"{label}: calibrated and uncalibrated artifacts differ; "
            "they must cover the same images"
        )
    return problems


def validate_formal_run_index(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every reason this run index may not be trusted, or an empty tuple."""

    problems: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {SCHEMA_VERSION!r}, got {document.get('schema_version')!r}"
        )
        return tuple(problems)

    runs = document.get("runs")
    if not isinstance(runs, Sequence) or not runs:
        return (*problems, "runs must be a non-empty list")

    protocol_sha256 = str(document.get("protocol_sha256"))
    dataset_manifest_sha256 = str(document.get("dataset_manifest_sha256"))
    raw_steps = document.get("expected_steps")
    if not isinstance(raw_steps, int) or isinstance(raw_steps, bool):
        return (*problems, "expected_steps must be an integer")
    expected_steps = raw_steps

    seen: set[tuple[Any, Any]] = set()
    for position, entry in enumerate(runs):
        problems.extend(
            _check_entry(
                entry,
                position,
                protocol_sha256=protocol_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                expected_steps=expected_steps,
            )
        )
        key = (entry.get("model"), entry.get("seed"))
        if key in seen:
            problems.append(f"duplicate run for model {key[0]!r} seed {key[1]!r}")
        seen.add(key)

    for model in APPROVED_MODELS:
        for seed in APPROVED_SEEDS:
            if (model, seed) not in seen:
                problems.append(f"missing run for model {model!r} seed {seed}")

    cohorts = {
        frozenset(entry.get("uncalibrated_sample_ids") or ())
        for entry in runs
        if entry.get("uncalibrated_sample_ids")
    }
    if len(cohorts) > 1:
        problems.append(
            "runs do not share one cohort; the paired bootstrap requires every run "
            "to sit on the identical image axis"
        )
    return tuple(problems)
