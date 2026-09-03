"""Assemble the immutable index of the nine formal runs from their run directories.

The index is what the formal-set gate validates and what aggregation consumes.
It is built from the run records and artifacts the pipeline itself wrote, never
typed by hand, because a hand-written index is exactly where a wrong seed, a
stale checkpoint hash, or a temperature fitted for different weights slips into
the published numbers. Every binding is re-checked here rather than trusted.

Directory convention, relative to ``runs_root``::

    <model>/seed-<seed>/train/run_record.json
    <model>/seed-<seed>/calibration/temperature.json
    <model>/seed-<seed>/eval/run_record.json
    <model>/seed-<seed>/eval_calibrated/run_record.json

The training engine writes ``final_checkpoint`` and ``status: succeeded`` only
after the last protocol step, so a succeeded record with that artifact is the
final-step checkpoint by construction.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from drivemetrics.artifacts.formal_set import (
    APPROVED_MODELS,
    APPROVED_SEEDS,
    SCHEMA_VERSION,
    validate_formal_run_index,
)
from drivemetrics.artifacts.run_record import RunRecordV1

# The manifest, protocol and calibration modules are imported inside the builder
# rather than here. This package is reached through `protocol.config`, which the
# manifest module itself imports, so a top-level import here would close a cycle
# and leave `drivemetrics.data.manifest` half-initialized for whoever imports it
# first. The pure schema in `formal_set` and the run-record model have no such edge.

INDEX_FILENAME = "formal_run_index.json"
LOCKED_SPLIT = "locked_validation"


@dataclass(frozen=True)
class FormalIndexResult:
    """Where the index was written and what it binds."""

    index_path: Path
    run_count: int
    protocol_sha256: str
    dataset_manifest_sha256: str


def _read_object(path: Path, model: str, seed: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path.name} for {model} seed-{seed}: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return document


def _read_run_record(path: Path, model: str, seed: int) -> RunRecordV1:
    try:
        return RunRecordV1.model_validate(_read_object(path, model, seed))
    except ValidationError as error:
        raise ValueError(f"{path} is not a valid run record: {error}") from error


def _check_record(
    record: RunRecordV1,
    label: str,
    *,
    protocol_sha256: str,
    seed: int,
    dataset_manifest_sha256: str | None,
) -> None:
    if record.status != "succeeded":
        raise ValueError(f"{label} status must be succeeded, got {record.status!r}")
    if record.protocol_sha256 != protocol_sha256:
        raise ValueError(f"{label} was produced under a different protocol hash")
    if record.seed != seed:
        raise ValueError(f"{label} carries seed {record.seed}, expected {seed}")
    if dataset_manifest_sha256 is not None and (
        record.dataset_manifest_sha256 != dataset_manifest_sha256
    ):
        raise ValueError(f"{label} was scored on a different dataset manifest")


def _validated_critical_ids(critical_class_ids: Sequence[int]) -> tuple[int, ...]:
    from drivemetrics.data.bdd100k import NUM_TRAIN_CLASSES

    ids = tuple(critical_class_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("critical_class_ids must be non-empty and unique")
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("critical_class_ids must be integers")
        if not 0 <= value < NUM_TRAIN_CLASSES:
            raise ValueError(f"critical class {value} is outside 0..{NUM_TRAIN_CLASSES - 1}")
    return ids


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def build_formal_run_index(
    runs_root: Path,
    config_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    critical_class_ids: Sequence[int],
) -> FormalIndexResult:
    """Build, gate, and write the index of every approved (model, seed) run.

    Refuses to overwrite an existing index, because analyses and claims cite it
    by content; refuses any cohort other than the locked validation split; and
    runs the assembled document through the formal-set gate before writing, so
    the builder and the gate can never disagree about what a complete matrix is.
    """

    if output_path.exists():
        raise FileExistsError(f"{INDEX_FILENAME} already exists: {output_path}")

    from drivemetrics.calibration.service import TEMPERATURE_SCHEMA_VERSION
    from drivemetrics.data.bdd100k import NUM_TRAIN_CLASSES
    from drivemetrics.data.manifest import load_manifest
    from drivemetrics.protocol.config import load_protocol

    loaded = load_protocol(config_path)
    protocol_sha256 = loaded.protocol_sha256
    expected_steps = loaded.protocol.training.steps
    manifest = load_manifest(manifest_path)
    if manifest.split_name != LOCKED_SPLIT:
        raise ValueError(
            f"the index must be built against the {LOCKED_SPLIT} cohort, "
            f"got {manifest.split_name!r}"
        )
    locked_sha256 = manifest.manifest_sha256
    critical = _validated_critical_ids(critical_class_ids)
    base = output_path.parent

    runs: list[dict[str, Any]] = []
    for model in APPROVED_MODELS:
        for seed in APPROVED_SEEDS:
            run_dir = runs_root / model / f"seed-{seed}"
            run_id = f"{model}-seed-{seed}"

            train = _read_run_record(run_dir / "train" / "run_record.json", model, seed)
            _check_record(
                train,
                f"training run {run_id}",
                protocol_sha256=protocol_sha256,
                seed=seed,
                dataset_manifest_sha256=None,
            )
            if train.run_id != run_id:
                raise ValueError(f"training run_id {train.run_id!r} does not match {run_id!r}")
            checkpoint_sha256 = train.artifacts.get("final_checkpoint")
            if checkpoint_sha256 is None:
                raise ValueError(f"training run {run_id} recorded no final_checkpoint artifact")

            temperature_doc = _read_object(
                run_dir / "calibration" / "temperature.json", model, seed
            )
            if temperature_doc.get("schema_version") != TEMPERATURE_SCHEMA_VERSION:
                raise ValueError(f"temperature for {run_id} has the wrong schema_version")
            if temperature_doc.get("protocol_sha256") != protocol_sha256:
                raise ValueError(f"temperature for {run_id} was fitted under another protocol")
            if temperature_doc.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(f"temperature for {run_id} was fitted for a different checkpoint")
            temperature = float(temperature_doc["temperature"])
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(f"temperature for {run_id} must be finite and positive")

            eval_raw = _read_run_record(run_dir / "eval" / "run_record.json", model, seed)
            eval_cal = _read_run_record(
                run_dir / "eval_calibrated" / "run_record.json", model, seed
            )
            for record, label in ((eval_raw, "uncalibrated"), (eval_cal, "calibrated")):
                _check_record(
                    record,
                    f"{label} evaluation of {run_id}",
                    protocol_sha256=protocol_sha256,
                    seed=seed,
                    dataset_manifest_sha256=locked_sha256,
                )
            raw_ids = list(eval_raw.artifacts)
            cal_ids = list(eval_cal.artifacts)
            if set(raw_ids) != set(cal_ids):
                raise ValueError(
                    f"calibrated and uncalibrated evaluations of {run_id} must cover the "
                    "same images"
                )

            runs.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_id": run_id,
                    "protocol_sha256": protocol_sha256,
                    "dataset_manifest_sha256": locked_sha256,
                    "checkpoint_sha256": checkpoint_sha256,
                    "final_step": expected_steps,
                    "status": "succeeded",
                    "temperature": temperature,
                    "artifacts_dir": _relative(run_dir / "eval", base),
                    "calibrated_artifacts_dir": _relative(run_dir / "eval_calibrated", base),
                    "uncalibrated_sample_ids": raw_ids,
                    "calibrated_sample_ids": cal_ids,
                }
            )

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "dataset_manifest_sha256": locked_sha256,
        "expected_steps": expected_steps,
        "cohort": LOCKED_SPLIT,
        "num_classes": NUM_TRAIN_CLASSES,
        "critical_class_ids": list(critical),
        "runs": runs,
    }
    violations = validate_formal_run_index(document)
    if violations:
        raise ValueError("assembled index failed its own gate: " + "; ".join(violations))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return FormalIndexResult(
        index_path=output_path,
        run_count=len(runs),
        protocol_sha256=protocol_sha256,
        dataset_manifest_sha256=locked_sha256,
    )
