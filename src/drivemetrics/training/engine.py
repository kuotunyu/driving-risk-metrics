"""Deterministic, backend-injected training orchestration for the locked protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from drivemetrics.artifacts.envelope import canonical_json_bytes
from drivemetrics.artifacts.run_record import (
    RunRecordV1,
    load_run_provenance,
)
from drivemetrics.data.manifest import load_manifest
from drivemetrics.models.registry import ModelName
from drivemetrics.protocol.config import load_protocol
from drivemetrics.training.losses import cross_entropy_spec
from drivemetrics.training.schedule import optimizer_spec, polynomial_learning_rate

APPROVED_SEEDS: tuple[int, ...] = (17, 42, 73)
CHECKPOINT_FILENAME = "final_checkpoint.pt"
RUN_RECORD_FILENAME = "run_record.json"

# Called once per optimizer step with (step, total_steps, mean micro-batch loss).
# Purely observational: the engine never reads anything back from it, so a run
# with an observer and a run without one produce byte-identical checkpoints.
StepObserver = Callable[[int, int, float], None]


class TrainingBackend(Protocol):
    """The framework boundary; no gradient or tensor code lives in this engine."""

    def seed_all(self, seed: int) -> None:
        """Seed every framework random source before any state is created."""

    def create_training_state(
        self,
        model_name: ModelName,
        optimizer: Mapping[str, float | str],
    ) -> object:
        """Build the model and optimizer described by the approved specification."""

    def run_step(
        self,
        state: object,
        batch: object,
        learning_rate: float,
        *,
        apply_update: bool,
    ) -> float:
        """Run one micro batch at the supplied learning rate and return its loss.

        ``batch`` is a tuple of ``(sample_id, flip_draw)`` pairs. Every random
        choice, including the augmentation draw, is made by the engine, so a
        rerun never depends on backend random state. ``apply_update`` is true
        only on the last micro batch of an optimizer step, which is the signal to
        apply the accumulated gradients exactly once per locked effective batch.
        """

    def save_checkpoint(
        self,
        state: object,
        path: Path,
        metadata: Mapping[str, object],
    ) -> str:
        """Persist the final state with its metadata and return the payload SHA-256."""

    def load_checkpoint(self, path: Path, expected_metadata: Mapping[str, object]) -> object:
        """Load a checkpoint, failing closed when its metadata does not match."""


@dataclass(frozen=True)
class TrainingResult:
    """Where one completed run stored its only checkpoint and its provenance."""

    final_step: int
    checkpoint_path: Path
    checkpoint_sha256: str
    run_record_path: Path


class TrainingRunConfigV1(BaseModel):
    """One (protocol, architecture, micro batch) selection for a single run."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        protected_namespaces=(),
    )

    schema_version: Literal["drivemetrics-training-run/v1"]
    protocol_path: str = Field(min_length=1)
    model: ModelName
    micro_batch_size: int

    @field_validator("protocol_path")
    @classmethod
    def validate_relative_protocol_path(cls, value: str) -> str:
        """Keep the protocol beside its run configuration and inside the repository."""

        path = PurePosixPath(value)
        has_drive_prefix = bool(path.parts and path.parts[0].endswith(":"))
        has_noncanonical_segment = any(part in {"", ".", ".."} for part in value.split("/"))
        if "\\" in value or path.is_absolute() or has_drive_prefix or has_noncanonical_segment:
            raise ValueError("protocol_path must be a safe relative POSIX path")
        return value


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_run_config(path: Path) -> tuple[TrainingRunConfigV1, str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("training run configuration must be a mapping")
    config = TrainingRunConfigV1.model_validate(document)
    digest = hashlib.sha256(canonical_json_bytes(config.model_dump(mode="json"))).hexdigest()
    return config, digest


def _sampling_plan(
    sample_ids: tuple[str, ...],
    seed: int,
    micro_batch_size: int,
    count: int,
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Return the whole deterministic micro-batch plan for one run.

    Each entry pairs a sample ID with its augmentation draw, so the data order
    and every random choice of a run are one value that depends only on the
    manifest, the seed, and the micro batch size. Each epoch is a fresh
    permutation, so no sample is dropped or repeated inside one pass.
    """

    generator = np.random.default_rng(seed)
    plan: list[tuple[tuple[str, float], ...]] = []
    pool: list[str] = []
    for _ in range(count):
        while len(pool) < micro_batch_size:
            order = generator.permutation(len(sample_ids))
            pool.extend(sample_ids[int(index)] for index in order)
        draws = generator.random(micro_batch_size)
        plan.append(
            tuple((pool[position], float(draws[position])) for position in range(micro_batch_size))
        )
        del pool[:micro_batch_size]
    return tuple(plan)


def _write_run_record(path: Path, document: dict[str, Any]) -> None:
    record = RunRecordV1.model_validate(document)
    path.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train(
    config_path: Path,
    manifest_path: Path,
    output_dir: Path,
    seed: int,
    *,
    backend: TrainingBackend,
    on_step: StepObserver | None = None,
) -> TrainingResult:
    """Run one locked training job and return its single final-step checkpoint.

    The engine owns the deterministic parts of the protocol: seeding order,
    epoch-shuffled micro batches, gradient accumulation up to the locked
    effective batch size, the learning-rate schedule, final-step-only
    checkpointing, and the run record. Everything framework specific happens
    behind ``backend``, so this orchestration is fully testable on CPU without a
    training framework. A failed run still writes a ``failed`` run record before
    the original error propagates, so a partial job never looks finished.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in APPROVED_SEEDS:
        raise ValueError(f"seed must be one of the approved seeds {APPROVED_SEEDS}")
    provenance = load_run_provenance()
    run_config, config_sha256 = load_run_config(config_path)
    loaded_protocol = load_protocol(config_path.parent / run_config.protocol_path)
    protocol = loaded_protocol.protocol

    effective_batch_size = protocol.training.effective_batch_size
    if run_config.micro_batch_size < 1 or effective_batch_size % run_config.micro_batch_size:
        raise ValueError("micro_batch_size must be a positive divisor of the effective batch size")

    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = protocol.training.steps
    accumulation_steps = effective_batch_size // run_config.micro_batch_size
    optimizer = optimizer_spec(run_config.model)
    base_lr = float(optimizer["learning_rate"])
    run_record_path = output_dir / RUN_RECORD_FILENAME
    identity: dict[str, Any] = {
        "schema_version": "driving-risk-run/v1",
        "run_id": f"{run_config.model}-seed-{seed}",
        "commit": provenance.commit,
        "config_sha256": config_sha256,
        "protocol_sha256": loaded_protocol.protocol_sha256,
        "dataset_manifest_sha256": manifest.manifest_sha256,
        "lock_sha256": provenance.lock_sha256,
        "hardware": dict(provenance.hardware),
        "seed": seed,
        "started_at_utc": _utc_now(),
    }

    try:
        backend.seed_all(seed)
        state = backend.create_training_state(run_config.model, optimizer)
        plan = _sampling_plan(
            manifest.sample_ids,
            seed,
            run_config.micro_batch_size,
            total_steps * accumulation_steps,
        )
        for step in range(1, total_steps + 1):
            learning_rate = polynomial_learning_rate(
                step,
                base_lr=base_lr,
                warmup_steps=protocol.training.warmup_steps,
                total_steps=total_steps,
            )
            start = (step - 1) * accumulation_steps
            window = plan[start : start + accumulation_steps]
            losses: list[float] = []
            for position, batch in enumerate(window, start=1):
                losses.append(
                    backend.run_step(
                        state,
                        batch,
                        learning_rate,
                        apply_update=position == accumulation_steps,
                    )
                )
            if on_step is not None:
                on_step(step, total_steps, sum(losses) / len(losses))

        checkpoint_path = output_dir / CHECKPOINT_FILENAME
        checkpoint_sha256 = backend.save_checkpoint(
            state,
            checkpoint_path,
            {
                "run_id": identity["run_id"],
                "model": run_config.model,
                "seed": seed,
                "final_step": total_steps,
                "config_sha256": config_sha256,
                "protocol_sha256": loaded_protocol.protocol_sha256,
                "dataset_manifest_sha256": manifest.manifest_sha256,
                "loss": cross_entropy_spec(),
            },
        )
    except Exception:
        _write_run_record(
            run_record_path,
            {**identity, "finished_at_utc": _utc_now(), "status": "failed", "artifacts": {}},
        )
        raise

    _write_run_record(
        run_record_path,
        {
            **identity,
            "finished_at_utc": _utc_now(),
            "status": "succeeded",
            "artifacts": {"final_checkpoint": checkpoint_sha256},
        },
    )
    return TrainingResult(
        final_step=total_steps,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_record_path=run_record_path,
    )
