"""Locked-cohort evaluation that publishes metric-sufficient prediction artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from drivemetrics.artifacts.predictions import (
    PredictionRecord,
    write_prediction_artifact,
)
from drivemetrics.artifacts.run_record import RunRecordV1, load_run_provenance
from drivemetrics.calibration.service import TEMPERATURE_SCHEMA_VERSION
from drivemetrics.calibration.temperature import apply_temperature, softmax_probabilities
from drivemetrics.data.manifest import DatasetManifest, load_manifest
from drivemetrics.data.transforms import (
    CANVAS_WIDTH,
    MASK_PAD_VALUE,
    prepare_sample,
    restore_index_map,
)
from drivemetrics.metrics.calibration import (
    classwise_ece_sufficient_statistics,
    multiclass_brier_sums,
    pack_correctness,
    quantize_confidence,
)
from drivemetrics.metrics.confusion import compute_confusion
from drivemetrics.models.adapters import SegmentationModel
from drivemetrics.protocol.config import load_protocol, split_paths
from drivemetrics.protocol.hashing import sha256_file

# Called once per scored image with (completed, total). Purely observational: the
# engine reads nothing back, so a run with an observer and a run without one
# produce identical artifacts.
SampleObserver = Callable[[int, int], None]

RUN_RECORD_FILENAME = "run_record.json"


class EvaluationBackend(Protocol):
    """The framework boundary; no tensor code lives in this evaluation engine."""

    def load_model(
        self,
        checkpoint_path: Path,
    ) -> tuple[SegmentationModel, Mapping[str, object]]:
        """Restore the model and the metadata recorded with its checkpoint."""


@dataclass(frozen=True)
class EvaluationResult:
    """What one locked-cohort evaluation produced and where it recorded itself."""

    evaluated_samples: int
    artifact_paths: tuple[Path, ...]
    run_record_path: Path


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _verified_path(root: Path, relative: str, expected_sha256: str) -> Path:
    path = root / relative
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"dataset file SHA-256 does not match the frozen manifest: {path}")
    return path


def load_temperature(
    temperature_path: Path,
    *,
    expected_protocol_sha256: str,
    expected_checkpoint_sha256: str,
) -> float:
    """Read one fitted temperature and prove it belongs to this exact run.

    A temperature is only meaningful for the weights it was fitted on, under the
    protocol it was fitted under. Both bindings are checked here rather than
    trusted, because a mismatched temperature produces confident-looking numbers
    that describe nothing.
    """

    document = json.loads(temperature_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("temperature artifact must be a JSON object")
    if document.get("schema_version") != TEMPERATURE_SCHEMA_VERSION:
        raise ValueError(
            f"temperature artifact must declare {TEMPERATURE_SCHEMA_VERSION}, "
            f"got {document.get('schema_version')!r}"
        )
    if document.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("temperature protocol hash does not match the evaluation protocol")
    if document.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("temperature was fitted for a different checkpoint")

    temperature = float(document["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    return temperature


def _evaluate_sample(
    model: SegmentationModel,
    image_path: Path,
    label_path: Path,
    sample_id: str,
    temperature: float,
) -> tuple[PredictionRecord, Any]:
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    with Image.open(label_path) as handle:
        mask = np.asarray(handle, dtype=np.uint8)

    prepared = prepare_sample(image, mask, training=False, flip_draw=1.0)
    logits = model.logits(prepared.image_chw[None, ...])
    num_classes = int(logits.shape[1])
    canvas = np.ascontiguousarray(logits[0].transpose(1, 2, 0))
    content = canvas[:, prepared.pad_left : CANVAS_WIDTH - prepared.pad_right]
    scaled = apply_temperature(content.reshape(-1, num_classes), temperature)
    canvas_probabilities = softmax_probabilities(scaled)

    valid = mask != MASK_PAD_VALUE
    probabilities = canvas_probabilities[restore_index_map(prepared)[valid]]
    predicted = np.argmax(probabilities, axis=-1)
    targets = mask[valid].astype(np.int64)
    confidence = probabilities[np.arange(predicted.size), predicted]

    record = PredictionRecord(
        sample_id=sample_id,
        predicted_class=predicted.astype(np.uint8),
        top1_confidence_q16=quantize_confidence(confidence),
        correctness_bitset=pack_correctness(predicted == targets),
        confusion=compute_confusion(targets, predicted, num_classes),
        brier_sum_by_class=multiclass_brier_sums(probabilities, targets, num_classes),
        valid_pixel_count=int(predicted.size),
    )
    ece = classwise_ece_sufficient_statistics(probabilities, targets, num_classes)
    return record, ece


def _write_run_record(path: Path, document: dict[str, Any]) -> None:
    record = RunRecordV1.model_validate(document)
    path.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def evaluate_checkpoint(
    config_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    backend: EvaluationBackend,
    temperature_path: Path | None = None,
    on_sample: SampleObserver | None = None,
) -> EvaluationResult:
    """Score one checkpoint over one frozen cohort and publish per-image evidence.

    Every sample is read in manifest order and its bytes are re-hashed against
    the frozen manifest, so a dataset that drifted after freezing can never be
    scored. Metrics are computed at source geometry rather than on the resized
    model canvas, because nearest-resizing the mask would quietly erase the small
    instances this project exists to measure. Only valid pixels reach the stored
    arrays, and the checkpoint must declare the same protocol hash that the
    evaluation is running under.
    """

    provenance = load_run_provenance()
    loaded_protocol = load_protocol(config_path)
    manifest: DatasetManifest = load_manifest(manifest_path)
    model, metadata = backend.load_model(checkpoint_path)

    if metadata.get("protocol_sha256") != loaded_protocol.protocol_sha256:
        raise ValueError("checkpoint protocol hash does not match the evaluation protocol")

    temperature = 1.0
    if temperature_path is not None:
        temperature = load_temperature(
            temperature_path,
            expected_protocol_sha256=loaded_protocol.protocol_sha256,
            expected_checkpoint_sha256=sha256_file(checkpoint_path),
        )

    image_root_name, label_root_name = split_paths(loaded_protocol.protocol, manifest.split_name)
    image_root = data_root / image_root_name
    label_root = data_root / label_root_name
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: list[Path] = []
    artifacts: dict[str, str] = {}
    started_at_utc = _utc_now()
    for position, sample_id in enumerate(manifest.sample_ids):
        image_path = _verified_path(
            image_root,
            manifest.relative_image_paths[position],
            manifest.file_sha256[2 * position],
        )
        label_path = _verified_path(
            label_root,
            manifest.relative_label_paths[position],
            manifest.file_sha256[2 * position + 1],
        )
        record, ece = _evaluate_sample(model, image_path, label_path, sample_id, temperature)
        artifact_path = output_dir / f"{sample_id}.json"
        artifact = write_prediction_artifact(
            artifact_path,
            record,
            ece,
            protocol_sha256=loaded_protocol.protocol_sha256,
            dataset_manifest_sha256=manifest.manifest_sha256,
        )
        artifact_paths.append(artifact_path)
        artifacts[sample_id] = artifact.payload_sha256
        if on_sample is not None:
            on_sample(position + 1, len(manifest.sample_ids))

    run_record_path = output_dir / RUN_RECORD_FILENAME
    _write_run_record(
        run_record_path,
        {
            "schema_version": "driving-risk-run/v1",
            "run_id": f"eval-{metadata['run_id']}",
            "commit": provenance.commit,
            "config_sha256": loaded_protocol.protocol_sha256,
            "protocol_sha256": loaded_protocol.protocol_sha256,
            "dataset_manifest_sha256": manifest.manifest_sha256,
            "lock_sha256": provenance.lock_sha256,
            "hardware": dict(provenance.hardware),
            "seed": int(metadata["seed"]),  # type: ignore[call-overload]
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_now(),
            "status": "succeeded",
            "artifacts": artifacts,
        },
    )
    return EvaluationResult(
        evaluated_samples=len(artifact_paths),
        artifact_paths=tuple(artifact_paths),
        run_record_path=run_record_path,
    )
