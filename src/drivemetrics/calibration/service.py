"""Fit one scalar temperature on the frozen calibration cohort.

Full per-pixel logits are infeasible for the formal cohort: 700 images at
1280x720 by 19 classes in float64 is roughly 98 GB. The fit therefore runs on a
deterministic seeded pixel sample per image, drawn only from labelled pixels.
That makes the sampling rule part of the evidence rather than an implementation
detail, so it is written into the artifact and can be reproduced exactly.

The sample is keyed by sample ID rather than by position, so reordering the
cohort cannot change which pixels were used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from drivemetrics.artifacts.run_record import RunRecordV1, load_run_provenance
from drivemetrics.calibration.temperature import (
    CalibrationProvenance,
    fit_provenance_checked_temperature,
)
from drivemetrics.data.manifest import DatasetManifest, load_manifest
from drivemetrics.data.transforms import (
    CANVAS_WIDTH,
    MASK_PAD_VALUE,
    prepare_sample,
    restore_index_map,
)
from drivemetrics.models.adapters import SegmentationModel
from drivemetrics.protocol.config import load_protocol, split_paths
from drivemetrics.protocol.hashing import sha256_file

TEMPERATURE_FILENAME = "temperature.json"
RUN_RECORD_FILENAME = "run_record.json"
TEMPERATURE_SCHEMA_VERSION = "drivemetrics-temperature/v1"
CALIBRATION_SPLIT_NAME = "calibration"

#: Labelled pixels drawn per calibration image. Large enough that the scalar fit
#: is stable, small enough that the whole cohort fits in memory as float64.
PIXELS_PER_IMAGE = 2048

#: Fixed so the draw is reproducible; distinct from the bootstrap seed so the
#: two never move together.
SAMPLING_SEED = 20260901


class CalibrationBackend(Protocol):
    """Restores the model and metadata recorded with one checkpoint."""

    def load_model(self, checkpoint_path: Path) -> tuple[SegmentationModel, Mapping[str, object]]:
        """Return the model and the metadata recorded with its checkpoint."""


@dataclass(frozen=True)
class CalibrationResult:
    """The fitted temperature and everything needed to reproduce it."""

    temperature: float
    artifact_path: Path
    run_record_path: Path
    dataset_manifest_sha256: str
    sampled_images: int
    pixels_per_image: int


def sample_pixel_indices(
    mask: npt.NDArray[np.uint8],
    *,
    sample_id: str,
    pixels: int = PIXELS_PER_IMAGE,
) -> npt.NDArray[np.int64]:
    """Draw flat indices of labelled pixels, deterministically for this sample.

    Ignore pixels carry no label, so drawing them would spend the budget on
    values the fit discards. When an image has fewer labelled pixels than the
    budget, every labelled pixel is returned once rather than resampled, because
    duplicating a pixel would silently reweight that image.
    """

    flat = mask.reshape(-1)
    valid = np.flatnonzero(flat != MASK_PAD_VALUE).astype(np.int64)
    if valid.size <= pixels:
        return valid

    digest = hashlib.sha256(f"{SAMPLING_SEED}:{sample_id}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    chosen = generator.choice(valid.size, size=pixels, replace=False)
    return np.sort(valid[chosen])


def _verified_path(root: Path, relative: str, expected_sha256: str) -> Path:
    path = root / relative
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"dataset file SHA-256 does not match the frozen manifest: {relative}")
    return path


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sample_logits(
    model: SegmentationModel,
    image_path: Path,
    label_path: Path,
    sample_id: str,
    pixels: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return `(pixels, classes)` logits and `(pixels,)` targets at source geometry."""

    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    with Image.open(label_path) as handle:
        mask = np.asarray(handle, dtype=np.uint8)

    prepared = prepare_sample(image, mask, training=False, flip_draw=1.0)
    logits = model.logits(prepared.image_chw[None, ...])
    num_classes = int(logits.shape[1])
    canvas = np.ascontiguousarray(logits[0].transpose(1, 2, 0))
    content = canvas[:, prepared.pad_left : CANVAS_WIDTH - prepared.pad_right]
    canvas_logits = content.reshape(-1, num_classes)

    indices = sample_pixel_indices(mask, sample_id=sample_id, pixels=pixels)
    source_to_canvas = restore_index_map(prepared).reshape(-1)
    drawn = canvas_logits[source_to_canvas[indices]]
    targets = mask.reshape(-1)[indices].astype(np.int64)

    # Pad short images with ignore targets so every image contributes the same
    # shape; the fitter drops ignore rows, so padding cannot bias the result.
    if drawn.shape[0] < pixels:
        pad = pixels - drawn.shape[0]
        drawn = np.concatenate([drawn, np.zeros((pad, num_classes), dtype=np.float64)])
        targets = np.concatenate([targets, np.full(pad, MASK_PAD_VALUE, dtype=np.int64)])
    return drawn, targets


def calibrate_checkpoint(
    config_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    backend: CalibrationBackend,
    pixels_per_image: int = PIXELS_PER_IMAGE,
) -> CalibrationResult:
    """Fit and publish one scalar temperature for one checkpoint.

    The cohort must be the frozen calibration split, and the checkpoint must
    declare the same protocol hash the fit runs under. Both are refused rather
    than warned about, because a temperature fitted on the wrong cohort silently
    corrupts every calibrated number downstream.
    """

    provenance = load_run_provenance()
    loaded_protocol = load_protocol(config_path)
    manifest: DatasetManifest = load_manifest(manifest_path)
    if manifest.split_name != CALIBRATION_SPLIT_NAME:
        raise ValueError(
            f"temperature fitting requires the {CALIBRATION_SPLIT_NAME} split, "
            f"got {manifest.split_name!r}"
        )

    model, metadata = backend.load_model(checkpoint_path)
    if metadata.get("protocol_sha256") != loaded_protocol.protocol_sha256:
        raise ValueError("checkpoint protocol hash does not match the calibration protocol")

    artifact_path = output_dir / TEMPERATURE_FILENAME
    if artifact_path.exists():
        raise FileExistsError(f"a frozen temperature already exists: {artifact_path}")

    image_root_name, label_root_name = split_paths(loaded_protocol.protocol, manifest.split_name)
    image_root = data_root / image_root_name
    label_root = data_root / label_root_name

    started_at_utc = _utc_now()
    all_logits: list[npt.NDArray[np.float64]] = []
    all_targets: list[npt.NDArray[np.int64]] = []
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
        logits, targets = _sample_logits(model, image_path, label_path, sample_id, pixels_per_image)
        all_logits.append(logits)
        all_targets.append(targets)

    temperature = fit_provenance_checked_temperature(
        np.stack(all_logits),
        np.stack(all_targets),
        CalibrationProvenance(
            split_name=manifest.split_name,
            sample_ids=manifest.sample_ids,
            dataset_manifest_sha256=manifest.manifest_sha256,
        ),
        expected_dataset_manifest_sha256=manifest.manifest_sha256,
        expected_sample_ids=manifest.sample_ids,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "schema_version": TEMPERATURE_SCHEMA_VERSION,
        "temperature": temperature,
        "protocol_sha256": loaded_protocol.protocol_sha256,
        "dataset_manifest_sha256": manifest.manifest_sha256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_id": str(metadata["run_id"]),
        "seed": int(metadata["seed"]),  # type: ignore[call-overload]
        "sampled_images": len(manifest.sample_ids),
        "pixels_per_image": pixels_per_image,
        "sampling_seed": SAMPLING_SEED,
    }
    artifact_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run_record_path = output_dir / RUN_RECORD_FILENAME
    record = RunRecordV1.model_validate(
        {
            "schema_version": "driving-risk-run/v1",
            "run_id": f"calibrate-{metadata['run_id']}",
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
            "artifacts": {"temperature": hashlib.sha256(artifact_path.read_bytes()).hexdigest()},
        }
    )
    run_record_path.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return CalibrationResult(
        temperature=temperature,
        artifact_path=artifact_path,
        run_record_path=run_record_path,
        dataset_manifest_sha256=manifest.manifest_sha256,
        sampled_images=len(manifest.sample_ids),
        pixels_per_image=pixels_per_image,
    )
