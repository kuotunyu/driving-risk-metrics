"""Deterministic freezing of the three formal BDD100K cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drivemetrics.data.manifest import (
    DatasetManifest,
    build_paired_manifest,
    save_manifest,
    subset_manifest,
)
from drivemetrics.data.splits import freeze_bdd100k_split, validate_locked_split
from drivemetrics.protocol.config import load_protocol

MANIFEST_NAMES: tuple[str, ...] = (
    "source_train",
    "train",
    "calibration",
    "locked_validation",
)


@dataclass(frozen=True)
class PreflightResult:
    """Where the frozen cohorts were written and exactly what they contain."""

    protocol_sha256: str
    manifest_paths: dict[str, Path]
    manifest_sha256: dict[str, str]
    counts: dict[str, int]


def _resolve_directory(data_root: Path, relative: str) -> Path:
    path = data_root / relative
    if not path.is_dir():
        raise ValueError(f"dataset path is not a directory: {path}")
    return path


def _require_count(name: str, manifest: DatasetManifest, expected: int) -> None:
    actual = len(manifest.sample_ids)
    if actual != expected:
        raise ValueError(
            f"{name} split must contain exactly {expected} paired samples, found {actual}"
        )


def run_preflight(config_path: Path, data_root: Path, output_dir: Path) -> PreflightResult:
    """Verify the dataset against the protocol and freeze the three formal cohorts.

    The locked validation split is built and counted first, so a broken locked
    cohort is reported before the much larger training split is hashed. Nothing
    is written until every count, the deterministic SHA-256 train/calibration
    split, and the contamination checks have passed, and an existing frozen
    manifest stops the run before any directory is scanned.
    """

    loaded = load_protocol(config_path)
    protocol = loaded.protocol

    paths = {name: output_dir / f"{name}.json" for name in MANIFEST_NAMES}
    existing = tuple(str(path) for path in paths.values() if path.exists())
    if existing:
        raise FileExistsError(f"frozen manifest already exists: {existing}")

    validation_images = _resolve_directory(data_root, protocol.paths.validation_images)
    validation_labels = _resolve_directory(data_root, protocol.paths.validation_labels)
    train_images = _resolve_directory(data_root, protocol.paths.train_images)
    train_labels = _resolve_directory(data_root, protocol.paths.train_labels)

    validation = build_paired_manifest(validation_images, validation_labels, "locked_validation")
    _require_count("locked_validation", validation, protocol.splits.locked_validation)

    source_train = build_paired_manifest(train_images, train_labels, "source_train")
    _require_count("source_train", source_train, protocol.splits.source_train)

    train_ids, calibration_ids = freeze_bdd100k_split(source_train.sample_ids)
    validate_locked_split(train_ids, calibration_ids, validation.sample_ids)

    manifests = {
        "source_train": source_train,
        "train": subset_manifest(source_train, train_ids, "train"),
        "calibration": subset_manifest(source_train, calibration_ids, "calibration"),
        "locked_validation": validation,
    }
    for name in MANIFEST_NAMES:
        save_manifest(manifests[name], paths[name])

    return PreflightResult(
        protocol_sha256=loaded.protocol_sha256,
        manifest_paths=dict(paths),
        manifest_sha256={name: manifests[name].manifest_sha256 for name in MANIFEST_NAMES},
        counts={name: len(manifests[name].sample_ids) for name in MANIFEST_NAMES},
    )
