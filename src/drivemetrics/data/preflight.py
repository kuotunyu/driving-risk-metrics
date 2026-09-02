"""Deterministic freezing of the three formal BDD100K cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from drivemetrics.data.manifest import (
    DatasetManifest,
    build_paired_manifest,
    mark_ineligible,
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
    """Where the frozen cohorts were written and exactly what they contain.

    ``counts`` are eligible samples per cohort; ``ineligible`` names, per cohort,
    every assigned sample that was excluded and why.
    """

    protocol_sha256: str
    manifest_paths: dict[str, Path]
    manifest_sha256: dict[str, str]
    counts: dict[str, int]
    ineligible: dict[str, dict[str, str]]


def _resolve_directory(data_root: Path, relative: str) -> Path:
    path = data_root / relative
    if not path.is_dir():
        raise ValueError(f"dataset path is not a directory: {path}")
    return path


def _require_count(name: str, manifest: DatasetManifest, expected: int) -> None:
    actual = len(manifest.sample_ids) + len(manifest.ineligible_sample_ids)
    if actual != expected:
        raise ValueError(
            f"{name} split must contain exactly {expected} paired samples, found {actual}"
        )


def _all_ids(manifest: DatasetManifest) -> tuple[str, ...]:
    return (*manifest.sample_ids, *manifest.ineligible_sample_ids)


def pair_geometry_reasons(
    manifest: DatasetManifest,
    image_root: Path,
    label_root: Path,
) -> dict[str, str]:
    """Name every eligible pair whose image and label pixel geometry disagree.

    Decided from file headers alone: no label content, no metric, no model.
    Such a pair has no usable supervision and no scorable prediction. BDD100K
    10K ships a few of them as 720x1280 portraits beside 1280x720 labels, with
    no EXIF orientation to undo and no rigid transform that aligns them.
    """

    reasons: dict[str, str] = {}
    pairs = zip(
        manifest.sample_ids,
        manifest.relative_image_paths,
        manifest.relative_label_paths,
        strict=True,
    )
    for sample_id, image_relative, label_relative in pairs:
        with Image.open(image_root / image_relative) as image:
            image_size = image.size
        with Image.open(label_root / label_relative) as label:
            label_size = label.size
        if image_size != label_size:
            reasons[sample_id] = (
                f"image {image_size[0]}x{image_size[1]} but label {label_size[0]}x{label_size[1]}"
            )
    return reasons


def run_preflight(config_path: Path, data_root: Path, output_dir: Path) -> PreflightResult:
    """Verify the dataset against the protocol and freeze the three formal cohorts.

    The locked validation split is built and counted first, so a broken locked
    cohort is reported before the much larger training split is hashed. Counts
    are checked against the official split sizes before any pair is judged;
    only then are pairs with disagreeing image and label geometry marked
    ineligible. The deterministic SHA-256 train/calibration split runs over
    every source ID, eligible or not, so an exclusion never moves anyone else.
    Nothing is written until every count, the split, and the contamination
    checks have passed, and an existing frozen manifest stops the run before
    any directory is scanned.
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
    validation = mark_ineligible(
        validation, pair_geometry_reasons(validation, validation_images, validation_labels)
    )

    source_train = build_paired_manifest(train_images, train_labels, "source_train")
    _require_count("source_train", source_train, protocol.splits.source_train)
    source_train = mark_ineligible(
        source_train, pair_geometry_reasons(source_train, train_images, train_labels)
    )

    train_ids, calibration_ids = freeze_bdd100k_split(_all_ids(source_train))
    validate_locked_split(train_ids, calibration_ids, _all_ids(validation))

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
        ineligible={
            name: dict(
                zip(
                    manifests[name].ineligible_sample_ids,
                    manifests[name].ineligibility_reasons,
                    strict=True,
                )
            )
            for name in MANIFEST_NAMES
        },
    )
