"""Portable, content-hashed paired image/label manifests."""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from drivemetrics.data.bdd100k import (
    DATASET_NAME,
    DATASET_VERSION,
    image_sample_id,
    label_sample_id,
)
from drivemetrics.protocol.hashing import canonical_manifest_sha256, sha256_file

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _validate_relative_posix_path(value: str) -> None:
    path = PurePosixPath(value)
    has_drive_prefix = bool(path.parts and path.parts[0].endswith(":"))
    has_noncanonical_segment = any(part in {"", ".", ".."} for part in value.split("/"))
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or has_drive_prefix
        or has_noncanonical_segment
    ):
        raise ValueError(f"expected a safe relative POSIX path, got {value!r}")


@dataclass(frozen=True)
class DatasetManifest:
    """A root-independent snapshot; file hashes interleave image then label by sample.

    ``sample_ids`` are the eligible samples a consumer may train on, calibrate on
    or score. ``ineligible_sample_ids`` were assigned to this cohort but carry a
    source defect named in ``ineligibility_reasons``; they are listed so that the
    exclusion is on record and hashed, never silent.
    """

    dataset_name: str
    dataset_version: str
    split_name: str
    sample_ids: tuple[str, ...]
    relative_image_paths: tuple[str, ...]
    relative_label_paths: tuple[str, ...]
    file_sha256: tuple[str, ...]
    manifest_sha256: str
    ineligible_sample_ids: tuple[str, ...] = ()
    ineligibility_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sample_count = len(self.sample_ids)
        ineligible = self.ineligible_sample_ids
        if len(ineligible) != len(self.ineligibility_reasons):
            raise ValueError("ineligible sample IDs and reasons must be aligned")
        if len(set(ineligible)) != len(ineligible):
            raise ValueError("manifest contains duplicate ineligible sample IDs")
        if set(ineligible) & set(self.sample_ids):
            raise ValueError("a sample cannot be both eligible and ineligible")
        if any(not reason for reason in self.ineligibility_reasons):
            raise ValueError("every ineligible sample needs a reason")
        if not self.dataset_name or not self.dataset_version or not self.split_name:
            raise ValueError("manifest identity fields must be nonempty")
        if sample_count == 0:
            raise ValueError("manifest must contain at least one sample")
        if not (
            sample_count == len(self.relative_image_paths) == len(self.relative_label_paths)
            and len(self.file_sha256) == sample_count * 2
        ):
            raise ValueError("manifest sample, path, and file-hash fields must be aligned")
        if len(set(self.sample_ids)) != sample_count:
            raise ValueError("manifest contains duplicate sample IDs")
        if any(
            not value or value in {".", ".."} or "/" in value or "\\" in value
            for value in (*self.sample_ids, *ineligible)
        ):
            raise ValueError("manifest sample IDs must be nonempty path-free names")
        for value in (*self.relative_image_paths, *self.relative_label_paths):
            _validate_relative_posix_path(value)
        if not all(_SHA256_PATTERN.fullmatch(value) for value in self.file_sha256):
            raise ValueError("file SHA-256 values must be lowercase 64-digit hex")
        if not _SHA256_PATTERN.fullmatch(self.manifest_sha256):
            raise ValueError("manifest SHA-256 must be lowercase 64-digit hex")
        actual = canonical_manifest_sha256(self.semantic_content())
        if not secrets.compare_digest(actual, self.manifest_sha256):
            raise ValueError("manifest SHA-256 mismatch")

    def semantic_content(self) -> dict[str, object]:
        """Return the exact hashable manifest fields, excluding the hash itself."""

        values = asdict(self)
        del values["manifest_sha256"]
        return values


def _finalize(**fields: Any) -> DatasetManifest:
    """Mint the manifest hash over exactly the semantic fields and build the manifest."""

    return DatasetManifest(**fields, manifest_sha256=canonical_manifest_sha256(fields))


def _index_files(
    root: Path,
    pattern: str,
    id_from_path: Callable[[Path], str],
) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in root.rglob(pattern):
        sample_id = id_from_path(path)
        if sample_id in indexed:
            raise ValueError(f"duplicate sample ID {sample_id!r} under {root}")
        indexed[sample_id] = path
    return indexed


def _relative_posix(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    _validate_relative_posix_path(value)
    return value


def build_paired_manifest(
    image_root: Path,
    label_root: Path,
    split_name: str,
) -> DatasetManifest:
    """Scan, pair, sort, and content-hash one BDD100K semantic split."""

    if not image_root.is_dir():
        raise ValueError(f"image root is not a directory: {image_root}")
    if not label_root.is_dir():
        raise ValueError(f"label root is not a directory: {label_root}")
    images = _index_files(image_root, "*.jpg", image_sample_id)
    labels = _index_files(label_root, "*.png", label_sample_id)
    image_ids = set(images)
    label_ids = set(labels)
    missing_labels = sorted(image_ids - label_ids)
    if missing_labels:
        raise ValueError(f"missing label for sample IDs: {missing_labels}")
    extra_labels = sorted(label_ids - image_ids)
    if extra_labels:
        raise ValueError(f"extra label for sample IDs: {extra_labels}")
    if not image_ids:
        raise ValueError("no paired samples found")

    sample_ids = tuple(sorted(image_ids))
    relative_image_paths = tuple(_relative_posix(images[value], image_root) for value in sample_ids)
    relative_label_paths = tuple(_relative_posix(labels[value], label_root) for value in sample_ids)
    file_sha256 = tuple(
        digest
        for sample_id in sample_ids
        for digest in (sha256_file(images[sample_id]), sha256_file(labels[sample_id]))
    )
    return _finalize(
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        split_name=split_name,
        sample_ids=sample_ids,
        relative_image_paths=relative_image_paths,
        relative_label_paths=relative_label_paths,
        file_sha256=file_sha256,
        ineligible_sample_ids=(),
        ineligibility_reasons=(),
    )


MANIFEST_FIELDS: tuple[str, ...] = (
    "dataset_name",
    "dataset_version",
    "split_name",
    "sample_ids",
    "relative_image_paths",
    "relative_label_paths",
    "file_sha256",
    "manifest_sha256",
    "ineligible_sample_ids",
    "ineligibility_reasons",
)


def load_manifest(path: Path) -> DatasetManifest:
    """Load one frozen manifest document and re-verify its own content hash."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("manifest document must be a mapping")
    if set(document) != set(MANIFEST_FIELDS):
        raise ValueError("manifest document must contain exactly the manifest fields")
    return DatasetManifest(
        dataset_name=document["dataset_name"],
        dataset_version=document["dataset_version"],
        split_name=document["split_name"],
        sample_ids=tuple(document["sample_ids"]),
        relative_image_paths=tuple(document["relative_image_paths"]),
        relative_label_paths=tuple(document["relative_label_paths"]),
        file_sha256=tuple(document["file_sha256"]),
        manifest_sha256=document["manifest_sha256"],
        ineligible_sample_ids=tuple(document["ineligible_sample_ids"]),
        ineligibility_reasons=tuple(document["ineligibility_reasons"]),
    )


def save_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Write one frozen manifest document, never replacing an existing file.

    A frozen cohort is evidence. Overwriting one in place would silently detach
    every run record and claim that referenced its previous hash.
    """

    if path.exists():
        raise FileExistsError(f"frozen manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def subset_manifest(
    manifest: DatasetManifest,
    sample_ids: Sequence[str],
    split_name: str,
) -> DatasetManifest:
    """Derive one cohort manifest from a superset without rehashing any file.

    Paths and the interleaved image/label hashes follow the requested sample
    order exactly, so a cohort can never inherit another sample file hashes.
    Cohort assignment covers every source ID, eligible or not, so a chosen ID
    that is ineligible in the source travels into the subset as ineligible,
    with its reason.
    """

    chosen = tuple(sample_ids)
    if len(set(chosen)) != len(chosen):
        raise ValueError("manifest subset contains duplicate sample IDs")
    index = {sample_id: position for position, sample_id in enumerate(manifest.sample_ids)}
    reasons = dict(zip(manifest.ineligible_sample_ids, manifest.ineligibility_reasons, strict=True))
    missing = tuple(value for value in chosen if value not in index and value not in reasons)
    if missing:
        raise ValueError(f"sample IDs are not present in the source manifest: {missing}")

    eligible = tuple(value for value in chosen if value in index)
    ineligible = tuple(value for value in chosen if value in reasons)
    positions = tuple(index[value] for value in eligible)
    return _finalize(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        split_name=split_name,
        sample_ids=eligible,
        relative_image_paths=tuple(manifest.relative_image_paths[value] for value in positions),
        relative_label_paths=tuple(manifest.relative_label_paths[value] for value in positions),
        file_sha256=tuple(
            digest
            for value in positions
            for digest in manifest.file_sha256[2 * value : 2 * value + 2]
        ),
        ineligible_sample_ids=ineligible,
        ineligibility_reasons=tuple(reasons[value] for value in ineligible),
    )


def mark_ineligible(manifest: DatasetManifest, reasons: Mapping[str, str]) -> DatasetManifest:
    """Move the named samples out of the eligible list, keeping each reason on record.

    Ineligibility is decided from file geometry alone, never from labels,
    metrics or models, and it never changes which cohort a sample was assigned
    to: the sample stays listed under its cohort, as ineligible, with its reason.
    """

    missing = tuple(value for value in reasons if value not in manifest.sample_ids)
    if missing:
        raise ValueError(f"sample IDs are not present in the manifest: {missing}")
    if any(not reason for reason in reasons.values()):
        raise ValueError("every ineligible sample needs a reason")

    kept = tuple(
        position for position, value in enumerate(manifest.sample_ids) if value not in reasons
    )
    newly = tuple(value for value in manifest.sample_ids if value in reasons)
    return _finalize(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        split_name=manifest.split_name,
        sample_ids=tuple(manifest.sample_ids[value] for value in kept),
        relative_image_paths=tuple(manifest.relative_image_paths[value] for value in kept),
        relative_label_paths=tuple(manifest.relative_label_paths[value] for value in kept),
        file_sha256=tuple(
            digest for value in kept for digest in manifest.file_sha256[2 * value : 2 * value + 2]
        ),
        ineligible_sample_ids=(*manifest.ineligible_sample_ids, *newly),
        ineligibility_reasons=(
            *manifest.ineligibility_reasons,
            *(reasons[value] for value in newly),
        ),
    )
