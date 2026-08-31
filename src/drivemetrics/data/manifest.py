"""Portable, content-hashed paired image/label manifests."""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

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
    """A root-independent snapshot; file hashes interleave image then label by sample."""

    dataset_name: str
    dataset_version: str
    split_name: str
    sample_ids: tuple[str, ...]
    relative_image_paths: tuple[str, ...]
    relative_label_paths: tuple[str, ...]
    file_sha256: tuple[str, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        sample_count = len(self.sample_ids)
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
            for value in self.sample_ids
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
    without_hash: dict[str, object] = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "split_name": split_name,
        "sample_ids": sample_ids,
        "relative_image_paths": relative_image_paths,
        "relative_label_paths": relative_label_paths,
        "file_sha256": file_sha256,
    }
    return DatasetManifest(
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        split_name=split_name,
        sample_ids=sample_ids,
        relative_image_paths=relative_image_paths,
        relative_label_paths=relative_label_paths,
        file_sha256=file_sha256,
        manifest_sha256=canonical_manifest_sha256(without_hash),
    )
