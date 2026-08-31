"""Byte-exact hashing helpers for files and semantic manifest content."""

from __future__ import annotations

import hashlib
from pathlib import Path

from drivemetrics.artifacts.envelope import canonical_json_bytes


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a file without loading the whole file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_sha256(manifest_without_hash: dict[str, object]) -> str:
    """Hash canonical semantic manifest content, independent of key order."""

    return hashlib.sha256(canonical_json_bytes(manifest_without_hash)).hexdigest()
