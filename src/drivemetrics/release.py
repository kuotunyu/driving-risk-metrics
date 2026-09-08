"""Validate release identities against the project, installed package and archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import re
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Sequence
from email.parser import BytesParser
from pathlib import Path

from drivemetrics import __version__


def verify_backend(project: Path) -> None:
    """Refuse a floating backend requirement or a different installed backend."""
    config = tomllib.loads(project.read_text(encoding="utf-8"))
    for requirement in config["build-system"]["requires"]:
        match = re.fullmatch(r"([A-Za-z0-9_-]+)==([0-9]+(?:\.[0-9]+)*)", requirement)
        if match is None:
            raise ValueError("backend dependencies must have exact version pins")
        name, expected = match.groups()
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"installed backend does not match {requirement}")
        print(f"backend: {requirement}")


def _check_metadata(payload: bytes, name: str, version: str) -> None:
    metadata = BytesParser().parsebytes(payload)
    if metadata.get_all("Name") != [name] or metadata.get_all("Version") != [version]:
        raise ValueError("archive metadata does not match project name and version")


def verify_release(
    project: Path,
    dist_dir: Path,
    tag: str,
    installed_version: str,
    runtime_version: str,
) -> None:
    """Write portable checksums only after all release identities agree."""
    config = tomllib.loads(project.read_text(encoding="utf-8"))["project"]
    name, version = config["name"], config["version"]
    if (tag, installed_version, runtime_version) != (f"v{version}", version, version):
        raise ValueError("tag, installed and runtime identity must match project version")
    stem = f"{re.sub(r'[-_.]+', '_', name)}-{version}"
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if [path.name for path in wheels] != [f"{stem}-py3-none-any.whl"]:
        raise ValueError("wheel filename or distribution count does not match project")
    if [path.name for path in sdists] != [f"{stem}.tar.gz"]:
        raise ValueError("sdist filename or distribution count does not match project")
    wheel, sdist = wheels[0], sdists[0]
    with zipfile.ZipFile(wheel) as archive:
        expected = f"{stem}.dist-info/METADATA"
        members = [item for item in archive.namelist() if item.endswith(".dist-info/METADATA")]
        if members != [expected]:
            raise ValueError("wheel must contain one canonical metadata member")
        _check_metadata(archive.read(expected), name, version)
    with tarfile.open(sdist, "r:gz") as archive:
        expected = f"{stem}/PKG-INFO"
        tar_members = [item for item in archive.getmembers() if item.name == expected]
        if len(tar_members) != 1 or not tar_members[0].isfile():
            raise ValueError("sdist must contain one canonical metadata file")
        payload = archive.extractfile(tar_members[0])
        assert payload is not None
        _check_metadata(payload.read(), name, version)
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in (wheel, sdist)
    )
    (dist_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8", newline="\n")
    print(f"verified: {name} {version} ({tag})")
    print(checksums, end="")


def normalize_sdist(dist_dir: Path, epoch: int) -> None:
    """Pin demonstrated gzip/TAR timestamps without changing archive payloads."""
    if not 0 <= epoch < 2**32:
        raise ValueError("epoch must fit the unsigned gzip timestamp")
    sdists = list(dist_dir.glob("*.tar.gz"))
    if len(sdists) != 1:
        raise ValueError("normalization requires exactly one sdist")
    path = sdists[0]
    buffer = io.BytesIO()
    with (
        tarfile.open(path, "r:gz") as source,
        gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=epoch) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target,
    ):
        for member in source:
            payload = source.extractfile(member) if member.isfile() else None
            member.mtime = epoch
            headers = dict(member.pax_headers)
            headers.pop("mtime", None)
            member.pax_headers = headers
            target.addfile(member, payload)
    path.write_bytes(buffer.getvalue())


def main(argv: Sequence[str] | None = None) -> int:
    """Expose backend and archive checks to the release workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend = subparsers.add_parser("backend")
    backend.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    verify.add_argument("--dist-dir", type=Path, required=True)
    verify.add_argument("--tag", required=True)
    normalize = subparsers.add_parser("normalize-sdist")
    normalize.add_argument("--dist-dir", type=Path, required=True)
    normalize.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "backend":
        verify_backend(args.project)
    elif args.command == "normalize-sdist":
        normalize_sdist(args.dist_dir, args.epoch)
    else:
        verify_release(
            args.project,
            args.dist_dir,
            args.tag,
            importlib.metadata.version("driving-risk-metrics"),
            __version__,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
