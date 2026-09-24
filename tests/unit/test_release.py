"""Release identities and portable checksums are checked on real archives."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import runpy
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def release_module() -> ModuleType:
    try:
        from drivemetrics import release
    except ImportError:
        pytest.fail("release identity verifier is missing", pytrace=False)
    return release


def distributions(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "driving-risk-metrics"\nversion = "1.0.2"\n')
    wheel = tmp_path / "driving_risk_metrics-1.0.2-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "driving_risk_metrics-1.0.2.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: driving-risk-metrics\nVersion: 1.0.2\n",
        )
    sdist = tmp_path / "driving_risk_metrics-1.0.2.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = b"Metadata-Version: 2.4\nName: driving-risk-metrics\nVersion: 1.0.2\n"
        member = tarfile.TarInfo("driving_risk_metrics-1.0.2/PKG-INFO")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return project, wheel, sdist


def test_verified_archives_get_relative_two_file_checksums(tmp_path: Path) -> None:
    project, wheel, sdist = distributions(tmp_path)
    release_module().verify_release(project, tmp_path, "v1.0.2", "1.0.2", "1.0.2")
    assert (tmp_path / "SHA256SUMS").read_bytes() == (
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n"
        f"{hashlib.sha256(sdist.read_bytes()).hexdigest()}  {sdist.name}\n"
    ).encode()


@pytest.mark.parametrize(
    ("tag", "installed", "runtime"),
    [
        ("v1.0.1", "1.0.2", "1.0.2"),
        ("1.0.2", "1.0.2", "1.0.2"),
        ("v1.0.2", "1.0.0", "1.0.2"),
        ("v1.0.2", "1.0.2", "1.0.0"),
    ],
)
def test_version_mismatch_is_refused_before_checksums(
    tmp_path: Path, tag: str, installed: str, runtime: str
) -> None:
    project, _, _ = distributions(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        release_module().verify_release(project, tmp_path, tag, installed, runtime)
    assert not (tmp_path / "SHA256SUMS").exists()


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("field", ["Name", "Version"])
def test_archive_metadata_mismatch_is_refused(tmp_path: Path, kind: str, field: str) -> None:
    project, wheel, sdist = distributions(tmp_path)
    payload = f"Name: {'other' if field == 'Name' else 'driving-risk-metrics'}\nVersion: {'9.9.9' if field == 'Version' else '1.0.2'}\n".encode()
    if kind == "wheel":
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("driving_risk_metrics-1.0.2.dist-info/METADATA", payload)
    else:
        with tarfile.open(sdist, "w:gz") as archive:
            member = tarfile.TarInfo("driving_risk_metrics-1.0.2/PKG-INFO")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="metadata"):
        release_module().verify_release(project, tmp_path, "v1.0.2", "1.0.2", "1.0.2")


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_wrong_filename_is_refused(tmp_path: Path, kind: str) -> None:
    project, wheel, sdist = distributions(tmp_path)
    artifact = wheel if kind == "wheel" else sdist
    artifact.rename(artifact.with_name(artifact.name.replace("1.0.2", "1.0.0")))
    with pytest.raises(ValueError, match="filename"):
        release_module().verify_release(project, tmp_path, "v1.0.2", "1.0.2", "1.0.2")


@pytest.mark.parametrize("requirement", ["setuptools>=84", "setuptools==83.0.0"])
def test_backend_refuses_unpinned_or_different_installed_version(
    tmp_path: Path, requirement: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(f'[build-system]\nrequires = ["{requirement}", "wheel==0.45.1"]\n')
    monkeypatch.setattr(
        importlib.metadata, "version", {"setuptools": "84.0.0", "wheel": "0.45.1"}.__getitem__
    )
    with pytest.raises(ValueError, match="backend"):
        release_module().verify_backend(project)


def test_backend_accepts_only_exact_installed_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text('[build-system]\nrequires = ["setuptools==84.0.0", "wheel==0.45.1"]\n')
    monkeypatch.setattr(
        importlib.metadata, "version", {"setuptools": "84.0.0", "wheel": "0.45.1"}.__getitem__
    )
    release_module().verify_backend(project)


def test_sdist_normalization_removes_only_timestamp_variation_and_preserves_payloads(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "one", tmp_path / "two"]
    for index, path in enumerate(paths):
        path.mkdir()
        with tarfile.open(path / "sample.tar.gz", "w:gz") as archive:
            directory = tarfile.TarInfo("sample")
            directory.type = tarfile.DIRTYPE
            directory.mtime = 900 + index
            archive.addfile(directory)
            member = tarfile.TarInfo("sample/run.py")
            member.mode = 0o755
            member.uid = 123
            member.gid = 456
            member.uname = "builder"
            member.gname = "team"
            member.mtime = 1000 + index
            member.pax_headers = {"mtime": str(1000 + index), "comment": "keep"}
            member.size = 7
            archive.addfile(member, io.BytesIO(b"payload"))
    module = release_module()
    assert hasattr(module, "normalize_sdist"), "sdist timestamp normalization is missing"
    for path in paths:
        module.normalize_sdist(path, 1788600000)
    first = (paths[0] / "sample.tar.gz").read_bytes()
    assert first == (paths[1] / "sample.tar.gz").read_bytes()
    assert int.from_bytes(first[4:8], "little") == 1788600000
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        assert [member.name for member in archive] == ["sample", "sample/run.py"]
        member = archive.getmember("sample/run.py")
        assert (member.mode, member.uid, member.gid, member.uname, member.gname) == (
            0o755,
            123,
            456,
            "builder",
            "team",
        )
        assert member.pax_headers == {"comment": "keep"}
        assert all(member.mtime == 1788600000 for member in archive)
        payload = archive.extractfile(member)
        assert payload is not None and payload.read() == b"payload"
    module.normalize_sdist(paths[0], 1788600000)
    assert (paths[0] / "sample.tar.gz").read_bytes() == first
    assert gzip.decompress(first)


@pytest.mark.parametrize("epoch", [-1, 2**32])
def test_invalid_normalization_epoch_keeps_original_bytes(tmp_path: Path, epoch: int) -> None:
    _, _, sdist = distributions(tmp_path)
    before = sdist.read_bytes()
    module = release_module()
    assert hasattr(module, "normalize_sdist"), "sdist timestamp normalization is missing"
    with pytest.raises(ValueError, match="epoch"):
        module.normalize_sdist(tmp_path, epoch)
    assert sdist.read_bytes() == before


def test_normalization_refuses_ambiguous_sdist_selection(tmp_path: Path) -> None:
    module = release_module()
    assert hasattr(module, "normalize_sdist"), "sdist timestamp normalization is missing"
    with pytest.raises(ValueError, match="one sdist"):
        module.normalize_sdist(tmp_path, 1788600000)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "noncanonical"])
def test_metadata_must_be_one_canonical_file(tmp_path: Path, kind: str, defect: str) -> None:
    project, wheel, sdist = distributions(tmp_path)
    payload = b"Name: driving-risk-metrics\nVersion: 1.0.2\n"
    if kind == "wheel":
        with zipfile.ZipFile(wheel, "w") as archive:
            if defect != "missing":
                archive.writestr("other-1.0.2.dist-info/METADATA", payload)
            if defect == "duplicate":
                archive.writestr("driving_risk_metrics-1.0.2.dist-info/METADATA", payload)
    else:
        with tarfile.open(sdist, "w:gz") as archive:
            if defect != "missing":
                member = tarfile.TarInfo("driving_risk_metrics-1.0.2/PKG-INFO")
                if defect == "noncanonical":
                    member.type = tarfile.DIRTYPE
                    archive.addfile(member)
                else:
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
                    archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="canonical metadata"):
        release_module().verify_release(project, tmp_path, "v1.0.2", "1.0.2", "1.0.2")


def test_sdist_nested_metadata_cannot_stand_in_for_root_metadata(tmp_path: Path) -> None:
    project, _, sdist = distributions(tmp_path)
    with tarfile.open(sdist, "w:gz") as archive:
        payload = b"Name: driving-risk-metrics\nVersion: 1.0.2\n"
        member = tarfile.TarInfo("driving_risk_metrics-1.0.2/src/example.egg-info/PKG-INFO")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="canonical metadata"):
        release_module().verify_release(project, tmp_path, "v1.0.2", "1.0.2", "1.0.2")


def test_cli_checks_actual_installed_version_and_normalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _, _ = distributions(tmp_path)
    module = release_module()
    # The fixture project is version 1.0.2 whatever the package's own version is,
    # so the runtime identity is pinned to it too; only the CLI wiring is tested.
    monkeypatch.setattr(module, "__version__", "1.0.2")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0.0")
    with pytest.raises(ValueError, match="identity"):
        module.main(
            ["verify", "--project", str(project), "--dist-dir", str(tmp_path), "--tag", "v1.0.2"]
        )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0.2")
    assert (
        module.main(
            ["verify", "--project", str(project), "--dist-dir", str(tmp_path), "--tag", "v1.0.2"]
        )
        == 0
    )
    assert (
        module.main(["normalize-sdist", "--dist-dir", str(tmp_path), "--epoch", "1788600000"]) == 0
    )


def test_module_entrypoint_runs_backend_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text('[build-system]\nrequires = ["setuptools==84.0.0"]\n')
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "84.0.0")
    monkeypatch.setattr(sys, "argv", ["release", "backend", "--project", str(project)])
    monkeypatch.delitem(sys.modules, "drivemetrics.release", raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("drivemetrics.release", run_name="__main__")
    assert stopped.value.code == 0
