"""Tests for deterministic, portable BDD100K paired-file manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "bdd100k_tiny_manifest_input.json"


def load_manifest_module() -> ModuleType:
    try:
        from drivemetrics.data import manifest
    except ImportError:
        pytest.fail("drivemetrics.data.manifest is missing", pytrace=False)
    return manifest


def materialize_fixture(tmp_path: Path, *, reverse: bool = False) -> tuple[Path, Path]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    samples = fixture["samples"]
    if reverse:
        samples = list(reversed(samples))
    for sample in samples:
        image_path = image_root / sample["image_path"]
        label_path = label_root / sample["label_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_text(sample["image_content"], encoding="utf-8")
        label_path.write_text(sample["label_content"], encoding="utf-8")
    return image_root, label_root


def test_build_manifest_is_sorted_and_independent_of_creation_order(tmp_path: Path) -> None:
    manifest = load_manifest_module()
    first_images, first_labels = materialize_fixture(tmp_path / "first")
    second_images, second_labels = materialize_fixture(tmp_path / "second", reverse=True)

    first = manifest.build_paired_manifest(first_images, first_labels, "train")
    second = manifest.build_paired_manifest(second_images, second_labels, "train")

    assert first == second
    assert first.dataset_name == "bdd100k"
    assert first.dataset_version == "10k-semantic-v1"
    assert first.sample_ids == ("sample-a", "sample-b")
    assert first.relative_image_paths == ("nested/sample-a.jpg", "sample-b.jpg")
    assert first.relative_label_paths == (
        "nested/sample-a_train_id.png",
        "sample-b_train_id.png",
    )
    assert len(first.file_sha256) == 4
    assert first.file_sha256 == tuple(
        hashlib.sha256(value.encode()).hexdigest()
        for value in ("image-a", "label-a", "image-b", "label-b")
    )
    assert len(first.manifest_sha256) == 64


@pytest.mark.parametrize(
    ("remove_glob", "expected"),
    [
        ("*.png", r"^missing label for sample IDs: \['sample-b'\]$"),
        ("*.jpg", r"^extra label for sample IDs: \['sample-b'\]$"),
    ],
)
def test_build_manifest_rejects_missing_or_extra_pair(
    tmp_path: Path,
    remove_glob: str,
    expected: str,
) -> None:
    manifest = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    search_root = label_root if remove_glob == "*.png" else image_root
    next(search_root.rglob(remove_glob)).unlink()

    with pytest.raises(ValueError, match=expected):
        manifest.build_paired_manifest(image_root, label_root, "train")


@pytest.mark.parametrize("kind", ["image", "label"])
def test_build_manifest_rejects_duplicate_sample_id(tmp_path: Path, kind: str) -> None:
    manifest = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    if kind == "image":
        duplicate = image_root / "duplicate" / "sample-b.jpg"
        source = image_root / "sample-b.jpg"
    else:
        duplicate = label_root / "duplicate" / "sample-b_train_id.png"
        source = label_root / "sample-b_train_id.png"
    duplicate.parent.mkdir()
    duplicate.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match=r"^duplicate sample ID 'sample-b' under"):
        manifest.build_paired_manifest(image_root, label_root, "train")


def test_build_manifest_detects_checksum_drift(tmp_path: Path) -> None:
    manifest = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    before = manifest.build_paired_manifest(image_root, label_root, "train")
    (label_root / "sample-b_train_id.png").write_text("changed", encoding="utf-8")

    after = manifest.build_paired_manifest(image_root, label_root, "train")

    assert before.file_sha256 != after.file_sha256
    assert before.manifest_sha256 != after.manifest_sha256


@pytest.mark.parametrize("missing", ["images", "labels"])
def test_build_manifest_rejects_missing_root(tmp_path: Path, missing: str) -> None:
    manifest = load_manifest_module()
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    (label_root if missing == "images" else image_root).mkdir()

    with pytest.raises(ValueError, match=f"{missing[:-1]} root"):
        manifest.build_paired_manifest(image_root, label_root, "train")


def test_build_manifest_rejects_empty_split(tmp_path: Path) -> None:
    manifest = load_manifest_module()
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir()
    label_root.mkdir()

    with pytest.raises(ValueError, match=r"^no paired samples found"):
        manifest.build_paired_manifest(image_root, label_root, "train")


@pytest.mark.parametrize(
    "path",
    [
        "../escape.jpg",
        "/absolute.jpg",
        "C:/absolute.jpg",
        r"nested\windows.jpg",
        ".",
        "",
        "nested//double.jpg",
    ],
)
def test_manifest_rejects_nonportable_or_traversing_relative_path(path: str) -> None:
    manifest = load_manifest_module()

    with pytest.raises(ValueError, match=r"^expected a safe relative POSIX path, got '"):
        manifest.DatasetManifest(
            dataset_name="bdd100k",
            dataset_version="10k-semantic-v1",
            split_name="train",
            sample_ids=("sample-a",),
            relative_image_paths=(path,),
            relative_label_paths=("sample-a_train_id.png",),
            file_sha256=("a" * 64, "b" * 64),
            manifest_sha256="c" * 64,
        )


def test_manifest_rejects_misaligned_fields_or_invalid_hash() -> None:
    manifest = load_manifest_module()

    with pytest.raises(ValueError, match=r"^manifest sample, path, and file-hash fields must"):
        manifest.DatasetManifest(
            dataset_name="bdd100k",
            dataset_version="10k-semantic-v1",
            split_name="train",
            sample_ids=("a", "b"),
            relative_image_paths=("a.jpg",),
            relative_label_paths=("a_train_id.png",),
            file_sha256=("a" * 64, "b" * 64),
            manifest_sha256="c" * 64,
        )

    with pytest.raises(ValueError, match=r"^file SHA-256 values must be lowercase 64-digit hex"):
        manifest.DatasetManifest(
            dataset_name="bdd100k",
            dataset_version="10k-semantic-v1",
            split_name="train",
            sample_ids=("a",),
            relative_image_paths=("a.jpg",),
            relative_label_paths=("a_train_id.png",),
            file_sha256=("not-a-hash", "b" * 64),
            manifest_sha256="c" * 64,
        )


def test_manifest_rejects_duplicate_ids_and_manifest_hash_drift(tmp_path: Path) -> None:
    manifest = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    built = manifest.build_paired_manifest(image_root, label_root, "train")

    with pytest.raises(ValueError, match=r"^manifest contains duplicate sample IDs"):
        replace(built, sample_ids=("sample-a", "sample-a"))
    with pytest.raises(ValueError, match=r"^manifest SHA-256 must be lowercase 64-digit hex"):
        replace(built, manifest_sha256="X" * 64)
    with pytest.raises(ValueError, match=r"^manifest SHA-256 mismatch"):
        replace(built, manifest_sha256="0" * 64)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"dataset_name": ""}, r"^manifest identity fields must be nonempty$"),
        (
            {
                "sample_ids": (),
                "relative_image_paths": (),
                "relative_label_paths": (),
                "file_sha256": (),
            },
            r"^manifest must contain at least one sample$",
        ),
        (
            {"sample_ids": ("../sample-a", "sample-b")},
            r"^manifest sample IDs must be nonempty path-free names$",
        ),
    ],
)
def test_manifest_rejects_empty_identity_cohort_or_path_like_id(
    tmp_path: Path,
    changes: dict[str, object],
    expected: str,
) -> None:
    manifest = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    built = manifest.build_paired_manifest(image_root, label_root, "train")

    with pytest.raises(ValueError, match=expected):
        replace(built, **changes)


@pytest.mark.parametrize(
    ("function_name", "filename", "expected"),
    [
        ("image_sample_id", "sample-a.png", r"^unsupported BDD100K image filename: sample-a\.png$"),
        (
            "label_sample_id",
            "sample-a_color.png",
            r"^unsupported BDD100K train-ID label filename: sample-a_color\.png$",
        ),
    ],
)
def test_bdd100k_filename_parsers_reject_wrong_artifact_type(
    function_name: str,
    filename: str,
    expected: str,
) -> None:
    try:
        from drivemetrics.data import bdd100k
    except ImportError:
        pytest.fail("drivemetrics.data.bdd100k is missing", pytrace=False)

    with pytest.raises(ValueError, match=expected):
        getattr(bdd100k, function_name)(Path(filename))


def dump_manifest(manifest: Any, path: Path) -> Path:
    from dataclasses import asdict

    path.write_text(json.dumps(asdict(manifest), sort_keys=True), encoding="utf-8")
    return path


def test_a_frozen_manifest_document_round_trips_without_drift(tmp_path: Path) -> None:
    """Reading a manifest through a lossy path would break every downstream hash check."""

    module = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    original = module.build_paired_manifest(image_root, label_root, "train")

    restored = module.load_manifest(dump_manifest(original, tmp_path / "manifest.json"))

    assert restored == original


def test_a_manifest_document_that_is_not_a_mapping_fails_closed(tmp_path: Path) -> None:
    """A JSON list would otherwise reach the dataclass as positional garbage."""

    module = load_manifest_module()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(["not", "a", "manifest"]), encoding="utf-8")

    with pytest.raises(TypeError, match=r"^manifest document must be a mapping"):
        module.load_manifest(path)


def test_a_manifest_document_with_wrong_fields_fails_closed(tmp_path: Path) -> None:
    """A renamed or extra field would silently drop part of the frozen cohort identity."""

    module = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    original = module.build_paired_manifest(image_root, label_root, "train")
    path = dump_manifest(original, tmp_path / "manifest.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["unexpected_field"] = "value"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^manifest document must contain exactly the manifest"):
        module.load_manifest(path)


def test_a_tampered_manifest_document_fails_its_own_hash(tmp_path: Path) -> None:
    """Editing a frozen cohort by hand must never pass as the original manifest."""

    module = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    original = module.build_paired_manifest(image_root, label_root, "train")
    path = dump_manifest(original, tmp_path / "manifest.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["split_name"] = "locked_validation"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^manifest SHA-256 mismatch"):
        module.load_manifest(path)


def test_a_saved_manifest_reloads_as_the_same_frozen_cohort(tmp_path: Path) -> None:
    """A writer and reader that disagree would make every frozen cohort unverifiable."""

    module = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    original = module.build_paired_manifest(image_root, label_root, "train")
    path = tmp_path / "saved.json"

    module.save_manifest(original, path)

    assert module.load_manifest(path) == original
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_saving_over_an_existing_manifest_fails_closed(tmp_path: Path) -> None:
    """Silently replacing a frozen manifest would destroy the provenance of a locked run."""

    module = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    original = module.build_paired_manifest(image_root, label_root, "train")
    path = tmp_path / "saved.json"
    module.save_manifest(original, path)

    with pytest.raises(FileExistsError, match=r"^frozen manifest already exists:"):
        module.save_manifest(original, path)


def test_a_subset_keeps_each_sample_paired_with_its_own_file_hashes(tmp_path: Path) -> None:
    """Misaligned image and label hashes would silently validate the wrong file pair."""

    module = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    original = module.build_paired_manifest(image_root, label_root, "train")
    chosen = (original.sample_ids[1], original.sample_ids[0])

    subset = module.subset_manifest(original, chosen, "calibration")

    assert subset.split_name == "calibration"
    assert subset.sample_ids == chosen
    for position, sample_id in enumerate(chosen):
        source = original.sample_ids.index(sample_id)
        assert subset.relative_image_paths[position] == original.relative_image_paths[source]
        assert subset.relative_label_paths[position] == original.relative_label_paths[source]
        assert subset.file_sha256[2 * position] == original.file_sha256[2 * source]
        assert subset.file_sha256[2 * position + 1] == original.file_sha256[2 * source + 1]
    assert subset.manifest_sha256 != original.manifest_sha256


def test_a_subset_of_an_unknown_sample_fails_closed(tmp_path: Path) -> None:
    """Inventing a sample would produce a cohort no file on disk can support."""

    module = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    original = module.build_paired_manifest(image_root, label_root, "train")

    with pytest.raises(ValueError, match=r"^sample IDs are not present in the source manifest:"):
        module.subset_manifest(original, ("missing-sample",), "calibration")


def test_a_subset_with_duplicate_samples_fails_closed(tmp_path: Path) -> None:
    """A duplicated sample would double its weight in every aggregated metric."""

    module = load_manifest_module()
    image_root, label_root = materialize_fixture(tmp_path)
    original = module.build_paired_manifest(image_root, label_root, "train")
    duplicated = (original.sample_ids[0], original.sample_ids[0])

    with pytest.raises(ValueError, match=r"^manifest subset contains duplicate sample IDs"):
        module.subset_manifest(original, duplicated, "calibration")
