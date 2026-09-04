"""Contracts for source-defective pairs: an image whose geometry disagrees with its label.

BDD100K 10K ships a few frames stored as 720x1280 portraits next to 1280x720
labels, with no EXIF orientation and no rigid transform that aligns them. Such a
pair carries no usable supervision and no scorable prediction. The rule that
removes it must be mechanical, label-blind, and applied after cohort assignment,
so that no other membership moves and the exclusion itself is on record.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

TRAIN_IMAGES = "images/10k/train"
TRAIN_LABELS = "labels/sem_seg/masks/train"
VALIDATION_IMAGES = "images/10k/val"
VALIDATION_LABELS = "labels/sem_seg/masks/val"
REASON = "image 4x8 but label 8x4"


def encoded(size: tuple[int, int], fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB" if fmt == "JPEG" else "L", size).save(buffer, fmt)
    return buffer.getvalue()


LANDSCAPE_JPEG = encoded((8, 4), "JPEG")
PORTRAIT_JPEG = encoded((4, 8), "JPEG")
LANDSCAPE_PNG = encoded((8, 4), "PNG")


def write_pairs(
    data_root: Path,
    image_dir: str,
    label_dir: str,
    count: int,
    portrait: tuple[str, ...] = (),
) -> None:
    images = data_root / image_dir
    labels = data_root / label_dir
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    prefix = "t" if "train" in image_dir else "v"
    for index in range(count):
        sample_id = f"{prefix}{index:05d}"
        image = PORTRAIT_JPEG if sample_id in portrait else LANDSCAPE_JPEG
        (images / f"{sample_id}.jpg").write_bytes(image)
        (labels / f"{sample_id}_train_id.png").write_bytes(LANDSCAPE_PNG)


def small_manifest(tmp_path: Path) -> tuple[Any, Path, Path]:
    from drivemetrics.data.manifest import build_paired_manifest

    write_pairs(tmp_path, TRAIN_IMAGES, TRAIN_LABELS, 3, portrait=("t00001",))
    image_root, label_root = tmp_path / TRAIN_IMAGES, tmp_path / TRAIN_LABELS
    return build_paired_manifest(image_root, label_root, "source_train"), image_root, label_root


def test_a_pair_whose_image_and_label_geometry_disagree_is_named_with_both_sizes(
    tmp_path: Path,
) -> None:
    """The reason must let a reader verify the defect from the files alone."""

    from drivemetrics.data.preflight import pair_geometry_reasons

    manifest, image_root, label_root = small_manifest(tmp_path)

    assert pair_geometry_reasons(manifest, image_root, label_root) == {"t00001": REASON}


def test_marking_pairs_ineligible_removes_them_from_the_eligible_list_and_rehashes(
    tmp_path: Path,
) -> None:
    """An ineligible pair is named, not silently dropped, and the hash binds that record."""

    from drivemetrics.data.manifest import load_manifest, mark_ineligible, save_manifest

    manifest, _, _ = small_manifest(tmp_path)

    marked = mark_ineligible(manifest, {"t00001": REASON})

    assert marked.sample_ids == ("t00000", "t00002")
    assert len(marked.relative_image_paths) == len(marked.relative_label_paths) == 2
    assert marked.file_sha256 == manifest.file_sha256[0:2] + manifest.file_sha256[4:6]
    assert marked.ineligible_sample_ids == ("t00001",)
    assert marked.ineligibility_reasons == (REASON,)
    assert marked.manifest_sha256 != manifest.manifest_sha256
    save_manifest(marked, tmp_path / "marked.json")
    assert load_manifest(tmp_path / "marked.json") == marked


@pytest.mark.parametrize(
    ("reasons", "message"),
    [
        ({"nobody": REASON}, r"^sample IDs are not present in the manifest: \('nobody',\)$"),
        ({"t00001": ""}, r"^every ineligible sample needs a reason$"),
    ],
)
def test_marking_an_unknown_or_unreasoned_pair_fails_closed(
    tmp_path: Path, reasons: dict[str, str], message: str
) -> None:
    from drivemetrics.data.manifest import mark_ineligible

    manifest, _, _ = small_manifest(tmp_path)

    with pytest.raises(ValueError, match=message):
        mark_ineligible(manifest, reasons)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ineligibility_reasons": ()}, r"^ineligible sample IDs and reasons must be aligned$"),
        (
            {
                "ineligible_sample_ids": ("t00001", "t00001"),
                "ineligibility_reasons": (REASON, REASON),
            },
            r"^manifest contains duplicate ineligible sample IDs$",
        ),
        (
            {"ineligible_sample_ids": ("t00000",)},
            r"^a sample cannot be both eligible and ineligible$",
        ),
        ({"ineligibility_reasons": ("",)}, r"^every ineligible sample needs a reason$"),
        (
            {"ineligible_sample_ids": ("a/b",)},
            r"^manifest sample IDs must be nonempty path-free names$",
        ),
    ],
)
def test_a_manifest_with_an_inconsistent_ineligible_record_fails_closed(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    """A defect record that cannot be trusted is worse than none."""

    from dataclasses import replace

    from drivemetrics.data.manifest import mark_ineligible

    manifest, _, _ = small_manifest(tmp_path)
    marked = mark_ineligible(manifest, {"t00001": REASON})

    with pytest.raises(ValueError, match=message):
        replace(marked, **changes)


def test_a_subset_carries_the_ineligible_members_it_was_assigned(tmp_path: Path) -> None:
    """Cohort assignment covers every source ID; the defect travels with its cohort."""

    from drivemetrics.data.manifest import mark_ineligible, subset_manifest

    manifest, _, _ = small_manifest(tmp_path)
    marked = mark_ineligible(manifest, {"t00001": REASON})

    with_defect = subset_manifest(marked, ("t00001", "t00002"), "train")
    without_defect = subset_manifest(marked, ("t00000",), "calibration")

    assert with_defect.sample_ids == ("t00002",)
    assert with_defect.ineligible_sample_ids == ("t00001",)
    assert with_defect.ineligibility_reasons == (REASON,)
    assert without_defect.sample_ids == ("t00000",)
    assert without_defect.ineligible_sample_ids == ()


@pytest.fixture(scope="module")
def frozen_with_defects(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One real-size preflight over a dataset with two train defects and one locked defect."""

    from drivemetrics.data import preflight

    tmp_path = tmp_path_factory.mktemp("defects")
    config_path = tmp_path / "protocol.yaml"
    shutil.copyfile(PROTOCOL_SOURCE, config_path)
    data_root = tmp_path / "data"
    write_pairs(data_root, TRAIN_IMAGES, TRAIN_LABELS, 7000, portrait=("t00003", "t00777"))
    write_pairs(data_root, VALIDATION_IMAGES, VALIDATION_LABELS, 1000, portrait=("v00001",))
    return preflight.run_preflight(config_path, data_root, tmp_path / "manifests")


def test_the_preflight_excludes_defects_after_assignment_without_moving_anyone_else(
    frozen_with_defects: Any,
) -> None:
    """The SHA-256 partition is computed over all 7,000 IDs; defects leave, nobody moves."""

    from drivemetrics.data.manifest import load_manifest
    from drivemetrics.data.splits import freeze_bdd100k_split

    result = frozen_with_defects
    all_ids = tuple(f"t{index:05d}" for index in range(7000))
    expected_train, expected_calibration = freeze_bdd100k_split(all_ids)
    defects = {"t00003", "t00777"}
    train_defects = defects & set(expected_train)
    calibration_defects = defects & set(expected_calibration)

    assert result.counts == {
        "source_train": 6998,
        "train": 6300 - len(train_defects),
        "calibration": 700 - len(calibration_defects),
        "locked_validation": 999,
    }
    assert result.ineligible == {
        "source_train": {"t00003": REASON, "t00777": REASON},
        "train": dict.fromkeys(sorted(train_defects), REASON),
        "calibration": dict.fromkeys(sorted(calibration_defects), REASON),
        "locked_validation": {"v00001": REASON},
    }
    train = load_manifest(result.manifest_paths["train"])
    calibration = load_manifest(result.manifest_paths["calibration"])
    locked = load_manifest(result.manifest_paths["locked_validation"])
    assert set(train.sample_ids) == set(expected_train) - defects
    assert set(calibration.sample_ids) == set(expected_calibration) - defects
    assert set(train.ineligible_sample_ids) == train_defects
    assert set(calibration.ineligible_sample_ids) == calibration_defects
    assert locked.ineligible_sample_ids == ("v00001",)
    assert "v00001" not in locked.sample_ids
    assert len(locked.sample_ids) == 999


def test_the_preflight_status_reports_the_ineligible_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notebook parses the status; a defect list it cannot see is one it cannot show."""

    from typer.testing import CliRunner

    import drivemetrics.cli.data as data_cli
    from drivemetrics.cli.app import app

    stub = SimpleNamespace(
        protocol_sha256="a" * 64,
        counts={"train": 6299},
        manifest_sha256={"train": "b" * 64},
        manifest_paths={"train": tmp_path / "train.json"},
        ineligible={"train": {"t00003": REASON}},
    )
    monkeypatch.setattr(data_cli, "PREFLIGHT_SERVICE", lambda *_: stub)
    (tmp_path / "protocol.yaml").write_text("", encoding="utf-8")
    (tmp_path / "data").mkdir()

    result = CliRunner().invoke(
        app,
        [
            "data",
            "preflight",
            "--config",
            str(tmp_path / "protocol.yaml"),
            "--data-root",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "manifests"),
        ],
    )

    assert result.exit_code == 0, result.output
    status = json.loads(result.stdout.strip().splitlines()[-1])
    assert status["ineligible"] == {"train": {"t00003": REASON}}
