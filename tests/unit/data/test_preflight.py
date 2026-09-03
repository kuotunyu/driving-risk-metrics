"""Contracts for the dataset preflight that freezes the three formal cohorts."""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

TRAIN_IMAGES = "images/10k/train"
TRAIN_LABELS = "labels/sem_seg/masks/train"
VALIDATION_IMAGES = "images/10k/val"
VALIDATION_LABELS = "labels/sem_seg/masks/val"

MANIFEST_NAMES = ("source_train", "train", "calibration", "locked_validation")


def load_preflight_module() -> ModuleType:
    try:
        from drivemetrics.data import preflight
    except ImportError:
        pytest.fail("drivemetrics.data.preflight is missing", pytrace=False)
    return preflight


def _encoded(size: tuple[int, int], fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB" if fmt == "JPEG" else "L", size).save(buffer, fmt)
    return buffer.getvalue()


TINY_JPEG = _encoded((8, 4), "JPEG")
TINY_PNG = _encoded((8, 4), "PNG")


def write_pairs(data_root: Path, image_dir: str, label_dir: str, count: int) -> None:
    """Create one tiny paired image and train-ID label per synthetic sample.

    The files are real 8x4 encodings because the preflight reads image headers
    to compare image and label geometry.
    """

    images = data_root / image_dir
    labels = data_root / label_dir
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    prefix = "t" if "train" in image_dir else "v"
    for index in range(count):
        sample_id = f"{prefix}{index:05d}"
        (images / f"{sample_id}.jpg").write_bytes(TINY_JPEG)
        (labels / f"{sample_id}_train_id.png").write_bytes(TINY_PNG)


def build_workspace(
    tmp_path: Path,
    *,
    train_count: int,
    validation_count: int,
) -> tuple[Path, Path, Path]:
    config_path = tmp_path / "protocol.yaml"
    shutil.copyfile(PROTOCOL_SOURCE, config_path)
    data_root = tmp_path / "data"
    write_pairs(data_root, TRAIN_IMAGES, TRAIN_LABELS, train_count)
    write_pairs(data_root, VALIDATION_IMAGES, VALIDATION_LABELS, validation_count)
    return config_path, data_root, tmp_path / "manifests"


@pytest.fixture(scope="module")
def frozen_cohort(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, Path]:
    """Run one real-size preflight once and share it across the success contracts."""

    preflight = load_preflight_module()
    tmp_path = tmp_path_factory.mktemp("preflight")
    config_path, data_root, output_dir = build_workspace(
        tmp_path,
        train_count=7000,
        validation_count=1000,
    )
    return preflight.run_preflight(config_path, data_root, output_dir), output_dir


def test_the_preflight_freezes_exactly_the_protocol_cohort_sizes(
    frozen_cohort: tuple[object, Path],
) -> None:
    """A cohort of the wrong size would silently change the locked experiment."""

    result, _ = frozen_cohort

    assert result.counts == {  # type: ignore[attr-defined]
        "source_train": 7000,
        "train": 6300,
        "calibration": 700,
        "locked_validation": 1000,
    }


def test_every_frozen_manifest_is_written_and_reloadable(
    frozen_cohort: tuple[object, Path],
) -> None:
    """A manifest that cannot be reloaded could never support a reproducible run."""

    from drivemetrics.data.manifest import load_manifest

    result, output_dir = frozen_cohort

    for name in MANIFEST_NAMES:
        path = result.manifest_paths[name]  # type: ignore[attr-defined]
        assert path.parent == output_dir
        reloaded = load_manifest(path)
        assert reloaded.split_name == name
        assert reloaded.manifest_sha256 == result.manifest_sha256[name]  # type: ignore[attr-defined]


def test_the_frozen_cohorts_never_share_a_sample(
    frozen_cohort: tuple[object, Path],
) -> None:
    """Any overlap would leak calibration or locked-validation data into training."""

    from drivemetrics.data.manifest import load_manifest

    result, _ = frozen_cohort
    cohorts = {
        name: set(load_manifest(result.manifest_paths[name]).sample_ids)  # type: ignore[attr-defined]
        for name in ("train", "calibration", "locked_validation")
    }

    assert not cohorts["train"] & cohorts["calibration"]
    assert not cohorts["train"] & cohorts["locked_validation"]
    assert not cohorts["calibration"] & cohorts["locked_validation"]


def test_the_protocol_hash_is_reported_for_the_run_record(
    frozen_cohort: tuple[object, Path],
) -> None:
    """Without the protocol hash a frozen cohort cannot be tied to its experiment."""

    from drivemetrics.protocol.config import load_protocol

    result, _ = frozen_cohort
    expected = load_protocol(PROTOCOL_SOURCE).protocol_sha256

    assert result.protocol_sha256 == expected  # type: ignore[attr-defined]


def test_an_existing_frozen_manifest_stops_the_preflight_before_any_scan(
    tmp_path: Path,
) -> None:
    """Re-freezing over a published cohort would invalidate every claim that cites it."""

    preflight = load_preflight_module()
    config_path, data_root, output_dir = build_workspace(
        tmp_path,
        train_count=2,
        validation_count=2,
    )
    output_dir.mkdir(parents=True)
    (output_dir / "train.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"^frozen manifest already exists:"):
        preflight.run_preflight(config_path, data_root, output_dir)


@pytest.mark.parametrize(
    "removed",
    [TRAIN_IMAGES, TRAIN_LABELS, VALIDATION_IMAGES, VALIDATION_LABELS],
)
def test_a_missing_dataset_directory_fails_closed(tmp_path: Path, removed: str) -> None:
    """A silently empty split would produce a cohort with no evidence behind it."""

    preflight = load_preflight_module()
    config_path, data_root, output_dir = build_workspace(
        tmp_path,
        train_count=2,
        validation_count=2,
    )
    shutil.rmtree(data_root / removed)

    with pytest.raises(ValueError, match=r"^dataset path is not a directory:"):
        preflight.run_preflight(config_path, data_root, output_dir)


def test_a_locked_validation_count_mismatch_fails_closed(tmp_path: Path) -> None:
    """A short locked split would change the denominator of every reported metric."""

    preflight = load_preflight_module()
    config_path, data_root, output_dir = build_workspace(
        tmp_path,
        train_count=2,
        validation_count=3,
    )

    with pytest.raises(ValueError, match=r"^locked_validation split must contain exactly"):
        preflight.run_preflight(config_path, data_root, output_dir)


def test_a_source_train_count_mismatch_fails_closed(tmp_path: Path) -> None:
    """A partial download must never be frozen as if it were the official split."""

    preflight = load_preflight_module()
    config_path, data_root, output_dir = build_workspace(
        tmp_path,
        train_count=3,
        validation_count=1000,
    )

    with pytest.raises(ValueError, match=r"^source_train split must contain exactly"):
        preflight.run_preflight(config_path, data_root, output_dir)


def test_data_package_exports_the_preflight_entry_points() -> None:
    """The CLI consumes the preflight through the package entry point."""

    import drivemetrics.data as data

    preflight = load_preflight_module()
    assert data.run_preflight is preflight.run_preflight
    assert data.PreflightResult is preflight.PreflightResult
