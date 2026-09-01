"""Contracts for the injected-backend locked-cohort evaluation service."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image

from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR
from drivemetrics.data.manifest import build_paired_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

VALIDATION_IMAGES = "images/10k/val"
VALIDATION_LABELS = "labels/sem_seg/masks/val"
TRAIN_IMAGES = "images/10k/train"
TRAIN_LABELS = "labels/sem_seg/masks/train"

SOURCE_HEIGHT = 90
SOURCE_WIDTH = 160
NUM_CLASSES = 4
PREDICTED_CLASS = 1

PROVENANCE = {
    "commit": "e" * 40,
    "lock_sha256": "f" * 64,
    "hardware": {"gpu": "cpu-only", "runtime": "pytest"},
}


def load_evaluation_module() -> ModuleType:
    try:
        from drivemetrics.evaluation import engine
    except ImportError:
        pytest.fail("drivemetrics.evaluation.engine is missing", pytrace=False)
    return engine


class FakeModel:
    """Return the same confident class everywhere, so every metric is hand computable."""

    def __init__(self) -> None:
        self.calls = 0

    def logits(self, image_nchw: np.ndarray) -> np.ndarray:
        self.calls += 1
        batch, _, height, width = image_nchw.shape
        values = np.zeros((batch, NUM_CLASSES, height, width), dtype=np.float64)
        values[:, PREDICTED_CLASS] = np.log(3.0)
        return values

    def trainable_parameters(self) -> object:
        return ("weight",)


class FakeBackend:
    """Load a model and the metadata that the training engine recorded with it."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.model = FakeModel()
        self.loaded: list[Path] = []

    def load_model(self, checkpoint_path: Path) -> tuple[FakeModel, dict[str, Any]]:
        self.loaded.append(checkpoint_path)
        return self.model, dict(self.metadata)


def write_sample(
    data_root: Path,
    sample_id: str,
    mask: np.ndarray,
    *,
    split_name: str = "locked_validation",
) -> None:
    validation = split_name == "locked_validation"
    images = data_root / (VALIDATION_IMAGES if validation else TRAIN_IMAGES)
    labels = data_root / (VALIDATION_LABELS if validation else TRAIN_LABELS)
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(images / f"{sample_id}.jpg")
    Image.fromarray(mask).save(labels / f"{sample_id}_train_id.png")


def default_mask(ignored_columns: int = 0) -> np.ndarray:
    """Half the columns hold the predicted class and half hold another class."""

    mask = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), PREDICTED_CLASS, dtype=np.uint8)
    mask[:, SOURCE_WIDTH // 2 :] = PREDICTED_CLASS + 1
    if ignored_columns:
        mask[:, :ignored_columns] = 255
    return mask


def build_workspace(
    tmp_path: Path,
    *,
    sample_ids: tuple[str, ...] = ("v00001", "v00002"),
    ignored_columns: int = 0,
    split_name: str = "locked_validation",
) -> tuple[Path, Path, Path, Path]:
    config_path = tmp_path / "protocol.yaml"
    shutil.copyfile(PROTOCOL_SOURCE, config_path)
    data_root = tmp_path / "data"
    for sample_id in sample_ids:
        write_sample(data_root, sample_id, default_mask(ignored_columns), split_name=split_name)

    validation = split_name == "locked_validation"
    manifest = build_paired_manifest(
        data_root / (VALIDATION_IMAGES if validation else TRAIN_IMAGES),
        data_root / (VALIDATION_LABELS if validation else TRAIN_LABELS),
        split_name,
    )
    manifest_path = tmp_path / f"{split_name}.json"
    manifest_path.write_text(
        json.dumps(dataclasses.asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )
    return config_path, manifest_path, data_root, tmp_path / "artifacts"


def checkpoint_metadata(config_path: Path) -> dict[str, Any]:
    from drivemetrics.protocol.config import load_protocol

    return {
        "run_id": "fcn_resnet50-seed-17",
        "model": "fcn_resnet50",
        "seed": 17,
        "final_step": 30000,
        "protocol_sha256": load_protocol(config_path).protocol_sha256,
    }


@pytest.fixture
def provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))


def test_one_prediction_artifact_is_published_for_every_manifest_sample(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A skipped image would change the denominator of every locked-cohort metric."""

    engine = load_evaluation_module()
    config_path, manifest_path, data_root, output_dir = build_workspace(tmp_path)
    backend = FakeBackend(checkpoint_metadata(config_path))

    result = engine.evaluate_checkpoint(
        config_path,
        manifest_path,
        tmp_path / "checkpoint.pt",
        data_root,
        output_dir,
        backend=backend,
    )

    assert result.evaluated_samples == 2
    assert len(result.artifact_paths) == 2
    assert [path.stem for path in result.artifact_paths] == ["v00001", "v00002"]
    assert all(path.exists() for path in result.artifact_paths)
    assert backend.model.calls == 2


def test_the_published_confusion_matches_the_hand_computed_counts(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A transposed or resampled confusion would misreport every per-class rate.

    The model always predicts class 1. Half of the 90 by 160 mask is class 1 and
    half is class 2, so exactly 7,200 pixels are correct and 7,200 are counted as
    true class 2 predicted as class 1.
    """

    from drivemetrics.artifacts.predictions import read_prediction_artifact

    engine = load_evaluation_module()
    config_path, manifest_path, data_root, output_dir = build_workspace(
        tmp_path,
        sample_ids=("v00001",),
    )
    backend = FakeBackend(checkpoint_metadata(config_path))

    result = engine.evaluate_checkpoint(
        config_path,
        manifest_path,
        tmp_path / "checkpoint.pt",
        data_root,
        output_dir,
        backend=backend,
    )
    _, record, _ = read_prediction_artifact(result.artifact_paths[0])

    assert record.valid_pixel_count == SOURCE_HEIGHT * SOURCE_WIDTH
    assert int(record.confusion[PREDICTED_CLASS, PREDICTED_CLASS]) == 7200
    assert int(record.confusion[PREDICTED_CLASS + 1, PREDICTED_CLASS]) == 7200
    assert int(record.confusion.sum()) == SOURCE_HEIGHT * SOURCE_WIDTH
    assert set(np.unique(record.predicted_class)) == {PREDICTED_CLASS}


def test_ignored_pixels_are_excluded_from_every_stored_array(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Storing padded or ignored pixels would score labels that do not exist."""

    from drivemetrics.artifacts.predictions import read_prediction_artifact

    engine = load_evaluation_module()
    config_path, manifest_path, data_root, output_dir = build_workspace(
        tmp_path,
        sample_ids=("v00001",),
        ignored_columns=40,
    )
    backend = FakeBackend(checkpoint_metadata(config_path))

    result = engine.evaluate_checkpoint(
        config_path,
        manifest_path,
        tmp_path / "checkpoint.pt",
        data_root,
        output_dir,
        backend=backend,
    )
    _, record, _ = read_prediction_artifact(result.artifact_paths[0])

    expected_valid = SOURCE_HEIGHT * (SOURCE_WIDTH - 40)
    assert record.valid_pixel_count == expected_valid
    assert record.predicted_class.size == expected_valid
    assert int(record.confusion.sum()) == expected_valid


def test_a_checkpoint_from_another_protocol_fails_closed(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Evaluating under a different protocol would silently compare incomparable runs."""

    engine = load_evaluation_module()
    config_path, manifest_path, data_root, output_dir = build_workspace(tmp_path)
    metadata = checkpoint_metadata(config_path)
    metadata["protocol_sha256"] = "0" * 64
    backend = FakeBackend(metadata)

    with pytest.raises(ValueError, match="protocol"):
        engine.evaluate_checkpoint(
            config_path,
            manifest_path,
            tmp_path / "checkpoint.pt",
            data_root,
            output_dir,
            backend=backend,
        )


def test_a_dataset_file_that_drifted_from_the_manifest_fails_closed(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A silently edited image would produce evidence the frozen cohort cannot support."""

    engine = load_evaluation_module()
    config_path, manifest_path, data_root, output_dir = build_workspace(
        tmp_path,
        sample_ids=("v00001",),
    )
    replacement = np.full((SOURCE_HEIGHT, SOURCE_WIDTH, 3), 200, dtype=np.uint8)
    Image.fromarray(replacement).save(data_root / VALIDATION_IMAGES / "v00001.jpg")
    backend = FakeBackend(checkpoint_metadata(config_path))

    with pytest.raises(ValueError, match="SHA-256"):
        engine.evaluate_checkpoint(
            config_path,
            manifest_path,
            tmp_path / "checkpoint.pt",
            data_root,
            output_dir,
            backend=backend,
        )


def test_missing_run_provenance_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An evaluation without provenance could never be tied to a commit or machine."""

    engine = load_evaluation_module()
    monkeypatch.delenv(PROVENANCE_ENV_VAR, raising=False)
    config_path, manifest_path, data_root, output_dir = build_workspace(tmp_path)
    backend = FakeBackend(checkpoint_metadata(config_path))

    with pytest.raises(ValueError, match=PROVENANCE_ENV_VAR):
        engine.evaluate_checkpoint(
            config_path,
            manifest_path,
            tmp_path / "checkpoint.pt",
            data_root,
            output_dir,
            backend=backend,
        )


def test_the_run_record_carries_the_evaluated_checkpoint_identity(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Without the evaluated run identity a metric table cannot name its own model."""

    engine = load_evaluation_module()
    config_path, manifest_path, data_root, output_dir = build_workspace(tmp_path)
    backend = FakeBackend(checkpoint_metadata(config_path))

    result = engine.evaluate_checkpoint(
        config_path,
        manifest_path,
        tmp_path / "checkpoint.pt",
        data_root,
        output_dir,
        backend=backend,
    )
    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))

    assert record["status"] == "succeeded"
    assert record["run_id"] == "eval-fcn_resnet50-seed-17"
    assert record["seed"] == 17
    assert record["commit"] == PROVENANCE["commit"]
    assert len(record["artifacts"]) == 2


def test_evaluation_package_exports_the_public_entry_points() -> None:
    """The CLI consumes the evaluation service through the package entry point."""

    import drivemetrics.evaluation as evaluation

    engine = load_evaluation_module()
    assert evaluation.evaluate_checkpoint is engine.evaluate_checkpoint
    assert evaluation.EvaluationResult is engine.EvaluationResult


def test_a_calibration_cohort_is_read_from_the_training_directories(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Reading the calibration cohort from the validation tree would evaluate the wrong files."""

    engine = load_evaluation_module()
    config_path, manifest_path, data_root, output_dir = build_workspace(
        tmp_path,
        sample_ids=("t00001",),
        split_name="calibration",
    )
    backend = FakeBackend(checkpoint_metadata(config_path))

    result = engine.evaluate_checkpoint(
        config_path,
        manifest_path,
        tmp_path / "checkpoint.pt",
        data_root,
        output_dir,
        backend=backend,
    )

    assert result.evaluated_samples == 1
    assert result.artifact_paths[0].stem == "t00001"
