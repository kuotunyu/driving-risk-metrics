"""Contracts for scoring a checkpoint through a fitted scalar temperature.

Temperature scaling must change how confident a prediction claims to be without
changing which class it predicts. If it moved the argmax it would no longer be
calibration, it would be a different model, and every accuracy number computed
beside it would describe something the uncalibrated run never produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from drivemetrics.artifacts.predictions import read_prediction_artifact
from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR
from drivemetrics.calibration.service import calibrate_checkpoint
from drivemetrics.data.manifest import build_paired_manifest, save_manifest
from drivemetrics.evaluation.engine import evaluate_checkpoint
from drivemetrics.protocol.config import load_protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

SOURCE_HEIGHT = 64
SOURCE_WIDTH = 128
NUM_CLASSES = 19
PREDICTED = 3

PROVENANCE = {
    "commit": "1" * 40,
    "lock_sha256": "2" * 64,
    "hardware": {"gpu": "cpu-only", "runtime": "pytest"},
}


class OverconfidentModel:
    def logits(self, image_nchw: np.ndarray) -> np.ndarray:
        batch, _, height, width = image_nchw.shape
        values = np.full((batch, NUM_CLASSES, height, width), -6.0, dtype=np.float64)
        values[:, PREDICTED] = 6.0
        return values

    def trainable_parameters(self) -> object:
        return ("weight",)


class FakeBackend:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def load_model(self, checkpoint_path: Path) -> tuple[OverconfidentModel, dict[str, Any]]:
        return OverconfidentModel(), dict(self.metadata)


def build(tmp_path: Path, split_name: str) -> dict[str, Any]:
    data_root = tmp_path / "data"
    subdir = "train" if split_name == "calibration" else "val"
    images = data_root / f"images/10k/{subdir}"
    labels = data_root / f"labels/sem_seg/masks/{subdir}"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(11)
    for index in range(2):
        sample_id = f"{split_name[:3]}{index:04d}"
        Image.fromarray(
            rng.integers(0, 256, (SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)
        ).save(images / f"{sample_id}.jpg")
        mask = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), PREDICTED, dtype=np.uint8)
        mask[: SOURCE_HEIGHT // 4] = (PREDICTED + 1) % NUM_CLASSES
        Image.fromarray(mask).save(labels / f"{sample_id}_train_id.png")

    manifest = build_paired_manifest(images, labels, split_name)
    manifest_path = tmp_path / f"{split_name}.json"
    save_manifest(manifest, manifest_path)
    return {"data_root": data_root, "manifest": manifest_path, "manifest_obj": manifest}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))
    protocol_dir = tmp_path / "configs" / "protocols"
    protocol_dir.mkdir(parents=True)
    protocol = protocol_dir / "bdd100k_semseg_v1.yaml"
    protocol.write_text(PROTOCOL_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    protocol_hash = load_protocol(protocol).protocol_sha256

    checkpoint = tmp_path / "final_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    metadata = {
        "model": "upernet_convnextv2_tiny",
        "run_id": "upernet_convnextv2_tiny-seed-17",
        "seed": 17,
        "protocol_sha256": protocol_hash,
        "final_step": 30000,
    }

    calibration = build(tmp_path / "cal", "calibration")
    locked = build(tmp_path / "val", "locked_validation")

    fitted = calibrate_checkpoint(
        protocol,
        calibration["manifest"],
        checkpoint,
        calibration["data_root"],
        tmp_path / "calibration-out",
        backend=FakeBackend(metadata),
    )
    return {
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "checkpoint": checkpoint,
        "metadata": metadata,
        "locked": locked,
        "temperature_path": fitted.artifact_path,
        "temperature": fitted.temperature,
        "tmp_path": tmp_path,
    }


def evaluate(workspace: dict[str, Any], output: Path, temperature: Path | None) -> Any:
    return evaluate_checkpoint(
        workspace["protocol"],
        workspace["locked"]["manifest"],
        workspace["checkpoint"],
        workspace["locked"]["data_root"],
        output,
        backend=FakeBackend(workspace["metadata"]),
        temperature_path=temperature,
    )


def test_calibration_changes_confidence_but_never_the_prediction(
    workspace: dict[str, Any],
) -> None:
    """A temperature that moved the argmax would be a different model, not calibration."""

    tmp_path = workspace["tmp_path"]
    raw = evaluate(workspace, tmp_path / "raw", None)
    calibrated = evaluate(workspace, tmp_path / "calibrated", workspace["temperature_path"])

    _, raw_record, _ = read_prediction_artifact(raw.artifact_paths[0])
    _, calibrated_record, _ = read_prediction_artifact(calibrated.artifact_paths[0])

    assert np.array_equal(raw_record.predicted_class, calibrated_record.predicted_class)
    assert raw_record.correctness_bitset == calibrated_record.correctness_bitset
    assert not np.array_equal(raw_record.top1_confidence_q16, calibrated_record.top1_confidence_q16)


def test_softening_an_overconfident_model_lowers_its_confidence(
    workspace: dict[str, Any],
) -> None:
    """A temperature above one must reduce confidence, or it is applied inverted."""

    tmp_path = workspace["tmp_path"]
    assert workspace["temperature"] > 1.0

    raw = evaluate(workspace, tmp_path / "raw", None)
    calibrated = evaluate(workspace, tmp_path / "calibrated", workspace["temperature_path"])

    _, raw_record, _ = read_prediction_artifact(raw.artifact_paths[0])
    _, calibrated_record, _ = read_prediction_artifact(calibrated.artifact_paths[0])

    assert calibrated_record.top1_confidence_q16.mean() < raw_record.top1_confidence_q16.mean()


def test_a_temperature_from_another_protocol_is_refused(workspace: dict[str, Any]) -> None:
    """Applying a temperature fitted under a different protocol is not comparable."""

    tmp_path = workspace["tmp_path"]
    document = json.loads(workspace["temperature_path"].read_text(encoding="utf-8"))
    document["protocol_sha256"] = "f" * 64
    foreign = tmp_path / "foreign-temperature.json"
    foreign.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"^temperature protocol hash does not match the evaluation"
    ):
        evaluate(workspace, tmp_path / "out", foreign)


def test_a_temperature_from_another_checkpoint_is_refused(workspace: dict[str, Any]) -> None:
    """A temperature belongs to the exact weights it was fitted on."""

    tmp_path = workspace["tmp_path"]
    document = json.loads(workspace["temperature_path"].read_text(encoding="utf-8"))
    document["checkpoint_sha256"] = "a" * 64
    foreign = tmp_path / "foreign-temperature.json"
    foreign.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^temperature was fitted for a different checkpoint"):
        evaluate(workspace, tmp_path / "out", foreign)


def test_a_malformed_temperature_artifact_fails_closed(workspace: dict[str, Any]) -> None:
    """A missing schema version means the file is not what the caller believes."""

    tmp_path = workspace["tmp_path"]
    broken = tmp_path / "broken-temperature.json"
    broken.write_text(json.dumps({"temperature": 1.5}), encoding="utf-8")

    with pytest.raises(
        (ValueError, KeyError),
        match=r"^temperature artifact must declare drivemetrics-temperature/v1, got ",
    ):
        evaluate(workspace, tmp_path / "out", broken)


def test_a_non_positive_temperature_is_refused(workspace: dict[str, Any]) -> None:
    """Dividing logits by zero or a negative value is not a calibration."""

    tmp_path = workspace["tmp_path"]
    document = json.loads(workspace["temperature_path"].read_text(encoding="utf-8"))
    document["temperature"] = 0.0
    broken = tmp_path / "zero-temperature.json"
    broken.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^temperature must be finite and positive, got"):
        evaluate(workspace, tmp_path / "out", broken)


def test_a_temperature_artifact_that_is_not_an_object_fails_closed(
    workspace: dict[str, Any],
) -> None:
    """A JSON list would reach the binding checks as positional garbage."""

    tmp_path = workspace["tmp_path"]
    listed = tmp_path / "list-temperature.json"
    listed.write_text(json.dumps([1.5]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^temperature artifact must be a JSON object"):
        evaluate(workspace, tmp_path / "out", listed)
