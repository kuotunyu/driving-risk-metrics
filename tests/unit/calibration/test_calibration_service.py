"""Contracts for fitting a scalar temperature on the frozen calibration cohort.

Full per-pixel logits for the formal cohort are infeasible: 700 images at
1280x720 by 19 classes in float64 is roughly 98 GB. The service therefore fits
on a deterministic seeded pixel sample per image. That makes the sampling rule
part of the evidence, so it is recorded in the artifact and pinned here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image

from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR
from drivemetrics.data.manifest import build_paired_manifest, save_manifest
from drivemetrics.protocol.config import load_protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

SOURCE_HEIGHT = 64
SOURCE_WIDTH = 128
NUM_CLASSES = 19
CONFIDENT_CLASS = 3

PROVENANCE = {
    "commit": "1" * 40,
    "lock_sha256": "2" * 64,
    "hardware": {"gpu": "cpu-only", "runtime": "pytest"},
}


def load_service() -> ModuleType:
    try:
        from drivemetrics.calibration import service
    except ImportError:
        pytest.fail("drivemetrics.calibration.service is missing", pytrace=False)
    return service


class OverconfidentModel:
    """Emit deliberately peaked logits so a temperature above one is the right fit."""

    def logits(self, image_nchw: np.ndarray) -> np.ndarray:
        batch, _, height, width = image_nchw.shape
        values = np.full((batch, NUM_CLASSES, height, width), -6.0, dtype=np.float64)
        values[:, CONFIDENT_CLASS] = 6.0
        return values

    def trainable_parameters(self) -> object:
        return ("weight",)


class FakeBackend:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def load_model(self, checkpoint_path: Path) -> tuple[OverconfidentModel, dict[str, Any]]:
        return OverconfidentModel(), dict(self.metadata)


def build_workspace(tmp_path: Path, *, split_name: str = "calibration") -> dict[str, Any]:
    data_root = tmp_path / "data"
    images = data_root / "images/10k/train"
    labels = data_root / "labels/sem_seg/masks/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    rng = np.random.default_rng(7)
    for index in range(3):
        sample_id = f"c{index:04d}"
        Image.fromarray(
            rng.integers(0, 256, (SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)
        ).save(images / f"{sample_id}.jpg")
        mask = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), CONFIDENT_CLASS, dtype=np.uint8)
        # A minority of pixels disagree, so the fit has something to correct.
        mask[: SOURCE_HEIGHT // 4] = (CONFIDENT_CLASS + 1) % NUM_CLASSES
        Image.fromarray(mask).save(labels / f"{sample_id}_train_id.png")

    manifest = build_paired_manifest(images, labels, split_name)
    manifest_path = tmp_path / f"{split_name}.json"
    save_manifest(manifest, manifest_path)

    protocol_dir = tmp_path / "configs" / "protocols"
    protocol_dir.mkdir(parents=True)
    protocol_path = protocol_dir / "bdd100k_semseg_v1.yaml"
    protocol_path.write_text(PROTOCOL_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    checkpoint = tmp_path / "final_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")

    return {
        "data_root": data_root,
        "manifest": manifest_path,
        "manifest_obj": manifest,
        "protocol": protocol_path,
        "protocol_hash": load_protocol(protocol_path).protocol_sha256,
        "checkpoint": checkpoint,
    }


def backend_for(workspace: dict[str, Any], **overrides: Any) -> FakeBackend:
    metadata: dict[str, Any] = {
        "model": "upernet_convnextv2_tiny",
        "run_id": "upernet_convnextv2_tiny-seed-17",
        "seed": 17,
        "protocol_sha256": workspace["protocol_hash"],
        "final_step": 30000,
    }
    metadata.update(overrides)
    return FakeBackend(metadata)


def run(workspace: dict[str, Any], output: Path, **overrides: Any) -> Any:
    service = load_service()
    return service.calibrate_checkpoint(
        workspace["protocol"],
        workspace["manifest"],
        workspace["checkpoint"],
        workspace["data_root"],
        output,
        backend=backend_for(workspace, **overrides),
    )


@pytest.fixture(autouse=True)
def provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))


def test_an_overconfident_model_receives_a_temperature_above_one(tmp_path: Path) -> None:
    """Temperature scaling exists to soften overconfidence; a T of 1 would be a no-op."""

    workspace = build_workspace(tmp_path)

    result = run(workspace, tmp_path / "calibration")

    assert result.temperature > 1.0


def test_the_artifact_records_the_sampling_rule_that_produced_it(tmp_path: Path) -> None:
    """A temperature fitted on an unrecorded pixel sample is not reproducible."""

    workspace = build_workspace(tmp_path)

    result = run(workspace, tmp_path / "calibration")
    document = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert document["schema_version"] == "drivemetrics-temperature/v1"
    assert document["protocol_sha256"] == workspace["protocol_hash"]
    assert document["dataset_manifest_sha256"] == workspace["manifest_obj"].manifest_sha256
    assert document["pixels_per_image"] > 0
    assert document["sampling_seed"] > 0
    assert document["temperature"] == pytest.approx(result.temperature)


def test_two_fits_agree_exactly(tmp_path: Path) -> None:
    """A resampled pixel set would move every calibrated number on every rerun."""

    workspace = build_workspace(tmp_path)

    first = run(workspace, tmp_path / "one")
    second = run(workspace, tmp_path / "two")

    assert first.temperature == second.temperature
    assert first.artifact_path.read_bytes() == second.artifact_path.read_bytes()


def test_a_checkpoint_from_another_protocol_is_refused(tmp_path: Path) -> None:
    """Fitting across protocol versions would silently mix incomparable runs."""

    workspace = build_workspace(tmp_path)

    with pytest.raises(
        ValueError, match=r"^checkpoint protocol hash does not match the calibration"
    ):
        run(workspace, tmp_path / "calibration", protocol_sha256="f" * 64)


def test_a_cohort_that_is_not_the_calibration_split_is_refused(tmp_path: Path) -> None:
    """Fitting on train or on the locked cohort is the contamination this forbids."""

    workspace = build_workspace(tmp_path, split_name="locked_validation")

    with pytest.raises(ValueError, match=r"^temperature fitting requires the calibration split,"):
        run(workspace, tmp_path / "calibration")


def test_the_fit_refuses_to_overwrite_an_existing_temperature(tmp_path: Path) -> None:
    """A replaced temperature detaches every calibrated artifact that cited it."""

    workspace = build_workspace(tmp_path)
    run(workspace, tmp_path / "calibration")

    with pytest.raises(FileExistsError, match=r"temperature\.json"):
        run(workspace, tmp_path / "calibration")


def test_the_sample_is_drawn_only_from_valid_pixels(tmp_path: Path) -> None:
    """Ignore pixels carry no label, so sampling them would waste the budget."""

    service = load_service()
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[0, :4] = CONFIDENT_CLASS

    indices = service.sample_pixel_indices(mask, sample_id="c0000", pixels=6)

    chosen = mask.reshape(-1)[indices]
    assert np.all(chosen[chosen != 255] == CONFIDENT_CLASS)
    assert int(np.sum(chosen != 255)) == 4


def test_sampling_is_keyed_by_sample_id_not_by_position(tmp_path: Path) -> None:
    """Position-keyed draws would change if the cohort were ever reordered."""

    service = load_service()
    mask = np.full((16, 16), CONFIDENT_CLASS, dtype=np.uint8)

    first = service.sample_pixel_indices(mask, sample_id="c0000", pixels=8)
    same = service.sample_pixel_indices(mask, sample_id="c0000", pixels=8)
    other = service.sample_pixel_indices(mask, sample_id="c0001", pixels=8)

    assert np.array_equal(first, same)
    assert not np.array_equal(first, other)


def test_the_result_names_the_cohort_it_used(tmp_path: Path) -> None:
    """The caller must be able to prove which frozen cohort produced the value."""

    workspace = build_workspace(tmp_path)

    result = run(workspace, tmp_path / "calibration")

    assert result.dataset_manifest_sha256 == workspace["manifest_obj"].manifest_sha256
    assert result.sampled_images == 3


def test_the_package_exports_the_service(tmp_path: Path) -> None:
    """The command line consumes the service through the package entry point."""

    import drivemetrics.calibration as calibration

    service = load_service()
    assert calibration.calibrate_checkpoint is service.calibrate_checkpoint


def test_a_manifest_whose_files_drifted_is_refused(tmp_path: Path) -> None:
    """Fitting on bytes that no longer match the frozen cohort is not calibration."""

    workspace = build_workspace(tmp_path)
    manifest = workspace["manifest_obj"]
    drifted = workspace["data_root"] / "images/10k/train" / manifest.relative_image_paths[0]
    drifted.write_bytes(b"different bytes entirely")

    with pytest.raises(ValueError, match=r"SHA-256 does not match"):
        run(workspace, tmp_path / "calibration")


def test_the_run_record_marks_the_fit_as_succeeded(tmp_path: Path) -> None:
    """A calibration fit is a run, and every run in this project leaves a record."""

    workspace = build_workspace(tmp_path)

    result = run(workspace, tmp_path / "calibration")
    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))

    assert record["status"] == "succeeded"
    assert record["run_id"] == "calibrate-upernet_convnextv2_tiny-seed-17"
    assert record["dataset_manifest_sha256"] == workspace["manifest_obj"].manifest_sha256


def test_dataclass_result_is_frozen(tmp_path: Path) -> None:
    """A mutable result could be edited between fitting and publication."""

    workspace = build_workspace(tmp_path)

    result = run(workspace, tmp_path / "calibration")

    assert dataclasses.is_dataclass(type(result))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.temperature = 2.0


def test_an_image_with_fewer_labelled_pixels_than_the_budget_is_padded(
    tmp_path: Path,
) -> None:
    """Resampling a short image would silently reweight it against the others."""

    service = load_service()
    workspace = build_workspace(tmp_path)

    # Ask for far more pixels than any 64x128 image can supply.
    result = service.calibrate_checkpoint(
        workspace["protocol"],
        workspace["manifest"],
        workspace["checkpoint"],
        workspace["data_root"],
        tmp_path / "padded",
        backend=backend_for(workspace),
        pixels_per_image=SOURCE_HEIGHT * SOURCE_WIDTH * 2,
    )

    assert result.temperature > 0.0
    assert result.pixels_per_image == SOURCE_HEIGHT * SOURCE_WIDTH * 2


def test_a_cohort_at_exactly_the_budget_is_returned_whole(tmp_path: Path) -> None:
    """The boundary decides whether an image is sampled or taken entire.

    At exactly the budget both branches return the same count, so only the
    identity of the returned pixels separates them: sampling would reorder and
    drop, while the contract says take every labelled pixel once.
    """

    del tmp_path
    service = load_service()
    mask = np.arange(4, dtype=np.uint8).reshape(2, 2)

    indices = service.sample_pixel_indices(mask, sample_id="c0000", pixels=4)

    assert indices.tolist() == [0, 1, 2, 3]


def test_the_sample_is_the_same_every_time_for_one_sample_id(tmp_path: Path) -> None:
    """A calibration fit that redraws its pixels is not reproducible."""

    del tmp_path
    service = load_service()
    mask = np.arange(64, dtype=np.uint8).reshape(8, 8)

    first = service.sample_pixel_indices(mask, sample_id="c0000", pixels=8)
    second = service.sample_pixel_indices(mask, sample_id="c0000", pixels=8)

    assert first.tolist() == second.tolist()
    assert len(set(first.tolist())) == 8, "a pixel was drawn twice and would be double-weighted"
    assert first.tolist() == sorted(first.tolist())


def test_two_sample_ids_draw_different_pixels(tmp_path: Path) -> None:
    """A seed that ignores the sample ID would give every image the same pixels."""

    del tmp_path
    service = load_service()
    mask = np.arange(64, dtype=np.uint8).reshape(8, 8)

    first = service.sample_pixel_indices(mask, sample_id="c0000", pixels=8)
    second = service.sample_pixel_indices(mask, sample_id="c0001", pixels=8)

    assert first.tolist() != second.tolist()
