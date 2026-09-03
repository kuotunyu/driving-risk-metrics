"""Contracts for fitting a scalar temperature on the frozen calibration cohort.

Full per-pixel logits for the formal cohort are infeasible: 700 images at
1280x720 by 19 classes in float64 is roughly 98 GB. The service therefore fits
on a deterministic seeded pixel sample per image. That makes the sampling rule
part of the evidence, so it is recorded in the artifact and pinned here.
"""

from __future__ import annotations

import dataclasses
import json
import re
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
        # The disagreeing class VARIES BY SAMPLE so the three label files have
        # distinct bytes and therefore distinct hashes. Identical label files
        # made the manifest's image/label hash pairing untestable: any index
        # into the label half matched any other.
        mask[: SOURCE_HEIGHT // 4] = (CONFIDENT_CLASS + 1 + index) % NUM_CLASSES
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


def test_calibration_creates_a_nested_output_directory(tmp_path: Path) -> None:
    """The same requirement the analysis stage has, for the same reason.

    Every real invocation writes several levels deep, and failing after the
    temperature has been fitted wastes the fit.
    """

    workspace = build_workspace(tmp_path)
    output = tmp_path / "artifacts" / "calibration" / "upernet_convnextv2_tiny-seed-17"

    run(workspace, output)

    assert (output / "temperature.json").is_file()
    assert (output / "run_record.json").is_file()


def test_the_written_calibration_has_sorted_keys(tmp_path: Path) -> None:
    """Key order is part of the bytes, and a claim cites the bytes.

    Without `sort_keys=True` the file follows the model's field declaration
    order, which is meaningful rather than alphabetical, so an unrelated field
    reorder would change the artifact hash and detach every claim that cited
    it.
    """

    workspace = build_workspace(tmp_path)
    output = tmp_path / "out"
    run(workspace, output)

    for name in ("temperature", "run_record"):
        text = (output / f"{name}.json").read_text(encoding="utf-8")
        keys = re.findall(r'^  "([^"]+)":', text, re.MULTILINE)
        assert keys == sorted(keys), f"{name}.json top-level keys are not sorted: {keys}"


def _write_square_sample(tmp_path: Path) -> tuple[Path, Path]:
    """Write one SQUARE image and mask, so the canvas is horizontally padded.

    The shared workspace uses a 2:1 source, which resizes to the full canvas
    width and leaves `pad_left` and `pad_right` both zero. Under that geometry
    `CANVAS_WIDTH - pad_right` and `CANVAS_WIDTH + pad_right` are the same
    number, so no test built on it can tell the two apart.
    """

    side = 64
    rng = np.random.default_rng(11)
    image_path = tmp_path / "square.jpg"
    label_path = tmp_path / "square_train_id.png"
    Image.fromarray(rng.integers(0, 256, (side, side, 3), dtype=np.uint8)).save(image_path)
    mask = np.full((side, side), CONFIDENT_CLASS, dtype=np.uint8)
    mask[: side // 4] = (CONFIDENT_CLASS + 1) % NUM_CLASSES
    Image.fromarray(mask).save(label_path)
    return image_path, label_path


class PositionCodedModel:
    """Emit a logit that identifies the canvas column each pixel came from.

    The confident class carries `column / width`, so the returned logits are a
    fingerprint of exactly which canvas pixels were gathered. A model that
    returns the same value everywhere, as `OverconfidentModel` does, cannot
    detect a gather that reads the wrong columns.
    """

    def logits(self, image_nchw: np.ndarray) -> np.ndarray:
        batch, _, height, width = image_nchw.shape
        values = np.zeros((batch, NUM_CLASSES, height, width), dtype=np.float64)
        columns = np.arange(width, dtype=np.float64) / float(width)
        values[:, CONFIDENT_CLASS] = columns[None, None, :]
        return values

    def trainable_parameters(self) -> object:
        return ("weight",)


def test_the_sampled_pixel_indices_are_the_rule_the_artifact_records() -> None:
    """The draw is published as a rule, so the values that rule produces are a contract.

    `temperature.json` records `sampling_seed` and `pixels_per_image` and
    nothing else, so a reader reproduces the sample by re-deriving it: the
    first eight bytes of `sha256(f"{SAMPLING_SEED}:{sample_id}")` read
    big-endian, seeding a PCG64 generator, drawn without replacement, sorted
    ascending. This pins what that recipe produces. A change to the digest
    slice, the key format or the draw silently re-selects the pixels every
    published temperature was fitted on, while every determinism test in this
    file keeps passing.
    """

    service = load_service()
    mask = np.full((8, 8), CONFIDENT_CLASS, dtype=np.uint8)

    indices = service.sample_pixel_indices(mask, sample_id="c0000", pixels=8)

    assert indices.tolist() == [15, 22, 28, 32, 42, 44, 51, 62]
    assert indices.dtype == np.int64


def test_the_gathered_logits_are_the_canvas_pixels_the_sample_selected(
    tmp_path: Path,
) -> None:
    """Every step from model canvas to sampled rows is checked against an independent gather.

    The service restores the canvas to source geometry, drops the horizontal
    padding, and gathers the drawn pixels. Recomputing that here with a model
    whose logits encode their own column separates four things a
    constant-valued model cannot see: the right-hand padding is SUBTRACTED
    rather than added, the flattening keeps the class axis last, the index map
    is applied to the drawn indices, and the sample id reaches the draw instead
    of being replaced by a constant.
    """

    from drivemetrics.data.transforms import prepare_sample, restore_index_map

    service = load_service()
    image_path, label_path = _write_square_sample(tmp_path)
    sample_id = "c0000"

    drawn, targets = service._sample_logits(
        PositionCodedModel(), image_path, label_path, sample_id, 6
    )

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(label_path), dtype=np.uint8)
    prepared = prepare_sample(image, mask, training=False, flip_draw=1.0)
    assert prepared.pad_right > 0, "a square source is used precisely so padding is non-zero"
    canvas = PositionCodedModel().logits(prepared.image_chw[None, ...])[0].transpose(1, 2, 0)
    canvas = np.ascontiguousarray(canvas)
    content = canvas[:, prepared.pad_left : service.CANVAS_WIDTH - prepared.pad_right]
    indices = service.sample_pixel_indices(mask, sample_id=sample_id, pixels=6)
    expected = content.reshape(-1, NUM_CLASSES)[restore_index_map(prepared).reshape(-1)[indices]]

    np.testing.assert_array_equal(drawn, expected)
    np.testing.assert_array_equal(targets, mask.reshape(-1)[indices].astype(np.int64))
    assert len({float(value) for value in drawn[:, CONFIDENT_CLASS]}) > 1


def test_an_image_shorter_than_the_budget_is_padded_with_ignored_rows(
    tmp_path: Path,
) -> None:
    """Short images contribute the same shape, and the padding must be discardable.

    Every image hands back exactly `pixels` rows so the batch stacks, so a
    short image is padded. The padding carries the ignore target, which the
    fitter drops, and that is the only reason padding cannot bias the result.
    A padding count computed by addition, or read from the wrong axis, would
    either overrun the budget or leave real rows unpadded.
    """

    service = load_service()
    image_path, label_path = _write_square_sample(tmp_path)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    mask[0, :3] = CONFIDENT_CLASS
    Image.fromarray(mask).save(label_path)

    drawn, targets = service._sample_logits(
        PositionCodedModel(), image_path, label_path, "c0000", 7
    )

    assert drawn.shape == (7, NUM_CLASSES)
    assert targets.shape == (7,)
    assert int(np.sum(targets == 255)) == 4
    np.testing.assert_array_equal(drawn[3:], np.zeros((4, NUM_CLASSES)))


def test_each_sample_is_verified_against_its_own_label_hash(tmp_path: Path) -> None:
    """The manifest interleaves image and label hashes, and the label is the odd entry.

    For sample `i` the image hash is at `2i` and the label hash at `2i + 1`.
    Any other index reads a different sample's hash, which is why the three
    fixture label files are deliberately distinct: while they were identical,
    every index into the label half matched every other and the pairing could
    not be tested at all.

    This asserts the layout directly and then proves the calibration accepts a
    manifest built that way, so an off-by-one in either index fails here rather
    than fitting a temperature against annotations nobody checked.
    """

    from drivemetrics.protocol.hashing import sha256_file

    service = load_service()
    workspace = build_workspace(tmp_path)
    manifest = workspace["manifest_obj"]
    images = workspace["data_root"] / "images/10k/train"
    labels = workspace["data_root"] / "labels/sem_seg/masks/train"

    label_hashes = {manifest.file_sha256[2 * position + 1] for position in range(3)}
    assert len(label_hashes) == 3, "the fixture's label files must be distinct"
    for position in range(len(manifest.sample_ids)):
        assert manifest.file_sha256[2 * position] == sha256_file(
            images / manifest.relative_image_paths[position]
        )
        assert manifest.file_sha256[2 * position + 1] == sha256_file(
            labels / manifest.relative_label_paths[position]
        )

    assert (
        service.calibrate_checkpoint(
            workspace["protocol"],
            workspace["manifest"],
            workspace["checkpoint"],
            workspace["data_root"],
            tmp_path / "out",
            backend=backend_for(workspace),
        ).temperature
        > 0.0
    )


def test_the_recorded_timestamps_are_read_from_a_utc_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `Z` suffix on a local reading is a lie that nothing downstream can detect.

    The run record stamps its times with a trailing `Z`, which asserts UTC. The
    host this project is developed on runs eight hours ahead of UTC, so a naive
    `now()` would produce a timestamp that parses, validates, and orders
    correctly against its own siblings while being wrong by eight hours against
    every other run in the study.
    """

    service = load_service()
    real_datetime = service.datetime
    observed: list[object] = []
    fixed = real_datetime(2026, 9, 4, 1, 2, 3, tzinfo=service.UTC)

    class RecordingDatetime:
        @staticmethod
        def now(tz: object = None) -> Any:
            observed.append(tz)
            return fixed

    workspace = build_workspace(tmp_path)
    monkeypatch.setattr(service, "datetime", RecordingDatetime)
    output = tmp_path / "out"
    run(workspace, output)

    assert observed, "the service never asked for the time"
    assert all(tz is service.UTC for tz in observed)
    record = json.loads((output / "run_record.json").read_text(encoding="utf-8"))
    assert record["started_at_utc"] == "2026-09-04T01:02:03Z"
    assert record["finished_at_utc"] == "2026-09-04T01:02:03Z"


def test_the_temperature_document_carries_exactly_the_declared_fields(
    tmp_path: Path,
) -> None:
    """The artifact is cited by hash, so its key set and its bytes are both the contract.

    A renamed or case-changed key produces a file that still parses, still
    hashes to something, and still looks like a temperature record, while every
    reader that asks for the old name gets nothing. The byte comparison covers
    the indentation and the trailing newline for the same reason: they are part
    of what was hashed.
    """

    workspace = build_workspace(tmp_path)
    output = tmp_path / "out"
    result = run(workspace, output)

    raw = (output / "temperature.json").read_bytes()
    document = json.loads(raw)

    assert set(document) == {
        "schema_version",
        "temperature",
        "protocol_sha256",
        "dataset_manifest_sha256",
        "checkpoint_sha256",
        "run_id",
        "seed",
        "sampled_images",
        "pixels_per_image",
        "sampling_seed",
    }
    assert document["run_id"] == "upernet_convnextv2_tiny-seed-17"
    assert document["seed"] == 17
    assert document["sampled_images"] == len(workspace["manifest_obj"].sample_ids)
    assert document["temperature"] == pytest.approx(result.temperature)
    assert raw == (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def test_the_run_record_names_the_temperature_artifact_it_produced(
    tmp_path: Path,
) -> None:
    """The record is the index from a run to its outputs, keyed by artifact name.

    A renamed key leaves the record valid and its own hash correct while the
    aggregate stage, which looks the artifact up by name, finds nothing. The
    bytes are pinned here too, because the record is itself hashed into the
    formal run index.
    """

    import hashlib

    workspace = build_workspace(tmp_path)
    output = tmp_path / "out"
    run(workspace, output)

    raw = (output / "run_record.json").read_bytes()
    record = json.loads(raw)

    assert set(record["artifacts"]) == {"temperature"}
    assert (
        record["artifacts"]["temperature"]
        == hashlib.sha256((output / "temperature.json").read_bytes()).hexdigest()
    )
    assert raw == (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def test_an_existing_empty_output_directory_is_usable(tmp_path: Path) -> None:
    """Only an existing TEMPERATURE is refused; an existing directory is not.

    The refusal that protects a frozen artifact is keyed on `temperature.json`
    and is tested separately. Refusing the directory as well would fail after
    the fit, which is the expensive part, rather than before it.
    """

    workspace = build_workspace(tmp_path)
    output = tmp_path / "out" / "nested"
    output.mkdir(parents=True)

    result = run(workspace, output)

    assert result.artifact_path.is_file()


def test_each_image_is_sampled_under_its_own_sample_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The draw is keyed by sample id, and the service must pass the real one.

    Replaced by a constant, every image in the cohort would draw the SAME pixel
    positions. That is not detectable from the fitted temperature alone, and it
    is invisible whenever the budget exceeds the image, because then every pixel
    is returned and the key is never consulted. This test therefore uses a
    budget smaller than the image and watches the call itself: the three ids the
    service passes must be the three ids in the manifest, in cohort order.
    """

    service = load_service()
    workspace = build_workspace(tmp_path)
    manifest = workspace["manifest_obj"]
    observed: list[object] = []
    real_draw = service.sample_pixel_indices

    def recording_draw(mask: Any, *, sample_id: Any, pixels: int) -> Any:
        observed.append(sample_id)
        return real_draw(mask, sample_id=sample_id, pixels=pixels)

    monkeypatch.setattr(service, "sample_pixel_indices", recording_draw)
    service.calibrate_checkpoint(
        workspace["protocol"],
        workspace["manifest"],
        workspace["checkpoint"],
        workspace["data_root"],
        tmp_path / "out",
        backend=backend_for(workspace),
        pixels_per_image=64,
    )

    assert observed == list(manifest.sample_ids)


def test_a_smaller_budget_actually_reduces_the_sampled_rows(tmp_path: Path) -> None:
    """The budget is a real limit, and the neighbouring test depends on it biting.

    The fixture's images hold 8,192 pixels, far below the default budget, so
    every pixel is returned and the keyed draw never runs. A test that means to
    exercise the draw has to ask for fewer pixels than the image holds, and
    this pins that 64 does so.
    """

    service = load_service()
    mask = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), CONFIDENT_CLASS, dtype=np.uint8)

    assert mask.size > 64
    assert service.sample_pixel_indices(mask, sample_id="c0000", pixels=64).size == 64
    assert (
        service.sample_pixel_indices(mask, sample_id="c0000", pixels=mask.size + 1).size
        == mask.size
    )
