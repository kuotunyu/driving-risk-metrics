"""Contracts for the post-release resolution sweep on the calibration split.

The sweep is pre-registered in docs/posthoc/resolution-v1/analysis-plan.md. These
tests pin the parts a reader has to trust without rerunning it: the formal arm
is the released geometry, no other cohort than calibration can be scored, the
instance rules are the released ones, and an interrupted run resumes without
scoring an image twice.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch", reason="the optional train extra is not installed")

from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR
from drivemetrics.data.bdd100k import NUM_TRAIN_CLASSES
from drivemetrics.data.manifest import build_paired_manifest
from drivemetrics.data.transforms import prepare_sample, restore_prediction
from drivemetrics.metrics.instances import instance_coverages
from drivemetrics.models.adapters import SegmentationAdapter
from drivemetrics.posthoc import resolution
from drivemetrics.protocol.config import load_protocol
from drivemetrics.protocol.hashing import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"
TERTILES_SOURCE = REPO_ROOT / "docs" / "evidence" / "bdd100k_semseg_v1" / "area_tertiles.json"
TRAIN_IMAGES = "images/10k/train"
TRAIN_LABELS = "labels/sem_seg/masks/train"
VALIDATION_IMAGES = "images/10k/val"
VALIDATION_LABELS = "labels/sem_seg/masks/val"
HEIGHT, WIDTH = 720, 1280
PERSON, RIDER, CAR, ROAD = 11, 12, 13, 0
A100 = "NVIDIA A100-SXM4-40GB, 40960 MiB"
L4 = "NVIDIA L4, 23034 MiB"
RUN_IDS = ("segformer_b2-seed-17", "segformer_b2-seed-42")


def provenance(gpu: str = A100) -> dict[str, Any]:
    return {"commit": "e" * 40, "lock_sha256": "f" * 64, "hardware": {"gpu": gpu}}


class TinySegmenter(torch.nn.Module):
    """A quarter-resolution random segmenter: enough structure for argmax to vary."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.conv = torch.nn.Conv2d(3, NUM_TRAIN_CLASSES, kernel_size=4, stride=4)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=generator))
            bias = self.conv.bias
            assert bias is not None
            bias.copy_(torch.randn(bias.shape, generator=generator))

    def forward(self, batch: Any) -> Any:
        return SimpleNamespace(logits=self.conv(batch))


class FakeBackend:
    """Restore a tiny model per checkpoint with the metadata training would have written."""

    def __init__(self, metadata: dict[str, dict[str, Any]]) -> None:
        self.metadata = metadata

    def load_model(self, checkpoint_path: Path) -> tuple[SegmentationAdapter, dict[str, Any]]:
        metadata = self.metadata[checkpoint_path.stem]
        return SegmentationAdapter(TinySegmenter(int(metadata["seed"]))), dict(metadata)


def scene(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One source-geometry image, its train-ID mask and its instance bitmask."""

    rng = np.random.default_rng(index)
    image = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    mask = np.full((HEIGHT, WIDTH), ROAD, dtype=np.uint8)
    mask[:8, :8] = 255
    bitmask = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    offset = 40 * index
    objects = [
        # (category, train id, instance id, rows, cols)
        (1, PERSON, 1, slice(100 + offset, 110 + offset), slice(100, 120)),  # 200 px, small
        (1, PERSON, 2, slice(200, 230), slice(300, 320)),  # 600 px, medium
        (1, PERSON, 3, slice(300, 350), slice(500, 540)),  # 2000 px, large
        (2, RIDER, 4, slice(400, 410), slice(600, 610)),  # 100 px, small
        (3, CAR, 5, slice(500, 560), slice(700, 800)),  # 6000 px, large
    ]
    for category, train_id, instance_id, rows, cols in objects:
        mask[rows, cols] = train_id
        bitmask[rows, cols, 0] = category
        bitmask[rows, cols, 3] = instance_id
    # A person annotation the semantic mask never agrees with: excluded and counted.
    bitmask[600:610, 900:910, 0] = 1
    bitmask[600:610, 900:910, 3] = 6
    # A person whose boundary the two annotations disagree on: footprint narrows.
    mask[650:660, 1000:1010] = PERSON
    bitmask[650:660, 1000:1012, 0] = 1
    bitmask[650:660, 1000:1012, 3] = 7
    return image, mask, bitmask


def write_split(
    root: Path, split_name: str, sample_ids: tuple[str, ...]
) -> tuple[Path, Path, Path]:
    validation = split_name == "locked_validation"
    images = root / "data" / (VALIDATION_IMAGES if validation else TRAIN_IMAGES)
    labels = root / "data" / (VALIDATION_LABELS if validation else TRAIN_LABELS)
    bitmasks = root / "bitmasks"
    for directory in (images, labels, bitmasks):
        directory.mkdir(parents=True, exist_ok=True)
    for index, sample_id in enumerate(sample_ids):
        image, mask, bitmask = scene(index)
        Image.fromarray(image).save(images / f"{sample_id}.jpg")
        Image.fromarray(mask).save(labels / f"{sample_id}_train_id.png")
        Image.fromarray(bitmask).save(bitmasks / f"{sample_id}.png")
    manifest = build_paired_manifest(images, labels, split_name)
    manifest_path = root / f"{split_name}.json"
    manifest_path.write_text(
        json.dumps(dataclasses.asdict(manifest), sort_keys=True), encoding="utf-8", newline="\n"
    )
    return manifest_path, root / "data", bitmasks


@dataclasses.dataclass
class Workspace:
    config: Path
    manifest: Path
    data_root: Path
    bitmasks: Path
    tertiles: Path
    checkpoints: Path
    backend: FakeBackend
    root: Path

    def parity(self, **kwargs: Any) -> resolution.ParityResult:
        options: dict[str, Any] = {"backend": self.backend, "images": 1}
        options.update(kwargs)
        return resolution.formal_parity(
            self.config,
            self.manifest,
            self.checkpoints,
            self.data_root,
            self.bitmasks,
            self.tertiles,
            self.root / "parity" / "parity.json",
            **options,
        )

    def sweep(self, **kwargs: Any) -> resolution.SweepResult:
        options: dict[str, Any] = {"backend": self.backend, "arms": resolution.DEFAULT_ARMS}
        options.update(kwargs)
        return resolution.resolution_sweep(
            self.config,
            self.manifest,
            self.checkpoints,
            self.data_root,
            self.bitmasks,
            self.tertiles,
            self.root / "parity" / "parity.json",
            self.root / "out",
            **options,
        )


def build_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    split_name: str = "calibration",
    sample_ids: tuple[str, ...] = ("c0001", "c0002", "c0003"),
    gpu: str = A100,
) -> Workspace:
    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(provenance(gpu)))
    config = tmp_path / "protocol.yaml"
    shutil.copyfile(PROTOCOL_SOURCE, config)
    manifest, data_root, bitmasks = write_split(tmp_path, split_name, sample_ids)
    protocol_sha256 = load_protocol(config).protocol_sha256
    checkpoints: dict[str, list[str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    (tmp_path / "checkpoints").mkdir()
    for run_id in RUN_IDS:
        path = tmp_path / "checkpoints" / f"{run_id}.pt"
        path.write_bytes(f"weights of {run_id}".encode())
        checkpoints[run_id] = [str(path), sha256_file(path)]
        model, seed = run_id.rsplit("-seed-", 1)
        metadata[run_id] = {
            "run_id": run_id,
            "model": model,
            "seed": int(seed),
            "protocol_sha256": protocol_sha256,
        }
    checkpoint_list = tmp_path / "checkpoints.json"
    checkpoint_list.write_text(json.dumps(checkpoints), encoding="utf-8", newline="\n")
    return Workspace(
        config=config,
        manifest=manifest,
        data_root=data_root,
        bitmasks=bitmasks,
        tertiles=TERTILES_SOURCE,
        checkpoints=checkpoint_list,
        backend=FakeBackend(metadata),
        root=tmp_path,
    )


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- arms


def test_the_default_arms_are_the_three_pre_registered_ones_in_scale_order() -> None:
    """Optional controls exist in code but are never run unless asked for."""

    arms = resolution.parse_arms(["native", " formal", "s085", ""])

    assert tuple(arm.name for arm in arms) == ("formal", "s085", "native")
    assert resolution.DEFAULT_ARMS == ("formal", "s085", "native")
    assert [round(arm.scale, 4) for arm in arms] == [0.7111, 0.85, 1.0]
    assert resolution.ARMS["s060"].scale == pytest.approx(0.6)
    assert resolution.ARMS["formal_unpadded"].formal is False


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["formal", "native", "s999"], "unknown arm"),
        (["formal", "native", "native"], "more than once"),
        (["formal", "s085"], "must include"),
    ],
)
def test_an_arm_list_outside_the_plan_is_refused(names: list[str], message: str) -> None:
    """Every pre-registered contrast is formal against native; without both there is none."""

    with pytest.raises(ValueError, match=message):
        resolution.parse_arms(names)


# ------------------------------------------------------------------ preprocessing


def test_the_formal_arm_is_the_frozen_preprocessing_itself() -> None:
    """A re-implementation that agreed only approximately would not be the released geometry."""

    image, mask, _ = scene(0)

    chw, prepared = resolution.model_input(image, mask, resolution.ARMS["formal"])

    expected = prepare_sample(image, mask, training=False, flip_draw=1.0)
    assert prepared is not None
    assert np.array_equal(chw, expected.image_chw)
    assert (prepared.pad_left, prepared.pad_right) == (expected.pad_left, expected.pad_right)


def test_the_unpadded_control_is_the_formal_content_without_its_padding() -> None:
    """The generic resize path must reproduce the frozen one bit for bit at the same size."""

    image, mask, _ = scene(0)
    formal, prepared = resolution.model_input(image, mask, resolution.ARMS["formal"])

    unpadded, restore = resolution.model_input(image, mask, resolution.ARMS["formal_unpadded"])

    assert restore is None
    assert prepared is not None
    content = formal[:, :, prepared.pad_left : formal.shape[2] - prepared.pad_right]
    assert np.array_equal(unpadded, content)


@pytest.mark.parametrize(("arm", "shape"), [("native", (3, 720, 1280)), ("s085", (3, 612, 1088))])
def test_non_formal_arms_feed_the_model_at_their_own_geometry(
    arm: str, shape: tuple[int, int, int]
) -> None:
    image, mask, _ = scene(0)

    chw, prepared = resolution.model_input(image, mask, resolution.ARMS[arm])

    assert prepared is None
    assert chw.shape == shape
    assert chw.dtype == np.float32


def test_a_source_that_is_not_720_by_1280_is_refused() -> None:
    """The arms are defined as fractions of the BDD100K source geometry and nothing else."""

    image = np.zeros((90, 160, 3), dtype=np.uint8)
    mask = np.zeros((90, 160), dtype=np.uint8)

    with pytest.raises(ValueError, match="720x1280"):
        resolution.model_input(image, mask, resolution.ARMS["formal"])


# ---------------------------------------------------------------------- inference


def test_labels_are_the_argmax_of_the_released_logits_path() -> None:
    """Device-side argmax must agree with the adapter's float64 logits the engine uses."""

    image, mask, _ = scene(0)
    adapter = SegmentationAdapter(TinySegmenter(3))
    chw, _ = resolution.model_input(image, mask, resolution.ARMS["formal"])

    labels = resolution.predict_labels(adapter, chw)

    reference = np.argmax(adapter.logits(chw[None, ...])[0], axis=0)
    assert labels.dtype == np.uint8
    assert labels.shape == (512, 1024)
    assert np.array_equal(labels, reference)


def test_predictions_are_restored_to_source_geometry_for_every_arm() -> None:
    """Metrics are never computed on a model canvas."""

    image, mask, _ = scene(0)
    adapter = SegmentationAdapter(TinySegmenter(5))
    for name in resolution.ARMS:
        chw, prepared = resolution.model_input(image, mask, resolution.ARMS[name])
        labels = resolution.predict_labels(adapter, chw)

        restored = resolution.source_labels(labels, prepared)

        assert restored.shape == (HEIGHT, WIDTH)
        if prepared is not None:
            assert np.array_equal(restored, restore_prediction(labels, prepared))
        if name == "native":
            assert restored is labels


# ------------------------------------------------------------------------ scoring


def test_footprint_restricted_coverage_equals_the_full_image_rule() -> None:
    """The ten-times-cheaper restriction must not change a single instance record."""

    _, mask, bitmask = scene(1)
    edges = resolution.load_tertile_edges(TERTILES_SOURCE)
    truth = resolution.load_image_truth(mask, bitmask, edges)
    predicted = np.random.default_rng(9).integers(0, 19, size=mask.shape).astype(np.uint8)
    predicted[mask == PERSON] = PERSON  # guarantee some covered persons too
    predicted[105:110, 100:120] = ROAD

    coverages, confusion = resolution.score_labels(truth, predicted)

    full = instance_coverages(
        mask.astype(np.int64).reshape(-1),
        predicted.astype(np.int64).reshape(-1),
        truth.footprint_ids.reshape(-1),
        truth.classes,
        edges,
    )
    assert coverages == full
    assert truth.excluded_without_semantic_pixels == 1
    assert [record.area_tertile for record in truth.instances][:4] == [
        "small",
        "medium",
        "large",
        "small",
    ]
    assert confusion.shape == (19, 19)
    assert int(confusion.sum()) == int(np.count_nonzero(mask != 255))


def test_a_bitmask_of_the_wrong_shape_is_refused() -> None:
    _, mask, _ = scene(0)
    edges = resolution.load_tertile_edges(TERTILES_SOURCE)

    with pytest.raises(ValueError, match="bitmask"):
        resolution.load_image_truth(mask, np.zeros((HEIGHT, WIDTH), dtype=np.uint8), edges)


def test_tertile_edges_are_translated_into_train_ids(tmp_path: Path) -> None:
    """Frozen edges are keyed by instance category; comparisons happen in train IDs."""

    document = json.loads(TERTILES_SOURCE.read_text(encoding="utf-8"))
    document["tertile_edges"]["99"] = [1, 2]
    path = tmp_path / "tertiles.json"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    edges = resolution.load_tertile_edges(path)

    assert edges[PERSON] == (349, 987)
    assert edges[RIDER] == (409, 1536)
    assert len(edges) == 8


# --------------------------------------------------------------------- checkpoints


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "JSON object"),
        ({}, "JSON object"),
        ({"r": "path"}, r"\[path, sha256\]"),
        ({"r": ["path", "ABC"]}, "SHA-256"),
    ],
)
def test_a_malformed_checkpoint_list_is_refused(
    tmp_path: Path, document: object, message: str
) -> None:
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match=message):
        resolution.read_checkpoint_list(path)


def test_a_checkpoint_that_is_not_the_experiment_card_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))
    (tmp_path / "checkpoints" / f"{RUN_IDS[0]}.pt").write_bytes(b"retrained")

    with pytest.raises(ValueError, match="SHA-256"):
        workspace.parity()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("protocol_sha256", "0" * 64, "formal protocol"), ("run_id", "other", "names itself")],
)
def test_checkpoint_metadata_must_match_the_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))
    workspace.backend.metadata[RUN_IDS[1]][field] = value

    with pytest.raises(ValueError, match=message):
        workspace.parity()


# -------------------------------------------------------------------------- cohort


def test_the_locked_cohort_is_refused_in_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan forbids looking at the locked cohort; convention alone is not enough."""

    workspace = build_workspace(tmp_path, monkeypatch, split_name="locked_validation")

    with pytest.raises(ValueError, match="calibration"):
        workspace.parity()
    with pytest.raises(ValueError, match="calibration"):
        workspace.sweep()


# -------------------------------------------------------------------------- parity


def test_the_formal_arm_reproduces_the_released_evaluation_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch)

    result = workspace.parity(images=2)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == resolution.PARITY_SCHEMA_VERSION
    assert report["passed"] is True
    assert report["criterion"] == "exact"
    assert report["sample_ids"] == ["c0001", "c0002"]
    assert report["hardware"]["gpu"] == A100
    for run_id in RUN_IDS:
        entry = report["runs"][run_id]
        assert entry["mismatched_pixels"] == 0
        assert entry["changed_critical_miss_decisions"] == 0
        assert entry["compared_pixels"] == 2 * (HEIGHT * WIDTH - 64)
        assert entry["compared_instances"] == 12
    assert result.mismatched_pixels == dict.fromkeys(RUN_IDS, 0)
    assert result.report_path.read_bytes().endswith(b"}\n")
    assert b"\r\n" not in result.report_path.read_bytes()


@pytest.mark.parametrize(("criterion", "images"), [("strict", 1), ("exact", 0)])
def test_parity_settings_outside_the_plan_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, criterion: str, images: int
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))

    with pytest.raises(ValueError, match=r"criterion|images"):
        workspace.parity(criterion=criterion, images=images)


def test_an_a100_keeps_exact_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The relaxed criterion is the owner's L4 fallback, not an A100 escape hatch."""

    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))

    with pytest.raises(ValueError, match="L4"):
        workspace.parity(criterion="decision")


def released_with_one_changed_pixel(original: Any) -> Any:
    """Wrap the released scorer so it disagrees on one pixel outside every instance."""

    def evaluate(*args: Any) -> Any:
        record, ece = original(*args)
        predicted = record.predicted_class.copy()
        predicted[-1] = (int(predicted[-1]) + 1) % 19
        return dataclasses.replace(record, predicted_class=predicted), ece

    return evaluate


def test_on_an_l4_a_pixel_difference_that_changes_no_decision_passes_the_relaxed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",), gpu=L4)
    monkeypatch.setattr(
        resolution, "_evaluate_sample", released_with_one_changed_pixel(resolution._evaluate_sample)
    )

    result = workspace.parity(criterion="decision")

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["criterion"] == "decision"
    assert result.mismatched_pixels == dict.fromkeys(RUN_IDS, 1)
    assert result.changed_decisions == dict.fromkeys(RUN_IDS, 0)


def test_the_same_pixel_difference_fails_the_exact_gate_and_still_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evidence of a failed gate is what the owner needs to decide what to do next."""

    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))
    monkeypatch.setattr(
        resolution, "_evaluate_sample", released_with_one_changed_pixel(resolution._evaluate_sample)
    )

    with pytest.raises(ValueError, match="parity gate failed"):
        workspace.parity()

    report = json.loads((tmp_path / "parity" / "parity.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["runs"][RUN_IDS[0]]["mismatched_pixels"] == 1


def test_a_changed_critical_miss_decision_fails_even_the_relaxed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",), gpu=L4)
    monkeypatch.setattr(
        resolution,
        "predict_labels",
        lambda _adapter, chw: np.full(chw.shape[1:], PERSON, dtype=np.uint8),
    )
    original = resolution._evaluate_sample

    def released_all_road(*args: Any) -> Any:
        record, ece = original(*args)
        predicted = np.zeros_like(record.predicted_class)
        return dataclasses.replace(record, predicted_class=predicted), ece

    monkeypatch.setattr(resolution, "_evaluate_sample", released_all_road)

    with pytest.raises(ValueError, match="parity gate failed"):
        workspace.parity(criterion="decision")

    report = json.loads((tmp_path / "parity" / "parity.json").read_text(encoding="utf-8"))
    assert report["runs"][RUN_IDS[0]]["changed_critical_miss_decisions"] == 4


# --------------------------------------------------------------------------- sweep


def passed_parity(workspace: Workspace) -> None:
    workspace.parity(images=1)


def test_the_sweep_scores_every_arm_and_run_and_resumes_where_it_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch)
    passed_parity(workspace)
    progress: list[tuple[int, int]] = []

    first = workspace.sweep(limit=2, on_sample=lambda done, total: progress.append((done, total)))
    second = workspace.sweep()

    assert (first.scored_samples, first.completed_samples, first.total_samples) == (2, 2, 3)
    assert (second.scored_samples, second.completed_samples) == (1, 3)
    assert progress == [(1, 2), (2, 2)]
    lines = read_lines(second.results_path)
    assert [line["sample_id"] for line in lines] == ["c0001", "c0002", "c0003"]
    line = lines[0]
    assert line["schema_version"] == resolution.SWEEP_SCHEMA_VERSION
    assert line["bitmask_sha256"] == sha256_file(workspace.bitmasks / "c0001.png")
    assert line["excluded_without_semantic_pixels"] == 1
    assert [entry["class_id"] for entry in line["instances"]] == [
        PERSON,
        PERSON,
        PERSON,
        RIDER,
        CAR,
        PERSON,
    ]
    assert set(line["results"]) == {"formal", "s085", "native"}
    for arm in line["results"].values():
        assert set(arm) == set(RUN_IDS)
        for entry in arm.values():
            assert len(entry["correct_fraction"]) == 6
            assert entry["critical_miss"] == [value < 0.5 for value in entry["correct_fraction"]]
            assert len(entry["confusion"]) == 19 * 19
            assert sum(entry["confusion"]) == HEIGHT * WIDTH - 64

    run = json.loads((tmp_path / "out" / resolution.RUN_FILENAME).read_text(encoding="utf-8"))
    assert run["schema_version"] == resolution.SWEEP_SCHEMA_VERSION
    assert run["split_name"] == "calibration"
    assert run["sample_ids"] == ["c0001", "c0002", "c0003"]
    assert [arm["name"] for arm in run["arms"]] == ["formal", "s085", "native"]
    assert [entry["run_id"] for entry in run["runs"]] == list(RUN_IDS)
    assert run["tertiles_sha256"] == sha256_file(TERTILES_SOURCE)
    sessions = read_lines(tmp_path / "out" / resolution.SESSIONS_FILENAME)
    assert [session["scored_samples"] for session in sessions] == [2, 1]
    assert sessions[0]["provenance"]["commit"] == "e" * 40
    assert sessions[0]["parity_criterion"] == "exact"


def test_the_swept_formal_arm_equals_the_released_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: what the sweep records for formal is what the release would have scored."""

    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))
    passed_parity(workspace)

    result = workspace.sweep(arms=("formal", "native"))

    line = read_lines(result.results_path)[0]
    adapter, _ = workspace.backend.load_model(tmp_path / "checkpoints" / f"{RUN_IDS[0]}.pt")
    image_path = workspace.data_root / TRAIN_IMAGES / "c0001.jpg"
    label_path = workspace.data_root / TRAIN_LABELS / "c0001_train_id.png"
    record, _ = resolution._evaluate_sample(adapter, image_path, label_path, "c0001", 1.0)
    assert (
        line["results"]["formal"][RUN_IDS[0]]["confusion"] == record.confusion.reshape(-1).tolist()
    )


def test_a_line_cut_off_by_an_interrupted_write_is_rescored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001", "c0002"))
    passed_parity(workspace)
    first = workspace.sweep(limit=1)
    with first.results_path.open("ab") as handle:
        handle.write(b'{"sample_id": "c00')

    second = workspace.sweep()

    assert second.scored_samples == 1
    assert [line["sample_id"] for line in read_lines(second.results_path)] == ["c0001", "c0002"]


@pytest.mark.parametrize("sample_id", ["c0001", "x9999"])
def test_a_results_file_that_is_not_this_cohort_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_id: str
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001", "c0002"))
    passed_parity(workspace)
    first = workspace.sweep(limit=1)
    with first.results_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"sample_id": sample_id}) + "\n")

    with pytest.raises(ValueError, match="results file"):
        workspace.sweep()


def test_a_resumed_sweep_must_use_the_same_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixing arms or checkpoints across sessions would pair numbers that are not paired."""

    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001", "c0002"))
    passed_parity(workspace)
    workspace.sweep(limit=1)

    with pytest.raises(ValueError, match="configuration"):
        workspace.sweep(arms=("formal", "native"))


def test_the_sweep_requires_a_passed_parity_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))

    with pytest.raises(FileNotFoundError):
        workspace.sweep()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "other", "parity report"),
        ("passed", False, "did not pass"),
        ("protocol_sha256", "0" * 64, "different protocol, cohort"),
        ("dataset_manifest_sha256", "0" * 64, "different protocol, cohort"),
        ("runs", {}, "different protocol, cohort"),
    ],
)
def test_a_parity_report_for_something_else_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))
    passed_parity(workspace)
    path = tmp_path / "parity" / "parity.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    path.write_text(json.dumps(report), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match=message):
        workspace.sweep()


def test_a_non_positive_limit_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))

    with pytest.raises(ValueError, match="limit"):
        workspace.sweep(limit=0)


@pytest.mark.parametrize("kind", ["image", "label"])
def test_a_file_that_drifted_from_the_frozen_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    workspace = build_workspace(tmp_path, monkeypatch, sample_ids=("c0001",))
    if kind == "image":
        target = workspace.data_root / TRAIN_IMAGES / "c0001.jpg"
    else:
        target = workspace.data_root / TRAIN_LABELS / "c0001_train_id.png"
    target.write_bytes(target.read_bytes() + b"\0")

    with pytest.raises(ValueError, match=f"{kind} drifted"):
        workspace.parity()
