"""Post-release resolution sweep of the released checkpoints on the calibration split.

The analysis is pre-registered in ``docs/posthoc/resolution-v1/analysis-plan.md``:
it asks whether small-person critical misses come from the models or from the
inference input being downscaled to about 0.711 of source resolution. Nothing
here changes a released definition. The frozen protocol, preprocessing and
evaluation engine are imported and called, never edited:

- the ``formal`` arm calls the frozen :func:`prepare_sample` and
  :func:`restore_prediction` unchanged, so it is the released geometry by
  construction, and a parity gate proves it against the released scorer;
- the other arms only change the size the source image is resized to before
  the same normalisation, and the prediction is restored to source geometry
  before anything is scored;
- instances are the released corroborated instances, scored with the released
  :func:`instance_coverages` against the frozen area tertiles.

Only a manifest whose split is ``calibration`` is accepted. The locked cohort is
refused in code rather than by convention.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from drivemetrics.analysis.extended import (
    INSTANCE_CATEGORY_TO_TRAIN_ID,
    _corroborated_instances,
)
from drivemetrics.artifacts.run_record import RunProvenance, load_run_provenance
from drivemetrics.data.bdd100k import NUM_TRAIN_CLASSES
from drivemetrics.data.manifest import DatasetManifest, load_manifest
from drivemetrics.data.transforms import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    MASK_PAD_VALUE,
    PreparedSample,
    prepare_sample,
    restore_prediction,
)
from drivemetrics.evaluation.engine import _evaluate_sample
from drivemetrics.metrics.confusion import compute_confusion
from drivemetrics.metrics.instances import InstanceCoverage, instance_coverages
from drivemetrics.models.adapters import SegmentationAdapter
from drivemetrics.protocol.config import load_protocol, split_paths
from drivemetrics.protocol.hashing import sha256_file

Int64Array = npt.NDArray[np.int64]
UInt8Array = npt.NDArray[np.uint8]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

SWEEP_SCHEMA_VERSION = "driving-risk-resolution-sweep/v1"
PARITY_SCHEMA_VERSION = "driving-risk-resolution-parity/v1"
ANALYSIS_PLAN = "docs/posthoc/resolution-v1/analysis-plan.md"
REQUIRED_SPLIT = "calibration"
SOURCE_HEIGHT = 720
SOURCE_WIDTH = 1280
PER_IMAGE_FILENAME = "per_image.jsonl"
RUN_FILENAME = "run.json"
SESSIONS_FILENAME = "sessions.jsonl"
PARITY_CRITERIA = ("exact", "decision")
DEFAULT_PARITY_IMAGES = 20
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: The owner's fallback: only an L4 may use the relaxed decision criterion.
_L4 = re.compile(r"\bL4\b")

SampleObserver = Callable[[int, int], None]


@dataclass(frozen=True)
class Arm:
    """One model-input geometry. ``formal`` means: call the frozen preprocessing."""

    name: str
    height: int
    width: int
    formal: bool = False

    @property
    def scale(self) -> float:
        """Linear scale of the model input relative to the 720-pixel source height."""

        return self.height / SOURCE_HEIGHT


#: Every arm the tool implements, in scale order. The pre-registered arms are
#: ``DEFAULT_ARMS``; ``s060`` and ``formal_unpadded`` are optional controls that
#: are off by default and outside the pre-registration.
ARMS: dict[str, Arm] = {
    "s060": Arm("s060", 432, 768),
    "formal": Arm("formal", 512, 910, formal=True),
    "formal_unpadded": Arm("formal_unpadded", 512, 910),
    "s085": Arm("s085", 612, 1088),
    "native": Arm("native", SOURCE_HEIGHT, SOURCE_WIDTH),
}
DEFAULT_ARMS: tuple[str, ...] = ("formal", "s085", "native")
REQUIRED_ARMS: tuple[str, ...] = ("formal", "native")


def parse_arms(names: Sequence[str]) -> tuple[Arm, ...]:
    """Validate an arm selection and return it in canonical scale order."""

    cleaned = [name.strip() for name in names if name.strip()]
    unknown = sorted(set(cleaned) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; known arms are {list(ARMS)}")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("an arm was listed more than once")
    missing = [name for name in REQUIRED_ARMS if name not in cleaned]
    if missing:
        raise ValueError(f"the arms must include {list(REQUIRED_ARMS)}; missing {missing}")
    return tuple(arm for name, arm in ARMS.items() if name in cleaned)


# ---------------------------------------------------------------------------
# Preprocessing and inference
# ---------------------------------------------------------------------------


def model_input(
    image_hwc: UInt8Array, mask_hw: UInt8Array, arm: Arm
) -> tuple[Float32Array, PreparedSample | None]:
    """Normalised CHW input for one arm; the formal arm is the frozen function itself.

    The non-formal path performs the same operations in the same order as the
    frozen preprocessing (bilinear resize of the uint8 image, scaling to [0, 1],
    ImageNet normalisation), without padding.
    """

    if image_hwc.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError(
            f"the arms are defined for {SOURCE_HEIGHT}x{SOURCE_WIDTH} RGB sources only, "
            f"got {image_hwc.shape}"
        )
    if arm.formal:
        prepared = prepare_sample(image_hwc, mask_hw, training=False, flip_draw=1.0)
        return prepared.image_chw, prepared
    if (arm.height, arm.width) != (SOURCE_HEIGHT, SOURCE_WIDTH):
        image_hwc = np.asarray(
            Image.fromarray(image_hwc).resize(
                (arm.width, arm.height), resample=Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    image_chw = np.transpose(image_hwc.astype(np.float32) / np.float32(255.0), (2, 0, 1))
    normalized = (image_chw - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(normalized, dtype=np.float32), None


def predict_labels(adapter: SegmentationAdapter, image_chw: Float32Array) -> UInt8Array:
    """Argmax labels at model-input size, computed on the model's device.

    This is the released adapter's inference path (evaluation mode, no
    gradients, bilinear upsampling of the logits to the input size) with the
    argmax taken before the logits leave the device. The released engine takes
    the argmax of a float64 softmax of the same float32 logits, which selects the
    same class; the parity gate measures that rather than assuming it.
    """

    import torch

    module = adapter.module
    module.eval()
    device = next(iter(module.parameters())).device
    height, width = int(image_chw.shape[1]), int(image_chw.shape[2])
    with torch.no_grad():
        raw_output = module(torch.from_numpy(image_chw[None, ...]).to(device))
        logits = adapter.extract_logits(raw_output)
        resized = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        labels = resized.argmax(dim=1)[0].to(torch.uint8)
    return np.asarray(labels.cpu().numpy(), dtype=np.uint8)


def source_labels(labels_hw: UInt8Array, prepared: PreparedSample | None) -> UInt8Array:
    """Restore model-input labels to the 720x1280 source geometry with nearest resampling."""

    if prepared is not None:
        return restore_prediction(labels_hw, prepared)
    if labels_hw.shape == (SOURCE_HEIGHT, SOURCE_WIDTH):
        return labels_hw
    return np.asarray(
        Image.fromarray(labels_hw).resize(
            (SOURCE_WIDTH, SOURCE_HEIGHT), resample=Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )


# ---------------------------------------------------------------------------
# Ground truth and scoring
# ---------------------------------------------------------------------------


def load_tertile_edges(tertiles_path: Path) -> dict[int, tuple[int, int]]:
    """Frozen area-tertile edges, translated from instance categories to train IDs."""

    document = json.loads(tertiles_path.read_text(encoding="utf-8"))
    return {
        INSTANCE_CATEGORY_TO_TRAIN_ID[int(category)]: (int(low), int(high))
        for category, (low, high) in document["tertile_edges"].items()
        if int(category) in INSTANCE_CATEGORY_TO_TRAIN_ID
    }


@dataclass(frozen=True)
class ImageTruth:
    """One image's ground truth, prepared once and shared by every arm and run."""

    truth: Int64Array
    valid: BoolArray
    footprint_ids: Int64Array
    classes: dict[int, int]
    excluded_without_semantic_pixels: int
    edges: Mapping[int, tuple[int, int]]
    keep: BoolArray
    truth_on_footprint: Int64Array
    ids_on_footprint: Int64Array
    instances: tuple[InstanceCoverage, ...]


def load_image_truth(
    mask_hw: UInt8Array, bitmask: UInt8Array, edges: Mapping[int, tuple[int, int]]
) -> ImageTruth:
    """Join the semantic mask and the instance bitmask with the released corroboration rule."""

    truth = mask_hw.astype(np.int64)
    if bitmask.ndim != 3 or bitmask.shape[:2] != truth.shape or bitmask.shape[2] != 4:
        raise ValueError(
            f"instance bitmask must be RGBA with the mask's shape {truth.shape}, "
            f"got {bitmask.shape}"
        )
    channels = bitmask.astype(np.int64)
    corroboration = _corroborated_instances(
        truth, channels[:, :, 0], (channels[:, :, 2] << 8) | channels[:, :, 3]
    )
    footprint = corroboration.footprint_ids.reshape(-1)
    keep = footprint != 0
    truth_on_footprint = truth.reshape(-1)[keep]
    ids_on_footprint = footprint[keep]
    # Scoring the ground truth against itself yields every model-independent
    # field of every instance record: ID, class, area and tertile.
    instances = instance_coverages(
        truth_on_footprint,
        truth_on_footprint,
        ids_on_footprint,
        corroboration.classes,
        edges,
    )
    return ImageTruth(
        truth=truth,
        valid=truth != MASK_PAD_VALUE,
        footprint_ids=corroboration.footprint_ids,
        classes=corroboration.classes,
        excluded_without_semantic_pixels=corroboration.without_semantic_pixels,
        edges=edges,
        keep=keep,
        truth_on_footprint=truth_on_footprint,
        ids_on_footprint=ids_on_footprint,
        instances=instances,
    )


def score_labels(
    image_truth: ImageTruth, labels_hw: UInt8Array
) -> tuple[tuple[InstanceCoverage, ...], Int64Array]:
    """Released instance coverage and the 19x19 confusion for one source-geometry prediction.

    The flat arrays are restricted to corroborated instance pixels before the
    released coverage rule runs. Every declared instance keeps all of its
    pixels, so the records are identical to the full-image call, at a fraction
    of the cost.
    """

    predicted = labels_hw.astype(np.int64)
    coverages = instance_coverages(
        image_truth.truth_on_footprint,
        predicted.reshape(-1)[image_truth.keep],
        image_truth.ids_on_footprint,
        image_truth.classes,
        image_truth.edges,
    )
    confusion = compute_confusion(
        image_truth.truth[image_truth.valid], predicted[image_truth.valid], NUM_TRAIN_CLASSES
    )
    return coverages, confusion


# ---------------------------------------------------------------------------
# Cohort, checkpoints and inputs
# ---------------------------------------------------------------------------


class ModelBackend(Protocol):
    """Restores one checkpoint; the evaluation backend satisfies this."""

    def load_model(self, checkpoint_path: Path) -> tuple[SegmentationAdapter, Mapping[str, object]]:
        """Restore the model and the metadata recorded with its checkpoint."""


@dataclass(frozen=True)
class Cohort:
    """The frozen calibration manifest and where its files live."""

    protocol_sha256: str
    manifest: DatasetManifest
    image_root: Path
    label_root: Path


@dataclass(frozen=True)
class LoadedRun:
    """One verified checkpoint, restored."""

    run_id: str
    model: str
    seed: int
    checkpoint_sha256: str
    adapter: SegmentationAdapter


def open_calibration_cohort(config_path: Path, manifest_path: Path, data_root: Path) -> Cohort:
    """Load the protocol and the manifest, refusing any cohort but calibration."""

    loaded = load_protocol(config_path)
    manifest = load_manifest(manifest_path)
    if manifest.split_name != REQUIRED_SPLIT:
        raise ValueError(
            f"the resolution analysis runs on the {REQUIRED_SPLIT} cohort only; "
            f"refusing split {manifest.split_name!r}"
        )
    image_root_name, label_root_name = split_paths(loaded.protocol, manifest.split_name)
    return Cohort(
        protocol_sha256=loaded.protocol_sha256,
        manifest=manifest,
        image_root=data_root / image_root_name,
        label_root=data_root / label_root_name,
    )


def read_checkpoint_list(path: Path) -> dict[str, tuple[Path, str]]:
    """Read ``{run_id: [checkpoint path, expected SHA-256]}``."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not document:
        raise ValueError("the checkpoint list must be a non-empty JSON object")
    entries: dict[str, tuple[Path, str]] = {}
    for run_id, entry in document.items():
        if not (
            isinstance(entry, list)
            and len(entry) == 2
            and all(isinstance(value, str) for value in entry)
        ):
            raise ValueError(f"checkpoint list entry {run_id!r} must be [path, sha256]")
        if not _SHA256.fullmatch(entry[1]):
            raise ValueError(f"checkpoint list entry {run_id!r} has no lowercase hex SHA-256")
        entries[run_id] = (Path(entry[0]), entry[1])
    return entries


def load_runs(
    checkpoints_path: Path, backend: ModelBackend, protocol_sha256: str
) -> tuple[LoadedRun, ...]:
    """Verify each checkpoint's bytes and metadata, then restore it."""

    runs: list[LoadedRun] = []
    for run_id, (path, expected_sha256) in sorted(read_checkpoint_list(checkpoints_path).items()):
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"checkpoint {run_id} does not match its recorded SHA-256")
        adapter, metadata = backend.load_model(path)
        if metadata.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"checkpoint {run_id} was not trained under the formal protocol")
        if metadata.get("run_id") != run_id:
            raise ValueError(
                f"checkpoint listed as {run_id} names itself {metadata.get('run_id')!r}"
            )
        runs.append(
            LoadedRun(
                run_id=run_id,
                model=str(metadata["model"]),
                seed=int(str(metadata["seed"])),
                checkpoint_sha256=expected_sha256,
                adapter=adapter,
            )
        )
    return tuple(runs)


def _verified_sample_paths(cohort: Cohort, position: int) -> tuple[Path, Path]:
    manifest = cohort.manifest
    image_path = cohort.image_root / manifest.relative_image_paths[position]
    label_path = cohort.label_root / manifest.relative_label_paths[position]
    sample_id = manifest.sample_ids[position]
    if sha256_file(image_path) != manifest.file_sha256[2 * position]:
        raise ValueError(f"image drifted from the frozen manifest: {sample_id}")
    if sha256_file(label_path) != manifest.file_sha256[2 * position + 1]:
        raise ValueError(f"label drifted from the frozen manifest: {sample_id}")
    return image_path, label_path


def _read_image(path: Path, *, rgb: bool) -> UInt8Array:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB") if rgb else handle, dtype=np.uint8)


@dataclass(frozen=True)
class _Sample:
    sample_id: str
    image_path: Path
    label_path: Path
    bitmask_path: Path
    image: UInt8Array
    mask: UInt8Array
    truth: ImageTruth


def _load_sample(
    cohort: Cohort, position: int, bitmask_dir: Path, edges: Mapping[int, tuple[int, int]]
) -> _Sample:
    sample_id = cohort.manifest.sample_ids[position]
    image_path, label_path = _verified_sample_paths(cohort, position)
    bitmask_path = bitmask_dir / f"{sample_id}.png"
    mask = _read_image(label_path, rgb=False)
    return _Sample(
        sample_id=sample_id,
        image_path=image_path,
        label_path=label_path,
        bitmask_path=bitmask_path,
        image=_read_image(image_path, rgb=True),
        mask=mask,
        truth=load_image_truth(mask, _read_image(bitmask_path, rgb=False), edges),
    )


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _gpu(provenance: RunProvenance) -> str:
    return provenance.hardware.get("gpu", "")


# ---------------------------------------------------------------------------
# Parity gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityResult:
    """A passed parity gate and where its report was written."""

    report_path: Path
    criterion: str
    mismatched_pixels: dict[str, int]
    changed_decisions: dict[str, int]


def formal_parity(
    config_path: Path,
    manifest_path: Path,
    checkpoints_path: Path,
    data_root: Path,
    bitmask_dir: Path,
    tertiles_path: Path,
    output_path: Path,
    *,
    backend: ModelBackend,
    images: int = DEFAULT_PARITY_IMAGES,
    criterion: str = "exact",
) -> ParityResult:
    """Prove the formal arm reproduces the released evaluation path on this machine.

    The first ``images`` calibration images are scored for every checkpoint by
    the released per-sample scorer at temperature 1 (argmax does not depend on
    the temperature) and by the formal arm. ``exact`` requires zero mismatched
    pixels. ``decision`` requires zero changed critical-miss decisions and is
    accepted only on an L4, the owner's pre-registered fallback; an A100 keeps
    exact parity. The report is written before a failure is raised, so the
    evidence of a failed gate is never lost.
    """

    if criterion not in PARITY_CRITERIA:
        raise ValueError(f"parity criterion must be one of {PARITY_CRITERIA}, got {criterion!r}")
    if images <= 0:
        raise ValueError("parity images must be a positive number")
    provenance = load_run_provenance()
    if criterion == "decision" and not _L4.search(_gpu(provenance)):
        raise ValueError(
            "the relaxed decision criterion is allowed only on an L4; "
            f"this runtime reports {_gpu(provenance)!r} and keeps exact parity"
        )
    cohort = open_calibration_cohort(config_path, manifest_path, data_root)
    runs = load_runs(checkpoints_path, backend, cohort.protocol_sha256)
    edges = load_tertile_edges(tertiles_path)
    count = min(images, len(cohort.manifest.sample_ids))

    tallies: dict[str, dict[str, Any]] = {
        run.run_id: {
            "checkpoint_sha256": run.checkpoint_sha256,
            "mismatched_pixels": 0,
            "changed_critical_miss_decisions": 0,
            "compared_pixels": 0,
            "compared_instances": 0,
        }
        for run in runs
    }
    formal = ARMS["formal"]
    for position in range(count):
        sample = _load_sample(cohort, position, bitmask_dir, edges)
        image_chw, prepared = model_input(sample.image, sample.mask, formal)
        valid = sample.truth.valid
        for run in runs:
            ours = source_labels(predict_labels(run.adapter, image_chw), prepared)
            record, _ = _evaluate_sample(
                run.adapter, sample.image_path, sample.label_path, sample.sample_id, 1.0
            )
            released = ours.copy()
            released[valid] = record.predicted_class
            ours_records, _ = score_labels(sample.truth, ours)
            released_records, _ = score_labels(sample.truth, released)
            tally = tallies[run.run_id]
            tally["mismatched_pixels"] += int(np.count_nonzero(ours[valid] != released[valid]))
            tally["changed_critical_miss_decisions"] += sum(
                mine.is_critical_miss != theirs.is_critical_miss
                for mine, theirs in zip(ours_records, released_records, strict=True)
            )
            tally["compared_pixels"] += int(np.count_nonzero(valid))
            tally["compared_instances"] += len(ours_records)

    measure = "mismatched_pixels" if criterion == "exact" else "changed_critical_miss_decisions"
    passed = all(tally[measure] == 0 for tally in tallies.values())
    _write_json(
        output_path,
        {
            "schema_version": PARITY_SCHEMA_VERSION,
            "analysis_plan": ANALYSIS_PLAN,
            "criterion": criterion,
            "passed": passed,
            "protocol_sha256": cohort.protocol_sha256,
            "dataset_manifest_sha256": cohort.manifest.manifest_sha256,
            "sample_ids": list(cohort.manifest.sample_ids[:count]),
            "commit": provenance.commit,
            "lock_sha256": provenance.lock_sha256,
            "hardware": dict(provenance.hardware),
            "created_at_utc": _utc_now(),
            "runs": tallies,
        },
    )
    if not passed:
        raise ValueError(
            f"parity gate failed under the {criterion} criterion: the formal arm disagrees "
            f"with the released evaluation path; see {output_path}"
        )
    return ParityResult(
        report_path=output_path,
        criterion=criterion,
        mismatched_pixels={key: int(value["mismatched_pixels"]) for key, value in tallies.items()},
        changed_decisions={
            key: int(value["changed_critical_miss_decisions"]) for key, value in tallies.items()
        },
    )


def _require_passed_parity(
    report_path: Path, cohort: Cohort, runs: Sequence[LoadedRun]
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != PARITY_SCHEMA_VERSION:
        raise ValueError(f"{report_path} is not a resolution parity report")
    if report.get("passed") is not True:
        raise ValueError("the parity gate did not pass; the sweep will not start")
    recorded = {run_id: entry.get("checkpoint_sha256") for run_id, entry in report["runs"].items()}
    if (
        report.get("protocol_sha256") != cohort.protocol_sha256
        or report.get("dataset_manifest_sha256") != cohort.manifest.manifest_sha256
        or recorded != {run.run_id: run.checkpoint_sha256 for run in runs}
    ):
        raise ValueError(
            "the parity report was made for a different protocol, cohort or set of checkpoints"
        )
    return dict(report)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepResult:
    """Where the sweep wrote and how far it has got."""

    results_path: Path
    scored_samples: int
    completed_samples: int
    total_samples: int


def _scored_sample_ids(results_path: Path, cohort_ids: set[str]) -> set[str]:
    """Sample IDs already on disk, after dropping a line an interruption cut short."""

    if not results_path.exists():
        return set()
    raw = results_path.read_bytes()
    if not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        with results_path.open("r+b") as handle:
            handle.truncate(cut)
        raw = raw[:cut]
    done: set[str] = set()
    for line in raw.decode("utf-8").splitlines():
        sample_id = json.loads(line)["sample_id"]
        if sample_id not in cohort_ids or sample_id in done:
            raise ValueError(
                f"the results file holds {sample_id!r} twice or from outside this cohort"
            )
        done.add(sample_id)
    return done


def _sweep_configuration(
    cohort: Cohort, runs: Sequence[LoadedRun], arms: Sequence[Arm], tertiles_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "analysis_plan": ANALYSIS_PLAN,
        "split_name": cohort.manifest.split_name,
        "protocol_sha256": cohort.protocol_sha256,
        "dataset_manifest_sha256": cohort.manifest.manifest_sha256,
        "tertiles_sha256": sha256_file(tertiles_path),
        "source_geometry": [SOURCE_HEIGHT, SOURCE_WIDTH],
        "arms": [
            {
                "name": arm.name,
                "height": arm.height,
                "width": arm.width,
                "scale": arm.scale,
                "formal": arm.formal,
            }
            for arm in arms
        ],
        "runs": [
            {
                "run_id": run.run_id,
                "model": run.model,
                "seed": run.seed,
                "checkpoint_sha256": run.checkpoint_sha256,
            }
            for run in runs
        ],
        "sample_ids": list(cohort.manifest.sample_ids),
    }


def _score_sample(
    sample: _Sample, arms: Sequence[Arm], runs: Sequence[LoadedRun]
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for arm in arms:
        image_chw, prepared = model_input(sample.image, sample.mask, arm)
        results[arm.name] = {}
        for run in runs:
            labels = source_labels(predict_labels(run.adapter, image_chw), prepared)
            coverages, confusion = score_labels(sample.truth, labels)
            results[arm.name][run.run_id] = {
                "correct_fraction": [record.correct_fraction for record in coverages],
                "critical_miss": [record.is_critical_miss for record in coverages],
                "confusion": confusion.reshape(-1).tolist(),
            }
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "bitmask_sha256": sha256_file(sample.bitmask_path),
        "excluded_without_semantic_pixels": sample.truth.excluded_without_semantic_pixels,
        "instances": [
            {
                "instance_id": record.instance_id,
                "class_id": record.class_id,
                "area_pixels": record.area_pixels,
                "area_tertile": record.area_tertile,
            }
            for record in sample.truth.instances
        ],
        "results": results,
    }


def resolution_sweep(
    config_path: Path,
    manifest_path: Path,
    checkpoints_path: Path,
    data_root: Path,
    bitmask_dir: Path,
    tertiles_path: Path,
    parity_report_path: Path,
    output_dir: Path,
    *,
    backend: ModelBackend,
    arms: Sequence[str] = DEFAULT_ARMS,
    limit: int | None = None,
    on_sample: SampleObserver | None = None,
) -> SweepResult:
    """Score up to ``limit`` not-yet-scored calibration images; resumable.

    Each image appends one JSON line holding the model-independent instance
    records, and per arm and run the correct fractions, the critical-miss
    decisions and the 19x19 confusion. A resumed sweep must use exactly the same
    configuration (``run.json``), and each invocation appends its provenance to
    ``sessions.jsonl``. The sweep refuses to start without a passed parity report
    for the same protocol, cohort and checkpoints.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive number of images")
    provenance = load_run_provenance()
    selected = parse_arms(arms)
    cohort = open_calibration_cohort(config_path, manifest_path, data_root)
    runs = load_runs(checkpoints_path, backend, cohort.protocol_sha256)
    parity = _require_passed_parity(parity_report_path, cohort, runs)
    edges = load_tertile_edges(tertiles_path)

    configuration = _sweep_configuration(cohort, runs, selected, tertiles_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / RUN_FILENAME
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError(
                f"{run_path} records a different sweep configuration; "
                "resume with the same arms, checkpoints and cohort, or use a new output directory"
            )
    else:
        _write_json(run_path, configuration)

    results_path = output_dir / PER_IMAGE_FILENAME
    manifest = cohort.manifest
    done = _scored_sample_ids(results_path, set(manifest.sample_ids))
    todo = [
        position for position, sample_id in enumerate(manifest.sample_ids) if sample_id not in done
    ][:limit]

    started_at_utc = _utc_now()
    for count, position in enumerate(todo, 1):
        sample = _load_sample(cohort, position, bitmask_dir, edges)
        line = _score_sample(sample, selected, runs)
        with results_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
        if on_sample is not None:
            on_sample(count, len(todo))

    session = {
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
        "provenance": provenance.model_dump(mode="json"),
        "parity_criterion": parity["criterion"],
        "parity_report_sha256": sha256_file(parity_report_path),
        "scored_samples": len(todo),
    }
    with (output_dir / SESSIONS_FILENAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(session, sort_keys=True, separators=(",", ":")) + "\n")
    return SweepResult(
        results_path=results_path,
        scored_samples=len(todo),
        completed_samples=len(done) + len(todo),
        total_samples=len(manifest.sample_ids),
    )
