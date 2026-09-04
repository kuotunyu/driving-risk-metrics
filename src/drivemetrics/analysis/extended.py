"""The published metrics that need more than the confusion matrices.

Three blocks, and they do not need the same inputs.

Selective risk is rebuilt from the confidence histogram every artifact already
carries, so it is always computed. Per-band pixel accuracy needs the ground-truth
semantic mask. Instance coverage and area tertiles need the instance bitmasks as
well, and the frozen tertile edges — which were learned from the training
intersection and must never be re-learned here, because learning them on the
evaluation cohort would let the locked split influence a decision.

Each block is therefore independently optional, and an absent input produces an
explicit ``not_computed`` record naming what was missing. A block that silently
vanished would be indistinguishable from a block whose value was zero, and the
release cites this document.

Every ground-truth metric is computed on the CALIBRATED evaluation. Temperature
scaling is monotonic, so argmax and therefore every one of these numbers is
identical in the two evaluations; naming the one that was read is still part of
saying what was measured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from drivemetrics.artifacts.formal_set import (
    APPROVED_MODELS,
    APPROVED_SEEDS,
    validate_formal_run_index,
)
from drivemetrics.artifacts.predictions import PredictionRecord, read_prediction_artifact
from drivemetrics.data.manifest import DatasetManifest, load_manifest
from drivemetrics.data.transforms import MASK_PAD_VALUE
from drivemetrics.metrics.calibration import unpack_correctness
from drivemetrics.metrics.instances import instance_coverages
from drivemetrics.metrics.selective import (
    area_under_risk_coverage,
    selective_risk_from_histogram,
)
from drivemetrics.metrics.spatial import NORMALIZED_IMAGE_BAND_NAMES, normalized_image_bands
from drivemetrics.protocol.hashing import sha256_file
from drivemetrics.protocol.risk_profiles import BDD100K_SEMANTIC_CLASS_NAMES

Int64Array = npt.NDArray[np.int64]

EXTENDED_SCHEMA_VERSION = "driving-risk-extended-metrics/v1"
GROUND_TRUTH_EVALUATION = "eval_calibrated"
#: The confidence axis the artifacts quantize onto. The curve is defined at these
#: boundaries and nowhere between them; inside a bin the pixels are
#: indistinguishable at the stored precision.
CONFIDENCE_LEVELS = 65536
BAND_DEFINITION = "normalized image rows only; not physical depth or metric distance"
NO_GROUND_TRUTH = "ground truth was not available to this analysis run"
NO_TERTILES = "the frozen area tertiles were not supplied"
#: BDD100K numbers its instance categories 1..8 in a space of its own while the
#: semantic masks carry the nineteen Cityscapes train IDs. They are two numbering
#: systems for the same eight things, and comparing them directly would compare a
#: pedestrian to a building. The correspondence is written as NAMES so that a
#: reader can check it against the semantic class table instead of trusting eight
#: magic numbers, and the train IDs are derived from that table rather than
#: restated, so a change to the class list breaks this loudly instead of quietly.
INSTANCE_CATEGORY_NAMES: tuple[tuple[int, str], ...] = (
    (1, "person"),  # BDD100K writes "pedestrian"; the semantic table writes "person"
    (2, "rider"),
    (3, "car"),
    (4, "truck"),
    (5, "bus"),
    (6, "train"),
    (7, "motorcycle"),
    (8, "bicycle"),
)
INSTANCE_CATEGORY_TO_TRAIN_ID: dict[int, int] = {
    category: BDD100K_SEMANTIC_CLASS_NAMES.index(name) for category, name in INSTANCE_CATEGORY_NAMES
}


@dataclass(frozen=True)
class ExtendedMetricsResult:
    """Where the document was written and which blocks it actually holds."""

    document_path: Path
    models: tuple[str, ...]
    computed: tuple[str, ...]


def _confidence_histogram(
    directory: Path, sample_ids: Sequence[str]
) -> tuple[Int64Array, Int64Array]:
    """Pool one evaluation's confidence histogram over the whole cohort.

    Counts add across images exactly, which is the only reason a cohort curve can
    be built at all: nine hundred million per-pixel confidences are not retained
    anywhere and never were.
    """

    counts = np.zeros(CONFIDENCE_LEVELS, dtype=np.int64)
    correct = np.zeros(CONFIDENCE_LEVELS, dtype=np.int64)
    for sample_id in sample_ids:
        _, record, _ = read_prediction_artifact(directory / f"{sample_id}.json")
        confidence = record.top1_confidence_q16.astype(np.int64)
        correctness = unpack_correctness(record.correctness_bitset, record.valid_pixel_count)
        counts += np.bincount(confidence, minlength=CONFIDENCE_LEVELS)
        correct += np.bincount(confidence[correctness], minlength=CONFIDENCE_LEVELS)
    return counts, correct


def _selective_block(directory: Path, sample_ids: Sequence[str]) -> dict[str, Any]:
    counts, correct = _confidence_histogram(directory, sample_ids)
    coverage, risk = selective_risk_from_histogram(counts, correct)
    # A model that emitted one confidence for every pixel gives a curve of one
    # point, which has no span and therefore no area. That is a real answer about
    # a degenerate model, and reporting a number for it would invent one.
    area = area_under_risk_coverage(coverage, risk) if coverage.size > 1 else None
    return {
        "aurc": area,
        "coverage_points": int(coverage.size),
        "defined_at": "confidence_bin_boundaries",
    }


def _read_mask(path: Path) -> Int64Array:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image, dtype=np.int64)


def _dense_prediction(record: PredictionRecord, truth: Int64Array) -> Int64Array:
    """Scatter an artifact's valid-pixel predictions back onto the image grid.

    The artifact holds one prediction per NON-IGNORED pixel, in row-major order
    of the native mask, because that is exactly the set of pixels the evaluation
    scored. Reshaping that flat array to the image's shape would misalign every
    pixel after the first ignored one and would still produce a plausible-looking
    number, so the values are scattered through the same ignore mask the
    evaluation used and the ignored positions keep the ignore value.

    The length check is the pairing proof. A mask from a different image, or from
    a different preprocessing of this one, cannot have exactly this many
    non-ignored pixels by accident.
    """

    valid = truth != MASK_PAD_VALUE
    expected = int(np.count_nonzero(valid))
    if record.predicted_class.size != expected:
        raise ValueError(
            f"artifact {record.sample_id} holds {record.predicted_class.size} predictions "
            f"but its ground-truth mask has {expected} non-ignored pixels"
        )
    dense = np.full(truth.shape, MASK_PAD_VALUE, dtype=np.int64)
    dense[valid] = record.predicted_class.astype(np.int64)
    return dense


def _band_block(
    directory: Path,
    sample_ids: Sequence[str],
    label_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Pixel accuracy per normalized image band, pooled over the cohort.

    Pooled rather than averaged per image: images differ in size, and a mean of
    per-image rates would weight a small image like a large one, which is the
    same conflation the cohort metrics exist to prevent.

    Ignored pixels are excluded from both halves of every rate. No model was
    asked about them, so counting them in a denominator would publish a model as
    wrong about pixels it was never shown.
    """

    correct = np.zeros(len(NORMALIZED_IMAGE_BAND_NAMES), dtype=np.int64)
    total = np.zeros(len(NORMALIZED_IMAGE_BAND_NAMES), dtype=np.int64)
    for sample_id in sample_ids:
        _, record, _ = read_prediction_artifact(directory / f"{sample_id}.json")
        truth = _read_mask(label_paths[sample_id])
        predicted = _dense_prediction(record, truth)
        scored = truth != MASK_PAD_VALUE
        bands = np.broadcast_to(normalized_image_bands(truth.shape[0])[:, None], truth.shape)
        hit = scored & (predicted == truth)
        for band_id in range(len(NORMALIZED_IMAGE_BAND_NAMES)):
            selected = bands == band_id
            total[band_id] += int(np.count_nonzero(scored & selected))
            correct[band_id] += int(np.count_nonzero(hit & selected))

    block: dict[str, Any] = {"definition": BAND_DEFINITION}
    for band_id, name in enumerate(NORMALIZED_IMAGE_BAND_NAMES):
        block[name] = {
            "pixels": int(total[band_id]),
            "pixel_accuracy": (
                float(correct[band_id]) / float(total[band_id]) if total[band_id] else None
            ),
        }
    return block


def _corroborated_instances(
    truth: Int64Array, categories: Int64Array, annotation_ids: Int64Array
) -> tuple[dict[int, int], int, int]:
    """Keep the instances the semantic mask agrees with, and count the rest.

    Two independent rasterizations are being joined. An instance bitmask and a
    semantic mask are separate annotations of the same image and they need not
    agree pixel for pixel, so an instance is scored only when it has at least one
    non-ignored semantic pixel and every one of those pixels carries its own
    class. An instance the two annotations disagree about says nothing about the
    model, and scoring it would attribute an annotation conflict to the network.

    The exclusions are returned rather than dropped. A silent drop would shrink
    the denominator of every rate below it without leaving a trace.
    """

    corroborated: dict[int, int] = {}
    without_semantic_pixels = 0
    disagreeing = 0
    for value in np.unique(annotation_ids[annotation_ids != 0]):
        instance_id = int(value)
        instance_pixels = annotation_ids == instance_id
        scored = instance_pixels & (truth != MASK_PAD_VALUE)
        if not np.any(scored):
            without_semantic_pixels += 1
            continue
        category = int(categories[instance_pixels][0])
        train_id = INSTANCE_CATEGORY_TO_TRAIN_ID.get(category)
        if train_id is None or bool(np.any(truth[scored] != train_id)):
            disagreeing += 1
            continue
        corroborated[instance_id] = train_id
    return corroborated, without_semantic_pixels, disagreeing


def _instance_block(
    directory: Path,
    sample_ids: Sequence[str],
    label_paths: Mapping[str, Path],
    instance_root: Path,
    tertiles_path: Path,
) -> dict[str, Any]:
    """Instance coverage against the FROZEN tertile edges, never re-learned edges.

    The frozen edges are keyed by BDD100K instance category, and everything the
    semantic mask carries is keyed by train ID, so the edges are translated once
    here into the space the comparison actually happens in.
    """

    edges_document = json.loads(tertiles_path.read_text(encoding="utf-8"))
    edges_by_train_id = {
        INSTANCE_CATEGORY_TO_TRAIN_ID[int(category)]: (int(low), int(high))
        for category, (low, high) in edges_document["tertile_edges"].items()
        if int(category) in INSTANCE_CATEGORY_TO_TRAIN_ID
    }

    by_tertile: dict[str, dict[str, Any]] = {
        name: {"instance_count": 0, "critical_misses": 0, "coverage_sum": 0.0}
        for name in ("small", "medium", "large")
    }
    instance_count = 0
    without_semantic_pixels = 0
    disagreeing = 0
    for sample_id in sample_ids:
        _, record, _ = read_prediction_artifact(directory / f"{sample_id}.json")
        truth = _read_mask(label_paths[sample_id])
        predicted = _dense_prediction(record, truth)
        bitmask = _read_mask(
            instance_root / "labels" / "ins_seg" / "bitmasks" / "val" / f"{sample_id}.png"
        )
        categories = bitmask[:, :, 0]
        annotation_ids = (bitmask[:, :, 2] << 8) | bitmask[:, :, 3]
        corroborated, unlabelled, conflicting = _corroborated_instances(
            truth, categories, annotation_ids
        )
        without_semantic_pixels += unlabelled
        disagreeing += conflicting
        for coverage in instance_coverages(
            truth.reshape(-1),
            predicted.reshape(-1),
            annotation_ids.reshape(-1),
            corroborated,
            edges_by_train_id,
        ):
            instance_count += 1
            bucket = by_tertile[coverage.area_tertile]
            bucket["instance_count"] += 1
            bucket["critical_misses"] += int(coverage.is_critical_miss)
            bucket["coverage_sum"] += coverage.correct_fraction

    for bucket in by_tertile.values():
        count = bucket.pop("instance_count")
        coverage_sum = bucket.pop("coverage_sum")
        bucket["instance_count"] = count
        bucket["mean_correct_fraction"] = coverage_sum / count if count else None
    return {
        "tertile_edges_from": str(tertiles_path),
        "instance_count": instance_count,
        "excluded_without_semantic_pixels": without_semantic_pixels,
        "excluded_semantic_class_disagreement": disagreeing,
        "by_tertile": by_tertile,
    }


def _resolve_label_paths(
    manifest_path: Path,
    labels_root: Path,
    sample_ids: Sequence[str],
    dataset_manifest_sha256: str,
) -> tuple[DatasetManifest, dict[str, Path]]:
    """Locate every ground-truth mask through the frozen manifest, and verify it.

    The paths are not assembled from a layout convention. They are the relative
    paths the manifest froze, which are the paths the nine runs actually read, so
    a directory that merely resembles a BDD100K tree cannot stand in for the
    cohort. Each file is then checked against the digest the manifest recorded,
    which is what makes the published band and instance numbers attributable: the
    masks that produced them are named, and no other file can be substituted
    without the check failing.
    """

    manifest = load_manifest(manifest_path)
    if manifest.manifest_sha256 != dataset_manifest_sha256:
        raise ValueError(
            "the ground-truth manifest is not the cohort this run index was built from: "
            f"manifest {manifest.manifest_sha256}, index {dataset_manifest_sha256}"
        )
    if tuple(sorted(manifest.sample_ids)) != tuple(sample_ids):
        raise ValueError("the manifest cohort and the run index cohort are not the same images")

    paths: dict[str, Path] = {}
    for position, sample_id in enumerate(manifest.sample_ids):
        path = labels_root / manifest.relative_label_paths[position]
        if sha256_file(path) != manifest.file_sha256[2 * position + 1]:
            raise ValueError(
                f"the ground-truth mask for {sample_id} is not the file the manifest froze"
            )
        paths[sample_id] = path
    return manifest, paths


def extended_metrics(
    index_path: Path,
    output_path: Path,
    *,
    manifest_path: Path | None = None,
    labels_root: Path | None = None,
    instance_root: Path | None = None,
    tertiles_path: Path | None = None,
) -> ExtendedMetricsResult:
    """Publish the blocks this evidence supports, and name what it does not."""

    if (manifest_path is None) != (labels_root is None):
        raise ValueError(
            "the ground-truth blocks need both the frozen manifest and the labels root, "
            "or neither; one without the other cannot place a single mask"
        )
    if output_path.exists():
        raise FileExistsError(f"an extended metrics document already exists: {output_path}")

    document = json.loads(index_path.read_text(encoding="utf-8"))
    violations = validate_formal_run_index(document)
    if violations:
        raise ValueError("formal run index is not valid: " + "; ".join(violations))

    index_dir = index_path.parent
    runs = sorted(
        document["runs"],
        key=lambda entry: (
            APPROVED_MODELS.index(str(entry["model"])),
            APPROVED_SEEDS.index(int(entry["seed"])),
        ),
    )
    sample_ids = tuple(sorted(runs[0]["uncalibrated_sample_ids"]))
    models = tuple(dict.fromkeys(str(entry["model"]) for entry in runs))

    manifest: DatasetManifest | None = None
    label_paths: dict[str, Path] = {}
    if manifest_path is not None and labels_root is not None:
        manifest, label_paths = _resolve_label_paths(
            manifest_path, labels_root, sample_ids, str(document["dataset_manifest_sha256"])
        )

    selective: dict[str, Any] = {}
    bands: dict[str, Any] = {}
    instances: dict[str, Any] = {}
    for model in models:
        entries = [entry for entry in runs if str(entry["model"]) == model]
        selective[model] = {
            kind: _mean_of_blocks(
                [_selective_block(index_dir / str(entry[key]), sample_ids) for entry in entries]
            )
            for kind, key in (
                ("uncalibrated", "artifacts_dir"),
                ("calibrated", "calibrated_artifacts_dir"),
            )
        }
        if manifest is not None:
            calibrated = [index_dir / str(entry["calibrated_artifacts_dir"]) for entry in entries]
            bands[model] = _mean_of_band_blocks(
                [_band_block(directory, sample_ids, label_paths) for directory in calibrated]
            )
            if instance_root is not None and tertiles_path is not None:
                instances[model] = _instance_block(
                    calibrated[0], sample_ids, label_paths, instance_root, tertiles_path
                )

    computed = ["selective_risk"]
    if bands:
        bands["definition"] = BAND_DEFINITION
        computed.append("normalized_image_bands")
    else:
        bands = {"not_computed": NO_GROUND_TRUTH}
    if instances:
        computed.append("instances")
    elif manifest is None:
        instances = {"not_computed": NO_GROUND_TRUTH}
    else:
        instances = {"not_computed": NO_TERTILES}

    payload = {
        "schema_version": EXTENDED_SCHEMA_VERSION,
        "protocol_hash": str(document["protocol_sha256"]),
        "dataset_manifest_hash": str(document["dataset_manifest_sha256"]),
        "evaluation_for_ground_truth_metrics": GROUND_TRUTH_EVALUATION,
        "ground_truth": (
            {
                "cohort_manifest_sha256": manifest.manifest_sha256,
                "split_name": manifest.split_name,
                "masks_verified": len(label_paths),
            }
            if manifest is not None
            else {"not_computed": NO_GROUND_TRUTH}
        ),
        "selective_risk": selective,
        "normalized_image_bands": bands,
        "instances": instances,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ExtendedMetricsResult(document_path=output_path, models=models, computed=tuple(computed))


def _mean_of_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average the seeds' curves by their summary, keeping the shared fields."""

    areas = [block["aurc"] for block in blocks if block["aurc"] is not None]
    return {
        "aurc": float(np.mean(areas)) if areas else None,
        "coverage_points": int(np.mean([block["coverage_points"] for block in blocks])),
        "defined_at": blocks[0]["defined_at"],
    }


def _mean_of_band_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average each band across seeds; the pixel count is the same in every seed."""

    merged: dict[str, Any] = {}
    for name in NORMALIZED_IMAGE_BAND_NAMES:
        rates = [block[name]["pixel_accuracy"] for block in blocks]
        present = [rate for rate in rates if rate is not None]
        merged[name] = {
            "pixels": blocks[0][name]["pixels"],
            "pixel_accuracy": float(np.mean(present)) if present else None,
        }
    return merged
