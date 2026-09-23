"""One verified read of every released prediction artifact, for the all-seed analysis.

The analysis is pre-registered in ``docs/posthoc/allseed-v1/analysis-plan.md``.
It re-aggregates the per-image artifacts that the released evaluation wrote for
the nine formal runs; no model is run and no released number is changed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from drivemetrics.analysis.extended import (
    CONFIDENCE_LEVELS,
    _bitmask_path,
    _corroborated_instances,
    _dense_prediction,
    _digest_bitmask_set,
    _read_mask,
    _resolve_label_paths,
)
from drivemetrics.artifacts.formal_index import _check_record, _read_run_record
from drivemetrics.artifacts.formal_set import (
    APPROVED_MODELS,
    APPROVED_SEEDS,
    validate_formal_run_index,
)
from drivemetrics.artifacts.predictions import PredictionRecord, read_prediction_artifact
from drivemetrics.metrics.calibration import ECEBinSufficientStatistics, unpack_correctness
from drivemetrics.metrics.instances import instance_coverages
from drivemetrics.posthoc.resolution import load_tertile_edges
from drivemetrics.protocol.hashing import sha256_file

Float64Array = npt.NDArray[np.float64]
UInt16Array = npt.NDArray[np.uint16]
BoolArray = npt.NDArray[np.bool_]

ANALYSIS_PLAN = "docs/posthoc/allseed-v1/analysis-plan.md"
EXTRACT_SCHEMA_VERSION = "driving-risk-allseed-extract/v1"
EXTRACT_FILENAME = "extract.json"
COMPONENTS_FILENAME = "components.npz"
INSTANCES_FILENAME = "instances.jsonl"
HISTOGRAM_DIRNAME = "histograms"
RUN_RECORD_FILENAME = "run_record.json"
#: The two evaluations of every run, and the index key naming each one's directory.
STATES: tuple[str, ...] = ("uncalibrated", "calibrated")
_DIRECTORY_KEYS = {"uncalibrated": "artifacts_dir", "calibrated": "calibrated_artifacts_dir"}
ECE_BINS = 15
#: Top-label ECE bins. 65,535 = 15 x 4,369, so the bins line up exactly with the
#: stored 65,536-level confidence grid and no stored level straddles a bin edge.
TOPLABEL_BINS = 15
_CONFIDENCE_SCALE = 65535
_LEVELS_PER_BIN = _CONFIDENCE_SCALE // TOPLABEL_BINS


def toplabel_ece_components(q16: UInt16Array, correct: BoolArray) -> Float64Array:
    """Per-bin pixel count, confidence sum and correct count of one image's top-1 predictions."""

    if q16.shape != correct.shape:
        raise ValueError("the confidences and the correctness flags must pair pixel for pixel")
    levels = q16.astype(np.int64)
    bins = np.minimum(levels // _LEVELS_PER_BIN, TOPLABEL_BINS - 1)
    stats = np.zeros((TOPLABEL_BINS, 3), dtype=np.float64)
    stats[:, 0] = np.bincount(bins, minlength=TOPLABEL_BINS)
    stats[:, 1] = np.bincount(bins, weights=levels / _CONFIDENCE_SCALE, minlength=TOPLABEL_BINS)
    stats[:, 2] = np.bincount(bins[correct], minlength=TOPLABEL_BINS)
    return stats


def toplabel_ece(stats: Float64Array) -> float:
    """Pixel-weighted mean gap between accuracy and confidence over the bins."""

    total = float(stats[:, 0].sum())
    if total == 0:
        raise ValueError("the histogram holds no pixel to compute an error over")
    return float(np.abs(stats[:, 2] - stats[:, 1]).sum() / total)


@dataclass(frozen=True)
class ExtractResult:
    """Where the extraction wrote, which runs it read, and how many images."""

    output_dir: Path
    runs: tuple[str, ...]
    images: int


def _recorded_payloads(
    directory: Path,
    entry: Mapping[str, Any],
    sample_ids: Sequence[str],
    protocol: str,
    dataset: str,
) -> dict[str, str]:
    """The payload hash the evaluation recorded for each image of this directory."""

    seed = int(entry["seed"])
    record = _read_run_record(directory / RUN_RECORD_FILENAME, str(entry["model"]), seed)
    _check_record(
        record,
        f"{directory.name} of {entry['run_id']}",
        protocol_sha256=protocol,
        seed=seed,
        dataset_manifest_sha256=dataset,
    )
    if set(record.artifacts) != set(sample_ids):
        raise ValueError(f"the run record of {directory} does not record every image of the cohort")
    return dict(record.artifacts)


def _read_verified(
    directory: Path, sample_id: str, payload_sha256: str, protocol: str, dataset: str
) -> tuple[PredictionRecord, ECEBinSufficientStatistics]:
    """Read one artifact and prove it is the one its evaluation wrote for this study."""

    manifest, record, ece = read_prediction_artifact(directory / f"{sample_id}.json")
    where = f"artifact {sample_id} in {directory}"
    if manifest.sample_id != sample_id:
        raise ValueError(f"{where} names itself {manifest.sample_id!r}")
    if manifest.protocol_sha256 != protocol:
        raise ValueError(f"{where} carries a different protocol hash than the run index")
    if manifest.dataset_manifest_sha256 != dataset:
        raise ValueError(f"{where} carries a different dataset manifest hash than the run index")
    if manifest.payload_sha256 != payload_sha256:
        raise ValueError(f"{where} is not the payload its evaluation recorded")
    return record, ece


def _sorted_runs(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The runs in the declared model and seed order, whatever order the index lists."""

    return sorted(
        document["runs"],
        key=lambda entry: (
            APPROVED_MODELS.index(str(entry["model"])),
            APPROVED_SEEDS.index(int(entry["seed"])),
        ),
    )


def _instance_line(sample_id: str, corroboration: Any) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "without_semantic_pixels": corroboration.without_semantic_pixels,
        "corroborated_fractions": corroboration.corroborated_fractions,
        "instances": None,
        "correct_fraction": {},
        "correct_pixels": {},
    }


def extract_all_seeds(
    index_path: Path,
    manifest_path: Path,
    labels_root: Path,
    instance_root: Path,
    tertiles_path: Path,
    output_dir: Path,
    *,
    on_image: Callable[[int, int], None] | None = None,
) -> ExtractResult:
    """Read every artifact of the nine runs once, verify it, and keep what the analysis needs.

    The loop is image-major, so each image's ground truth and instance footprints
    are computed once and shared by all nine runs. ``extract.json`` is written
    last: a directory without it is an extraction that did not finish.
    """

    if (output_dir / EXTRACT_FILENAME).exists():
        raise FileExistsError(f"an extraction already exists: {output_dir / EXTRACT_FILENAME}")
    document = json.loads(index_path.read_text(encoding="utf-8"))
    violations = validate_formal_run_index(document)
    if violations:
        raise ValueError("formal run index is not valid: " + "; ".join(violations))
    runs = _sorted_runs(document)
    sample_ids = tuple(sorted(runs[0]["uncalibrated_sample_ids"]))
    protocol = str(document["protocol_sha256"])
    dataset = str(document["dataset_manifest_sha256"])
    num_classes = int(document["num_classes"])
    _, label_paths = _resolve_label_paths(manifest_path, labels_root, sample_ids, dataset)
    bitmask_digest = _digest_bitmask_set(instance_root, sample_ids)
    edges = load_tertile_edges(tertiles_path)

    directories = {
        (position, state): index_path.parent / str(entry[_DIRECTORY_KEYS[state]])
        for position, entry in enumerate(runs)
        for state in STATES
    }
    payloads = {
        key: _recorded_payloads(directory, runs[key[0]], sample_ids, protocol, dataset)
        for key, directory in directories.items()
    }

    run_count, image_count = len(runs), len(sample_ids)
    per_state = (len(STATES), run_count, image_count)
    arrays: dict[str, npt.NDArray[Any]] = {
        "confusion": np.zeros((run_count, image_count, num_classes * num_classes), np.int64),
        "valid_pixels": np.zeros((run_count, image_count), np.int64),
        "brier": np.zeros((*per_state, num_classes), np.float64),
        "ece_counts": np.zeros((*per_state, num_classes, ECE_BINS), np.int64),
        "ece_confidence_sums": np.zeros((*per_state, num_classes, ECE_BINS), np.float64),
        "ece_positive_counts": np.zeros((*per_state, num_classes, ECE_BINS), np.int64),
        "toplabel": np.zeros((*per_state, TOPLABEL_BINS, 3), np.float64),
    }
    histogram_dir = output_dir / HISTOGRAM_DIRNAME
    histogram_dir.mkdir(parents=True, exist_ok=True)
    histograms = {
        (position, state_index): np.lib.format.open_memmap(
            histogram_dir / f"{runs[position]['run_id']}.{state}.npy",
            mode="w+",
            dtype=np.int32,
            shape=(image_count, 2, CONFIDENCE_LEVELS),
        )
        for position in range(run_count)
        for state_index, state in enumerate(STATES)
    }

    with (output_dir / INSTANCES_FILENAME).open("w", encoding="utf-8", newline="\n") as lines:
        for image, sample_id in enumerate(sample_ids):
            truth = _read_mask(label_paths[sample_id])
            bitmask = _read_mask(_bitmask_path(instance_root, sample_id))
            corroboration = _corroborated_instances(
                truth, bitmask[:, :, 0], (bitmask[:, :, 2] << 8) | bitmask[:, :, 3]
            )
            line = _instance_line(sample_id, corroboration)
            for position, entry in enumerate(runs):
                records: list[PredictionRecord] = []
                for state_index, state in enumerate(STATES):
                    record, ece = _read_verified(
                        directories[(position, state)],
                        sample_id,
                        payloads[(position, state)][sample_id],
                        protocol,
                        dataset,
                    )
                    correct = unpack_correctness(
                        record.correctness_bitset, record.valid_pixel_count
                    )
                    levels = record.top1_confidence_q16.astype(np.int64)
                    where = (state_index, position, image)
                    arrays["brier"][where] = record.brier_sum_by_class
                    arrays["ece_counts"][where] = ece.counts
                    arrays["ece_confidence_sums"][where] = ece.confidence_sums
                    arrays["ece_positive_counts"][where] = ece.positive_counts
                    arrays["toplabel"][where] = toplabel_ece_components(
                        record.top1_confidence_q16, correct
                    )
                    histogram = histograms[(position, state_index)]
                    histogram[image, 0] = np.bincount(levels, minlength=CONFIDENCE_LEVELS)
                    histogram[image, 1] = np.bincount(levels[correct], minlength=CONFIDENCE_LEVELS)
                    records.append(record)
                uncalibrated, calibrated = records
                if not (
                    np.array_equal(uncalibrated.predicted_class, calibrated.predicted_class)
                    and np.array_equal(uncalibrated.confusion, calibrated.confusion)
                ):
                    raise ValueError(
                        f"temperature scaling changed a predicted class of {sample_id} "
                        f"in {entry['run_id']}"
                    )
                arrays["confusion"][position, image] = uncalibrated.confusion.reshape(-1)
                arrays["valid_pixels"][position, image] = uncalibrated.valid_pixel_count
                coverages = instance_coverages(
                    truth.reshape(-1),
                    _dense_prediction(calibrated, truth).reshape(-1),
                    corroboration.footprint_ids.reshape(-1),
                    corroboration.classes,
                    edges,
                )
                if line["instances"] is None:
                    line["instances"] = [
                        {
                            "instance_id": coverage.instance_id,
                            "class_id": coverage.class_id,
                            "area_pixels": coverage.area_pixels,
                            "area_tertile": coverage.area_tertile,
                        }
                        for coverage in coverages
                    ]
                run_id = str(entry["run_id"])
                line["correct_fraction"][run_id] = [c.correct_fraction for c in coverages]
                # The fraction is correct / area rounded once, so this count is exact.
                line["correct_pixels"][run_id] = [
                    round(c.correct_fraction * c.area_pixels) for c in coverages
                ]
            lines.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
            if on_image is not None:
                on_image(image + 1, image_count)

    for histogram in histograms.values():
        histogram.flush()
    histograms.clear()
    np.savez(
        output_dir / COMPONENTS_FILENAME,
        confusion=arrays["confusion"],
        valid_pixels=arrays["valid_pixels"],
        brier=arrays["brier"],
        ece_counts=arrays["ece_counts"],
        ece_confidence_sums=arrays["ece_confidence_sums"],
        ece_positive_counts=arrays["ece_positive_counts"],
        toplabel=arrays["toplabel"],
    )
    summary = {
        "schema_version": EXTRACT_SCHEMA_VERSION,
        "analysis_plan": ANALYSIS_PLAN,
        "index_sha256": sha256_file(index_path),
        "protocol_sha256": protocol,
        "dataset_manifest_sha256": dataset,
        "num_classes": num_classes,
        "critical_class_ids": [int(value) for value in document["critical_class_ids"]],
        "tertiles_sha256": sha256_file(tertiles_path),
        "instance_bitmasks": bitmask_digest,
        "runs": [
            {
                "run_id": str(entry["run_id"]),
                "model": str(entry["model"]),
                "seed": int(entry["seed"]),
                "temperature": float(entry["temperature"]),
            }
            for entry in runs
        ],
        "sample_ids": list(sample_ids),
        "states": list(STATES),
        "confidence_levels": CONFIDENCE_LEVELS,
        "ece_bins": ECE_BINS,
        "gates": {"integrity": "passed"},
    }
    (output_dir / EXTRACT_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return ExtractResult(
        output_dir=output_dir,
        runs=tuple(str(entry["run_id"]) for entry in runs),
        images=image_count,
    )
