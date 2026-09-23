"""A tiny formal study shaped like the real one, for the all-seed analysis tests.

It follows the fixture of tests/unit/analysis/test_extended.py on every point that
let defects through before: masks carry IGNORED pixels and artifacts store one
prediction per non-ignored pixel; instance bitmasks number categories 1..8 while
masks carry nineteen train IDs; annotation IDs need two bytes. It adds what the
all-seed extraction reads and the extended fixture does not: a run_record.json
beside every artifact directory, whose artifacts map is checked against every
payload.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from drivemetrics.analysis.aggregate import aggregate_runs
from drivemetrics.analysis.extended import extended_metrics
from drivemetrics.artifacts.predictions import PredictionRecord, write_prediction_artifact
from drivemetrics.data.manifest import build_paired_manifest, save_manifest
from drivemetrics.metrics.calibration import (
    classwise_ece_sufficient_statistics,
    multiclass_brier_sums,
    pack_correctness,
    quantize_confidence,
)

MODELS = ("segformer_b2", "upernet_convnextv2_tiny", "upernet_dinov2_small")
SEEDS = (17, 42, 73)
PROTOCOL = "a" * 64
NUM_CLASSES = 19
IGNORE = 255
SAMPLES = ("v0001", "v0002", "v0003")
HEIGHT, WIDTH = 6, 4
TRUTH = np.array(
    [
        [10, 10, 10, IGNORE],
        [1, 10, 10, 10],
        [2, 2, 11, 11],
        [2, 2, 11, 11],
        [0, 0, 13, 13],
        [0, 0, IGNORE, IGNORE],
    ],
    dtype=np.int64,
)
VALID = TRUTH != IGNORE
#: The four-pixel person footprint is SMALL and the two-pixel car footprint is
#: LARGE, so the small-person statistics of the analysis have something to count.
TERTILE_EDGES = {"1": [4, 8], "3": [1, 1]}
PERSON_PIXELS = ((2, 2), (2, 3), (3, 2), (3, 3))

Prediction = Callable[[str, str, int], np.ndarray]


def default_prediction(sample_id: str, model: str, seed: int) -> np.ndarray:
    """Correct except for errors that make models, seeds and images distinguishable.

    The person is missed (three of its four pixels wrong) by UperNet-DINOv2 in
    every seed and image, by UperNet-ConvNeXtV2 in seed 42 only, and never by
    SegFormer. One top-band pixel is wrong for every model but SegFormer, and the
    bottom band is wrong in the first image.
    """

    predicted = TRUTH.copy()
    missed = model == "upernet_dinov2_small" or (model == "upernet_convnextv2_tiny" and seed == 42)
    for row, column in PERSON_PIXELS[: 3 if missed else 0]:
        predicted[row, column] = 0
    if model != "segformer_b2":
        predicted[1, 1] = 0
    if sample_id == "v0001":
        predicted[4, :] = 1
        predicted[5, :2] = 1
    return predicted


def write_artifacts(
    directory: Path,
    dataset_manifest_sha256: str,
    grid: Callable[[str], np.ndarray],
    *,
    spread_offset: float = 0.0,
) -> dict[str, str]:
    """One run's artifacts, and the payload hash of each, as the evaluation writes them."""

    directory.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, str] = {}
    for sample_id in SAMPLES:
        targets = TRUTH[VALID]
        predicted = grid(sample_id)[VALID]
        size = int(targets.size)
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(confusion, (targets, predicted), 1)
        spread = 0.01 + spread_offset + 0.02 * (np.arange(size) % 5) / 4.0
        probabilities = np.tile(spread[:, None], (1, NUM_CLASSES))
        probabilities[np.arange(size), predicted] = 1.0 - spread * (NUM_CLASSES - 1)
        artifact = write_prediction_artifact(
            directory / f"{sample_id}.json",
            PredictionRecord(
                sample_id=sample_id,
                predicted_class=predicted.astype(np.uint8),
                top1_confidence_q16=quantize_confidence(probabilities[np.arange(size), predicted]),
                correctness_bitset=pack_correctness(predicted == targets),
                confusion=confusion,
                brier_sum_by_class=multiclass_brier_sums(probabilities, targets, NUM_CLASSES),
                valid_pixel_count=size,
            ),
            classwise_ece_sufficient_statistics(probabilities, targets, NUM_CLASSES),
            protocol_sha256=PROTOCOL,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
        payloads[sample_id] = artifact.payload_sha256
    return payloads


def write_run_record(
    directory: Path, run_id: str, seed: int, dataset_manifest_sha256: str, payloads: dict[str, str]
) -> None:
    record = {
        "schema_version": "driving-risk-run/v1",
        "run_id": f"eval-{run_id}",
        "commit": "c" * 40,
        "config_sha256": PROTOCOL,
        "protocol_sha256": PROTOCOL,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "lock_sha256": "d" * 64,
        "hardware": {"gpu": "test"},
        "seed": seed,
        "started_at_utc": "2026-09-01T00:00:00Z",
        "finished_at_utc": "2026-09-01T01:00:00Z",
        "status": "succeeded",
        "artifacts": payloads,
    }
    (directory / "run_record.json").write_text(json.dumps(record), encoding="utf-8")


def place(bitmask: np.ndarray, rows: Any, columns: Any, *, category: int, annotation_id: int):
    bitmask[rows, columns, 0] = category
    bitmask[rows, columns, 2] = annotation_id >> 8
    bitmask[rows, columns, 3] = annotation_id & 0xFF


@dataclass(frozen=True)
class Study:
    """Everything the extraction reads, laid out as the released study lays it out."""

    root: Path
    index_path: Path
    manifest_path: Path
    labels_root: Path
    instance_root: Path
    tertiles_path: Path
    dataset_manifest_sha256: str

    def run_dir(self, model: str, seed: int, *, calibrated: bool = False) -> Path:
        suffix = "eval_calibrated" if calibrated else "eval"
        return self.root / "runs" / model / f"seed-{seed}" / suffix


def build_study(
    root: Path,
    prediction: Prediction = default_prediction,
    *,
    index_order: tuple[tuple[str, int], ...] | None = None,
) -> Study:
    images = root / "gt" / "images"
    labels = root / "gt" / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLES:
        Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).save(
            images / f"{sample_id}.jpg"
        )
        Image.fromarray(TRUTH.astype(np.uint8)).save(labels / f"{sample_id}_train_id.png")
    manifest = build_paired_manifest(images, labels, "locked_validation")
    manifest_path = root / "gt" / "locked_validation.json"
    save_manifest(manifest, manifest_path)
    digest = manifest.manifest_sha256

    instance_root = root / "instances"
    bitmask_dir = instance_root / "labels" / "ins_seg" / "bitmasks" / "val"
    bitmask_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLES:
        bitmask = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        place(bitmask, slice(2, 4), slice(2, 4), category=1, annotation_id=1)
        place(bitmask, 3, 1, category=1, annotation_id=1)
        place(bitmask, slice(4, 6), slice(2, 4), category=3, annotation_id=256)
        place(bitmask, 0, 3, category=2, annotation_id=3)
        Image.fromarray(bitmask, mode="RGBA").save(bitmask_dir / f"{sample_id}.png")

    tertiles_path = root / "area_tertiles.json"
    tertiles_path.write_text(json.dumps({"tertile_edges": TERTILE_EDGES}), encoding="utf-8")

    runs_root = root / "runs"
    runs: list[dict[str, Any]] = []
    order = index_order or tuple((model, seed) for model in MODELS for seed in SEEDS)
    for model, seed in order:
        run_id = f"{model}-seed-{seed}"
        for calibrated, offset in ((False, 0.0), (True, 0.005)):
            directory = (
                runs_root / model / f"seed-{seed}" / ("eval_calibrated" if calibrated else "eval")
            )
            payloads = write_artifacts(
                directory,
                digest,
                lambda sample_id, model=model, seed=seed: prediction(sample_id, model, seed),
                spread_offset=offset,
            )
            write_run_record(directory, run_id, seed, digest, payloads)
        runs.append(
            {
                "model": model,
                "seed": seed,
                "run_id": run_id,
                "protocol_sha256": PROTOCOL,
                "dataset_manifest_sha256": digest,
                "checkpoint_sha256": f"{MODELS.index(model)}{seed}".ljust(64, "0"),
                "final_step": 30000,
                "status": "succeeded",
                "temperature": 1.2,
                "artifacts_dir": f"{model}/seed-{seed}/eval",
                "calibrated_artifacts_dir": f"{model}/seed-{seed}/eval_calibrated",
                "uncalibrated_sample_ids": list(SAMPLES),
                "calibrated_sample_ids": list(SAMPLES),
            }
        )
    index_path = runs_root / "formal_run_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "drivemetrics-formal-set/v1",
                "protocol_sha256": PROTOCOL,
                "dataset_manifest_sha256": digest,
                "expected_steps": 30000,
                "cohort": "locked_validation",
                "num_classes": NUM_CLASSES,
                "critical_class_ids": [11, 12, 17, 18],
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return Study(
        root=root,
        index_path=index_path,
        manifest_path=manifest_path,
        labels_root=labels,
        instance_root=instance_root,
        tertiles_path=tertiles_path,
        dataset_manifest_sha256=digest,
    )


@pytest.fixture
def study_builder() -> Callable[..., Study]:
    """Build a study under a directory the test chooses."""

    return build_study


@pytest.fixture
def study(tmp_path: Path) -> Study:
    return build_study(tmp_path / "study")


@pytest.fixture
def study_tools() -> SimpleNamespace:
    """The fixture's writers and constants, for tests that tamper with one directory."""

    return SimpleNamespace(
        write_artifacts=write_artifacts,
        write_run_record=write_run_record,
        default_prediction=default_prediction,
        SAMPLES=SAMPLES,
        TRUTH=TRUTH,
        VALID=VALID,
    )


@pytest.fixture
def released(tmp_path: Path, study: Any) -> tuple[Path, Any]:
    """The study's released evidence, written by the released aggregate and extended code."""

    evidence = tmp_path / "evidence"
    aggregate_runs(study.index_path, evidence, resamples=200)
    extended_metrics(
        study.index_path,
        evidence / "extended-metrics.json",
        manifest_path=study.manifest_path,
        labels_root=study.labels_root,
        instance_root=study.instance_root,
        tertiles_path=study.tertiles_path,
    )
    from drivemetrics.posthoc import allseed, allseed_statistics

    allseed.extract_all_seeds(
        study.index_path,
        study.manifest_path,
        study.labels_root,
        study.instance_root,
        study.tertiles_path,
        tmp_path / "extract",
    )
    return evidence, allseed_statistics.load_extraction(tmp_path / "extract")
