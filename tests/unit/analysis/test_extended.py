"""Contracts for the metrics that need more than the confusion matrices.

Three blocks, and they do not need the same inputs. Selective risk is rebuilt
from the confidence histogram every artifact already carries, so it is always
computed. Per-band pixel accuracy needs the ground-truth semantic mask.
Instance coverage and area tertiles need the instance bitmasks as well, and the
frozen tertile edges that were learned from the training split and must never be
re-learned here.

Each block is therefore independently optional, and an absent input produces an
explicit `not_computed` record naming what was missing. A block that silently
vanished would be indistinguishable from a block whose value was zero, and the
release cites this document.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image

from drivemetrics.artifacts.predictions import PredictionRecord, write_prediction_artifact
from drivemetrics.metrics.calibration import (
    classwise_ece_sufficient_statistics,
    multiclass_brier_sums,
    pack_correctness,
    quantize_confidence,
)

MODELS = ("upernet_convnextv2_tiny", "upernet_dinov2_small", "segformer_b2")
SEEDS = (17, 42, 73)
PROTOCOL = "a" * 64
MANIFEST = "b" * 64
NUM_CLASSES = 4
SAMPLES = ("v0001", "v0002")
HEIGHT, WIDTH = 6, 4
#: One instance of category 1 covering six pixels, and one of category 2
#: covering two. The frozen edges below put the first in `large` and the second
#: in `small`, so both tertiles are exercised by two instances.
TERTILE_EDGES = {"1": [2, 4], "2": [8, 16]}


def load_extended() -> ModuleType:
    try:
        from drivemetrics.analysis import extended
    except ImportError:
        pytest.fail("drivemetrics.analysis.extended is missing", pytrace=False)
    return extended


def truth_and_prediction(sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    """A fixed ground truth and a prediction that is wrong only in the bottom band."""

    targets = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    targets[:2] = 1
    targets[2:4] = 2
    targets[4:] = 3
    predicted = targets.copy()
    if sample_id == "v0001":
        predicted[4:] = 0  # the whole bottom band is wrong
    else:
        predicted[5, 0] = 0  # one pixel of the bottom band is wrong
    return targets, predicted


def write_run_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLES:
        targets, predicted = truth_and_prediction(sample_id)
        flat_targets = targets.reshape(-1)
        flat_predicted = predicted.reshape(-1)
        size = flat_targets.size

        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(confusion, (flat_targets, flat_predicted), 1)
        # Confidence varies along the image so the selective-risk curve has more
        # than one point. A model that emitted one confidence for every pixel
        # would be degenerate, and that case is covered separately.
        spread = 0.05 + 0.10 * (np.arange(size) % 5) / 4.0
        probabilities = np.tile(spread[:, None], (1, NUM_CLASSES))
        probabilities[np.arange(size), flat_predicted] = 1.0 - spread * (NUM_CLASSES - 1)

        write_prediction_artifact(
            directory / f"{sample_id}.json",
            PredictionRecord(
                sample_id=sample_id,
                predicted_class=flat_predicted.astype(np.uint8),
                top1_confidence_q16=quantize_confidence(
                    probabilities[np.arange(size), flat_predicted]
                ),
                correctness_bitset=pack_correctness(flat_predicted == flat_targets),
                confusion=confusion,
                brier_sum_by_class=multiclass_brier_sums(probabilities, flat_targets, NUM_CLASSES),
                valid_pixel_count=size,
            ),
            classwise_ece_sufficient_statistics(probabilities, flat_targets, NUM_CLASSES),
            protocol_sha256=PROTOCOL,
            dataset_manifest_sha256=MANIFEST,
        )


def write_ground_truth(root: Path) -> Path:
    """Semantic masks under the BDD100K validation layout."""

    directory = root / "labels" / "sem_seg" / "masks" / "val"
    directory.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLES:
        targets, _ = truth_and_prediction(sample_id)
        Image.fromarray(targets.astype(np.uint8)).save(directory / f"{sample_id}.png")
    return root


def write_instance_bitmasks(root: Path) -> Path:
    """RGBA bitmasks in the official packing: category in red, annotation id in blue/alpha."""

    directory = root / "labels" / "ins_seg" / "bitmasks" / "val"
    directory.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLES:
        bitmask = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        # Instance 1, category 1, six pixels in the top band.
        bitmask[:2, :3, 0] = 1
        bitmask[:2, :3, 3] = 1
        # Instance 2, category 2, two pixels in the middle band.
        bitmask[2, :2, 0] = 2
        bitmask[2, :2, 3] = 2
        Image.fromarray(bitmask, mode="RGBA").save(directory / f"{sample_id}.png")
    return root


def build_index(root: Path) -> Path:
    runs: list[dict[str, Any]] = []
    for model_position, model in enumerate(MODELS):
        for seed in SEEDS:
            run_id = f"{model}-seed-{seed}"
            write_run_artifacts(root / run_id)
            write_run_artifacts(root / f"{run_id}-calibrated")
            runs.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_id": run_id,
                    "protocol_sha256": PROTOCOL,
                    "dataset_manifest_sha256": MANIFEST,
                    "checkpoint_sha256": f"{model_position}{seed}".ljust(64, "0"),
                    "final_step": 30000,
                    "status": "succeeded",
                    "temperature": 1.2,
                    "artifacts_dir": run_id,
                    "calibrated_artifacts_dir": f"{run_id}-calibrated",
                    "uncalibrated_sample_ids": list(SAMPLES),
                    "calibrated_sample_ids": list(SAMPLES),
                }
            )

    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "formal_run_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "drivemetrics-formal-set/v1",
                "protocol_sha256": PROTOCOL,
                "dataset_manifest_sha256": MANIFEST,
                "expected_steps": 30000,
                "cohort": "locked_validation",
                "num_classes": NUM_CLASSES,
                "critical_class_ids": [2, 3],
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return index_path


def compute(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    extended = load_extended()
    output_path = tmp_path / "extended-metrics.json"
    extended.extended_metrics(build_index(tmp_path / "runs"), output_path, **kwargs)
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_selective_risk_needs_no_ground_truth_and_is_always_computed(tmp_path: Path) -> None:
    """It is rebuilt from the histogram the artifacts already carry, for both evaluations.

    Every other block here depends on a file that may not have travelled with the
    run. This one depends on nothing outside the artifacts, so a release can
    always report it, and the document says which evaluation each curve is from
    because temperature changes confidence and this metric reads confidence.
    """

    document = compute(tmp_path)

    for model in MODELS:
        block = document["selective_risk"][model]
        assert set(block) == {"uncalibrated", "calibrated"}
        for kind in ("uncalibrated", "calibrated"):
            assert block[kind]["aurc"] is not None
            assert 0.0 <= block[kind]["aurc"] <= 1.0
            assert block[kind]["defined_at"] == "confidence_bin_boundaries"


def test_the_band_accuracy_is_the_one_the_fixture_was_built_to_show(tmp_path: Path) -> None:
    """The bottom band is where the fixture put the errors, so it must score lowest.

    `v0001` has its whole bottom band wrong and `v0002` one pixel of it, so over
    the two-image cohort the bottom band's accuracy is `(0 + 7) / 16` while the
    top and middle bands are perfect. A band metric that read the rows in the
    wrong order, or that averaged per-image rates instead of pooling pixels,
    would not produce this number.
    """

    document = compute(tmp_path, labels_root=write_ground_truth(tmp_path / "gt"))

    bands = document["normalized_image_bands"][MODELS[0]]
    assert bands["top"]["pixel_accuracy"] == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert bands["middle"]["pixel_accuracy"] == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert bands["bottom"]["pixel_accuracy"] == pytest.approx(7.0 / 16.0, rel=0.0, abs=1e-12)
    assert bands["bottom"]["pixels"] == 16


def test_the_band_names_are_image_regions_and_say_so(tmp_path: Path) -> None:
    """The field has been mistaken for depth before, so the document disclaims it."""

    document = compute(tmp_path, labels_root=write_ground_truth(tmp_path / "gt"))

    assert document["normalized_image_bands"]["definition"] == (
        "normalized image rows only; not physical depth or metric distance"
    )


def test_instance_coverage_uses_the_frozen_tertile_edges(tmp_path: Path) -> None:
    """Re-learning the edges on the evaluation cohort would tune on the locked split.

    The edges were learned from the training intersection and frozen. Passing
    them in rather than computing them is what keeps the locked cohort out of a
    decision, and the document records which file they came from so a reader can
    check that it is the frozen one.
    """

    tertiles = tmp_path / "area_tertiles.json"
    tertiles.write_text(json.dumps({"tertile_edges": TERTILE_EDGES}), encoding="utf-8")

    document = compute(
        tmp_path,
        labels_root=write_ground_truth(tmp_path / "gt"),
        instance_root=write_instance_bitmasks(tmp_path / "gt"),
        tertiles_path=tertiles,
    )

    block = document["instances"][MODELS[0]]
    assert block["tertile_edges_from"] == str(tertiles)
    assert block["instance_count"] == 4  # two instances in each of two images
    assert set(block["by_tertile"]) == {"small", "medium", "large"}
    # The category-1 instance covers six pixels against edges (2, 4): large.
    # The category-2 instance covers two against (8, 16): small.
    assert block["by_tertile"]["large"]["instance_count"] == 2
    assert block["by_tertile"]["small"]["instance_count"] == 2
    assert block["by_tertile"]["medium"]["instance_count"] == 0


def test_a_missing_input_is_recorded_as_not_computed_with_its_reason(tmp_path: Path) -> None:
    """A block that vanished silently is indistinguishable from one whose value was zero."""

    document = compute(tmp_path)

    for name in ("normalized_image_bands", "instances"):
        assert document[name] == {
            "not_computed": "ground truth was not available to this analysis run"
        }


def test_instances_alone_are_refused_without_the_frozen_edges(tmp_path: Path) -> None:
    """Bitmasks without edges would invite the edges to be learned here, and they must not be."""

    document = compute(
        tmp_path,
        labels_root=write_ground_truth(tmp_path / "gt"),
        instance_root=write_instance_bitmasks(tmp_path / "gt"),
    )

    assert document["instances"] == {"not_computed": "the frozen area tertiles were not supplied"}
    assert "not_computed" not in document["normalized_image_bands"]


def test_the_document_binds_itself_to_the_study_and_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    """It is cited by claims, so it carries its hashes and is written exactly once."""

    extended = load_extended()
    index_path = build_index(tmp_path / "runs")
    output_path = tmp_path / "extended-metrics.json"
    extended.extended_metrics(index_path, output_path)

    raw = output_path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    assert raw == (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert document["protocol_hash"] == PROTOCOL
    assert document["dataset_manifest_hash"] == MANIFEST
    assert document["evaluation_for_ground_truth_metrics"] == "eval_calibrated"

    with pytest.raises(FileExistsError, match=r"^an extended metrics document already exists: "):
        extended.extended_metrics(index_path, output_path)


def test_an_index_that_fails_its_own_gate_is_refused(tmp_path: Path) -> None:
    """Every published document starts from the same gate."""

    extended = load_extended()
    index_path = build_index(tmp_path / "runs")
    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["runs"] = document["runs"][:-1]
    index_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^formal run index is not valid: "):
        extended.extended_metrics(index_path, tmp_path / "extended-metrics.json")
