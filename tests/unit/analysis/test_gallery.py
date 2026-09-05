"""Contracts for selecting the failure gallery from the frozen per-image evidence.

The gallery is the one part of a release that invites a reader to look at
individual images, which is exactly why its selection rule is fixed before any
number is seen. A gallery chosen after the metrics are known is a gallery chosen
to make a point; a gallery chosen by a rule written down first is evidence. Its
counts are descriptive and never inferential: the release may truthfully report
an empty category and must never manufacture one to fill a quota.

The fixture here is deliberately not the one `test_aggregate.py` uses. That one
gives every image of a run the same expected accuracy, which is right for a
cohort metric and useless for a gallery: with no spread between images, "worst"
and "best" are whatever the sort happened to do. This fixture varies accuracy
PER IMAGE by a known amount, so the expected selection is derivable by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

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
#: Five images whose accuracies are spread far enough apart that their order is
#: the same for every model, so the expected worst and best are known without
#: recomputing the metric in the test.
SAMPLES = ("v0001", "v0002", "v0003", "v0004", "v0005")
PER_IMAGE_ACCURACY = {"v0001": 0.30, "v0002": 0.45, "v0003": 0.60, "v0004": 0.75, "v0005": 0.90}
SIZE = 600


def load_gallery() -> ModuleType:
    try:
        from drivemetrics.analysis import gallery
    except ImportError:
        pytest.fail("drivemetrics.analysis.gallery is missing", pytrace=False)
    return gallery


def write_run_artifacts(directory: Path, *, offset: float) -> None:
    """One run whose per-image accuracy is the declared value plus a model offset."""

    directory.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLES:
        accuracy = min(0.99, PER_IMAGE_ACCURACY[sample_id] + offset)
        rng = np.random.default_rng(abs(hash((sample_id, offset))) % (2**32))
        targets = rng.integers(0, NUM_CLASSES, SIZE, dtype=np.int64)
        correct = rng.random(SIZE) < accuracy
        predicted = np.where(correct, targets, (targets + 1) % NUM_CLASSES).astype(np.int64)

        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(confusion, (targets, predicted), 1)
        probabilities = np.full((SIZE, NUM_CLASSES), 0.1, dtype=np.float64)
        probabilities[np.arange(SIZE), predicted] = 1.0 - 0.1 * (NUM_CLASSES - 1)

        write_prediction_artifact(
            directory / f"{sample_id}.json",
            PredictionRecord(
                sample_id=sample_id,
                predicted_class=predicted.astype(np.uint8),
                top1_confidence_q16=quantize_confidence(probabilities[np.arange(SIZE), predicted]),
                correctness_bitset=pack_correctness(predicted == targets),
                confusion=confusion,
                brier_sum_by_class=multiclass_brier_sums(probabilities, targets, NUM_CLASSES),
                valid_pixel_count=SIZE,
            ),
            classwise_ece_sufficient_statistics(probabilities, targets, NUM_CLASSES),
            protocol_sha256=PROTOCOL,
            dataset_manifest_sha256=MANIFEST,
        )


def build_index(root: Path) -> Path:
    """Nine runs on one cohort, with the same per-image ordering in every model."""

    runs: list[dict[str, Any]] = []
    for model_position, model in enumerate(MODELS):
        for seed in SEEDS:
            run_id = f"{model}-seed-{seed}"
            offset = 0.02 * model_position + 0.001 * (seed % 7)
            write_run_artifacts(root / run_id, offset=offset)
            write_run_artifacts(root / f"{run_id}-calibrated", offset=offset)
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

    document = {
        "schema_version": "drivemetrics-formal-set/v1",
        "protocol_sha256": PROTOCOL,
        "dataset_manifest_sha256": MANIFEST,
        "expected_steps": 30000,
        "cohort": "locked_validation",
        "num_classes": NUM_CLASSES,
        "critical_class_ids": [2, 3],
        "runs": runs,
    }
    index_path = root / "formal_run_index.json"
    root.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return index_path


def select(tmp_path: Path, *, per_model: int = 2) -> dict[str, Any]:
    gallery = load_gallery()
    output_path = tmp_path / "gallery-manifest.json"
    gallery.select_gallery(build_index(tmp_path / "runs"), output_path, per_model=per_model)
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_the_manifest_records_the_rule_it_applied_as_data(tmp_path: Path) -> None:
    """A reader has to be able to check the selection without reading the source.

    The rule travels with the manifest as fields rather than as prose, so a later
    reader can reproduce the choice, and so changing the rule is a visible change
    to every published manifest rather than an invisible change in code.
    """

    manifest = select(tmp_path)

    assert manifest["schema_version"] == "driving-risk-gallery-manifest/v1"
    assert manifest["evaluation"] == "eval_calibrated"
    assert manifest["rule"] == {
        "metric": "image_mean_iou",
        "aggregate": "mean_over_seeds",
        "per_model": 2,
        "tie_break": "sample_id",
    }
    assert set(manifest["per_model"]) == set(MODELS)


def test_the_worst_images_are_the_ones_the_fixture_made_hardest(tmp_path: Path) -> None:
    """Reversing worst and best would publish a model's easiest images as its failures.

    The fixture's accuracies are ordered `v0001 < ... < v0005` for every model, so
    the expected answer is known without recomputing the metric here.
    """

    manifest = select(tmp_path)

    for model in MODELS:
        block = manifest["per_model"][model]
        assert [entry["sample_id"] for entry in block["worst"]] == ["v0001", "v0002"]
        assert [entry["sample_id"] for entry in block["best"]] == ["v0005", "v0004"]
        worst = [entry["mean_iou_over_seeds"] for entry in block["worst"]]
        best = [entry["mean_iou_over_seeds"] for entry in block["best"]]
        assert worst == sorted(worst)
        assert best == sorted(best, reverse=True)
        assert max(worst) < min(best)


def test_every_entry_carries_the_per_seed_values_behind_its_mean(tmp_path: Path) -> None:
    """One bad seed and three consistent failures look identical in a mean alone."""

    manifest = select(tmp_path)

    for model in MODELS:
        for entry in manifest["per_model"][model]["worst"]:
            assert set(entry["per_seed"]) == {str(seed) for seed in SEEDS}
            values = list(entry["per_seed"].values())
            assert entry["mean_iou_over_seeds"] == pytest.approx(
                sum(values) / len(values), rel=0.0, abs=1e-12
            )


def test_asking_for_more_images_than_the_cohort_holds_returns_the_cohort(
    tmp_path: Path,
) -> None:
    """A gallery is descriptive, so a request it cannot fill is not an error.

    Refusing here would make the cohort's size a precondition of publishing at
    all, and the honest response to a small cohort is a small gallery.
    """

    manifest = select(tmp_path, per_model=len(SAMPLES) + 5)

    for model in MODELS:
        block = manifest["per_model"][model]
        assert len(block["worst"]) == len(SAMPLES)
        assert len(block["best"]) == len(SAMPLES)


def test_images_that_score_the_same_are_ordered_by_sample_id() -> None:
    """Without a declared tie-break the manifest depends on dictionary order."""

    gallery = load_gallery()
    scores = {"v0003": 0.5, "v0001": 0.5, "v0002": 0.9}

    assert gallery.rank_samples(scores, count=2, worst=True) == ["v0001", "v0003"]
    assert gallery.rank_samples(scores, count=2, worst=False) == ["v0002", "v0001"]


def test_the_manifest_is_byte_exact_and_refuses_to_overwrite(tmp_path: Path) -> None:
    """The manifest is cited by claims, so a second run must not quietly replace it."""

    gallery = load_gallery()
    index_path = build_index(tmp_path / "runs")
    output_path = tmp_path / "gallery-manifest.json"
    gallery.select_gallery(index_path, output_path, per_model=2)

    raw = output_path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    assert raw == (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with pytest.raises(FileExistsError, match=r"^a gallery manifest already exists: "):
        gallery.select_gallery(index_path, output_path, per_model=2)


def test_an_index_that_fails_its_own_gate_is_refused(tmp_path: Path) -> None:
    """The gallery cites the same runs the metrics do, so it uses the same gate."""

    gallery = load_gallery()
    index_path = build_index(tmp_path / "runs")
    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["runs"] = document["runs"][:-2]
    index_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^formal run index is not valid: [^;]+; [^;]+$"):
        gallery.select_gallery(index_path, tmp_path / "gallery-manifest.json", per_model=2)


def test_the_hashes_bind_the_gallery_to_the_study_that_produced_it(tmp_path: Path) -> None:
    """A manifest without its protocol and cohort hashes cannot be placed."""

    gallery = load_gallery()
    index_path = build_index(tmp_path / "runs")
    output_path = tmp_path / "gallery-manifest.json"
    gallery.select_gallery(index_path, output_path, per_model=2)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest["protocol_hash"] == index["protocol_sha256"]
    assert manifest["dataset_manifest_hash"] == index["dataset_manifest_sha256"]


def test_a_gallery_of_zero_images_is_refused_because_it_publishes_nothing(
    tmp_path: Path,
) -> None:
    """Zero is not a small gallery; it is a manifest that shows no evidence at all."""

    gallery = load_gallery()

    with pytest.raises(ValueError, match=r"^per_model must be a positive integer, got "):
        gallery.select_gallery(
            build_index(tmp_path / "runs"), tmp_path / "gallery-manifest.json", per_model=0
        )


def test_the_gallery_reads_the_calibrated_evaluation(tmp_path: Path) -> None:
    """Argmax is temperature-invariant, so the choice is stated rather than assumed.

    Both evaluations produce identical confusions and therefore an identical
    gallery. Naming which one was read still matters: it is the condition the
    release publishes, and a reader who later finds a different rule needs to
    know which directory the images came from.
    """

    gallery = load_gallery()
    index_path = build_index(tmp_path / "runs")
    output_path = tmp_path / "gallery-manifest.json"
    gallery.select_gallery(index_path, output_path, per_model=2)
    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest["evaluation"] == "eval_calibrated"
    assert manifest["per_model"][MODELS[0]]["worst"][0]["sample_id"] == "v0001"


def test_a_gallery_of_one_image_per_model_is_the_smallest_allowed(tmp_path: Path) -> None:
    """One is a small gallery, not an empty one; the refusal boundary is at zero."""

    manifest = select(tmp_path, per_model=1)

    for model in MODELS:
        assert [entry["sample_id"] for entry in manifest["per_model"][model]["worst"]] == ["v0001"]
        assert [entry["sample_id"] for entry in manifest["per_model"][model]["best"]] == ["v0005"]


def test_a_boolean_gallery_size_is_refused(tmp_path: Path) -> None:
    """`True` is an int to Python and a mistake to a reader; it is refused by name."""

    gallery = load_gallery()

    with pytest.raises(ValueError, match=r"^per_model must be a positive integer, got True$"):
        gallery.select_gallery(
            build_index(tmp_path / "runs"), tmp_path / "gallery-manifest.json", per_model=True
        )


def test_the_output_directory_is_created_when_it_does_not_exist(tmp_path: Path) -> None:
    """The evidence directory is named by commit and does not exist until the first run."""

    gallery = load_gallery()
    output_path = tmp_path / "analysis" / "deadbeef" / "gallery-manifest.json"

    gallery.select_gallery(build_index(tmp_path / "runs"), output_path, per_model=2)

    assert output_path.is_file()


def test_each_model_is_ranked_on_its_own_runs(tmp_path: Path) -> None:
    """Every model's worst image is the same image here by design, so order alone cannot tell.

    The per-image mIoU behind the selection is recomputed from the model's OWN
    three calibrated evaluations. A gallery that averaged the other models' runs
    would pick the same sample IDs and publish different numbers beside them.
    """

    from drivemetrics.artifacts.predictions import read_prediction_artifact
    from drivemetrics.metrics.confusion import summarize_confusion

    gallery = load_gallery()
    index_path = build_index(tmp_path / "runs")
    output_path = tmp_path / "gallery-manifest.json"
    gallery.select_gallery(index_path, output_path, per_model=1)
    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    for model in MODELS:
        worst = manifest["per_model"][model]["worst"][0]
        expected = []
        for seed in SEEDS:
            path = (
                tmp_path / "runs" / f"{model}-seed-{seed}-calibrated" / f"{worst['sample_id']}.json"
            )
            _, record, _ = read_prediction_artifact(path)
            expected.append(summarize_confusion(record.confusion).mean_iou)
        assert worst["mean_iou_over_seeds"] == pytest.approx(
            sum(expected) / len(expected), rel=0.0, abs=1e-12
        )
        assert worst["per_seed"] == {
            str(seed): pytest.approx(value, rel=0.0, abs=1e-12)
            for seed, value in zip(SEEDS, expected, strict=True)
        }
