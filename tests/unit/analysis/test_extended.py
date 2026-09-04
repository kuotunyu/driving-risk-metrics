"""Contracts for the metrics that need more than the confusion matrices.

Three blocks, and they do not need the same inputs. Selective risk is rebuilt
from the confidence histogram every artifact already carries, so it is always
computed. Per-band pixel accuracy needs the ground-truth semantic mask. Instance
coverage and area tertiles need the instance bitmasks as well, and the frozen
tertile edges that were learned from the training split and must never be
re-learned here.

The fixture is deliberately shaped like the real dataset rather than like a
convenient array, because the previous one was not and that is exactly what let
four defects through. Real BDD100K masks carry IGNORED pixels, and an artifact
stores one prediction per non-ignored pixel rather than a dense image; real
instance bitmasks number their categories 1..8 while the semantic masks carry
nineteen Cityscapes train IDs; and real annotation IDs need two bytes. A fixture
where every pixel is valid, the class space is four wide and every ID fits in one
byte agrees with a wrong implementation on every one of those points.
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
from drivemetrics.data.manifest import build_paired_manifest, save_manifest
from drivemetrics.metrics.calibration import (
    classwise_ece_sufficient_statistics,
    multiclass_brier_sums,
    pack_correctness,
    quantize_confidence,
)

MODELS = ("upernet_convnextv2_tiny", "upernet_dinov2_small", "segformer_b2")
SEEDS = (17, 42, 73)
PROTOCOL = "a" * 64
NUM_CLASSES = 19
IGNORE = 255
SAMPLES = ("v0001", "v0002")
HEIGHT, WIDTH = 6, 4

#: Train IDs, with three IGNORED pixels: one in the top band and two in the
#: bottom. Their placement is the point. A band metric that counts them, or an
#: artifact reader that reshapes instead of scattering, cannot reproduce the
#: numbers the band test asserts.
TRUTH = np.array(
    [
        [10, 10, 10, IGNORE],
        [10, 10, 10, 10],
        [2, 2, 11, 11],
        [2, 2, 11, 11],
        [0, 0, 13, 13],
        [0, 0, IGNORE, IGNORE],
    ],
    dtype=np.int64,
)
VALID = TRUTH != IGNORE

#: Category 1 is "person", train ID 11; category 3 is "car", train ID 13. The
#: edges put the four-pixel person in `medium` and the two-pixel car in `large`,
#: leaving `small` empty so an empty bucket is exercised too.
TERTILE_EDGES = {"1": [2, 8], "3": [1, 1]}


def load_extended() -> ModuleType:
    try:
        from drivemetrics.analysis import extended
    except ImportError:
        pytest.fail("drivemetrics.analysis.extended is missing", pytrace=False)
    return extended


def prediction_grid(sample_id: str) -> np.ndarray:
    """Correct everywhere except the errors this fixture is built to measure.

    One middle-band pixel is wrong in both images, which makes the person
    instance three-quarters covered. The bottom band is entirely wrong in the
    first image and wrong in one pixel in the second, which makes the car
    instance a critical miss in one image and perfect in the other.
    """

    predicted = TRUTH.copy()
    predicted[2, 2] = 0
    if sample_id == "v0001":
        predicted[4, :] = 1
        predicted[5, :2] = 1
    else:
        predicted[5, 0] = 1
    return predicted


def write_run_artifacts(directory: Path, dataset_manifest_sha256: str) -> None:
    """One run's artifacts, holding a prediction per NON-IGNORED pixel only."""

    directory.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLES:
        targets = TRUTH[VALID]
        predicted = prediction_grid(sample_id)[VALID]
        size = int(targets.size)

        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(confusion, (targets, predicted), 1)
        # Confidence varies along the image so the selective-risk curve has more
        # than one point; a model emitting one confidence everywhere is covered
        # separately. The spread stays small enough that the predicted class is
        # still the argmax across nineteen classes.
        spread = 0.01 + 0.02 * (np.arange(size) % 5) / 4.0
        probabilities = np.tile(spread[:, None], (1, NUM_CLASSES))
        probabilities[np.arange(size), predicted] = 1.0 - spread * (NUM_CLASSES - 1)

        write_prediction_artifact(
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


def write_ground_truth(root: Path) -> tuple[Path, Path, str]:
    """Images and train-ID masks in the official filenames, plus a frozen manifest.

    The manifest is built the way the pipeline builds one, so its hashes are real
    and the analysis has to verify them rather than be told they are fine.
    """

    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLES:
        Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).save(
            images / f"{sample_id}.jpg"
        )
        Image.fromarray(TRUTH.astype(np.uint8)).save(labels / f"{sample_id}_train_id.png")

    manifest = build_paired_manifest(images, labels, "locked_validation")
    manifest_path = root / "locked_validation.json"
    save_manifest(manifest, manifest_path)
    return manifest_path, labels, manifest.manifest_sha256


def place(
    bitmask: np.ndarray, rows: Any, columns: Any, *, category: int, annotation_id: int
) -> None:
    """Paint one instance in the official packing, splitting the ID across two bytes."""

    bitmask[rows, columns, 0] = category
    bitmask[rows, columns, 2] = annotation_id >> 8
    bitmask[rows, columns, 3] = annotation_id & 0xFF


def write_instance_bitmasks(root: Path) -> Path:
    """RGBA bitmasks in the official packing: category in red, annotation ID in blue and alpha.

    Five instances, chosen so that every outcome of the corroboration rule occurs.
    Instance 1 is a person on four pixels the semantic mask agrees with. Instance
    300 is a car on four pixels of which two are semantically ignored, and its ID
    does not fit in one byte, so a reader that drops the high byte fails here.
    Instance 3 is a rider lying entirely on ignored pixels. Instance 4 is a car
    where the semantic mask says building. Instance 5 carries category 9, which
    is not an instance category at all.
    """

    directory = root / "labels" / "ins_seg" / "bitmasks" / "val"
    directory.mkdir(parents=True, exist_ok=True)
    # Deliberately not under the semantic label root: that root is scanned for
    # every PNG it contains, and an instance bitmask found there would be
    # mistaken for a semantic mask with an unreadable filename.
    for sample_id in SAMPLES:
        bitmask = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        place(bitmask, slice(2, 4), slice(2, 4), category=1, annotation_id=1)
        place(bitmask, slice(4, 6), slice(2, 4), category=3, annotation_id=300)
        place(bitmask, 0, 3, category=2, annotation_id=3)
        place(bitmask, 2, 0, category=3, annotation_id=4)
        place(bitmask, 1, 0, category=9, annotation_id=5)
        Image.fromarray(bitmask, mode="RGBA").save(directory / f"{sample_id}.png")
    return root


def build_index(root: Path, dataset_manifest_sha256: str) -> Path:
    runs: list[dict[str, Any]] = []
    for model_position, model in enumerate(MODELS):
        for seed in SEEDS:
            run_id = f"{model}-seed-{seed}"
            write_run_artifacts(root / run_id, dataset_manifest_sha256)
            write_run_artifacts(root / f"{run_id}-calibrated", dataset_manifest_sha256)
            runs.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_id": run_id,
                    "protocol_sha256": PROTOCOL,
                    "dataset_manifest_sha256": dataset_manifest_sha256,
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
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "expected_steps": 30000,
                "cohort": "locked_validation",
                "num_classes": NUM_CLASSES,
                "critical_class_ids": [11, 13],
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return index_path


def write_tertiles(tmp_path: Path) -> Path:
    tertiles = tmp_path / "area_tertiles.json"
    tertiles.write_text(json.dumps({"tertile_edges": TERTILE_EDGES}), encoding="utf-8")
    return tertiles


def compute(tmp_path: Path, *, ground_truth: bool = False, **kwargs: Any) -> dict[str, Any]:
    """Run the analysis over a freshly built study and return the document."""

    extended = load_extended()
    manifest_path, labels_root, digest = write_ground_truth(tmp_path / "gt")
    index_path = build_index(tmp_path / "runs", digest)
    if ground_truth:
        kwargs.setdefault("manifest_path", manifest_path)
        kwargs.setdefault("labels_root", labels_root)
    output_path = tmp_path / "extended-metrics.json"
    extended.extended_metrics(index_path, output_path, **kwargs)
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


def test_the_band_accuracy_counts_only_the_pixels_a_model_was_asked_about(
    tmp_path: Path,
) -> None:
    """Ignored pixels belong in neither half of a rate, and never in a denominator.

    The fixture ignores one pixel of the top band and two of the bottom, so the
    per-image band sizes are 7, 8 and 6 rather than 8, 8 and 8. Over the two
    images the top band is perfect, the middle loses one pixel per image, and the
    bottom loses all six in the first image and one in the second. An
    implementation that counted the ignored pixels, or that reshaped the stored
    predictions instead of scattering them through the ignore mask, cannot land
    on these three numbers together.
    """

    document = compute(tmp_path, ground_truth=True)

    bands = document["normalized_image_bands"][MODELS[0]]
    assert bands["top"]["pixels"] == 14
    assert bands["middle"]["pixels"] == 16
    assert bands["bottom"]["pixels"] == 12
    assert bands["top"]["pixel_accuracy"] == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert bands["middle"]["pixel_accuracy"] == pytest.approx(14 / 16, rel=0.0, abs=1e-12)
    assert bands["bottom"]["pixel_accuracy"] == pytest.approx(5 / 12, rel=0.0, abs=1e-12)


def test_a_mask_that_does_not_pair_with_the_artifact_is_refused() -> None:
    """The valid-pixel count is the pairing proof, and a mismatch is not recoverable.

    Two masks cannot hold the same number of non-ignored pixels by accident at
    this size, so a mismatch means the mask and the artifact are not describing
    the same image and every number derived from the pair would be fiction.
    """

    extended = load_extended()
    record = PredictionRecord(
        sample_id="v0001",
        predicted_class=np.zeros(5, dtype=np.uint8),
        top1_confidence_q16=np.zeros(5, dtype=np.uint16),
        correctness_bitset=pack_correctness(np.ones(5, dtype=bool)),
        confusion=np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
        brier_sum_by_class=np.zeros(NUM_CLASSES, dtype=np.float64),
        valid_pixel_count=5,
    )

    with pytest.raises(ValueError, match=r"^artifact v0001 holds 5 predictions but its "):
        extended._dense_prediction(record, TRUTH)


def test_the_band_names_are_image_regions_and_say_so(tmp_path: Path) -> None:
    """The field has been mistaken for depth before, so the document disclaims it."""

    document = compute(tmp_path, ground_truth=True)

    assert document["normalized_image_bands"]["definition"] == (
        "normalized image rows only; not physical depth or metric distance"
    )


def test_instance_categories_are_translated_into_semantic_train_ids() -> None:
    """The two annotations number the same eight things differently.

    A bitmask calls a car category 3 and the semantic mask calls it train ID 13.
    Comparing the two numbering systems directly would ask whether a car is a
    wall. The table is derived from the semantic class list rather than restated,
    so renaming or reordering that list breaks this test instead of silently
    remapping every published instance number.
    """

    from drivemetrics.protocol.risk_profiles import BDD100K_SEMANTIC_CLASS_NAMES

    extended = load_extended()

    assert extended.INSTANCE_CATEGORY_TO_TRAIN_ID == {
        1: 11,
        2: 12,
        3: 13,
        4: 14,
        5: 15,
        6: 16,
        7: 17,
        8: 18,
    }
    for category, name in extended.INSTANCE_CATEGORY_NAMES:
        train_id = extended.INSTANCE_CATEGORY_TO_TRAIN_ID[category]
        assert BDD100K_SEMANTIC_CLASS_NAMES[train_id] == name


def test_instance_coverage_uses_the_frozen_tertile_edges(tmp_path: Path) -> None:
    """Re-learning the edges on the evaluation cohort would tune on the locked split.

    The edges were learned from the training intersection and frozen. Passing
    them in rather than computing them is what keeps the locked cohort out of a
    decision, and the document records which file they came from so a reader can
    check that it is the frozen one. The edges arrive keyed by instance category
    and are applied in train-ID space, so a translation that dropped them would
    show up here as instances with no edges at all.
    """

    tertiles = write_tertiles(tmp_path)
    document = compute(
        tmp_path,
        ground_truth=True,
        instance_root=write_instance_bitmasks(tmp_path / "ins"),
        tertiles_path=tertiles,
    )

    block = document["instances"][MODELS[0]]
    assert block["tertile_edges_from"] == str(tertiles)
    assert block["instance_count"] == 4  # the person and the car, in each of two images
    assert set(block["by_tertile"]) == {"small", "medium", "large"}
    assert block["by_tertile"]["medium"]["instance_count"] == 2  # person, four pixels
    assert block["by_tertile"]["large"]["instance_count"] == 2  # car, two scored pixels
    assert block["by_tertile"]["small"]["instance_count"] == 0
    assert block["by_tertile"]["small"]["mean_correct_fraction"] is None
    assert block["by_tertile"]["medium"]["mean_correct_fraction"] == pytest.approx(
        0.75, rel=0.0, abs=1e-12
    )
    # The car is wholly missed in the first image and wholly found in the second.
    assert block["by_tertile"]["large"]["critical_misses"] == 1
    assert block["by_tertile"]["large"]["mean_correct_fraction"] == pytest.approx(
        0.5, rel=0.0, abs=1e-12
    )


def test_instances_the_semantic_mask_does_not_corroborate_are_excluded_and_counted(
    tmp_path: Path,
) -> None:
    """Two independent rasterizations need not agree, and a silent drop hides that.

    Three of the fixture's five instances are not corroborated: one lies entirely
    on ignored pixels, one sits where the semantic mask says building, and one
    carries a category that is not an instance category. None of them is evidence
    about the model. Dropping them without saying so would shrink the denominator
    of every rate above and leave no trace that it happened.
    """

    document = compute(
        tmp_path,
        ground_truth=True,
        instance_root=write_instance_bitmasks(tmp_path / "ins"),
        tertiles_path=write_tertiles(tmp_path),
    )

    block = document["instances"][MODELS[0]]
    assert block["excluded_without_semantic_pixels"] == 2  # the rider, in each image
    assert block["excluded_semantic_class_disagreement"] == 4  # two per image


def test_a_missing_input_is_recorded_as_not_computed_with_its_reason(tmp_path: Path) -> None:
    """A block that vanished silently is indistinguishable from one whose value was zero."""

    document = compute(tmp_path)

    for name in ("normalized_image_bands", "instances", "ground_truth"):
        assert document[name] == {
            "not_computed": "ground truth was not available to this analysis run"
        }


def test_instances_alone_are_refused_without_the_frozen_edges(tmp_path: Path) -> None:
    """Bitmasks without edges would invite the edges to be learned here, and they must not be."""

    document = compute(
        tmp_path,
        ground_truth=True,
        instance_root=write_instance_bitmasks(tmp_path / "ins"),
    )

    assert document["instances"] == {"not_computed": "the frozen area tertiles were not supplied"}
    assert "not_computed" not in document["normalized_image_bands"]


def test_the_document_records_which_cohort_the_masks_came_from(tmp_path: Path) -> None:
    """A band number is only attributable if the masks behind it are named."""

    extended = load_extended()
    manifest_path, labels_root, digest = write_ground_truth(tmp_path / "gt")
    index_path = build_index(tmp_path / "runs", digest)
    output_path = tmp_path / "extended-metrics.json"
    extended.extended_metrics(
        index_path, output_path, manifest_path=manifest_path, labels_root=labels_root
    )
    document = json.loads(output_path.read_text(encoding="utf-8"))

    assert document["ground_truth"] == {
        "cohort_manifest_sha256": digest,
        "split_name": "locked_validation",
        "masks_verified": len(SAMPLES),
    }
    assert document["dataset_manifest_hash"] == digest


def test_ground_truth_needs_both_the_manifest_and_the_root(tmp_path: Path) -> None:
    """Half the inputs cannot place a single mask, and guessing the other half was the old bug."""

    extended = load_extended()
    manifest_path, labels_root, digest = write_ground_truth(tmp_path / "gt")
    index_path = build_index(tmp_path / "runs", digest)

    for kwargs in ({"manifest_path": manifest_path}, {"labels_root": labels_root}):
        with pytest.raises(ValueError, match=r"^the ground-truth blocks need both "):
            extended.extended_metrics(index_path, tmp_path / "out.json", **kwargs)


def test_a_manifest_from_a_different_cohort_is_refused(tmp_path: Path) -> None:
    """The masks must be the cohort the index was built from, not merely a valid one."""

    extended = load_extended()
    manifest_path, labels_root, digest = write_ground_truth(tmp_path / "gt")
    index_path = build_index(tmp_path / "runs", digest)
    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["dataset_manifest_sha256"] = "c" * 64
    for entry in document["runs"]:
        entry["dataset_manifest_sha256"] = "c" * 64
    index_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^the ground-truth manifest is not the cohort "):
        extended.extended_metrics(
            index_path,
            tmp_path / "out.json",
            manifest_path=manifest_path,
            labels_root=labels_root,
        )


def test_a_manifest_listing_different_images_is_refused(tmp_path: Path) -> None:
    """Equal hashes are not enough; the two documents must name the same images."""

    extended = load_extended()
    manifest_path, labels_root, digest = write_ground_truth(tmp_path / "gt")
    index_path = build_index(tmp_path / "runs", digest)
    document = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in document["runs"]:
        entry["uncalibrated_sample_ids"] = ["v0001"]
        entry["calibrated_sample_ids"] = ["v0001"]
    index_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^the manifest cohort and the run index cohort "):
        extended.extended_metrics(
            index_path,
            tmp_path / "out.json",
            manifest_path=manifest_path,
            labels_root=labels_root,
        )


def test_a_mask_that_is_not_the_frozen_file_is_refused(tmp_path: Path) -> None:
    """A mask edited after the runs read it would rewrite history without a trace."""

    extended = load_extended()
    manifest_path, labels_root, digest = write_ground_truth(tmp_path / "gt")
    index_path = build_index(tmp_path / "runs", digest)
    tampered = TRUTH.copy()
    tampered[0, 0] = 0
    Image.fromarray(tampered.astype(np.uint8)).save(labels_root / "v0001_train_id.png")

    with pytest.raises(ValueError, match=r"^the ground-truth mask for v0001 is not the file "):
        extended.extended_metrics(
            index_path,
            tmp_path / "out.json",
            manifest_path=manifest_path,
            labels_root=labels_root,
        )


def test_the_document_binds_itself_to_the_study_and_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    """It is cited by claims, so it carries its hashes and is written exactly once."""

    extended = load_extended()
    _, _, digest = write_ground_truth(tmp_path / "gt")
    index_path = build_index(tmp_path / "runs", digest)
    output_path = tmp_path / "extended-metrics.json"
    extended.extended_metrics(index_path, output_path)

    raw = output_path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    assert raw == (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert document["protocol_hash"] == PROTOCOL
    assert document["dataset_manifest_hash"] == digest
    assert document["evaluation_for_ground_truth_metrics"] == "eval_calibrated"

    with pytest.raises(FileExistsError, match=r"^an extended metrics document already exists: "):
        extended.extended_metrics(index_path, output_path)


def test_an_index_that_fails_its_own_gate_is_refused(tmp_path: Path) -> None:
    """Every published document starts from the same gate."""

    extended = load_extended()
    _, _, digest = write_ground_truth(tmp_path / "gt")
    index_path = build_index(tmp_path / "runs", digest)
    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["runs"] = document["runs"][:-1]
    index_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^formal run index is not valid: "):
        extended.extended_metrics(index_path, tmp_path / "extended-metrics.json")
