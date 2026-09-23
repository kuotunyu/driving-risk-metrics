"""Contracts for the one verified read of every released prediction artifact.

The extraction is the only step that touches the 18 artifact directories, so
it is also where every integrity gate of the analysis plan lives: each payload
must be the one its evaluation recorded, each artifact must belong to the study,
and temperature scaling must not have changed a single predicted class.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drivemetrics.analysis.extended import (
    INSTANCE_CATEGORY_TO_TRAIN_ID,
    _corroborated_instances,
    _dense_prediction,
)
from drivemetrics.artifacts.predictions import read_prediction_artifact
from drivemetrics.metrics.calibration import unpack_correctness
from drivemetrics.metrics.instances import instance_coverages
from drivemetrics.posthoc import allseed

SORTED_RUNS = [
    f"{model}-seed-{seed}"
    for model in ("segformer_b2", "upernet_convnextv2_tiny", "upernet_dinov2_small")
    for seed in (17, 42, 73)
]


def extract(study: Any, output_dir: Path, **kwargs: Any) -> allseed.ExtractResult:
    return allseed.extract_all_seeds(
        study.index_path,
        study.manifest_path,
        study.labels_root,
        study.instance_root,
        study.tertiles_path,
        output_dir,
        **kwargs,
    )


def artifact(study: Any, run_id: str, sample_id: str, *, calibrated: bool = False):
    model, seed = run_id.rsplit("-seed-", 1)
    return read_prediction_artifact(
        study.run_dir(model, int(seed), calibrated=calibrated) / f"{sample_id}.json"
    )


def test_runs_and_images_are_extracted_in_the_declared_order(
    tmp_path: Path, study_builder: Any
) -> None:
    shuffled = tuple(
        (model, seed)
        for seed in (73, 17, 42)
        for model in ("upernet_dinov2_small", "segformer_b2", "upernet_convnextv2_tiny")
    )
    study = study_builder(tmp_path / "study", index_order=shuffled)

    result = extract(study, tmp_path / "out")

    document = json.loads((tmp_path / "out" / "extract.json").read_text(encoding="utf-8"))
    assert result.runs == tuple(SORTED_RUNS)
    assert result.images == 3
    assert [entry["run_id"] for entry in document["runs"]] == SORTED_RUNS
    assert document["sample_ids"] == ["v0001", "v0002", "v0003"]
    assert document["states"] == ["uncalibrated", "calibrated"]


def test_every_component_is_the_artifacts_own_value(tmp_path: Path, study: Any) -> None:
    extract(study, tmp_path / "out")
    components = np.load(tmp_path / "out" / "components.npz")

    assert components["confusion"].shape == (9, 3, 361)
    for r, run_id in enumerate(SORTED_RUNS):
        for i, sample_id in enumerate(("v0001", "v0002", "v0003")):
            for s, calibrated in enumerate((False, True)):
                _, record, ece = artifact(study, run_id, sample_id, calibrated=calibrated)
                correct = unpack_correctness(record.correctness_bitset, record.valid_pixel_count)
                assert np.array_equal(components["confusion"][r, i], record.confusion.reshape(-1))
                assert components["valid_pixels"][r, i] == record.valid_pixel_count
                assert np.array_equal(components["brier"][s, r, i], record.brier_sum_by_class)
                assert np.array_equal(components["ece_counts"][s, r, i], ece.counts)
                assert np.array_equal(
                    components["ece_confidence_sums"][s, r, i], ece.confidence_sums
                )
                assert np.array_equal(
                    components["ece_positive_counts"][s, r, i], ece.positive_counts
                )
                assert np.array_equal(
                    components["toplabel"][s, r, i],
                    allseed.toplabel_ece_components(record.top1_confidence_q16, correct),
                )


def test_histograms_hold_every_pixel_and_every_correct_pixel(tmp_path: Path, study: Any) -> None:
    extract(study, tmp_path / "out")

    for run_id in SORTED_RUNS:
        for state, calibrated in (("uncalibrated", False), ("calibrated", True)):
            histograms = np.load(tmp_path / "out" / "histograms" / f"{run_id}.{state}.npy")
            assert histograms.shape == (3, 2, 65536)
            assert histograms.dtype == np.int32
            for i, sample_id in enumerate(("v0001", "v0002", "v0003")):
                _, record, _ = artifact(study, run_id, sample_id, calibrated=calibrated)
                correct = unpack_correctness(record.correctness_bitset, record.valid_pixel_count)
                q16 = record.top1_confidence_q16.astype(np.int64)
                assert np.array_equal(histograms[i, 0], np.bincount(q16, minlength=65536))
                assert np.array_equal(histograms[i, 1], np.bincount(q16[correct], minlength=65536))


def test_instance_records_follow_the_released_coverage_rule(tmp_path: Path, study: Any) -> None:
    extract(study, tmp_path / "out")
    lines = (tmp_path / "out" / "instances.jsonl").read_text(encoding="utf-8").splitlines()
    from drivemetrics.analysis.extended import _read_mask

    edges = {INSTANCE_CATEGORY_TO_TRAIN_ID[1]: (4, 8), INSTANCE_CATEGORY_TO_TRAIN_ID[3]: (1, 1)}
    assert len(lines) == 3
    for line, sample_id in zip(lines, ("v0001", "v0002", "v0003"), strict=True):
        record = json.loads(line)
        truth = _read_mask(study.labels_root / f"{sample_id}_train_id.png")
        bitmask = _read_mask(
            study.instance_root / "labels" / "ins_seg" / "bitmasks" / "val" / f"{sample_id}.png"
        )
        corroboration = _corroborated_instances(
            truth, bitmask[:, :, 0], (bitmask[:, :, 2] << 8) | bitmask[:, :, 3]
        )
        assert record["sample_id"] == sample_id
        assert record["without_semantic_pixels"] == corroboration.without_semantic_pixels
        assert record["corroborated_fractions"] == corroboration.corroborated_fractions
        for run_id in SORTED_RUNS:
            _, prediction, _ = artifact(study, run_id, sample_id, calibrated=True)
            coverages = instance_coverages(
                truth.reshape(-1),
                _dense_prediction(prediction, truth).reshape(-1),
                corroboration.footprint_ids.reshape(-1),
                corroboration.classes,
                edges,
            )
            assert record["instances"] == [
                {
                    "instance_id": c.instance_id,
                    "class_id": c.class_id,
                    "area_pixels": c.area_pixels,
                    "area_tertile": c.area_tertile,
                }
                for c in coverages
            ]
            assert record["correct_fraction"][run_id] == [c.correct_fraction for c in coverages]
            assert record["correct_pixels"][run_id] == [
                round(c.correct_fraction * c.area_pixels) for c in coverages
            ]


def test_the_extraction_records_what_it_read_and_that_its_gates_passed(
    tmp_path: Path, study: Any
) -> None:
    extract(study, tmp_path / "out")

    document = json.loads((tmp_path / "out" / "extract.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == allseed.EXTRACT_SCHEMA_VERSION
    assert document["analysis_plan"] == allseed.ANALYSIS_PLAN
    assert document["index_sha256"] == hashlib.sha256(study.index_path.read_bytes()).hexdigest()
    assert (
        document["tertiles_sha256"] == hashlib.sha256(study.tertiles_path.read_bytes()).hexdigest()
    )
    assert document["dataset_manifest_sha256"] == study.dataset_manifest_sha256
    assert document["instance_bitmasks"]["count"] == 3
    assert document["gates"] == {"integrity": "passed"}
    assert document["critical_class_ids"] == [11, 12, 17, 18]
    assert [entry["seed"] for entry in document["runs"]][:3] == [17, 42, 73]


def test_progress_is_reported_per_image(tmp_path: Path, study: Any) -> None:
    seen: list[tuple[int, int]] = []

    extract(study, tmp_path / "out", on_image=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 3), (2, 3), (3, 3)]


def rewrite_record(directory: Path, change: Any) -> None:
    path = directory / "run_record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_a_payload_the_evaluation_did_not_record_is_refused(tmp_path: Path, study: Any) -> None:
    rewrite_record(
        study.run_dir("upernet_convnextv2_tiny", 42),
        lambda record: record["artifacts"].__setitem__("v0002", "f" * 64),
    )

    with pytest.raises(ValueError, match="not the payload its evaluation recorded"):
        extract(study, tmp_path / "out")
    assert not (tmp_path / "out" / "extract.json").exists()


def test_an_artifact_missing_from_its_run_record_is_refused(tmp_path: Path, study: Any) -> None:
    rewrite_record(
        study.run_dir("segformer_b2", 73, calibrated=True),
        lambda record: record["artifacts"].pop("v0003"),
    )

    with pytest.raises(ValueError, match="does not record every image"):
        extract(study, tmp_path / "out")


def test_a_run_record_for_another_seed_is_refused(tmp_path: Path, study: Any) -> None:
    rewrite_record(study.run_dir("segformer_b2", 17), lambda record: record.__setitem__("seed", 42))

    with pytest.raises(ValueError, match="seed"):
        extract(study, tmp_path / "out")


def test_a_temperature_that_changed_a_predicted_class_is_refused(
    tmp_path: Path, study: Any, study_tools: Any
) -> None:
    directory = study.run_dir("upernet_dinov2_small", 17, calibrated=True)

    def changed(sample_id: str) -> np.ndarray:
        grid = study_tools.default_prediction(sample_id, "upernet_dinov2_small", 17)
        if sample_id == "v0002":
            grid[0, 0] = 3
        return grid

    payloads = study_tools.write_artifacts(
        directory, study.dataset_manifest_sha256, changed, spread_offset=0.005
    )
    study_tools.write_run_record(
        directory, "upernet_dinov2_small-seed-17", 17, study.dataset_manifest_sha256, payloads
    )

    with pytest.raises(ValueError, match="temperature scaling changed"):
        extract(study, tmp_path / "out")


@pytest.mark.parametrize(
    ("protocol", "dataset", "message"),
    [("e" * 64, None, "protocol"), (None, "e" * 64, "dataset manifest")],
)
def test_an_artifact_from_another_study_is_refused(
    tmp_path: Path,
    study: Any,
    study_tools: Any,
    protocol: str | None,
    dataset: str | None,
    message: str,
) -> None:
    from drivemetrics.artifacts.predictions import PredictionRecord, write_prediction_artifact
    from drivemetrics.metrics.calibration import (
        classwise_ece_sufficient_statistics,
        multiclass_brier_sums,
        pack_correctness,
        quantize_confidence,
    )

    directory = study.run_dir("segformer_b2", 42)
    sample_id = study_tools.SAMPLES[0]
    targets = study_tools.TRUTH[study_tools.VALID]
    probabilities = np.full((targets.size, 19), 0.01)
    probabilities[np.arange(targets.size), targets] = 1 - 0.18
    written = write_prediction_artifact(
        directory / f"{sample_id}.json",
        PredictionRecord(
            sample_id=sample_id,
            predicted_class=targets.astype(np.uint8),
            top1_confidence_q16=quantize_confidence(np.full(targets.size, 0.82)),
            correctness_bitset=pack_correctness(np.ones(targets.size, dtype=bool)),
            confusion=np.diag(np.bincount(targets, minlength=19)).astype(np.int64),
            brier_sum_by_class=multiclass_brier_sums(probabilities, targets, 19),
            valid_pixel_count=int(targets.size),
        ),
        classwise_ece_sufficient_statistics(probabilities, targets, 19),
        protocol_sha256=protocol or "a" * 64,
        dataset_manifest_sha256=dataset or study.dataset_manifest_sha256,
    )
    rewrite_record(
        directory, lambda record: record["artifacts"].__setitem__(sample_id, written.payload_sha256)
    )

    with pytest.raises(ValueError, match=message):
        extract(study, tmp_path / "out")


def test_an_artifact_filed_under_another_image_is_refused(tmp_path: Path, study: Any) -> None:
    directory = study.run_dir("upernet_convnextv2_tiny", 73, calibrated=True)
    (directory / "v0001.json").write_bytes((directory / "v0002.json").read_bytes())

    with pytest.raises(ValueError, match="names itself"):
        extract(study, tmp_path / "out")


def test_a_finished_extraction_is_never_overwritten(tmp_path: Path, study: Any) -> None:
    extract(study, tmp_path / "out")

    with pytest.raises(FileExistsError):
        extract(study, tmp_path / "out")


def test_an_index_that_fails_its_own_gate_is_refused(tmp_path: Path, study: Any) -> None:
    document = json.loads(study.index_path.read_text(encoding="utf-8"))
    document["runs"] = document["runs"][:-1]
    study.index_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="formal run index"):
        extract(study, tmp_path / "out")
