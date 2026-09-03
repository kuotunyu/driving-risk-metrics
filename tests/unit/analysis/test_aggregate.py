"""Contracts for turning per-image artifacts into the three published documents.

Every metric this project reports is a function of the summed confusion matrix,
so one component array of per-image confusions supports all of them under one
shared bootstrap draw. That is what makes the intervals comparable across
metrics rather than each metric answering a slightly different question.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from drivemetrics.artifacts.predictions import PredictionRecord, write_prediction_artifact
from drivemetrics.metrics.calibration import (
    classwise_ece_sufficient_statistics,
    pack_correctness,
    quantize_confidence,
)

MODELS = ("upernet_convnextv2_tiny", "upernet_dinov2_small", "segformer_b2")
SEEDS = (17, 42, 73)
PROTOCOL = "a" * 64
MANIFEST = "b" * 64
NUM_CLASSES = 4
SAMPLES = ("v0001", "v0002", "v0003")
CRITICAL_CLASSES = (2, 3)


def load_aggregate() -> ModuleType:
    try:
        from drivemetrics.analysis import aggregate
    except ImportError:
        pytest.fail("drivemetrics.analysis.aggregate is missing", pytrace=False)
    return aggregate


def write_run_artifacts(
    directory: Path,
    *,
    accuracy: float,
    sizes: tuple[int, ...],
    protocol: str = PROTOCOL,
    manifest: str = MANIFEST,
) -> None:
    """Publish one run's per-image artifacts with a controllable correctness rate.

    Image sizes deliberately differ, so a ratio of summed confusions and a mean
    of per-image ratios cannot coincide by accident.
    """

    directory.mkdir(parents=True, exist_ok=True)
    for sample_id, size in zip(SAMPLES, sizes, strict=True):
        rng = np.random.default_rng(abs(hash((sample_id, accuracy, size))) % (2**32))
        targets = rng.integers(0, NUM_CLASSES, size, dtype=np.int64)
        correct = rng.random(size) < accuracy
        predicted = np.where(correct, targets, (targets + 1) % NUM_CLASSES).astype(np.int64)

        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(confusion, (targets, predicted), 1)
        probabilities = np.full((size, NUM_CLASSES), 0.1, dtype=np.float64)
        probabilities[np.arange(size), predicted] = 1.0 - 0.1 * (NUM_CLASSES - 1)
        confidence = probabilities[np.arange(size), predicted]

        record = PredictionRecord(
            sample_id=sample_id,
            predicted_class=predicted.astype(np.uint8),
            top1_confidence_q16=quantize_confidence(confidence),
            correctness_bitset=pack_correctness(predicted == targets),
            confusion=confusion,
            brier_sum_by_class=np.zeros(NUM_CLASSES, dtype=np.float64),
            valid_pixel_count=int(size),
        )
        write_prediction_artifact(
            directory / f"{sample_id}.json",
            record,
            classwise_ece_sufficient_statistics(probabilities, targets, NUM_CLASSES),
            protocol_sha256=protocol,
            dataset_manifest_sha256=manifest,
        )


def build_index(root: Path, **overrides: Any) -> Path:
    """Nine runs whose accuracy separates the models but overlaps across seeds."""

    runs = []
    for model_position, model in enumerate(MODELS):
        for seed in SEEDS:
            run_id = f"{model}-seed-{seed}"
            directory = root / run_id
            write_run_artifacts(
                directory,
                accuracy=0.55 + 0.08 * model_position + 0.01 * (seed % 7),
                sizes=(400, 1200, 800),
            )
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
                    "uncalibrated_sample_ids": list(SAMPLES),
                    "calibrated_sample_ids": list(SAMPLES),
                }
            )

    document: dict[str, Any] = {
        "schema_version": "drivemetrics-formal-set/v1",
        "protocol_sha256": PROTOCOL,
        "dataset_manifest_sha256": MANIFEST,
        "expected_steps": 30000,
        "cohort": "locked_validation",
        "num_classes": NUM_CLASSES,
        "critical_class_ids": list(CRITICAL_CLASSES),
        "runs": runs,
    }
    document.update(overrides)
    index_path = root / "formal_run_index.json"
    index_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return index_path


def run(root: Path, output: Path, **overrides: Any) -> Any:
    aggregate = load_aggregate()
    return aggregate.aggregate_runs(build_index(root, **overrides), output, resamples=60)


def documents(output: Path) -> dict[str, Any]:
    return {
        name: json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("metrics", "intervals", "rankings")
    }


def test_it_writes_exactly_the_three_documents_the_report_requires(tmp_path: Path) -> None:
    """The report refuses to build without all three, so a partial set is useless."""

    run(tmp_path / "runs", tmp_path / "out")

    for name in ("metrics", "intervals", "rankings"):
        assert (tmp_path / "out" / f"{name}.json").is_file()


def test_the_metrics_document_carries_the_fields_the_page_states(tmp_path: Path) -> None:
    """A chart without cohort, sample count, seeds and method is not interpretable."""

    run(tmp_path / "runs", tmp_path / "out")
    metrics = documents(tmp_path / "out")["metrics"]

    assert metrics["protocol_hash"] == PROTOCOL
    assert metrics["dataset_manifest_hash"] == MANIFEST
    assert metrics["cohort"] == "locked_validation"
    assert metrics["sample_count"] == len(SAMPLES)
    assert metrics["seed_count"] == len(SEEDS)
    assert "bootstrap" in metrics["interval_method"]
    assert set(metrics["metrics"]) == set(MODELS)


def test_every_model_reports_the_same_metric_names(tmp_path: Path) -> None:
    """A metric present for one model and absent for another cannot be ranked."""

    run(tmp_path / "runs", tmp_path / "out")
    table = documents(tmp_path / "out")["metrics"]["metrics"]

    names = {frozenset(values) for values in table.values()}
    assert len(names) == 1
    assert {"miou", "pixel_accuracy", "critical_recall"} <= next(iter(names))


def test_a_cohort_metric_is_a_ratio_of_sums_not_a_mean_of_per_image_ratios(
    tmp_path: Path,
) -> None:
    """This is the conflation the component estimator exists to prevent."""

    aggregate = load_aggregate()
    run(tmp_path / "runs", tmp_path / "out")
    table = documents(tmp_path / "out")["metrics"]["metrics"]

    # Recompute one model's mIoU straight from its summed confusions.
    from drivemetrics.metrics.confusion import summarize_confusion

    summed = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for seed in SEEDS:
        summed += aggregate.summed_confusion(
            tmp_path / "runs" / f"{MODELS[0]}-seed-{seed}", SAMPLES
        )
    per_seed = []
    for seed in SEEDS:
        confusion = aggregate.summed_confusion(
            tmp_path / "runs" / f"{MODELS[0]}-seed-{seed}", SAMPLES
        )
        per_seed.append(summarize_confusion(confusion).mean_iou)

    assert table[MODELS[0]]["miou"] == pytest.approx(float(np.mean(per_seed)))


def test_every_interval_names_the_estimator_that_produced_it(tmp_path: Path) -> None:
    """A ratio interval and a per-image-mean interval answer different questions."""

    run(tmp_path / "runs", tmp_path / "out")
    intervals = documents(tmp_path / "out")["intervals"]["intervals"]

    assert intervals
    for entry in intervals.values():
        assert entry["estimator"] in {"ratio_of_sums", "per_image_mean"}
        assert entry["confidence"] == pytest.approx(0.95)
        assert entry["resamples"] > 0
        assert entry["low"] <= entry["estimate"] <= entry["high"]


def test_intervals_cover_every_model_pair_for_every_metric(tmp_path: Path) -> None:
    """A missing pair would leave a comparison in the report with no uncertainty."""

    run(tmp_path / "runs", tmp_path / "out")
    intervals = documents(tmp_path / "out")["intervals"]["intervals"]

    for metric in ("miou", "critical_recall"):
        for left, right in ((0, 1), (0, 2), (1, 2)):
            key = f"{MODELS[left]} minus {MODELS[right]} ({metric})"
            assert key in intervals


def test_the_rankings_document_reports_the_observation(tmp_path: Path) -> None:
    """A reversal is an outcome to report, never a success criterion."""

    run(tmp_path / "runs", tmp_path / "out")
    rankings = documents(tmp_path / "out")["rankings"]

    assert rankings["baseline_metric"] == "miou"
    assert isinstance(rankings["comparisons"], list)
    for comparison in rankings["comparisons"]:
        assert set(comparison) >= {
            "metric_name",
            "baseline_order",
            "comparison_order",
            "reversal_observed",
        }


def test_two_aggregations_are_byte_identical(tmp_path: Path) -> None:
    """A moving interval could never appear in a verifiable published claim."""

    run(tmp_path / "runs", tmp_path / "first")
    run(tmp_path / "runs", tmp_path / "second")

    for name in ("metrics", "intervals", "rankings"):
        assert (tmp_path / "first" / f"{name}.json").read_bytes() == (
            tmp_path / "second" / f"{name}.json"
        ).read_bytes()


def test_an_index_that_fails_the_formal_gate_is_refused(tmp_path: Path) -> None:
    """Analysing an incomplete matrix would publish a narrowed interval."""

    with pytest.raises(ValueError, match=r"^formal run index is not valid: run"):
        run(tmp_path / "runs", tmp_path / "out", expected_steps=15000)


def test_an_artifact_from_another_cohort_is_refused(tmp_path: Path) -> None:
    """Pooling two cohorts produces a number that describes neither."""

    aggregate = load_aggregate()
    root = tmp_path / "runs"
    index_path = build_index(root)
    write_run_artifacts(
        root / f"{MODELS[0]}-seed-17", sizes=(400, 1200, 800), accuracy=0.6, manifest="c" * 64
    )

    with pytest.raises(ValueError, match=r"^artifact v0001 in upernet_convnextv2_tiny-seed-17"):
        aggregate.aggregate_runs(index_path, tmp_path / "out", resamples=20)


def test_it_refuses_to_overwrite_an_existing_analysis(tmp_path: Path) -> None:
    """A replaced analysis silently detaches every claim that cited its numbers."""

    run(tmp_path / "runs", tmp_path / "out")

    with pytest.raises(FileExistsError, match=r"^an analysis already exists:"):
        run(tmp_path / "runs", tmp_path / "out")


def test_the_package_exports_the_entry_point() -> None:
    """The command line consumes aggregation through the package entry point."""

    import drivemetrics.analysis as analysis

    assert analysis.aggregate_runs is load_aggregate().aggregate_runs


def test_a_cohort_with_no_artifacts_is_refused(tmp_path: Path) -> None:
    """An empty run directory would sum to nothing and report it as a result."""

    aggregate = load_aggregate()

    with pytest.raises(ValueError, match=r"^no artifacts found for the cohort under"):
        aggregate.summed_confusion(tmp_path, ())


def test_critical_recall_is_refused_when_the_classes_have_no_support(
    tmp_path: Path,
) -> None:
    """Reporting an undefined rate as zero would invent a perfect safety score."""

    aggregate = load_aggregate()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion[0, 0] = 10

    with pytest.raises(ValueError, match=r"^the critical classes have no ground-truth support"):
        aggregate._metric_from_confusion(confusion, "critical_recall", CRITICAL_CLASSES)


def test_an_artifact_from_another_protocol_is_refused(tmp_path: Path) -> None:
    """A run scored under a different protocol belongs to a different study."""

    aggregate = load_aggregate()
    root = tmp_path / "runs"
    index_path = build_index(root)
    write_run_artifacts(
        root / f"{MODELS[0]}-seed-17", sizes=(400, 1200, 800), accuracy=0.6, protocol="d" * 64
    )

    with pytest.raises(ValueError, match=r"^artifact v0001 in upernet_convnextv2_tiny-seed-17"):
        aggregate.aggregate_runs(index_path, tmp_path / "out", resamples=20)


def test_a_written_document_is_byte_exact_and_key_sorted(tmp_path: Path) -> None:
    """Analysis documents are cited by hash, so their bytes are the contract.

    Indentation, key order, encoding and the trailing newline are all part of
    what a claim's ``artifact_path`` resolves to. A reformat that changes any of
    them changes every hash that ever cited the file.
    """

    aggregate = load_aggregate()
    path = tmp_path / "document.json"

    aggregate._write(path, {"b": 2, "a": {"d": 4, "c": 3}})

    assert path.read_bytes() == b'{\n  "a": {\n    "c": 3,\n    "d": 4\n  },\n  "b": 2\n}\n'


def test_the_paired_statistic_is_oriented_left_minus_right() -> None:
    """A flipped sign turns "A beats B" into "B beats A" with the same interval.

    Nothing downstream can detect it: the interval is still finite, still the
    right width, and still centred on a plausible number. Only the sign of the
    orientation says which model the estimate is about.
    """

    aggregate = load_aggregate()

    signed = aggregate._signed_difference_statistic(lambda summed: summed, (0, 1))
    values = signed(np.array([3.0, 5.0], dtype=np.float64))

    assert values.tolist() == [3.0, -5.0]


def test_the_paired_statistic_orients_every_run_by_its_own_label() -> None:
    """Runs arrive interleaved, so the sign must follow the label, not the index."""

    aggregate = load_aggregate()

    signed = aggregate._signed_difference_statistic(lambda summed: summed, (1, 0, 1))
    values = signed(np.array([2.0, 2.0, 2.0], dtype=np.float64))

    assert values.tolist() == [-2.0, 2.0, -2.0]


def test_the_analysis_creates_a_nested_output_directory(tmp_path: Path) -> None:
    """An operator names the output path; the stage does not get to require its parents.

    `mkdir(parents=True)` is the difference between `--output-dir
    artifacts/analysis/bdd100k_semseg_v1` working on a clean checkout and
    failing with a bare FileNotFoundError after the analysis has already been
    computed. Every real invocation writes several levels deep.
    """

    output = tmp_path / "artifacts" / "analysis" / "bdd100k_semseg_v1"

    run(tmp_path / "runs", output)

    assert (output / "metrics.json").is_file()


def test_the_written_analysis_has_sorted_keys(tmp_path: Path) -> None:
    """Key order is part of the bytes, and the bytes are what a claim cites.

    Without `sort_keys=True` the file follows dict insertion order, so the same
    analysis rewritten under a different Python or after an unrelated field
    reorder produces different bytes and a different hash. Nothing else in the
    pipeline pins this, because the models themselves are declared in a
    meaningful rather than alphabetical order.
    """

    output = tmp_path / "out"
    run(tmp_path / "runs", output)

    for name in ("metrics", "intervals", "rankings"):
        text = (output / f"{name}.json").read_text(encoding="utf-8")
        keys = re.findall(r'^  "([^"]+)":', text, re.MULTILINE)
        assert keys == sorted(keys), f"{name}.json top-level keys are not sorted: {keys}"
