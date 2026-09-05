"""Contracts for turning per-image artifacts into the three published documents.

Every metric this project reports is a function of the summed confusion matrix,
so one component array of per-image confusions supports all of them under one
shared bootstrap draw. That is what makes the intervals comparable across
metrics rather than each metric answering a slightly different question.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from drivemetrics.artifacts.predictions import (
    PredictionRecord,
    read_prediction_artifact,
    write_prediction_artifact,
)
from drivemetrics.metrics.calibration import (
    classwise_ece_sufficient_statistics,
    multiclass_brier_sums,
    pack_correctness,
    quantize_confidence,
)

MODELS = ("upernet_convnextv2_tiny", "upernet_dinov2_small", "segformer_b2")
# The order the analysis publishes pairs in. It is the APPROVED model order,
# deliberately NOT the order this file lists them in, because the published
# orientation must not depend on how an index was assembled. Written out here
# rather than imported, so that changing the approved order fails this file.
PUBLISHED_MODELS = ("segformer_b2", "upernet_convnextv2_tiny", "upernet_dinov2_small")
SEEDS = (17, 42, 73)
PROTOCOL = "a" * 64
MANIFEST = "b" * 64
NUM_CLASSES = 4
SAMPLES = ("v0001", "v0002", "v0003")
#: The profile schema pins each class ID to its BDD100K train-ID name, so a
#: fixture profile has to use the real first four rather than invented labels.
#: That guard is the reason a profile cannot be applied to another taxonomy.
CLASS_NAMES = ("road", "sidewalk", "building", "wall")
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
    off_class_mass: float = 0.1,
    present_classes: int = NUM_CLASSES,
) -> None:
    """Publish one run's per-image artifacts with a controllable correctness rate.

    Image sizes deliberately differ, so a ratio of summed confusions and a mean
    of per-image ratios cannot coincide by accident.

    The ground truth of an image depends on the image alone — its ID and size —
    never on the run's accuracy, because nine runs of one study score the SAME
    masks. A fixture whose targets moved with the accuracy would give every run
    its own cohort, and a check that the runs agree on their ground truth could
    never be written against it. `present_classes` narrows the taxonomy so that a
    class can be absent from truth and prediction alike.

    `off_class_mass` moves probability off the predicted class WITHOUT changing
    which class is predicted, which is exactly what temperature scaling does: the
    confusions of a calibrated and an uncalibrated evaluation are identical and
    only their confidences differ. A calibrated copy of a run is therefore
    written with a larger mass, and any metric that separates the two has to be
    reading confidence rather than argmax.
    """

    directory.mkdir(parents=True, exist_ok=True)
    for sample_id, size in zip(SAMPLES, sizes, strict=True):
        digest = hashlib.sha256(f"{sample_id}:{size}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:4], "big"))
        targets = rng.integers(0, present_classes, size, dtype=np.int64)
        correct = rng.random(size) < accuracy
        predicted = np.where(correct, targets, (targets + 1) % present_classes).astype(np.int64)

        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(confusion, (targets, predicted), 1)
        probabilities = np.full((size, NUM_CLASSES), off_class_mass, dtype=np.float64)
        probabilities[np.arange(size), predicted] = 1.0 - off_class_mass * (NUM_CLASSES - 1)
        confidence = probabilities[np.arange(size), predicted]

        record = PredictionRecord(
            sample_id=sample_id,
            predicted_class=predicted.astype(np.uint8),
            top1_confidence_q16=quantize_confidence(confidence),
            correctness_bitset=pack_correctness(predicted == targets),
            confusion=confusion,
            brier_sum_by_class=multiclass_brier_sums(probabilities, targets, NUM_CLASSES),
            valid_pixel_count=int(size),
        )
        write_prediction_artifact(
            directory / f"{sample_id}.json",
            record,
            classwise_ece_sufficient_statistics(probabilities, targets, NUM_CLASSES),
            protocol_sha256=protocol,
            dataset_manifest_sha256=manifest,
        )


def write_profile(directory: Path, name: str, *, critical: list[int], sensitivity: float) -> Path:
    """A four-class cost profile, so the fixture's taxonomy is the one being scored.

    The production profiles declare all nineteen BDD100K classes and cannot score
    a four-class fixture; `compute_cost_risk` refuses a profile whose costs do
    not cover the confusion exactly, which is the check that keeps a profile from
    being applied to a taxonomy it was not written for.
    """

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "bdd100k-risk-profile/v1",
                "name": name,
                "taxonomy": "bdd100k-semantic-train-id/v1",
                "sensitivity": sensitivity,
                "class_costs": [
                    {"class_id": class_id, "class_name": CLASS_NAMES[class_id], "cost": 1.0}
                    for class_id in range(NUM_CLASSES)
                ],
                "critical_class_ids": critical,
            }
        ),
        encoding="utf-8",
    )
    return path


def build_index(root: Path, *, present_classes: int = NUM_CLASSES, **overrides: Any) -> Path:
    """Nine runs whose accuracy separates the models but overlaps across seeds."""

    runs = []
    for model_position, model in enumerate(MODELS):
        for seed in SEEDS:
            run_id = f"{model}-seed-{seed}"
            directory = root / run_id
            accuracy = 0.55 + 0.08 * model_position + 0.01 * (seed % 7)
            write_run_artifacts(
                directory,
                accuracy=accuracy,
                sizes=(400, 1200, 800),
                present_classes=present_classes,
            )
            write_run_artifacts(
                root / f"{run_id}-calibrated",
                accuracy=accuracy,
                sizes=(400, 1200, 800),
                off_class_mass=0.2,
                present_classes=present_classes,
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
                    "calibrated_artifacts_dir": f"{run_id}-calibrated",
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


def fixture_profiles(root: Path) -> Path:
    """Two four-class profiles for the fixture's taxonomy, one of them uncritical."""

    directory = root / "risk_profiles"
    write_profile(directory, "balanced", critical=[], sensitivity=1.0)
    write_profile(directory, "vru_priority", critical=[2, 3], sensitivity=2.0)
    return directory


def run(root: Path, output: Path, **overrides: Any) -> Any:
    aggregate = load_aggregate()
    return aggregate.aggregate_runs(
        build_index(root, **overrides),
        output,
        resamples=60,
        risk_profiles_dir=fixture_profiles(root.parent),
    )


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
            key = f"{PUBLISHED_MODELS[left]} minus {PUBLISHED_MODELS[right]} ({metric})"
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

    with pytest.raises(
        ValueError,
        match=(
            r"^artifact v0001 in upernet_convnextv2_tiny-seed-17 carries a different "
            r"dataset manifest hash than the run index$"
        ),
    ):
        aggregate.aggregate_runs(
            index_path, tmp_path / "out", resamples=20, risk_profiles_dir=fixture_profiles(tmp_path)
        )


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

    with pytest.raises(
        ValueError,
        match=(
            r"^the critical classes have no ground-truth support in this cohort, so "
            r"critical recall is undefined and must not be reported as zero$"
        ),
    ):
        aggregate._metric_from_confusion(confusion, "critical_recall", CRITICAL_CLASSES)


def test_an_artifact_from_another_protocol_is_refused(tmp_path: Path) -> None:
    """A run scored under a different protocol belongs to a different study."""

    aggregate = load_aggregate()
    root = tmp_path / "runs"
    index_path = build_index(root)
    write_run_artifacts(
        root / f"{MODELS[0]}-seed-17", sizes=(400, 1200, 800), accuracy=0.6, protocol="d" * 64
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^artifact v0001 in upernet_convnextv2_tiny-seed-17 carries a different "
            r"protocol hash than the run index$"
        ),
    ):
        aggregate.aggregate_runs(
            index_path, tmp_path / "out", resamples=20, risk_profiles_dir=fixture_profiles(tmp_path)
        )


def test_a_written_document_is_byte_exact_and_key_sorted(tmp_path: Path) -> None:
    """Analysis documents are cited by hash, so their bytes are the contract.

    The format itself is pinned in `test_documents.py`; this shows the aggregate
    writer applies exactly that format to a document it has validated.
    """

    from drivemetrics.artifacts.documents import serialised_document, validated_document

    aggregate = load_aggregate()
    document = {
        "schema_version": "driving-risk-intervals/v1",
        "protocol_hash": PROTOCOL,
        "dataset_manifest_hash": MANIFEST,
        "intervals": {},
    }
    path = tmp_path / "document.json"

    aggregate._write(path, document)

    assert path.read_bytes() == serialised_document(validated_document(document)).encode("utf-8")


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


def test_the_cohort_name_is_read_from_the_index(tmp_path: Path) -> None:
    """The recorded cohort names WHICH locked set produced the numbers.

    The existing test asserts `locked_validation`, which is also the fallback,
    so it passes whether the key is read or not. A different name is what
    proves the lookup happens: if the key were misspelled in the reader, every
    analysis would silently claim to be over the default cohort.
    """

    output = tmp_path / "out"
    run(tmp_path / "runs", output, cohort="locked_validation_rev2")

    assert documents(output)["metrics"]["cohort"] == "locked_validation_rev2"


def test_an_index_without_a_cohort_falls_back_to_the_locked_validation_set(
    tmp_path: Path,
) -> None:
    """The pair to the test above, which is what pins the fallback itself.

    With the key present and equal to the fallback, nothing distinguishes a
    changed default from a correct one.
    """

    output = tmp_path / "out"
    index = build_index(tmp_path / "runs")
    document = json.loads(index.read_text(encoding="utf-8"))
    del document["cohort"]
    index.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    aggregate = load_aggregate()
    aggregate.aggregate_runs(
        index, output, resamples=60, risk_profiles_dir=fixture_profiles(tmp_path)
    )

    assert documents(output)["metrics"]["cohort"] == "locked_validation"


def test_the_intervals_cover_exactly_the_unordered_model_pairs(tmp_path: Path) -> None:
    """Every metric gets one interval per unordered pair, and nothing else.

    The neighbouring test checks that the expected keys are present, which a
    loop producing extra keys still satisfies. This one pins the whole key set,
    so a self-pair, a duplicated pair, or a reversed pair is a failure rather
    than an unnoticed extra row in the report.
    """

    run(tmp_path / "runs", tmp_path / "out")
    intervals = documents(tmp_path / "out")["intervals"]["intervals"]

    expected = {
        f"{PUBLISHED_MODELS[left]} minus {PUBLISHED_MODELS[right]} ({metric})"
        for metric in ("miou", "pixel_accuracy", "critical_recall")
        for left, right in ((0, 1), (0, 2), (1, 2))
    }
    assert set(intervals) == expected


def test_the_paired_estimate_is_the_difference_of_the_two_model_means(
    tmp_path: Path,
) -> None:
    """The key names a difference, so the number must be that difference.

    The bootstrap orients the pair by giving the left model's runs `+metric` and
    the right model's `-metric`, then combines the two group means. Combining
    them by a SUM is what makes the estimate `metric_left - metric_right`.

    Until P1-17 they were combined by a mean, which published half the
    difference under a key that named the whole of it. Sign and zero-crossing
    were unaffected, so the ranking comparison was sound throughout and no
    published ordering changes; the reported effect size was wrong by a factor
    of two. A strict `xfail` carrying this estimand stood here until the fix
    landed, so the correction could not be made silently.

    Checking every pair and every metric rather than one of each is what makes
    this able to detect a selection that grabs the wrong runs, a label that
    collapses both models into one group, and an orientation that silently swaps
    the two models.
    """

    run(tmp_path / "runs", tmp_path / "out")
    published = documents(tmp_path / "out")
    table = published["metrics"]["metrics"]
    intervals = published["intervals"]["intervals"]

    for metric in ("miou", "pixel_accuracy", "critical_recall"):
        for left, right in ((0, 1), (0, 2), (1, 2)):
            key = f"{PUBLISHED_MODELS[left]} minus {PUBLISHED_MODELS[right]} ({metric})"
            difference = (
                table[PUBLISHED_MODELS[left]][metric] - table[PUBLISHED_MODELS[right]][metric]
            )
            assert intervals[key]["estimate"] == pytest.approx(difference, rel=0.0, abs=1e-12)


def test_each_metric_is_computed_by_its_own_statistic(tmp_path: Path) -> None:
    """Three metrics, three different numbers; a shared statistic would tie them.

    The metric name is threaded from the loop into `_statistic_for` and on into
    `_metric_from_confusion`. Nulled at either hop, every metric would be
    computed by whichever branch a missing name falls through to, and all three
    published values would agree for reasons no reader could see.
    """

    run(tmp_path / "runs", tmp_path / "out")
    table = documents(tmp_path / "out")["metrics"]["metrics"]

    for model in MODELS:
        values = [table[model][name] for name in ("miou", "pixel_accuracy", "critical_recall")]
        assert len(set(values)) == 3, f"{model} reports the same number for different metrics"


def test_critical_recall_is_one_minus_the_pooled_false_negative_rate(tmp_path: Path) -> None:
    """The published safety metric is recomputed here from the summed confusions.

    A sign slip or a wrong constant leaves a number in the same range and of
    the same shape, so only an independent recomputation can catch it. This
    also pins the branch selection: asking for `pixel_accuracy` must return
    pixel accuracy, not fall through to the critical-recall branch.
    """

    from drivemetrics.metrics.confusion import summarize_confusion
    from drivemetrics.metrics.risk import critical_false_negative_rate

    aggregate = load_aggregate()
    run(tmp_path / "runs", tmp_path / "out")
    table = documents(tmp_path / "out")["metrics"]["metrics"]

    for model in MODELS:
        per_seed_recall = []
        per_seed_accuracy = []
        for seed in SEEDS:
            confusion = aggregate.summed_confusion(
                tmp_path / "runs" / f"{model}-seed-{seed}", SAMPLES
            )
            rate = critical_false_negative_rate(confusion, CRITICAL_CLASSES)
            assert rate is not None
            per_seed_recall.append(1.0 - rate)
            per_seed_accuracy.append(summarize_confusion(confusion).pixel_accuracy)
        assert table[model]["critical_recall"] == pytest.approx(float(np.mean(per_seed_recall)))
        assert table[model]["pixel_accuracy"] == pytest.approx(float(np.mean(per_seed_accuracy)))


def test_the_seed_count_is_a_whole_number_of_seeds(tmp_path: Path) -> None:
    """Nine runs over three models is three seeds, not 3.0.

    The value is published and read back from JSON, where an integer division
    written as a true division becomes a float and the document says
    `"seed_count": 3.0`. Nothing downstream would crash; the experiment card
    would simply describe a fractional number of seeds.
    """

    run(tmp_path / "runs", tmp_path / "out")
    metrics = documents(tmp_path / "out")["metrics"]

    assert metrics["seed_count"] == len(SEEDS)
    assert isinstance(metrics["seed_count"], int)


def test_the_result_names_the_three_documents_it_wrote(tmp_path: Path) -> None:
    """The caller is handed paths, and those paths must be the files on disk.

    The command line writes a run record from this result, so a null or
    misspelled path here becomes a run record pointing at nothing while the
    analysis itself sits correctly on disk under its real name.
    """

    result = run(tmp_path / "runs", tmp_path / "out")
    output = tmp_path / "out"

    assert result.metrics_path == output / "metrics.json"
    assert result.intervals_path == output / "intervals.json"
    assert result.rankings_path == output / "rankings.json"
    assert result.sample_count == len(SAMPLES)
    assert sorted(path.name for path in output.iterdir()) == [
        "intervals.json",
        "metrics.json",
        "rankings.json",
    ]


def test_an_existing_empty_output_directory_is_usable(tmp_path: Path) -> None:
    """Only an existing ANALYSIS is refused; an existing directory is not.

    Callers create the output directory themselves, and a run that refuses to
    write into a directory somebody has already made would fail after the
    expensive part of the work rather than before it. The refusal that matters
    is the one keyed on the three document names, and it is tested separately.
    """

    output = tmp_path / "out" / "nested"
    output.mkdir(parents=True)

    result = run(tmp_path / "runs", output)

    assert result.metrics_path.is_file()


def test_the_requested_resample_count_and_seed_reach_the_interval(tmp_path: Path) -> None:
    """Both draw settings are forwarded, and the interval records what it used.

    An interval that reports settings it did not use cannot be reproduced from
    its own record. Dropping either argument on the way into the estimator
    substitutes the protocol default silently, and the recorded value is the
    only place that would show it.
    """

    aggregate = load_aggregate()
    aggregate.aggregate_runs(
        build_index(tmp_path / "runs"),
        tmp_path / "out",
        resamples=37,
        seed=11,
        risk_profiles_dir=fixture_profiles(tmp_path),
    )

    for entry in documents(tmp_path / "out")["intervals"]["intervals"].values():
        assert entry["resamples"] == 37
        assert entry["seed"] == 11


def test_the_protocol_resample_default_reaches_the_interval(tmp_path: Path) -> None:
    """`resamples` has a default so a caller may vary it, which is why it needs pinning.

    A run that quietly used 5,001 resamples would reproduce from its own record
    and not from the protocol, and the two intervals are not the same estimator.
    """

    aggregate = load_aggregate()
    aggregate.aggregate_runs(
        build_index(tmp_path / "runs"),
        tmp_path / "out",
        risk_profiles_dir=fixture_profiles(tmp_path),
    )

    for entry in documents(tmp_path / "out")["intervals"]["intervals"].values():
        assert entry["resamples"] == 5000


def test_the_cohort_order_comes_from_the_first_run_not_from_any_other(
    tmp_path: Path,
) -> None:
    """The image axis is fixed once, and a later run's listing order cannot move it.

    The index validator compares cohorts as sets, so two runs may legitimately
    list the same images in different orders. The analysis must not depend on
    which of them it happens to read: the bootstrap draws image POSITIONS, so a
    permuted axis makes a fixed seed select different images and moves the
    published interval.

    Recorded for P1-17: the order is taken from the first run rather than being
    canonical, so this test pins the current behaviour rather than the ideal
    one. Sorting the cohort, or having the validator compare ordered tuples,
    would make the choice unnecessary.
    """

    aggregate = load_aggregate()
    reference_index = build_index(tmp_path / "reference")
    aggregate.aggregate_runs(
        reference_index,
        tmp_path / "reference-out",
        resamples=60,
        risk_profiles_dir=fixture_profiles(tmp_path),
    )

    shuffled_index = build_index(tmp_path / "shuffled")
    document = json.loads(shuffled_index.read_text(encoding="utf-8"))
    document["runs"][1]["uncalibrated_sample_ids"] = list(reversed(SAMPLES))
    document["runs"][1]["calibrated_sample_ids"] = list(reversed(SAMPLES))
    shuffled_index.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    aggregate.aggregate_runs(
        shuffled_index,
        tmp_path / "shuffled-out",
        resamples=60,
        risk_profiles_dir=fixture_profiles(tmp_path),
    )

    for name in ("metrics", "intervals", "rankings"):
        assert (tmp_path / "reference-out" / f"{name}.json").read_bytes() == (
            tmp_path / "shuffled-out" / f"{name}.json"
        ).read_bytes()


def test_an_invalid_index_reports_every_violation_in_one_message(tmp_path: Path) -> None:
    """One run of the validator must list everything wrong, joined readably.

    An operator fixing a run index one error at a time pays a full validation
    cycle per fix, so every violation has to arrive together.

    The message is rebuilt here from the validator's own output rather than
    pattern-matched. A substring check cannot police the separator: any joiner
    that merely CONTAINS `"; "` satisfies it, so the sentence could be glued
    together with padding on both sides of the separator and still pass.
    """

    from drivemetrics.artifacts.formal_set import validate_formal_run_index

    index_path = build_index(tmp_path / "runs", expected_steps=15000)
    violations = validate_formal_run_index(json.loads(index_path.read_text(encoding="utf-8")))
    assert len(violations) >= 2, "this index must fail more than once for the join to matter"

    aggregate = load_aggregate()
    with pytest.raises(
        ValueError,
        match=r"^formal run index is not valid: run 0 \(upernet_convnextv2_tiny, seed 17\): checkpoint is at step ",
    ) as failure:
        aggregate.aggregate_runs(
            index_path, tmp_path / "out", resamples=20, risk_profiles_dir=fixture_profiles(tmp_path)
        )

    assert str(failure.value) == "formal run index is not valid: " + "; ".join(violations)


def test_the_published_interval_does_not_depend_on_the_order_the_samples_are_listed_in(
    tmp_path: Path,
) -> None:
    """The image axis is the cohort, and a cohort is a set; its listing order is not data.

    Every replicate draws POSITIONS on the image axis, so the axis order decides
    which images a given seed selects. Taking that order from whichever run
    happens to be first in the index makes the published bound depend on how the
    index was assembled, while the index validator proves only that the runs
    share the same SET of samples. Two indexes describing the identical study
    would then publish different intervals, which is the one thing a
    reproducibility claim cannot survive.
    """

    aggregate = load_aggregate()

    forward = build_index(tmp_path / "forward")
    aggregate.aggregate_runs(
        forward,
        tmp_path / "out-forward",
        resamples=60,
        risk_profiles_dir=fixture_profiles(tmp_path),
    )

    reversed_index = build_index(tmp_path / "reversed")
    document = json.loads(reversed_index.read_text(encoding="utf-8"))
    for entry in document["runs"]:
        entry["uncalibrated_sample_ids"] = list(reversed(entry["uncalibrated_sample_ids"]))
        entry["calibrated_sample_ids"] = list(reversed(entry["calibrated_sample_ids"]))
    reversed_index.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    aggregate.aggregate_runs(
        reversed_index,
        tmp_path / "out-reversed",
        resamples=60,
        risk_profiles_dir=fixture_profiles(tmp_path),
    )

    assert (
        documents(tmp_path / "out-forward")["intervals"]
        == documents(tmp_path / "out-reversed")["intervals"]
    )


def test_the_analysis_does_not_depend_on_the_order_the_runs_are_listed_in(
    tmp_path: Path,
) -> None:
    """The run order is how the index was assembled, and it must not reach a published number.

    It reaches two things if it is not canonicalised. The order the models first
    appear becomes the orientation of every pair, so the same study publishes
    `A minus B` from one index and `B minus A` from another. And the order of
    the seeds inside a model becomes the axis the seed resample draws positions
    on, exactly as the image order does, so the bounds move as well.

    The index validator requires all nine model-and-seed combinations to be
    present, so a canonical order always exists: the approved models in their
    declared order, each with the approved seeds in theirs.
    """

    aggregate = load_aggregate()

    forward = build_index(tmp_path / "forward")
    aggregate.aggregate_runs(
        forward,
        tmp_path / "out-forward",
        resamples=60,
        risk_profiles_dir=fixture_profiles(tmp_path),
    )

    shuffled_index = build_index(tmp_path / "shuffled")
    document = json.loads(shuffled_index.read_text(encoding="utf-8"))
    document["runs"] = list(reversed(document["runs"]))
    shuffled_index.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    aggregate.aggregate_runs(
        shuffled_index,
        tmp_path / "out-shuffled",
        resamples=60,
        risk_profiles_dir=fixture_profiles(tmp_path),
    )

    forward_docs = documents(tmp_path / "out-forward")
    shuffled_docs = documents(tmp_path / "out-shuffled")

    assert forward_docs["intervals"] == shuffled_docs["intervals"]
    assert forward_docs["metrics"] == shuffled_docs["metrics"]
    assert forward_docs["rankings"] == shuffled_docs["rankings"]


def test_the_metrics_document_carries_per_class_iou_and_recall(tmp_path: Path) -> None:
    """A mean over classes hides which class the model actually fails on.

    mIoU is one number over nineteen classes, and a model can hold it up while
    being unusable on the two or three that matter for a safety argument. The
    per-class values are recomputed here from the summed confusions rather than
    read back, so the document cannot agree with itself by construction.

    A class with no ground-truth support in the cohort is `None`, never `0.0`:
    an IoU of zero is the worst possible score and an absent class has no score.
    """

    from drivemetrics.metrics.confusion import summarize_confusion

    aggregate = load_aggregate()
    run(tmp_path / "runs", tmp_path / "out")
    published = documents(tmp_path / "out")["metrics"]

    for model in MODELS:
        per_seed_iou = []
        per_seed_recall = []
        for seed in SEEDS:
            summary = summarize_confusion(
                aggregate.summed_confusion(tmp_path / "runs" / f"{model}-seed-{seed}", SAMPLES)
            )
            per_seed_iou.append(summary.class_iou)
            per_seed_recall.append(summary.class_recall)

        for name, per_seed in (("iou", per_seed_iou), ("recall", per_seed_recall)):
            published_values = published["per_class"]["by_model"][model][name]
            assert len(published_values) == NUM_CLASSES
            for class_id in range(NUM_CLASSES):
                supported: list[float] = [
                    value for row in per_seed if (value := row[class_id]) is not None
                ]
                if not supported:
                    assert published_values[class_id] is None
                else:
                    assert published_values[class_id] == pytest.approx(
                        sum(supported) / len(supported), rel=0.0, abs=1e-12
                    )


def test_the_metrics_document_separates_calibration_before_and_after_temperature(
    tmp_path: Path,
) -> None:
    """Temperature changes confidence and nothing else, so only a confidence metric can see it.

    Argmax is invariant under a positive scalar temperature, so the confusions of
    the two evaluations are identical and every metric derived from them agrees.
    ECE and Brier are the two numbers that separate them, and they are the reason
    the calibrated artifacts were written at all. A document that reported one
    calibration block for both, or that read the uncalibrated directory twice,
    would publish the temperature fit as having done nothing.
    """

    run(tmp_path / "runs", tmp_path / "out")
    published = documents(tmp_path / "out")["metrics"]

    for model in MODELS:
        block = published["calibration"][model]
        assert set(block) == {"uncalibrated", "calibrated"}
        for kind in ("uncalibrated", "calibrated"):
            assert set(block[kind]) == {"ece", "brier", "per_seed"}
            assert block[kind]["ece"] is not None
            assert block[kind]["brier"] is not None
        assert block["uncalibrated"]["ece"] != block["calibrated"]["ece"]
        assert block["uncalibrated"]["brier"] != block["calibrated"]["brier"]

    # And the values are the ones the kernels give, not something recomputed here
    # by a different route: sum the statistics over the cohort and finalise once.
    from drivemetrics.artifacts.predictions import read_prediction_artifact
    from drivemetrics.metrics.calibration import (
        ECEBinSufficientStatistics,
        mean_classwise_expected_calibration_error,
        multiclass_brier_score,
    )

    seed_eces: list[float] = []
    for seed in SEEDS:
        directory = tmp_path / "runs" / f"{MODELS[0]}-seed-{seed}"
        counts = np.zeros((NUM_CLASSES, 15), dtype=np.int64)
        sums = np.zeros((NUM_CLASSES, 15), dtype=np.float64)
        positives = np.zeros((NUM_CLASSES, 15), dtype=np.int64)
        brier = np.zeros(NUM_CLASSES, dtype=np.float64)
        pixels = 0
        for sample_id in SAMPLES:
            _, record, ece = read_prediction_artifact(directory / f"{sample_id}.json")
            counts += ece.counts
            sums += ece.confidence_sums
            positives += ece.positive_counts
            brier += record.brier_sum_by_class
            pixels += record.valid_pixel_count
        value = mean_classwise_expected_calibration_error(
            ECEBinSufficientStatistics(
                counts=counts, confidence_sums=sums, positive_counts=positives
            )
        )
        assert value is not None
        assert multiclass_brier_score(brier, pixels) is not None
        seed_eces.append(value)

    expected_ece = sum(seed_eces) / len(seed_eces)
    block = published["calibration"][MODELS[0]]["uncalibrated"]
    assert block["ece"] == pytest.approx(expected_ece, rel=0.0, abs=1e-12)


def test_the_metrics_document_carries_every_declared_risk_profile(tmp_path: Path) -> None:
    """Publishing one cost profile would make the study's own question unanswerable.

    The question is whether the ranking depends on how cost is assigned, so every
    profile in the declared directory has to appear, including the one whose
    critical set is empty — that is the comparison the others are read against.
    The profiles are written here rather than taken from `configs/`, because the
    production profiles declare nineteen classes and this fixture has four; a
    test that reached into the repository's own configuration would be asserting
    something about the fixture's arity rather than about the aggregation.
    """

    from drivemetrics.metrics.risk import compute_cost_risk
    from drivemetrics.protocol.risk_profiles import load_risk_profile

    aggregate = load_aggregate()
    profiles = tmp_path / "profiles"
    write_profile(profiles, "balanced", critical=[], sensitivity=1.0)
    write_profile(profiles, "vru_priority", critical=[2, 3], sensitivity=2.0)

    aggregate.aggregate_runs(
        build_index(tmp_path / "runs"),
        tmp_path / "out",
        resamples=60,
        risk_profiles_dir=profiles,
    )
    published = documents(tmp_path / "out")["metrics"]

    assert sorted(published["risk_profiles"]) == ["balanced", "vru_priority"]
    for name in ("balanced", "vru_priority"):
        profile = load_risk_profile(profiles / f"{name}.yaml")
        block = published["risk_profiles"][name]
        assert block["sensitivity"] == profile.sensitivity
        assert block["critical_class_ids"] == list(profile.critical_class_ids)
        for model in MODELS:
            per_seed = [
                compute_cost_risk(
                    aggregate.summed_confusion(tmp_path / "runs" / f"{model}-seed-{seed}", SAMPLES),
                    profile,
                )
                for seed in SEEDS
            ]
            assert block["cost_risk"][model] == pytest.approx(
                sum(per_seed) / len(per_seed), rel=0.0, abs=1e-12
            )
    assert (
        published["risk_profiles"]["vru_priority"]["cost_risk"]
        != (published["risk_profiles"]["balanced"]["cost_risk"])
    )


def test_the_default_risk_profile_directory_is_the_repositorys_own(tmp_path: Path) -> None:
    """The default must be the shipped configuration, not a path a caller happens to pass."""

    aggregate = load_aggregate()
    profiles = aggregate.DEFAULT_RISK_PROFILES_DIR

    assert profiles.is_dir()
    assert sorted(path.stem for path in profiles.glob("*.yaml")) == [
        "balanced",
        "drivable_boundary",
        "vru_priority",
    ]


def test_summing_calibration_over_an_empty_cohort_is_refused(tmp_path: Path) -> None:
    """An empty cohort has no calibration to finalise, and zero error is a perfect score.

    The accumulator learns its array shapes from the first artifact it reads, so
    with nothing to read it has no shapes and no sums. Returning zeros would
    publish a perfectly calibrated model for a run that produced nothing.
    """

    aggregate = load_aggregate()
    directory = tmp_path / "runs" / f"{MODELS[0]}-seed-17"
    build_index(tmp_path / "runs")

    with pytest.raises(ValueError, match=r"^no artifacts found for the cohort under "):
        aggregate._accumulate_calibration(directory, ())


def test_per_class_values_carry_their_names_and_their_support(tmp_path: Path) -> None:
    """Nineteen unlabelled floats are not a table, and a score without support is not a result.

    The real cohort has a class that appears in seven images out of 998. Its IoU
    of zero is a measurement, but a reader who cannot see the seven cannot tell a
    catastrophic model from a class the cohort barely holds. The names come from
    the same table the risk-profile schema pins classes to, so the two can never
    disagree about what class 2 is called.
    """

    from drivemetrics.protocol.risk_profiles import BDD100K_SEMANTIC_CLASS_NAMES

    aggregate = load_aggregate()
    run(tmp_path / "runs", tmp_path / "out")
    per_class = documents(tmp_path / "out")["metrics"]["per_class"]

    assert per_class["class_names"] == list(BDD100K_SEMANTIC_CLASS_NAMES[:NUM_CLASSES])
    assert set(per_class["by_model"]) == set(MODELS)

    # Recomputed from the artifacts of one run — any run, because they share it.
    directory = tmp_path / "runs" / f"{MODELS[0]}-seed-{SEEDS[0]}"
    expected_support = aggregate.summed_confusion(directory, SAMPLES).sum(axis=1)
    expected_images = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sample_id in SAMPLES:
        _, record, _ = read_prediction_artifact(directory / f"{sample_id}.json")
        expected_images += record.confusion.sum(axis=1) > 0

    assert per_class["support_pixels"] == [int(value) for value in expected_support]
    assert per_class["images_with_class"] == [int(value) for value in expected_images]
    assert all(isinstance(value, int) for value in per_class["support_pixels"])
    assert sum(per_class["support_pixels"]) == sum((400, 1200, 800))


def test_a_class_absent_from_the_cohort_reports_null_scores_and_zero_support(
    tmp_path: Path,
) -> None:
    """Zero support and a zero score must never be the same character on the page."""

    run(tmp_path / "runs", tmp_path / "out", present_classes=NUM_CLASSES - 1)
    per_class = documents(tmp_path / "out")["metrics"]["per_class"]

    absent = NUM_CLASSES - 1
    assert per_class["support_pixels"][absent] == 0
    assert per_class["images_with_class"][absent] == 0
    for model in MODELS:
        assert per_class["by_model"][model]["iou"][absent] is None
        assert per_class["by_model"][model]["recall"][absent] is None
    assert per_class["support_pixels"][0] > 0


def test_runs_that_do_not_share_one_ground_truth_are_refused(tmp_path: Path) -> None:
    """Nine runs of one study score the same masks; a run that did not is another study.

    The index validator proves the runs share a SET of sample IDs. It cannot see
    whether the confusions behind those IDs were built from the same truth, and a
    run evaluated against different masks would pass every hash check while its
    numbers described a different cohort. The divergent run here is the SECOND in
    the canonical order, and the message names both it and the run it disagrees
    with, so a check that started one run too late, or named the wrong reference,
    is caught.
    """

    aggregate = load_aggregate()
    index_path = build_index(tmp_path / "runs")
    write_run_artifacts(
        tmp_path / "runs" / f"{PUBLISHED_MODELS[0]}-seed-{SEEDS[1]}",
        accuracy=0.7,
        sizes=(400, 1200, 800),
        present_classes=NUM_CLASSES - 1,
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"^the runs do not share one ground truth: {PUBLISHED_MODELS[0]}-seed-{SEEDS[1]} "
            rf"has different per-class support than {PUBLISHED_MODELS[0]}-seed-{SEEDS[0]}$"
        ),
    ):
        aggregate.aggregate_runs(
            index_path,
            tmp_path / "out",
            resamples=60,
            risk_profiles_dir=fixture_profiles(tmp_path),
        )


def test_a_divergent_first_run_is_caught_by_the_runs_that_follow_it(tmp_path: Path) -> None:
    """The reference run can be the odd one out; the check must not trust it by position.

    If the FIRST run were evaluated against different masks, every later run would
    disagree with it. A check that compared the later runs against the second run
    instead would find them all in agreement and publish the first run's numbers
    as the cohort's.
    """

    aggregate = load_aggregate()
    index_path = build_index(tmp_path / "runs")
    write_run_artifacts(
        tmp_path / "runs" / f"{PUBLISHED_MODELS[0]}-seed-{SEEDS[0]}",
        accuracy=0.7,
        sizes=(400, 1200, 800),
        present_classes=NUM_CLASSES - 1,
    )

    with pytest.raises(ValueError, match=r"^the runs do not share one ground truth: "):
        aggregate.aggregate_runs(
            index_path,
            tmp_path / "out",
            resamples=60,
            risk_profiles_dir=fixture_profiles(tmp_path),
        )


def test_an_image_counts_for_a_class_from_its_first_pixel() -> None:
    """Support is a count of images with ANY pixel of the class, and one pixel is any.

    Built directly from per-image confusions: two runs, two images, two classes.
    The second image holds exactly one pixel of the second class, so a threshold
    of two would report that class in one image instead of two.
    """

    aggregate = load_aggregate()
    first_image = np.array([[3, 0], [0, 1]], dtype=np.float64)  # class 1 has one pixel
    second_image = np.array([[2, 0], [0, 0]], dtype=np.float64)  # class 1 absent
    per_run = np.stack([first_image.reshape(-1), second_image.reshape(-1)])
    components = np.stack([per_run, per_run])
    runs = [{"run_id": "a"}, {"run_id": "b"}]

    support, images = aggregate._ground_truth_support(components, runs, 2, 2)

    assert support == [5, 1]
    assert images == [2, 1]


def test_an_interval_excludes_zero_only_when_neither_bound_touches_it() -> None:
    """The flag is a strict reading of two bounds, and the boundaries are where it matters."""

    aggregate = load_aggregate()

    assert aggregate._excludes_zero(0.1, 0.5) is True
    assert aggregate._excludes_zero(-0.5, -0.1) is True
    assert aggregate._excludes_zero(-0.5, 0.5) is False
    assert aggregate._excludes_zero(0.0, 0.5) is False
    assert aggregate._excludes_zero(-0.5, 0.0) is False


def test_a_taxonomy_the_class_table_cannot_name_is_refused(tmp_path: Path) -> None:
    """A class without a name cannot be published, and inventing one would be worse."""

    from drivemetrics.protocol.risk_profiles import BDD100K_SEMANTIC_CLASS_NAMES

    too_many = len(BDD100K_SEMANTIC_CLASS_NAMES) + 1
    with pytest.raises(ValueError, match=rf"^the index declares {too_many} classes but "):
        run(tmp_path / "runs", tmp_path / "out", num_classes=too_many)


def test_calibration_is_published_per_seed_and_the_mean_is_their_mean(tmp_path: Path) -> None:
    """One surprising calibration result cannot be read from a mean of three numbers.

    On the real cohort temperature scaling raised one model's ECE while lowering
    the other two models'. Whether that is one seed or all three is the whole
    question, and a seed mean cannot answer it. The per-seed values are the
    kernel's own, recomputed here from the artifacts by the same route.
    """

    from drivemetrics.metrics.calibration import (
        ECEBinSufficientStatistics,
        mean_classwise_expected_calibration_error,
        multiclass_brier_score,
    )

    run(tmp_path / "runs", tmp_path / "out")
    block = documents(tmp_path / "out")["metrics"]["calibration"][MODELS[0]]["uncalibrated"]

    assert set(block["per_seed"]) == {str(seed) for seed in SEEDS}
    for seed in SEEDS:
        directory = tmp_path / "runs" / f"{MODELS[0]}-seed-{seed}"
        counts = np.zeros((NUM_CLASSES, 15), dtype=np.int64)
        sums = np.zeros((NUM_CLASSES, 15), dtype=np.float64)
        positives = np.zeros((NUM_CLASSES, 15), dtype=np.int64)
        brier = np.zeros(NUM_CLASSES, dtype=np.float64)
        pixels = 0
        for sample_id in SAMPLES:
            _, record, ece = read_prediction_artifact(directory / f"{sample_id}.json")
            counts += ece.counts
            sums += ece.confidence_sums
            positives += ece.positive_counts
            brier += record.brier_sum_by_class
            pixels += record.valid_pixel_count
        expected_ece = mean_classwise_expected_calibration_error(
            ECEBinSufficientStatistics(
                counts=counts, confidence_sums=sums, positive_counts=positives
            )
        )
        published = block["per_seed"][str(seed)]
        assert set(published) == {"ece", "brier"}
        assert published["ece"] == pytest.approx(expected_ece, rel=0.0, abs=1e-12)
        assert published["brier"] == pytest.approx(
            multiclass_brier_score(brier, pixels), rel=0.0, abs=1e-12
        )

    seed_values = [block["per_seed"][str(seed)]["ece"] for seed in SEEDS]
    assert block["ece"] == pytest.approx(sum(seed_values) / len(seed_values), rel=0.0, abs=1e-12)


def test_every_document_declares_its_schema_version(tmp_path: Path) -> None:
    """A version string is how a later reader knows which contract a file was written under."""

    run(tmp_path / "runs", tmp_path / "out")
    published = documents(tmp_path / "out")

    assert published["metrics"]["schema_version"] == "driving-risk-metrics-table/v1"
    assert published["intervals"]["schema_version"] == "driving-risk-intervals/v1"
    assert published["rankings"]["schema_version"] == "driving-risk-rankings/v1"


def test_the_rankings_document_carries_the_separability_of_every_pair(tmp_path: Path) -> None:
    """ "No reversal" must not be the only thing this document can say about two models.

    On the real cohort the two best models are inseparable on mIoU and separable
    on VRU recall at the same confidence. The ranking is stable under both, so a
    document that reported only the order would hide the finding. Each pair's
    interval is copied from the intervals document — not recomputed — and the
    flag is a plain reading of it: the interval excludes zero or it does not.
    """

    run(tmp_path / "runs", tmp_path / "out")
    published = documents(tmp_path / "out")
    separability = published["rankings"]["separability"]
    intervals = published["intervals"]["intervals"]

    assert set(separability) == {"miou", "pixel_accuracy", "critical_recall"}
    pair_count = len(MODELS) * (len(MODELS) - 1) // 2
    for metric, pairs in separability.items():
        assert len(pairs) == pair_count
        for pair in pairs:
            assert set(pair) == {"left", "right", "estimate", "low", "high", "excludes_zero"}
            source = intervals[f"{pair['left']} minus {pair['right']} ({metric})"]
            assert (pair["estimate"], pair["low"], pair["high"]) == (
                source["estimate"],
                source["low"],
                source["high"],
            )
            assert pair["excludes_zero"] is (pair["low"] > 0.0 or pair["high"] < 0.0)
    # The pairs follow the approved model order, like the intervals they mirror.
    assert [(pair["left"], pair["right"]) for pair in separability["miou"]] == [
        (PUBLISHED_MODELS[0], PUBLISHED_MODELS[1]),
        (PUBLISHED_MODELS[0], PUBLISHED_MODELS[2]),
        (PUBLISHED_MODELS[1], PUBLISHED_MODELS[2]),
    ]
