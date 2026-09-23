"""Contracts for the pre-registered statistics of the resolution analysis.

Every threshold, level and seed here is fixed by
docs/posthoc/resolution-v1/analysis-plan.md. The tests pin them so that a later
edit cannot quietly tune the analysis after the results are in.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drivemetrics.posthoc import resolution, statistics

PERSON, RIDER, CAR, MOTORCYCLE, BICYCLE = 11, 12, 13, 17, 18
MODELS = ("segformer_b2", "upernet_convnextv2_tiny", "upernet_dinov2_small")
SEEDS = (17, 42, 73)
ARM_SCALES = {"formal": 512 / 720, "s085": 0.85, "native": 1.0}


def run_id(model: str, seed: int) -> str:
    return f"{model}-seed-{seed}"


# ------------------------------------------------------------------ bootstrap


def test_the_pre_registered_settings_are_pinned() -> None:
    assert statistics.BOOTSTRAP_RESAMPLES == 5000
    assert statistics.BOOTSTRAP_SEED == 20260831
    assert math.isclose(statistics.PRIMARY_CONFIDENCE, 1 - 0.05 / 3)
    assert statistics.DESCRIPTIVE_CONFIDENCE == 0.95
    assert statistics.SUPPORT_THRESHOLD == 0.10
    assert statistics.EQUIVALENCE_BOUND == 0.05
    assert statistics.MIOU_DROP_LIMIT == 0.02


def test_the_bootstrap_resamples_images_and_takes_a_ratio_of_sums() -> None:
    """Images are the unit; a per-image mean would answer a different question."""

    numerator = np.array([3.0, 0.0, 1.0, 0.0])
    denominator = np.array([4.0, 0.0, 6.0, 2.0])

    result = statistics.paired_ratio_bootstrap(numerator, denominator, resamples=200, seed=7)

    assert result.estimate == pytest.approx(4.0 / 12.0)
    generator = np.random.default_rng(7)
    replicates = []
    for _ in range(200):
        weights = np.bincount(generator.integers(0, 4, size=4), minlength=4)
        if weights @ denominator > 0:
            replicates.append((weights @ numerator) / (weights @ denominator))
    low, high = np.quantile(replicates, [0.025, 0.975])
    assert result.intervals[statistics.DESCRIPTIVE_CONFIDENCE] == pytest.approx((low, high))
    assert result.undefined_resamples == 200 - len(replicates)
    assert (result.resamples, result.seed) == (200, 7)


def test_a_ratio_with_no_denominator_is_undefined_rather_than_zero() -> None:
    result = statistics.paired_ratio_bootstrap(np.zeros(3), np.zeros(3), resamples=10, seed=1)

    assert result.estimate is None
    assert result.intervals == {
        statistics.PRIMARY_CONFIDENCE: None,
        statistics.DESCRIPTIVE_CONFIDENCE: None,
    }
    assert result.undefined_resamples == 10


@pytest.mark.parametrize(
    ("numerator", "denominator", "message"),
    [
        (np.zeros(2), np.zeros(3), "same shape"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "one-dimensional"),
        (np.zeros(0), np.zeros(0), "at least one image"),
        (np.array([np.nan]), np.ones(1), "finite"),
        (np.ones(1), np.array([-1.0]), "nonnegative"),
    ],
)
def test_bootstrap_inputs_outside_the_contract_are_refused(
    numerator: np.ndarray, denominator: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        statistics.paired_ratio_bootstrap(numerator, denominator)


# -------------------------------------------------------------------- decisions


@pytest.mark.parametrize(
    ("estimate", "interval", "confounded", "expected"),
    [
        (0.12, (0.01, 0.2), False, ("A", None)),
        (0.10, (0.001, 0.2), False, ("A", None)),
        (0.099, (0.02, 0.18), False, ("B", None)),
        (0.02, (0.01, 0.04), False, ("B", None)),
        (0.0, (-0.049, 0.049), False, ("C", None)),
        (0.0, (-0.049, 0.049), True, ("D", "C")),
        (0.0, (-0.05, 0.03), False, ("D", None)),
        (0.2, (0.0, 0.3), False, ("D", None)),
        (None, None, False, ("D", None)),
    ],
)
def test_the_decision_rules_apply_in_the_listed_order(
    estimate: float | None,
    interval: tuple[float, float] | None,
    confounded: bool,
    expected: tuple[str, str | None],
) -> None:
    """A lower bound of exactly zero is not above zero; confounding only ever removes a C."""

    assert statistics.classify_contrast(estimate, interval, confounded=confounded) == expected


@pytest.mark.parametrize(
    ("categories", "expected"),
    [
        ({"a": "A", "b": "A", "c": "D"}, "A"),
        ({"a": "C", "b": "B", "c": "C"}, "C"),
        ({"a": "A", "b": "B", "c": "C"}, "model-dependent"),
        ({"a": "A"}, "model-dependent"),
    ],
)
def test_an_overall_conclusion_needs_two_models_in_one_category(
    categories: dict[str, str], expected: str
) -> None:
    assert statistics.overall_conclusion(categories) == expected


def test_the_exact_mcnemar_test_is_a_two_sided_binomial_on_discordant_pairs() -> None:
    assert statistics.exact_mcnemar(0, 0) == 1.0
    assert statistics.exact_mcnemar(5, 0) == pytest.approx(2 * 0.5**5)
    assert statistics.exact_mcnemar(3, 3) == pytest.approx(1.0)


def test_holm_adjusts_in_rank_order_and_stays_monotone() -> None:
    adjusted = statistics.holm_adjust([0.01, 0.04, 0.03, 0.5])

    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert statistics.holm_adjust([0.6, 0.9]) == pytest.approx([1.0, 1.0])
    assert statistics.holm_adjust([]) == []


# ------------------------------------------------------------ synthetic sweeps

MissRule = Callable[[str, str, str, int, dict[str, Any]], bool]


def instance_list(image: int) -> list[dict[str, Any]]:
    """Every image: four small persons, one large person, one small rider, one car."""

    instances = [
        {
            "instance_id": k + 1,
            "class_id": PERSON,
            "area_pixels": 40 + 60 * k,
            "area_tertile": "small",
        }
        for k in range(4)
    ]
    instances += [
        {"instance_id": 5, "class_id": PERSON, "area_pixels": 5000, "area_tertile": "large"},
        {"instance_id": 6, "class_id": RIDER, "area_pixels": 100, "area_tertile": "small"},
        {"instance_id": 7, "class_id": CAR, "area_pixels": 9000, "area_tertile": "large"},
    ]
    if image == 0:
        instances.append(
            {"instance_id": 8, "class_id": MOTORCYCLE, "area_pixels": 90, "area_tertile": "small"}
        )
    return instances


def diagonal_confusion(errors: int = 0) -> list[int]:
    matrix = np.diag(np.full(19, 100, dtype=np.int64))
    matrix[0, 1] += errors
    return [int(value) for value in matrix.reshape(-1)]


def write_sweep(
    directory: Path,
    miss: MissRule,
    *,
    images: int = 4,
    arms: tuple[str, ...] = ("formal", "s085", "native"),
    confusion: Callable[[str, str], list[int]] = lambda _arm, _run: diagonal_confusion(),
    sessions: list[dict[str, Any]] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    sample_ids = [f"c{index:04d}" for index in range(images)]
    run_ids = [run_id(model, seed) for model in MODELS for seed in SEEDS]
    configuration = {
        "schema_version": resolution.SWEEP_SCHEMA_VERSION,
        "tertiles_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "dataset_manifest_sha256": "3" * 64,
        "arms": [{"name": arm, "scale": ARM_SCALES.get(arm, 0.6)} for arm in arms],
        "runs": [
            {"run_id": run_id(model, seed), "model": model, "seed": seed}
            for model in MODELS
            for seed in reversed(SEEDS)
        ],
        "sample_ids": sample_ids,
    }
    (directory / resolution.RUN_FILENAME).write_text(
        json.dumps(configuration), encoding="utf-8", newline="\n"
    )
    lines = []
    for index, sample_id in enumerate(sample_ids):
        instances = instance_list(index)
        results = {
            arm: {
                identifier: {
                    "critical_miss": [
                        miss(arm, identifier.rsplit("-seed-", 1)[0], identifier, index, entry)
                        for entry in instances
                    ],
                    "confusion": confusion(arm, identifier),
                }
                for identifier in run_ids
            }
            for arm in arms
        }
        lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "bitmask_sha256": f"{index:064x}",
                    "instances": instances,
                    "results": results,
                }
            )
        )
    (directory / resolution.PER_IMAGE_FILENAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    if sessions is None:
        sessions = [session()]
    (directory / resolution.SESSIONS_FILENAME).write_text(
        "".join(json.dumps(entry) + "\n" for entry in sessions), encoding="utf-8", newline="\n"
    )
    return directory


def session(commit: str = "e" * 40, lock: str = "f" * 64) -> dict[str, Any]:
    return {
        "provenance": {"commit": commit, "lock_sha256": lock, "hardware": {"gpu": "A100"}},
        "parity_criterion": "exact",
        "scored_samples": 4,
    }


def small_person(entry: dict[str, Any]) -> bool:
    return bool(entry["class_id"] == PERSON and entry["area_tertile"] == "small")


def three_categories(arm: str, model: str, run: str, _image: int, entry: dict[str, Any]) -> bool:
    """SegFormer recovers every small person (A); ConvNeXt V2 changes nothing (C);
    DINOv2 misses one extra small person in four at formal, in seed 17 only (B)."""

    if not small_person(entry):
        return False
    if model == "segformer_b2":
        return arm == "formal"
    if model == "upernet_convnextv2_tiny":
        return bool(entry["instance_id"] == 1)
    extra = arm == "formal" and run.endswith("-seed-17") and entry["instance_id"] == 2
    return bool(entry["instance_id"] == 1 or extra)


def test_the_pre_registered_analysis_classifies_each_model(tmp_path: Path) -> None:
    directory = write_sweep(tmp_path / "out", three_categories)

    result = statistics.analyse_sweep(directory)

    assert result.categories == {
        "segformer_b2": "A",
        "upernet_convnextv2_tiny": "C",
        "upernet_dinov2_small": "B",
    }
    assert result.overall == "model-dependent"
    raw = result.summary_path.read_bytes()
    assert raw.endswith(b"}\n")
    assert b"\r\n" not in raw
    summary = json.loads(raw)
    assert summary["schema_version"] == statistics.SUMMARY_SCHEMA_VERSION
    assert summary["analysis_plan"] == resolution.ANALYSIS_PLAN
    assert summary["overall"] == "model-dependent"
    assert summary["counts"] == {
        "images": 4,
        "small_person_instances": 16,
        "images_with_small_person": 4,
    }
    assert summary["sweep"]["commit"] == "e" * 40
    assert summary["sweep"]["parity_criteria"] == ["exact"]

    segformer = summary["primary"]["segformer_b2"]
    contrast = segformer["formal_minus_native"]
    assert contrast["estimate"] == pytest.approx(1.0)
    assert contrast["per_seed"] == pytest.approx([1.0, 1.0, 1.0])
    assert contrast["intervals"] == [
        {"confidence": statistics.PRIMARY_CONFIDENCE, "low": 1.0, "high": 1.0},
        {"confidence": statistics.DESCRIPTIVE_CONFIDENCE, "low": 1.0, "high": 1.0},
    ]
    assert contrast["resamples"] == 5000
    assert contrast["seed"] == 20260831
    assert segformer["runs"] == [run_id("segformer_b2", seed) for seed in SEEDS]
    assert segformer["critical_misses"]["formal"] == {
        run_id("segformer_b2", seed): 16 for seed in SEEDS
    }
    assert segformer["scale_mismatch"] == {
        "miou_drop_formal_minus_native": pytest.approx(0.0),
        "miou_drop_exceeds_limit": False,
        "large_person_native_worse": False,
        "confounded": False,
    }
    assert segformer["category"] == "A"
    assert segformer["downgraded_from"] is None

    dinov2 = summary["primary"]["upernet_dinov2_small"]["formal_minus_native"]
    assert dinov2["estimate"] == pytest.approx(1 / 12)
    assert dinov2["per_seed"] == pytest.approx([0.25, 0.0, 0.0])
    assert dinov2["intervals"][0]["low"] == pytest.approx(1 / 12)


def test_the_secondary_analyses_are_reported_but_decide_nothing(tmp_path: Path) -> None:
    directory = write_sweep(tmp_path / "out", three_categories)

    summary = json.loads(statistics.analyse_sweep(directory).summary_path.read_text("utf-8"))

    secondary = summary["secondary"]
    mcnemar = {entry["run_id"]: entry for entry in secondary["mcnemar_formal_vs_native"]}
    assert len(mcnemar) == 9
    segformer = mcnemar[run_id("segformer_b2", 17)]
    assert (segformer["formal_only_misses"], segformer["native_only_misses"]) == (16, 0)
    assert segformer["p_value"] == pytest.approx(2 * 0.5**16)
    assert segformer["holm_adjusted_p_value"] == pytest.approx(9 * 2 * 0.5**16)
    assert mcnemar[run_id("upernet_dinov2_small", 17)]["holm_adjusted_p_value"] == pytest.approx(
        6 * 2 * 0.5**4
    )
    assert mcnemar[run_id("upernet_convnextv2_tiny", 42)]["p_value"] == 1.0

    dose = secondary["dose_response"]["upernet_dinov2_small"]
    assert [step["arm"] for step in dose["steps"]] == ["formal", "s085", "native"]
    assert [step["miss_rate"] for step in dose["steps"]] == pytest.approx([1 / 3, 0.25, 0.25])
    assert dose["monotone_non_increasing"] is True
    assert secondary["dose_response"]["segformer_b2"]["monotone_non_increasing"] is True

    assert secondary["small_rider"]["segformer_b2"]["estimate"] == pytest.approx(0.0)
    assert secondary["small_rider"]["segformer_b2"]["instances"] == 4
    pooled = secondary["small_person_and_rider"]["segformer_b2"]
    assert pooled["estimate"] == pytest.approx(16 / 20)

    motorcycle = secondary["small_counts"]["motorcycle"]
    assert motorcycle["instances"] == 1
    assert motorcycle["critical_misses"]["native"][run_id("segformer_b2", 17)] == 0
    assert secondary["small_counts"]["bicycle"]["instances"] == 0

    bins = secondary["input_area_bins"]["segformer_b2"]["native"]
    assert [(entry["low"], entry["high"]) for entry in bins] == [
        (32, 64),
        (64, 128),
        (128, 256),
        (4096, 8192),
    ]
    assert [entry["instances"] for entry in bins] == [12, 12, 24, 12]
    assert [entry["miss_rate"] for entry in bins] == [0.0, 0.0, 0.0, 0.0]
    formal_bins = secondary["input_area_bins"]["segformer_b2"]["formal"]
    assert formal_bins[0]["low"] == 16
    assert formal_bins[0]["miss_rate"] == 1.0

    residual = secondary["residual_native_miss_rate"]["upernet_convnextv2_tiny"]
    assert residual["seed_mean"] == pytest.approx(0.25)
    assert residual["per_seed"] == pytest.approx([0.25, 0.25, 0.25])

    iou = secondary["iou"]["segformer_b2"]["native"]
    assert iou["seed_mean_miou"] == pytest.approx(1.0)
    assert iou["seed_mean_class_iou"]["person"] == pytest.approx(1.0)
    assert set(iou["per_run"]) == {run_id("segformer_b2", seed) for seed in SEEDS}


def confounded(arm: str, model: str, _run: str, _image: int, entry: dict[str, Any]) -> bool:
    """Every model leaves small persons alone; the DINOv2 native arm loses large persons."""

    if small_person(entry):
        return bool(entry["instance_id"] == 1)
    large_person = entry["class_id"] == PERSON and entry["area_tertile"] == "large"
    return bool(large_person and model == "upernet_dinov2_small" and arm == "native")


def test_scale_mismatch_turns_an_equivalence_into_inconclusive(tmp_path: Path) -> None:
    """A native arm that broke the model cannot be read as the model being the limit."""

    def confusion(arm: str, run: str) -> list[int]:
        broken = arm == "native" and run.startswith("upernet_convnextv2_tiny")
        return diagonal_confusion(errors=5000 if broken else 0)

    directory = write_sweep(tmp_path / "out", confounded, confusion=confusion)

    result = statistics.analyse_sweep(directory)

    assert result.categories == {
        "segformer_b2": "C",
        "upernet_convnextv2_tiny": "D",
        "upernet_dinov2_small": "D",
    }
    assert result.overall == "D"
    summary = json.loads(result.summary_path.read_text("utf-8"))
    convnext = summary["primary"]["upernet_convnextv2_tiny"]
    assert convnext["downgraded_from"] == "C"
    assert convnext["scale_mismatch"]["miou_drop_exceeds_limit"] is True
    assert convnext["scale_mismatch"]["miou_drop_formal_minus_native"] > 0.02
    dinov2 = summary["primary"]["upernet_dinov2_small"]
    assert dinov2["scale_mismatch"]["large_person_native_worse"] is True
    assert dinov2["large_person_formal_minus_native"]["estimate"] == pytest.approx(-1.0)
    assert dinov2["formal_minus_s085"]["estimate"] == pytest.approx(0.0)


def test_without_the_s085_arm_the_supplement_is_reported_as_unavailable(tmp_path: Path) -> None:
    directory = write_sweep(tmp_path / "out", three_categories, arms=("formal", "native"))

    summary = json.loads(statistics.analyse_sweep(directory).summary_path.read_text("utf-8"))

    assert summary["primary"]["segformer_b2"]["formal_minus_s085"] is None
    steps = summary["secondary"]["dose_response"]["segformer_b2"]["steps"]
    assert [step["arm"] for step in steps] == ["formal", "native"]


def test_a_non_monotone_dose_response_is_reported_as_such(tmp_path: Path) -> None:
    def bump(arm: str, _model: str, _run: str, _image: int, entry: dict[str, Any]) -> bool:
        return small_person(entry) and arm == "s085"

    directory = write_sweep(tmp_path / "out", bump)

    summary = json.loads(statistics.analyse_sweep(directory).summary_path.read_text("utf-8"))

    assert summary["secondary"]["dose_response"]["segformer_b2"]["monotone_non_increasing"] is False


def test_a_difference_that_changes_sign_across_images_is_inconclusive(tmp_path: Path) -> None:
    def alternating(arm: str, _model: str, _run: str, image: int, entry: dict[str, Any]) -> bool:
        if not small_person(entry) or entry["instance_id"] != 1:
            return False
        return (arm == "formal") == (image < 2)

    directory = write_sweep(tmp_path / "out", alternating)

    result = statistics.analyse_sweep(directory)

    assert set(result.categories.values()) == {"D"}
    summary = json.loads(result.summary_path.read_text("utf-8"))
    contrast = summary["primary"]["segformer_b2"]["formal_minus_native"]
    assert contrast["estimate"] == pytest.approx(0.0)
    assert contrast["intervals"][0]["low"] < -statistics.EQUIVALENCE_BOUND
    assert math.isclose(contrast["per_seed"][0], 0.0, abs_tol=1e-12)


def nothing_missed(*_: Any) -> bool:
    return False


def test_a_sweep_that_is_not_a_resolution_sweep_is_refused(tmp_path: Path) -> None:
    directory = write_sweep(tmp_path / "out", nothing_missed)
    run = json.loads((directory / resolution.RUN_FILENAME).read_text("utf-8"))
    run["schema_version"] = "something-else/v1"
    (directory / resolution.RUN_FILENAME).write_text(json.dumps(run), "utf-8", newline="\n")

    with pytest.raises(ValueError, match="resolution sweep"):
        statistics.analyse_sweep(directory)


@pytest.mark.parametrize("change", ["drop", "duplicate"])
def test_an_incomplete_sweep_is_not_analysed(tmp_path: Path, change: str) -> None:
    """The pre-registered denominators are the whole cohort, not whatever finished."""

    directory = write_sweep(tmp_path / "out", nothing_missed)
    path = directory / resolution.PER_IMAGE_FILENAME
    lines = path.read_text("utf-8").splitlines()
    lines = lines[:-1] if change == "drop" else [*lines, lines[0]]
    path.write_text("\n".join(lines) + "\n", "utf-8", newline="\n")

    with pytest.raises(ValueError, match="incomplete"):
        statistics.analyse_sweep(directory)


@pytest.mark.parametrize(
    "sessions",
    [[], [session(), session(commit="d" * 40)], [session(), session(lock="0" * 64)]],
)
def test_sessions_from_different_code_are_not_pooled(
    tmp_path: Path, sessions: list[dict[str, Any]]
) -> None:
    directory = write_sweep(tmp_path / "out", nothing_missed, sessions=sessions)

    with pytest.raises(ValueError, match="one commit"):
        statistics.analyse_sweep(directory)


def test_a_cohort_without_small_persons_has_nothing_to_analyse(tmp_path: Path) -> None:
    directory = write_sweep(tmp_path / "out", nothing_missed)
    path = directory / resolution.PER_IMAGE_FILENAME
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    for record in records:
        for entry in record["instances"]:
            entry["area_tertile"] = "medium"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8", newline="\n")

    with pytest.raises(ValueError, match="no small-tertile person"):
        statistics.analyse_sweep(directory)


def test_descriptive_contrasts_without_instances_are_undefined_not_zero(tmp_path: Path) -> None:
    directory = write_sweep(tmp_path / "out", nothing_missed)
    path = directory / resolution.PER_IMAGE_FILENAME
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    for record in records:
        for entry in record["instances"]:
            if entry["class_id"] == RIDER or entry["area_tertile"] == "large":
                entry["class_id"] = CAR
    path.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8", newline="\n")

    result = statistics.analyse_sweep(directory)

    summary = json.loads(result.summary_path.read_text("utf-8"))
    rider = summary["secondary"]["small_rider"]["segformer_b2"]
    assert rider["estimate"] is None
    assert rider["intervals"] == [None, None]
    assert rider["per_seed"] == [None, None, None]
    large = summary["primary"]["segformer_b2"]["scale_mismatch"]
    assert large["large_person_native_worse"] is False
    assert summary["sweep"]["hardware"] == [{"gpu": "A100"}]
