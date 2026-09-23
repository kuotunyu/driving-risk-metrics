"""Contracts for the pre-registered all-seed statistics.

The tiny study's predictions are built so that each outcome the plan names
occurs: UperNet-DINOv2 misses the small person in every seed (majority missed),
SegFormer never does (minority missed), and UperNet-ConvNeXtV2 misses it in seed
42 only, so its seed stage spans one half (undetermined).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from drivemetrics.posthoc import allseed_statistics

PRIMARY = 1 - 0.05 / 3


def analyse(released: tuple[Path, Any], output: Path) -> tuple[Any, dict[str, Any]]:
    evidence, extraction = released
    result = allseed_statistics.analyse_all_seeds(
        extraction.directory, evidence, output, resamples=120
    )
    return result, json.loads(result.summary_path.read_text(encoding="utf-8"))


@pytest.fixture
def analysed(released: tuple[Path, Any], tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    return analyse(released, tmp_path / "analysis")


def test_each_model_is_classified_by_the_pre_registered_rule(analysed: Any) -> None:
    result, summary = analysed

    assert result.categories == {
        "segformer_b2": "minority missed",
        "upernet_convnextv2_tiny": "undetermined",
        "upernet_dinov2_small": "majority missed",
    }
    assert summary["primary"]["small_person_miss_rate"]["upernet_dinov2_small"]["category"] == (
        "majority missed"
    )


def test_the_rate_reports_every_seed_and_how_far_seed_17_is_from_the_mean(analysed: Any) -> None:
    _, summary = analysed
    block = summary["primary"]["small_person_miss_rate"]["upernet_convnextv2_tiny"]

    assert block["per_seed"] == {"17": 0.0, "42": 1.0, "73": 0.0}
    assert block["estimate"] == pytest.approx(1 / 3)
    assert block["seed_17_minus_mean"] == pytest.approx(-1 / 3)
    assert [interval["confidence"] for interval in block["intervals"]] == [
        pytest.approx(PRIMARY),
        0.95,
    ]
    assert summary["counts"] == {
        "images": 3,
        "small_person_instances": 3,
        "images_with_small_person": 3,
    }


def test_a_pair_is_separable_only_when_its_primary_interval_excludes_zero(analysed: Any) -> None:
    result, summary = analysed
    pairs = summary["primary"]["pairs"]

    assert result.separable == {
        "segformer_b2 minus upernet_convnextv2_tiny": False,
        "segformer_b2 minus upernet_dinov2_small": True,
        "upernet_convnextv2_tiny minus upernet_dinov2_small": False,
    }
    assert pairs["segformer_b2 minus upernet_dinov2_small"]["estimate"] == -1.0
    assert pairs["segformer_b2 minus upernet_convnextv2_tiny"]["estimate"] == pytest.approx(-1 / 3)


def test_the_threshold_sweep_compares_integers_and_reports_the_order(analysed: Any) -> None:
    _, summary = analysed
    sweep = {entry["threshold"]: entry for entry in summary["secondary"]["threshold_sweep"]}

    assert sorted(sweep) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # The missed person has one of its four pixels right: a fraction of 0.25.
    assert set(sweep[0.2]["rates"].values()) == {0.0}
    assert sweep[0.3]["rates"] == {
        "segformer_b2": 0.0,
        "upernet_convnextv2_tiny": pytest.approx(1 / 3),
        "upernet_dinov2_small": 1.0,
    }
    assert sweep[0.5]["order"] == [
        "segformer_b2",
        "upernet_convnextv2_tiny",
        "upernet_dinov2_small",
    ]
    assert all(entry["order_equals_threshold_05"] for entry in sweep.values())


def test_the_secondary_intervals_cover_every_pre_registered_metric(
    analysed: Any, released: tuple[Path, Any]
) -> None:
    _, summary = analysed
    evidence, _ = released
    metrics = json.loads((evidence / "metrics.json").read_text(encoding="utf-8"))
    extended = json.loads((evidence / "extended-metrics.json").read_text(encoding="utf-8"))
    intervals = summary["secondary"]["intervals"]

    assert sorted(intervals) == sorted(
        [
            f"{metric} {state}"
            for metric in ("classwise_ece", "toplabel_ece", "brier", "aurc")
            for state in ("uncalibrated", "calibrated")
        ]
        + ["iou person", "cost vru_priority", "cost drivable_boundary"]
    )
    for state in ("uncalibrated", "calibrated"):
        for model in ("segformer_b2", "upernet_convnextv2_tiny", "upernet_dinov2_small"):
            level = intervals[f"classwise_ece {state}"]["levels"][model]
            assert level["estimate"] == pytest.approx(metrics["calibration"][model][state]["ece"])
            assert intervals[f"brier {state}"]["levels"][model]["estimate"] == pytest.approx(
                metrics["calibration"][model][state]["brier"]
            )
            assert intervals[f"aurc {state}"]["levels"][model]["estimate"] == pytest.approx(
                extended["selective_risk"][model][state]["aurc"]
            )
            assert level["low"] <= level["estimate"] <= level["high"]
    assert sorted(intervals["iou person"]["pairs"]) == sorted(summary["primary"]["pairs"])
    assert summary["secondary"]["not_computed"] == {
        "iou rider": "no pixel of this class in the cohort",
        "iou motorcycle": "no pixel of this class in the cohort",
        "iou bicycle": "no pixel of this class in the cohort",
    }


def test_every_run_has_an_instance_block_and_every_model_a_seed_mean(analysed: Any) -> None:
    _, summary = analysed
    blocks = summary["secondary"]["instance_blocks"]
    means = summary["secondary"]["instance_seed_means"]

    assert len(blocks) == 9
    small = means["upernet_convnextv2_tiny"]["by_class"]["person"]["by_tertile"]["small"]
    assert small == {"instance_count": 3, "critical_misses": 1.0, "mean_correct_fraction": 0.75}
    assert means["segformer_b2"]["by_class"]["rider"]["by_tertile"]["small"] == {
        "instance_count": 0,
        "critical_misses": 0.0,
        "mean_correct_fraction": None,
    }


def test_the_summary_names_no_image(analysed: Any) -> None:
    result, summary = analysed
    text = result.summary_path.read_text(encoding="utf-8")

    assert "v0001" not in text
    assert "sample_id" not in text
    assert summary["gates"] == {"integrity": "passed", "reproduction": "passed"}
    assert summary["schema_version"] == allseed_statistics.SUMMARY_SCHEMA_VERSION


def test_a_failed_reproduction_stops_before_any_result(
    released: tuple[Path, Any], tmp_path: Path
) -> None:
    evidence, _ = released
    path = evidence / "metrics.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["calibration"]["segformer_b2"]["calibrated"]["ece"] = 0.5
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="reproduction gate failed"):
        analyse(released, tmp_path / "analysis")
    report = json.loads((tmp_path / "analysis" / "reproduction.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["mismatches"] == ["calibration: segformer_b2 calibrated"]
    assert not (tmp_path / "analysis" / "summary.json").exists()


def test_an_existing_summary_is_never_overwritten(
    released: tuple[Path, Any], tmp_path: Path
) -> None:
    analyse(released, tmp_path / "analysis")

    with pytest.raises(FileExistsError):
        analyse(released, tmp_path / "analysis")


def test_a_one_point_selective_risk_curve_has_no_area_to_bootstrap() -> None:
    """A model with one confidence everywhere has no AURC, and the analysis must say so."""

    import numpy as np

    counts = np.zeros(8, dtype=np.int64)
    counts[5] = 10
    correct = np.zeros(8, dtype=np.int64)
    correct[5] = 7

    assert allseed_statistics.aurc_of(counts, correct) is None
    with pytest.raises(ValueError, match="no area to bootstrap"):
        allseed_statistics._finite_aurc(counts, correct)
