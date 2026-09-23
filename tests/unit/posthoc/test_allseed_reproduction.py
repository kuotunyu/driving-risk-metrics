"""Contracts for the reproduction gate of the all-seed analysis.

Before any new number is read, the new code must reproduce every released number
it overlaps with, exactly: the nine paired intervals, seed 17's instance block,
the seed-mean AURC and the seed-mean ECE and Brier. The released evidence for the
tiny study is produced here by the released code itself, so the gate is tested
against the functions that wrote the real evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from drivemetrics.analysis.aggregate import aggregate_runs
from drivemetrics.analysis.extended import extended_metrics
from drivemetrics.posthoc import allseed, allseed_statistics


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
    allseed.extract_all_seeds(
        study.index_path,
        study.manifest_path,
        study.labels_root,
        study.instance_root,
        study.tertiles_path,
        tmp_path / "extract",
    )
    return evidence, allseed_statistics.load_extraction(tmp_path / "extract")


def edit(path: Path, change: Any) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_every_released_number_is_reproduced_exactly(released: tuple[Path, Any]) -> None:
    evidence, extraction = released

    result = allseed_statistics.reproduce_released(extraction, evidence)

    assert result.mismatches == ()
    assert result.passed
    assert len([name for name in result.checked if name.startswith("interval:")]) == 9
    assert len([name for name in result.checked if name.startswith("instances:")]) == 3
    assert len([name for name in result.checked if name.startswith("aurc:")]) == 6
    assert len([name for name in result.checked if name.startswith("calibration:")]) == 6


def test_a_released_interval_bound_that_differs_is_named(released: tuple[Path, Any]) -> None:
    evidence, extraction = released
    key = "segformer_b2 minus upernet_dinov2_small (critical_recall)"
    edit(
        evidence / "intervals.json",
        lambda document: document["intervals"][key].__setitem__(
            "low", document["intervals"][key]["low"] + 1e-15
        ),
    )

    result = allseed_statistics.reproduce_released(extraction, evidence)

    assert not result.passed
    assert result.mismatches == (f"interval: {key}",)


def test_a_released_instance_count_that_differs_is_named(released: tuple[Path, Any]) -> None:
    evidence, extraction = released
    edit(
        evidence / "extended-metrics.json",
        lambda document: document["instances"]["upernet_dinov2_small"]["by_class"]["person"][
            "by_tertile"
        ]["small"].__setitem__("critical_misses", 0),
    )

    result = allseed_statistics.reproduce_released(extraction, evidence)

    assert result.mismatches == ("instances: upernet_dinov2_small",)


def test_released_aurc_and_calibration_that_differ_are_named(released: tuple[Path, Any]) -> None:
    evidence, extraction = released
    edit(
        evidence / "extended-metrics.json",
        lambda document: document["selective_risk"]["segformer_b2"]["calibrated"].__setitem__(
            "aurc", 0.5
        ),
    )
    edit(
        evidence / "metrics.json",
        lambda document: document["calibration"]["upernet_convnextv2_tiny"][
            "uncalibrated"
        ].__setitem__("brier", 0.5),
    )

    result = allseed_statistics.reproduce_released(extraction, evidence)

    assert result.mismatches == (
        "aurc: segformer_b2 calibrated",
        "calibration: upernet_convnextv2_tiny uncalibrated",
    )


def test_evidence_from_another_study_is_refused(released: tuple[Path, Any]) -> None:
    evidence, extraction = released
    edit(
        evidence / "intervals.json",
        lambda document: document.__setitem__("protocol_hash", "e" * 64),
    )

    with pytest.raises(ValueError, match="another study"):
        allseed_statistics.reproduce_released(extraction, evidence)


def test_an_unfinished_extraction_cannot_be_loaded(tmp_path: Path) -> None:
    (tmp_path / "extract").mkdir()

    with pytest.raises(FileNotFoundError):
        allseed_statistics.load_extraction(tmp_path / "extract")


def test_an_extraction_whose_gate_did_not_pass_cannot_be_loaded(
    released: tuple[Path, Any],
) -> None:
    _, extraction = released
    edit(extraction.directory / "extract.json", lambda document: document.__setitem__("gates", {}))

    with pytest.raises(ValueError, match="integrity"):
        allseed_statistics.load_extraction(extraction.directory)


def test_a_directory_that_is_not_an_all_seed_extraction_is_refused(
    released: tuple[Path, Any],
) -> None:
    _, extraction = released
    edit(
        extraction.directory / "extract.json",
        lambda document: document.__setitem__("schema_version", "something-else/v1"),
    )

    with pytest.raises(ValueError, match="does not hold an all-seed extraction"):
        allseed_statistics.load_extraction(extraction.directory)
