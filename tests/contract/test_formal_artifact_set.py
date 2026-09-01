"""Contract for the immutable index of the nine formal runs.

This gate exists to be run before any paid GPU job and again before analysis.
Every failure it names is one that produces a complete-looking result set: a
missing seed silently narrows the interval, a duplicate sample double-counts,
a drifted cohort hash compares two different studies, a mid-training checkpoint
is selection on the evaluation cohort, and a missing calibrated artifact turns
a calibration comparison into a comparison against nothing.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

MODELS = ("upernet_convnextv2_tiny", "upernet_dinov2_small", "segformer_b2")
SEEDS = (17, 42, 73)
PROTOCOL = "a" * 64
MANIFEST = "b" * 64


def load_formal_set() -> ModuleType:
    try:
        from drivemetrics.artifacts import formal_set
    except ImportError:
        pytest.fail("drivemetrics.artifacts.formal_set is missing", pytrace=False)
    return formal_set


def run_entry(model: str, seed: int, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "run_id": f"{model}-seed-{seed}",
        "protocol_sha256": PROTOCOL,
        "dataset_manifest_sha256": MANIFEST,
        "checkpoint_sha256": f"{abs(hash((model, seed))):064x}"[:64],
        "final_step": 30000,
        "status": "succeeded",
        "temperature": 1.25,
        "uncalibrated_sample_ids": ["v0001", "v0002"],
        "calibrated_sample_ids": ["v0001", "v0002"],
    }
    entry.update(overrides)
    return entry


def index(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "drivemetrics-formal-set/v1",
        "protocol_sha256": PROTOCOL,
        "dataset_manifest_sha256": MANIFEST,
        "expected_steps": 30000,
        "runs": [run_entry(model, seed) for model in MODELS for seed in SEEDS],
    }
    document.update(overrides)
    return document


def violations(document: dict[str, Any]) -> tuple[str, ...]:
    return load_formal_set().validate_formal_run_index(document)


def test_a_complete_matrix_is_accepted() -> None:
    """A validator that rejects the intended result set would block every release."""

    assert violations(index()) == ()


def test_a_missing_model_and_seed_pair_is_rejected() -> None:
    """Eight runs reported as nine silently narrows every interval."""

    incomplete = index()
    incomplete["runs"] = [
        entry
        for entry in incomplete["runs"]
        if not (entry["model"] == "segformer_b2" and entry["seed"] == 73)
    ]

    found = violations(incomplete)

    assert any("segformer_b2" in message and "73" in message for message in found)


def test_an_unapproved_seed_is_rejected() -> None:
    """A seed outside 17, 42 and 73 was never part of the declared matrix."""

    broken = index()
    broken["runs"][0]["seed"] = 5

    assert any("seed" in message for message in violations(broken))


def test_a_duplicated_run_is_rejected() -> None:
    """The same run counted twice weights one seed above the others."""

    duplicated = index()
    duplicated["runs"].append(run_entry("upernet_convnextv2_tiny", 17))

    assert any("duplicate" in message.lower() for message in violations(duplicated))


def test_a_duplicate_sample_id_within_a_run_is_rejected() -> None:
    """One image counted twice changes the cohort the numbers describe."""

    broken = index()
    broken["runs"][0]["uncalibrated_sample_ids"] = ["v0001", "v0001"]

    assert any("duplicate" in message.lower() for message in violations(broken))


def test_a_cohort_hash_mismatch_is_rejected() -> None:
    """Two runs on different cohorts are not comparable at all."""

    drifted = index()
    drifted["runs"][2]["dataset_manifest_sha256"] = "c" * 64

    assert any("manifest" in message for message in violations(drifted))


def test_a_protocol_hash_mismatch_is_rejected() -> None:
    """A run scored under another protocol belongs to a different study."""

    drifted = index()
    drifted["runs"][1]["protocol_sha256"] = "d" * 64

    assert any("protocol" in message for message in violations(drifted))


def test_a_non_final_checkpoint_is_rejected() -> None:
    """Selecting a mid-training checkpoint is selection on the evaluation cohort."""

    broken = index()
    broken["runs"][4]["final_step"] = 15000

    assert any("final" in message.lower() for message in violations(broken))


def test_a_missing_calibrated_artifact_is_rejected() -> None:
    """Comparing calibrated against uncalibrated needs both to exist."""

    broken = index()
    broken["runs"][3]["calibrated_sample_ids"] = []

    assert any("calibrated" in message for message in violations(broken))


def test_a_calibrated_cohort_that_differs_from_the_uncalibrated_one_is_rejected() -> None:
    """The two artifact sets must describe the same images to be comparable."""

    broken = index()
    broken["runs"][5]["calibrated_sample_ids"] = ["v0001", "v9999"]

    assert any(
        "same" in message.lower() or "differ" in message.lower() for message in violations(broken)
    )


def test_a_missing_temperature_is_rejected() -> None:
    """A calibrated artifact with no recorded temperature cannot be reproduced."""

    broken = index()
    broken["runs"][6]["temperature"] = None

    assert any("temperature" in message for message in violations(broken))


def test_a_run_that_did_not_succeed_is_rejected() -> None:
    """An aborted run must never be pooled with complete ones."""

    broken = index()
    broken["runs"][7]["status"] = "aborted"

    assert any("status" in message or "succeeded" in message for message in violations(broken))


def test_runs_must_share_one_cohort_across_the_whole_matrix() -> None:
    """The paired bootstrap requires every run to sit on the identical image axis."""

    broken = index()
    for entry in broken["runs"]:
        entry["uncalibrated_sample_ids"] = ["v0001", "v0002"]
        entry["calibrated_sample_ids"] = ["v0001", "v0002"]
    broken["runs"][0]["uncalibrated_sample_ids"] = ["v0001", "v0003"]
    broken["runs"][0]["calibrated_sample_ids"] = ["v0001", "v0003"]

    assert any("cohort" in message.lower() for message in violations(broken))


def test_every_violation_is_reported_not_only_the_first() -> None:
    """Fixing one defect at a time across nine paid runs is not affordable."""

    broken = index()
    broken["runs"][0]["final_step"] = 15000
    broken["runs"][1]["seed"] = 5

    assert len(violations(broken)) >= 2


def test_a_wrong_schema_version_fails_closed() -> None:
    """A document from another tool must never be validated as if it were ours."""

    broken = index(schema_version="something-else/v9")

    assert any("schema" in message for message in violations(broken))


def test_an_unapproved_architecture_is_rejected() -> None:
    """Only the three declared architectures were ever part of this study."""

    broken = index()
    broken["runs"][0]["model"] = "unet_resnet18"

    assert any("model" in message for message in violations(broken))


def test_a_non_positive_temperature_is_rejected() -> None:
    """Dividing logits by zero or a negative value is not a calibration."""

    broken = index()
    broken["runs"][2]["temperature"] = 0.0

    assert any("positive" in message for message in violations(broken))


def test_an_empty_run_list_fails_closed() -> None:
    """Zero runs reported as a complete matrix is the worst possible pass."""

    assert any("non-empty" in message for message in violations(index(runs=[])))


def test_a_missing_expected_step_count_fails_closed() -> None:
    """Without a declared final step there is nothing to check checkpoints against."""

    broken = index()
    del broken["expected_steps"]

    assert any("expected_steps" in message for message in violations(broken))


def test_a_missing_uncalibrated_artifact_is_rejected() -> None:
    """The uncalibrated set is the baseline the calibrated one is compared against."""

    broken = index()
    broken["runs"][8]["uncalibrated_sample_ids"] = []

    assert any("no uncalibrated" in message for message in violations(broken))
