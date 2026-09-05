"""Contracts for the six documents the release cites.

Each published document declared a version string, or nothing, with no schema
behind it. These tests pin the shape of every document to a model that forbids
unknown fields, so a producer that drifts fails here and a document that drifts
fails the verifier. The payloads are minimal on purpose: the real ones are
produced by the analysis fixtures, whose tests exercise validate-on-write.
"""

from __future__ import annotations

import copy
import json
from types import ModuleType
from typing import Any, get_args

import pytest
from pydantic import ValidationError

HASH_A = "a" * 64
HASH_B = "b" * 64


def load_documents() -> ModuleType:
    try:
        from drivemetrics.artifacts import documents
    except ImportError:
        pytest.fail("drivemetrics.artifacts.documents is missing", pytrace=False)
    return documents


def bucket(count: int = 0) -> dict[str, Any]:
    return {"instance_count": count, "critical_misses": 0, "mean_correct_fraction": None}


def tertiles() -> dict[str, Any]:
    return {"small": bucket(), "medium": bucket(), "large": bucket()}


MINIMAL: dict[str, dict[str, Any]] = {
    "driving-risk-metrics-table/v1": {
        "schema_version": "driving-risk-metrics-table/v1",
        "protocol_hash": HASH_A,
        "dataset_manifest_hash": HASH_B,
        "cohort": "locked_validation",
        "sample_count": 1,
        "seed_count": 1,
        "interval_method": "two-stage paired bootstrap",
        "metrics": {"a": {"miou": 0.5, "pixel_accuracy": 0.9, "critical_recall": 0.7}},
        "per_class": {
            "class_names": ["road", "sidewalk"],
            "support_pixels": [10, 0],
            "images_with_class": [1, 0],
            "by_model": {"a": {"iou": [0.5, None], "recall": [0.6, None]}},
        },
        "calibration": {
            "a": {
                "uncalibrated": {
                    "ece": 0.1,
                    "brier": 0.2,
                    "per_seed": {"17": {"ece": 0.1, "brier": 0.2}},
                },
                "calibrated": {
                    "ece": 0.05,
                    "brier": 0.15,
                    "per_seed": {"17": {"ece": 0.05, "brier": 0.15}},
                },
            }
        },
        "risk_profiles": {
            "balanced": {"sensitivity": 1.0, "critical_class_ids": [], "cost_risk": {"a": 0.1}}
        },
    },
    "driving-risk-intervals/v1": {
        "schema_version": "driving-risk-intervals/v1",
        "protocol_hash": HASH_A,
        "dataset_manifest_hash": HASH_B,
        "intervals": {
            "a minus b (miou)": {
                "estimate": 0.1,
                "low": -0.1,
                "high": 0.3,
                "confidence": 0.95,
                "resamples": 5000,
                "seed": 20260831,
                "estimator": "ratio_of_sums",
            }
        },
    },
    "driving-risk-rankings/v1": {
        "schema_version": "driving-risk-rankings/v1",
        "protocol_hash": HASH_A,
        "dataset_manifest_hash": HASH_B,
        "baseline_metric": "miou",
        "comparisons": [
            {
                "metric_name": "critical_recall",
                "baseline_order": ["a", "b"],
                "comparison_order": ["a", "b"],
                "reversal_observed": False,
            }
        ],
        "separability": {
            "miou": [
                {
                    "left": "a",
                    "right": "b",
                    "estimate": 0.1,
                    "low": -0.1,
                    "high": 0.3,
                    "excludes_zero": False,
                }
            ]
        },
    },
    "driving-risk-gallery-manifest/v1": {
        "schema_version": "driving-risk-gallery-manifest/v1",
        "protocol_hash": HASH_A,
        "dataset_manifest_hash": HASH_B,
        "evaluation": "eval_calibrated",
        "rule": {
            "metric": "image_mean_iou",
            "aggregate": "mean_over_seeds",
            "per_model": 8,
            "tie_break": "sample_id",
        },
        "per_model": {
            "a": {
                "worst": [{"sample_id": "v1", "mean_iou_over_seeds": 0.2, "per_seed": {"17": 0.2}}],
                "best": [],
            }
        },
    },
    "driving-risk-extended-metrics/v1": {
        "schema_version": "driving-risk-extended-metrics/v1",
        "protocol_hash": HASH_A,
        "dataset_manifest_hash": HASH_B,
        "evaluation_for_ground_truth_metrics": "eval_calibrated",
        "ground_truth": {
            "cohort_manifest_sha256": HASH_B,
            "split_name": "locked_validation",
            "masks_verified": 998,
            "instance_bitmasks": {"count": 998, "set_sha256": HASH_A},
        },
        "selective_risk": {
            "a": {
                "uncalibrated": {
                    "aurc": 0.01,
                    "coverage_points": 5,
                    "defined_at": "confidence_bin_boundaries",
                },
                "calibrated": {
                    "aurc": None,
                    "coverage_points": 1,
                    "defined_at": "confidence_bin_boundaries",
                },
            }
        },
        "normalized_image_bands": {
            "definition": "normalized image rows only; not physical depth or metric distance",
            "by_model": {
                "a": {
                    "top": {"pixels": 10, "pixel_accuracy": 0.9},
                    "middle": {"pixels": 10, "pixel_accuracy": 0.8},
                    "bottom": {"pixels": 0, "pixel_accuracy": None},
                }
            },
        },
        "instances": {
            "a": {
                "tertile_edges_sha256": HASH_A,
                "instance_count": 1,
                "excluded_without_semantic_pixels": 0,
                "mean_corroborated_fraction": 0.95,
                "by_tertile": {**tertiles(), "small": bucket(1)},
                "by_class": {
                    "person": {**bucket(1), "by_tertile": {**tertiles(), "small": bucket(1)}}
                },
            }
        },
    },
}

TERTILES_DOCUMENT: dict[str, Any] = {
    "eligible_images": 6296,
    "instances_per_category": {"1": 8946, "2": 419},
    "learned_from": "train",
    "tertile_edges": {"1": [349, 987], "2": [409, 1536]},
    "total_instances": 80249,
}


def test_every_registered_model_declares_the_version_it_is_registered_under() -> None:
    """A registry that mapped a version to a model expecting another would validate nothing."""

    documents = load_documents()

    assert set(documents.DOCUMENT_MODELS) == set(MINIMAL)
    for version, model in documents.DOCUMENT_MODELS.items():
        assert get_args(model.model_fields["schema_version"].annotation) == (version,)


@pytest.mark.parametrize("version", sorted(MINIMAL))
def test_a_well_formed_document_is_accepted(version: str) -> None:
    documents = load_documents()

    documents.DOCUMENT_MODELS[version].model_validate(MINIMAL[version])


@pytest.mark.parametrize("version", sorted(MINIMAL))
def test_an_unknown_field_is_refused(version: str) -> None:
    """A field the contract does not know is a shape change nobody wrote down."""

    documents = load_documents()
    payload = {**copy.deepcopy(MINIMAL[version]), "unexpected": 1}

    with pytest.raises(ValidationError):
        documents.DOCUMENT_MODELS[version].model_validate(payload)


@pytest.mark.parametrize("version", sorted(MINIMAL))
def test_the_wrong_version_is_refused(version: str) -> None:
    documents = load_documents()
    payload = {**copy.deepcopy(MINIMAL[version]), "schema_version": "driving-risk-other/v1"}

    with pytest.raises(ValidationError):
        documents.DOCUMENT_MODELS[version].model_validate(payload)


@pytest.mark.parametrize("version", sorted(MINIMAL))
def test_a_missing_hash_is_refused(version: str) -> None:
    documents = load_documents()
    payload = copy.deepcopy(MINIMAL[version])
    del payload["protocol_hash"]

    with pytest.raises(ValidationError):
        documents.DOCUMENT_MODELS[version].model_validate(payload)


def test_blocks_that_may_be_absent_accept_the_not_computed_record() -> None:
    """An absent block is a record naming what was missing, never a missing key."""

    documents = load_documents()
    payload = copy.deepcopy(MINIMAL["driving-risk-extended-metrics/v1"])
    reason = {"not_computed": "ground truth was not available to this analysis run"}
    payload["ground_truth"] = reason
    payload["normalized_image_bands"] = reason
    payload["instances"] = reason

    documents.ExtendedMetricsV1.model_validate(payload)


def test_the_frozen_tertiles_have_a_contract_matched_by_filename() -> None:
    """The one document frozen before versioning is not touched; the contract is written around it."""

    documents = load_documents()

    assert {"area_tertiles.json": documents.AreaTertilesV1} == documents.UNVERSIONED_DOCUMENT_MODELS
    documents.AreaTertilesV1.model_validate(TERTILES_DOCUMENT)
    with pytest.raises(ValidationError):
        documents.AreaTertilesV1.model_validate(
            {**TERTILES_DOCUMENT, "tertile_edges": {"1": [349]}}
        )


def test_validated_document_returns_plain_json_and_round_trips() -> None:
    """Producers write what this returns, so it must be JSON, not model objects or tuples."""

    documents = load_documents()
    payload = MINIMAL["driving-risk-rankings/v1"]

    out = documents.validated_document(payload)

    assert out == json.loads(json.dumps(out))
    assert isinstance(out["comparisons"], list)
    assert out == documents.validated_document(out)


def test_a_document_without_a_registered_version_cannot_be_validated() -> None:
    """Validating against a guessed contract could pass; refusing cannot."""

    documents = load_documents()

    with pytest.raises(ValueError, match=r"^no contract is registered for schema_version 'nope'$"):
        documents.validated_document({"schema_version": "nope"})
    with pytest.raises(ValueError, match=r"^no contract is registered for schema_version None$"):
        documents.validated_document({"protocol_hash": HASH_A})


def test_integer_fields_refuse_a_whole_float() -> None:
    """3.0 seeds is not a count; a producer that divided instead of floor-dividing must fail."""

    documents = load_documents()
    payload = copy.deepcopy(MINIMAL["driving-risk-metrics-table/v1"])
    payload["seed_count"] = 3.0

    with pytest.raises(ValidationError):
        documents.MetricsTableV1.model_validate(payload)
