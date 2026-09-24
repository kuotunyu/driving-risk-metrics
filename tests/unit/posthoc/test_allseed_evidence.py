"""Contracts for the claim-auditable evidence derived from an all-seed summary.

The claims audit reads numbers under one JSON pointer, and a rounded claim marker
names scalar fields under that pointer. The summary keeps each primary interval in
a list and each seed under its number, so the published report cites a flat
evidence file derived from it. Nothing is recomputed: every number is copied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from drivemetrics.posthoc import allseed_evidence, allseed_statistics

PRIMARY = allseed_statistics.PRIMARY_CONFIDENCE


def rate(estimate: float, primary: tuple[float, float], descriptive: tuple[float, float]):
    return {
        "estimate": estimate,
        "per_seed": {"17": estimate + 0.01, "42": estimate - 0.02, "73": estimate + 0.01},
        "seed_17_minus_mean": 0.01,
        "intervals": [
            {"confidence": PRIMARY, "low": primary[0], "high": primary[1]},
            {"confidence": 0.95, "low": descriptive[0], "high": descriptive[1]},
        ],
        "category": "majority missed",
    }


def bucket(count: int, misses: float) -> dict[str, Any]:
    return {"instance_count": count, "critical_misses": misses, "mean_correct_fraction": 0.3}


def summary_document() -> dict[str, Any]:
    level = {"estimate": 0.01, "low": 0.009, "high": 0.012, "confidence": 0.95}
    pair = {**level, "excludes_zero": True}
    small = {
        name: {"by_tertile": {"small": bucket(count, misses)}}
        for name, count, misses in (
            ("person", 462, 305.5),
            ("rider", 17, 16.5),
            ("motorcycle", 14, 13.0),
            ("bicycle", 32, 24.5),
        )
    }
    return {
        "schema_version": allseed_statistics.SUMMARY_SCHEMA_VERSION,
        "analysis_plan": "docs/posthoc/allseed-v1/analysis-plan.md",
        "status": allseed_statistics.STATUS,
        "protocol_hash": "a" * 64,
        "dataset_manifest_hash": "b" * 64,
        "settings": {"primary_confidence": PRIMARY, "descriptive_confidence": 0.95},
        "gates": {"integrity": "passed", "reproduction": "passed"},
        "counts": {"images": 998, "small_person_instances": 462, "images_with_small_person": 214},
        "primary": {
            "small_person_miss_rate": {
                "segformer_b2": rate(0.75, (0.68, 0.82), (0.70, 0.81)),
                "upernet_convnextv2_tiny": rate(0.66, (0.59, 0.73), (0.60, 0.72)),
            },
            "pairs": {
                "segformer_b2 minus upernet_convnextv2_tiny": {
                    "estimate": 0.09,
                    "intervals": [
                        {"confidence": PRIMARY, "low": 0.03, "high": 0.15},
                        {"confidence": 0.95, "low": 0.04, "high": 0.14},
                    ],
                    "separable": True,
                }
            },
        },
        "secondary": {
            "intervals": {
                "aurc uncalibrated": {"levels": {"segformer_b2": level}, "pairs": {"x": pair}}
            },
            "not_computed": {},
            "instance_blocks": {"segformer_b2-seed-17": {"instance_count": 1}},
            "instance_seed_means": {"segformer_b2": {"by_class": small}},
            "threshold_sweep": [
                {
                    "threshold": 0.1,
                    "rates": {"segformer_b2": 0.6, "upernet_convnextv2_tiny": 0.5},
                    "order": ["upernet_convnextv2_tiny", "segformer_b2"],
                    "order_equals_threshold_05": True,
                },
                {
                    "threshold": 0.5,
                    "rates": {"segformer_b2": 0.75, "upernet_convnextv2_tiny": 0.66},
                    "order": ["upernet_convnextv2_tiny", "segformer_b2"],
                    "order_equals_threshold_05": True,
                },
            ],
        },
    }


def test_the_evidence_carries_the_hashes_and_gates_the_audit_reads() -> None:
    document = allseed_evidence.allseed_evidence(summary_document(), "e" * 64)

    assert document["schema_version"] == allseed_evidence.EVIDENCE_SCHEMA_VERSION
    assert document["protocol_hash"] == "a" * 64
    assert document["dataset_manifest_hash"] == "b" * 64
    assert document["source_summary_sha256"] == "e" * 64
    assert document["gates"] == {"integrity": "passed", "reproduction": "passed"}
    assert document["counts"]["small_person_instances"] == 462


def test_each_primary_rate_is_flattened_to_named_scalars() -> None:
    document = allseed_evidence.allseed_evidence(summary_document(), "e" * 64)

    assert document["primary"]["segformer_b2"] == {
        "estimate": 0.75,
        "low": 0.68,
        "high": 0.82,
        "low_95": 0.70,
        "high_95": 0.81,
        "seed_17": 0.76,
        "seed_42": 0.73,
        "seed_73": 0.76,
        "seed_17_minus_mean": 0.01,
        "category": "majority missed",
    }
    assert document["pairs"]["segformer_b2 minus upernet_convnextv2_tiny"] == {
        "estimate": 0.09,
        "low": 0.03,
        "high": 0.15,
        "low_95": 0.04,
        "high_95": 0.14,
        "separable": True,
    }


def test_the_secondary_results_are_copied_under_names_a_claim_can_point_at() -> None:
    summary = summary_document()
    document = allseed_evidence.allseed_evidence(summary, "e" * 64)

    assert document["intervals"] == summary["secondary"]["intervals"]
    assert document["threshold_sweep"] == {
        "0.1": {"segformer_b2": 0.6, "upernet_convnextv2_tiny": 0.5, "order_holds": True},
        "0.5": {"segformer_b2": 0.75, "upernet_convnextv2_tiny": 0.66, "order_holds": True},
    }
    assert document["small_instances"]["segformer_b2"] == {
        "person_count": 462,
        "person_misses": 305.5,
        "rider_count": 17,
        "rider_misses": 16.5,
        "motorcycle_count": 14,
        "motorcycle_misses": 13.0,
        "bicycle_count": 32,
        "bicycle_misses": 24.5,
    }


def test_a_document_that_is_not_an_all_seed_summary_is_refused() -> None:
    document = summary_document()
    document["schema_version"] = "driving-risk-resolution-summary/v1"

    with pytest.raises(ValueError, match="not an all-seed summary"):
        allseed_evidence.allseed_evidence(document, "e" * 64)


def test_the_evidence_file_is_written_from_the_summary_bytes(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    raw = (json.dumps(summary_document(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    summary_path.write_bytes(raw)
    output = tmp_path / "evidence" / "allseed-evidence.json"

    result = allseed_evidence.write_allseed_evidence(summary_path, output)

    assert result.evidence_path == output
    written = output.read_bytes()
    assert written.endswith(b"}\n")
    assert b"\r\n" not in written
    assert json.loads(written)["source_summary_sha256"] == hashlib.sha256(raw).hexdigest()
    allseed_evidence.write_allseed_evidence(summary_path, output)
    assert output.read_bytes() == written
