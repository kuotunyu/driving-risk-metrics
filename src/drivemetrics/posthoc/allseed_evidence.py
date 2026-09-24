"""Claim-auditable evidence derived from a finished all-seed summary.

A rounded claim marker names scalar fields under one JSON pointer. The
pre-registered ``summary.json`` keeps each primary interval in a list and each
seed under its number, so the published report cites this flat file instead.
Nothing is recomputed here: every number is copied from the summary, and the
summary's own SHA-256 is recorded so the evidence can be traced back to it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drivemetrics.posthoc.allseed_statistics import SUMMARY_SCHEMA_VERSION
from drivemetrics.posthoc.evidence import _interval

EVIDENCE_SCHEMA_VERSION = "driving-risk-allseed-evidence/v1"
#: The vulnerable-road-user classes whose small-tertile counts the report cites.
SMALL_INSTANCE_CLASSES: tuple[str, ...] = ("person", "rider", "motorcycle", "bicycle")


@dataclass(frozen=True)
class EvidenceResult:
    """Where the evidence was written."""

    evidence_path: Path


def _bounds(block: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, float]:
    low, high = _interval(block, settings["primary_confidence"])
    low_95, high_95 = _interval(block, settings["descriptive_confidence"])
    return {
        "estimate": block["estimate"],
        "low": low,
        "high": high,
        "low_95": low_95,
        "high_95": high_95,
    }


def allseed_evidence(summary: Mapping[str, Any], summary_sha256: str) -> dict[str, Any]:
    """Flatten a pre-registered all-seed summary into the evidence the report cites."""

    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("the document is not an all-seed summary")
    settings = summary["settings"]
    primary = {
        model: {
            **_bounds(block, settings),
            **{f"seed_{seed}": value for seed, value in block["per_seed"].items()},
            "seed_17_minus_mean": block["seed_17_minus_mean"],
            "category": block["category"],
        }
        for model, block in summary["primary"]["small_person_miss_rate"].items()
    }
    pairs = {
        key: {**_bounds(block, settings), "separable": block["separable"]}
        for key, block in summary["primary"]["pairs"].items()
    }
    secondary = summary["secondary"]
    sweep = {
        str(entry["threshold"]): {
            **entry["rates"],
            "order_holds": entry["order_equals_threshold_05"],
        }
        for entry in secondary["threshold_sweep"]
    }
    small_instances = {
        model: {
            field: block["by_class"][name]["by_tertile"]["small"][key]
            for name in SMALL_INSTANCE_CLASSES
            for field, key in (
                (f"{name}_count", "instance_count"),
                (f"{name}_misses", "critical_misses"),
            )
        }
        for model, block in secondary["instance_seed_means"].items()
    }
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": summary["status"],
        "analysis_plan": summary["analysis_plan"],
        "source_summary_sha256": summary_sha256,
        "protocol_hash": summary["protocol_hash"],
        "dataset_manifest_hash": summary["dataset_manifest_hash"],
        "gates": dict(summary["gates"]),
        "counts": dict(summary["counts"]),
        "primary": primary,
        "pairs": pairs,
        "intervals": secondary["intervals"],
        "threshold_sweep": sweep,
        "small_instances": small_instances,
    }


def write_allseed_evidence(summary_path: Path, output_path: Path) -> EvidenceResult:
    """Derive the evidence from the exact bytes of ``summary_path`` and write it."""

    raw = summary_path.read_bytes()
    document = allseed_evidence(json.loads(raw), hashlib.sha256(raw).hexdigest())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return EvidenceResult(evidence_path=output_path)
