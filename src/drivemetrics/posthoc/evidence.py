"""Claim-auditable evidence derived from a finished resolution summary.

The claims audit reads an artifact's top-level ``protocol_hash`` and
``dataset_manifest_hash`` and the numbers under one JSON pointer. The
pre-registered ``summary.json`` keeps the hashes under ``sweep`` and its
intervals in lists, so the published report cites this flat file instead.
Nothing is recomputed here: every number is copied from the summary, and the
summary's own SHA-256 is recorded so the evidence can be traced back to it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drivemetrics.posthoc.statistics import SUMMARY_SCHEMA_VERSION

EVIDENCE_SCHEMA_VERSION = "driving-risk-resolution-evidence/v1"


@dataclass(frozen=True)
class EvidenceResult:
    """Where the evidence was written and the overall pre-registered conclusion."""

    evidence_path: Path
    overall: str


def _interval(contrast: Mapping[str, Any], confidence: float) -> tuple[float, float]:
    for interval in contrast["intervals"]:
        if math.isclose(interval["confidence"], confidence):
            return interval["low"], interval["high"]
    raise ValueError(f"the summary has no interval at confidence {confidence}")


def _model_block(summary: Mapping[str, Any], model: str, arms: Sequence[str]) -> dict[str, Any]:
    primary = summary["primary"][model]
    settings = summary["settings"]
    main = primary["formal_minus_native"]
    large = primary["large_person_formal_minus_native"]
    rates = {
        step["arm"]: step["miss_rate"]
        for step in summary["secondary"]["dose_response"][model]["steps"]
    }
    iou = summary["secondary"]["iou"][model]
    block: dict[str, Any] = {"category": primary["category"]}
    for arm in arms:
        block[f"{arm}_miss_rate"] = rates[arm]
        block[f"{arm}_miou"] = iou[arm]["seed_mean_miou"]
    block["delta"] = main["estimate"]
    block["delta_low"], block["delta_high"] = _interval(main, settings["primary_confidence"])
    block["delta_low_95"], block["delta_high_95"] = _interval(
        main, settings["descriptive_confidence"]
    )
    # The plan reports each seed's difference beside the seed mean; the summary
    # lists them in the order of the model's runs.
    for run_id, value in zip(primary["runs"], main["per_seed"], strict=True):
        block[f"delta_seed_{run_id.rsplit('-seed-', 1)[1]}"] = value
    if "formal_minus_s085" in primary:
        supplement = primary["formal_minus_s085"]
        block["formal_minus_s085"] = supplement["estimate"]
        block["formal_minus_s085_low"], block["formal_minus_s085_high"] = _interval(
            supplement, settings["primary_confidence"]
        )
    block["large_person_delta"] = large["estimate"]
    block["large_person_delta_low_95"], block["large_person_delta_high_95"] = _interval(
        large, settings["descriptive_confidence"]
    )
    block["miou_drop"] = primary["scale_mismatch"]["miou_drop_formal_minus_native"]
    block["confounded"] = primary["scale_mismatch"]["confounded"]
    return block


def resolution_evidence(summary: Mapping[str, Any], summary_sha256: str) -> dict[str, Any]:
    """Flatten a pre-registered resolution summary into the evidence the report cites."""

    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("the document is not a resolution summary")
    sweep = summary["sweep"]
    arms = list(sweep["arms"])
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": summary["status"],
        "analysis_plan": summary["analysis_plan"],
        "source_summary_sha256": summary_sha256,
        "protocol_hash": sweep["protocol_sha256"],
        "dataset_manifest_hash": sweep["dataset_manifest_sha256"],
        "tertiles_sha256": sweep["tertiles_sha256"],
        "commit": sweep["commit"],
        "gpus": [entry["gpu"] for entry in sweep["hardware"]],
        "parity_criteria": list(sweep["parity_criteria"]),
        "arms": arms,
        "counts": dict(summary["counts"]),
        "overall": summary["overall"],
        "models": {model: _model_block(summary, model, arms) for model in summary["primary"]},
    }


def write_resolution_evidence(summary_path: Path, output_path: Path) -> EvidenceResult:
    """Derive the evidence from the exact bytes of ``summary_path`` and write it."""

    raw = summary_path.read_bytes()
    document = resolution_evidence(json.loads(raw), hashlib.sha256(raw).hexdigest())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return EvidenceResult(evidence_path=output_path, overall=document["overall"])
