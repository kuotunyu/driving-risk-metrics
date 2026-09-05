"""Select the failure gallery deterministically from the frozen per-image evidence.

A gallery is the one place a release invites a reader to look at individual
images, and that is exactly why its rule is fixed before any number is seen. A
selection made after the metrics are known is a selection made to support a
point; a selection made by a rule written down first is evidence. The rule
travels with the manifest as data rather than as prose, so a later reader can
reproduce the choice without reading this file, and so changing the rule changes
every published manifest visibly.

The counts are descriptive. If a cohort holds fewer images than the gallery asks
for, the gallery is smaller — the release may truthfully show less and must never
manufacture an example to fill a quota.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drivemetrics.artifacts.documents import validated_document
from drivemetrics.artifacts.formal_set import (
    APPROVED_MODELS,
    APPROVED_SEEDS,
    validate_formal_run_index,
)
from drivemetrics.artifacts.predictions import read_prediction_artifact
from drivemetrics.metrics.confusion import summarize_confusion

GALLERY_SCHEMA_VERSION = "driving-risk-gallery-manifest/v1"
#: The calibrated evaluation is the condition the release publishes. Temperature
#: scaling is monotonic, so argmax and therefore every confusion is identical in
#: the two evaluations and the gallery would be the same either way; naming the
#: one that was read is still part of saying what the images are.
GALLERY_EVALUATION = "eval_calibrated"
DEFAULT_PER_MODEL = 8


@dataclass(frozen=True)
class GalleryResult:
    """Where the manifest was written and how much of it each model received."""

    manifest_path: Path
    per_model: int
    models: tuple[str, ...]


def rank_samples(scores: Mapping[str, float], *, count: int, worst: bool) -> list[str]:
    """Return the `count` lowest or highest sample IDs, ties broken by sample ID.

    The tie-break is declared rather than incidental. Runs that agree exactly
    produce ties, and without a rule the published order would come from whatever
    order the artifacts happened to be read in — the same class of defect as
    taking a cohort's order from whichever run an index listed first.
    """

    ordered = sorted(scores.items(), key=lambda item: (item[1] if worst else -item[1], item[0]))
    return [sample_id for sample_id, _ in ordered[:count]]


def _image_mean_iou(directory: Path, sample_ids: Sequence[str]) -> dict[str, float]:
    """One mIoU per image, from that image's own confusion matrix."""

    scores: dict[str, float] = {}
    for sample_id in sample_ids:
        _, record, _ = read_prediction_artifact(directory / f"{sample_id}.json")
        scores[sample_id] = summarize_confusion(record.confusion).mean_iou
    return scores


def select_gallery(
    index_path: Path,
    output_path: Path,
    *,
    per_model: int = DEFAULT_PER_MODEL,
) -> GalleryResult:
    """Choose each model's hardest and easiest images and write the manifest.

    The score is the mean over a model's seeds of that image's own mIoU, so a
    single unlucky seed cannot put an image in the gallery on its own — and the
    per-seed values travel with every entry, because one bad seed and three
    consistent failures are indistinguishable in a mean.
    """

    if not isinstance(per_model, int) or isinstance(per_model, bool) or per_model <= 0:
        raise ValueError(f"per_model must be a positive integer, got {per_model!r}")
    if output_path.exists():
        raise FileExistsError(f"a gallery manifest already exists: {output_path}")

    document = json.loads(index_path.read_text(encoding="utf-8"))
    violations = validate_formal_run_index(document)
    if violations:
        raise ValueError("formal run index is not valid: " + "; ".join(violations))

    index_dir = index_path.parent
    runs = sorted(
        document["runs"],
        key=lambda entry: (
            APPROVED_MODELS.index(str(entry["model"])),
            APPROVED_SEEDS.index(int(entry["seed"])),
        ),
    )
    sample_ids = tuple(sorted(runs[0]["uncalibrated_sample_ids"]))
    models = tuple(dict.fromkeys(str(entry["model"]) for entry in runs))

    per_model_block: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model in models:
        entries = [entry for entry in runs if str(entry["model"]) == model]
        by_seed = {
            int(entry["seed"]): _image_mean_iou(
                index_dir / str(entry["calibrated_artifacts_dir"]), sample_ids
            )
            for entry in entries
        }
        means = {
            sample_id: sum(scores[sample_id] for scores in by_seed.values()) / len(by_seed)
            for sample_id in sample_ids
        }
        count = min(per_model, len(sample_ids))
        per_model_block[model] = {
            kind: [
                {
                    "sample_id": sample_id,
                    "mean_iou_over_seeds": means[sample_id],
                    "per_seed": {str(seed): by_seed[seed][sample_id] for seed in sorted(by_seed)},
                }
                for sample_id in rank_samples(means, count=count, worst=kind == "worst")
            ]
            for kind in ("worst", "best")
        }

    manifest = {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "protocol_hash": str(document["protocol_sha256"]),
        "dataset_manifest_hash": str(document["dataset_manifest_sha256"]),
        "evaluation": GALLERY_EVALUATION,
        "rule": {
            "metric": "image_mean_iou",
            "aggregate": "mean_over_seeds",
            "per_model": per_model,
            "tie_break": "sample_id",
        },
        "per_model": per_model_block,
    }
    manifest = validated_document(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return GalleryResult(manifest_path=output_path, per_model=per_model, models=models)
