"""The published analysis documents, as contracts.

Six documents leave the analysis and are cited by claims: the metric table, the
paired intervals, the ranking comparison, the failure-gallery manifest, the
extended metrics, and the frozen area tertiles the extended metrics consume. Until
now each carried a `schema_version` string with nothing behind it, or no version
at all. A version string without a schema is a label; these models are the
contract the label names.

Every model forbids unknown fields. A producer that adds a field must add it here
first, which is the point: a shape change becomes a visible change to a contract
rather than an invisible change to a file. The producers validate their payload
through these models BEFORE writing, so a regression fails the producer, and the
verifier validates every document under `docs/evidence/` against them, so a hand
edit or a stale file fails the build.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
# Strict: a whole float such as 3.0 is not an integer count and is refused rather
# than coerced. A producer that divided where it should have floor-divided fails
# its own contract instead of publishing a number that merely looks right.
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(ge=1, strict=True)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]

METRICS_TABLE_SCHEMA_VERSION = "driving-risk-metrics-table/v1"
INTERVALS_SCHEMA_VERSION = "driving-risk-intervals/v1"
RANKINGS_SCHEMA_VERSION = "driving-risk-rankings/v1"
GALLERY_MANIFEST_SCHEMA_VERSION = "driving-risk-gallery-manifest/v1"
EXTENDED_METRICS_SCHEMA_VERSION = "driving-risk-extended-metrics/v1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ── metrics.json ────────────────────────────────────────────────────


class HeadlineMetricsV1(_Strict):
    miou: UnitFloat
    pixel_accuracy: UnitFloat
    critical_recall: UnitFloat


class PerClassScoresV1(_Strict):
    iou: tuple[UnitFloat | None, ...]
    recall: tuple[UnitFloat | None, ...]


class PerClassV1(_Strict):
    class_names: tuple[str, ...]
    support_pixels: tuple[NonNegativeInt, ...]
    images_with_class: tuple[NonNegativeInt, ...]
    by_model: dict[str, PerClassScoresV1]


class CalibrationValuesV1(_Strict):
    ece: float | None
    brier: float | None


class CalibrationEntryV1(CalibrationValuesV1):
    per_seed: dict[str, CalibrationValuesV1]


class RiskProfileResultV1(_Strict):
    sensitivity: float
    critical_class_ids: tuple[NonNegativeInt, ...]
    cost_risk: dict[str, float]


class MetricsTableV1(_Strict):
    schema_version: Literal["driving-risk-metrics-table/v1"]
    protocol_hash: Sha256
    dataset_manifest_hash: Sha256
    cohort: str
    sample_count: PositiveInt
    seed_count: PositiveInt
    interval_method: str
    metrics: dict[str, HeadlineMetricsV1]
    per_class: PerClassV1
    calibration: dict[str, dict[Literal["uncalibrated", "calibrated"], CalibrationEntryV1]]
    risk_profiles: dict[str, RiskProfileResultV1]


# ── intervals.json ──────────────────────────────────────────────────


class IntervalV1(_Strict):
    estimate: float
    low: float
    high: float
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    resamples: PositiveInt
    seed: int
    estimator: str


class IntervalsV1(_Strict):
    schema_version: Literal["driving-risk-intervals/v1"]
    protocol_hash: Sha256
    dataset_manifest_hash: Sha256
    intervals: dict[str, IntervalV1]


# ── rankings.json ───────────────────────────────────────────────────


class RankingComparisonV1(_Strict):
    metric_name: str
    baseline_order: tuple[str, ...]
    comparison_order: tuple[str, ...]
    reversal_observed: bool


class PairSeparabilityV1(_Strict):
    left: str
    right: str
    estimate: float
    low: float
    high: float
    excludes_zero: bool


class RankingsV1(_Strict):
    schema_version: Literal["driving-risk-rankings/v1"]
    protocol_hash: Sha256
    dataset_manifest_hash: Sha256
    baseline_metric: str
    comparisons: tuple[RankingComparisonV1, ...]
    separability: dict[str, tuple[PairSeparabilityV1, ...]]


# ── gallery-manifest.json ───────────────────────────────────────────


class GalleryRuleV1(_Strict):
    metric: str
    aggregate: str
    per_model: PositiveInt
    tie_break: str


class GalleryEntryV1(_Strict):
    sample_id: str
    mean_iou_over_seeds: UnitFloat
    per_seed: dict[str, UnitFloat]


class GallerySelectionV1(_Strict):
    worst: tuple[GalleryEntryV1, ...]
    best: tuple[GalleryEntryV1, ...]


class GalleryManifestV1(_Strict):
    schema_version: Literal["driving-risk-gallery-manifest/v1"]
    protocol_hash: Sha256
    dataset_manifest_hash: Sha256
    evaluation: str
    rule: GalleryRuleV1
    per_model: dict[str, GallerySelectionV1]


# ── extended-metrics.json ───────────────────────────────────────────


class NotComputedV1(_Strict):
    not_computed: str


class BitmaskSetV1(_Strict):
    count: NonNegativeInt
    set_sha256: Sha256


class GroundTruthV1(_Strict):
    cohort_manifest_sha256: Sha256
    split_name: str
    masks_verified: NonNegativeInt
    instance_bitmasks: BitmaskSetV1 | None


class SelectiveRiskV1(_Strict):
    aurc: float | None
    coverage_points: NonNegativeInt
    defined_at: str


class BandV1(_Strict):
    pixels: NonNegativeInt
    pixel_accuracy: UnitFloat | None


class BandsByModelV1(_Strict):
    top: BandV1
    middle: BandV1
    bottom: BandV1


class BandsV1(_Strict):
    definition: str
    by_model: dict[str, BandsByModelV1]


class CoverageBucketV1(_Strict):
    instance_count: NonNegativeInt
    critical_misses: NonNegativeInt
    mean_correct_fraction: UnitFloat | None


class TertileBucketsV1(_Strict):
    small: CoverageBucketV1
    medium: CoverageBucketV1
    large: CoverageBucketV1


class ClassCoverageV1(CoverageBucketV1):
    by_tertile: TertileBucketsV1


class InstanceBlockV1(_Strict):
    tertile_edges_sha256: Sha256
    instance_count: NonNegativeInt
    excluded_without_semantic_pixels: NonNegativeInt
    mean_corroborated_fraction: UnitFloat | None
    by_tertile: TertileBucketsV1
    by_class: dict[str, ClassCoverageV1]


class ExtendedMetricsV1(_Strict):
    schema_version: Literal["driving-risk-extended-metrics/v1"]
    protocol_hash: Sha256
    dataset_manifest_hash: Sha256
    evaluation_for_ground_truth_metrics: str
    ground_truth: GroundTruthV1 | NotComputedV1
    selective_risk: dict[str, dict[Literal["uncalibrated", "calibrated"], SelectiveRiskV1]]
    normalized_image_bands: BandsV1 | NotComputedV1
    instances: dict[str, InstanceBlockV1] | NotComputedV1


# ── area_tertiles.json (frozen at P1-14; consumed, never produced, by the analysis) ──


class AreaTertilesV1(_Strict):
    """The frozen file carries no `schema_version`; it is matched by filename.

    It was frozen before the documents had contracts, and its bytes are the ones
    the published evidence consumed, so the file is not touched. The contract is
    written around it.
    """

    eligible_images: PositiveInt
    instances_per_category: dict[str, NonNegativeInt]
    learned_from: str
    tertile_edges: dict[str, tuple[NonNegativeInt, NonNegativeInt]]
    total_instances: PositiveInt


#: Which model validates a document, by the version string it declares.
DOCUMENT_MODELS: dict[str, type[BaseModel]] = {
    METRICS_TABLE_SCHEMA_VERSION: MetricsTableV1,
    INTERVALS_SCHEMA_VERSION: IntervalsV1,
    RANKINGS_SCHEMA_VERSION: RankingsV1,
    GALLERY_MANIFEST_SCHEMA_VERSION: GalleryManifestV1,
    EXTENDED_METRICS_SCHEMA_VERSION: ExtendedMetricsV1,
}

#: Documents that predate versioning, matched by their fixed filename.
UNVERSIONED_DOCUMENT_MODELS: dict[str, type[BaseModel]] = {
    "area_tertiles.json": AreaTertilesV1,
}


def document_model_for(payload: Mapping[str, Any]) -> type[BaseModel]:
    """Return the contract a document declares, or refuse a document that declares none.

    A document without a recognised version cannot be validated, and validating it
    against a guess would be worse than refusing: the guess could pass.
    """

    version = payload.get("schema_version")
    model = DOCUMENT_MODELS.get(version) if isinstance(version, str) else None
    if model is None:
        raise ValueError(f"no contract is registered for schema_version {version!r}")
    return model


def validated_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a document through its contract and return it as plain JSON values.

    Every producer calls this before writing, so a shape regression fails the
    producer that made it rather than the reader that found it.
    """

    return document_model_for(payload).model_validate(payload).model_dump(mode="json")


def serialised_document(document: Mapping[str, Any]) -> str:
    """The exact text a document is cited by: two-space indent, sorted keys, one trailing newline.

    Documents are cited by hash, so their bytes are the contract. Indentation, key
    order, encoding and the trailing newline are all part of what a claim's
    `artifact_path` resolves to, and a reformat that changed any of them would
    change every hash that ever cited the file.
    """

    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def write_document(path: Path, payload: Mapping[str, Any]) -> None:
    """Validate a document through its contract, then write it in the cited byte format.

    Every producer writes through here, so the shape check and the byte format
    live in one place: a producer cannot publish a document its contract rejects,
    and cannot publish one in a format a hash would not recognise.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        serialised_document(validated_document(payload)), encoding="utf-8", newline="\n"
    )
