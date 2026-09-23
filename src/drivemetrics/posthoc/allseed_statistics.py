"""The pre-registered all-seed statistics, and the gate that must pass before any is read.

The analysis is pre-registered in ``docs/posthoc/allseed-v1/analysis-plan.md``.
Everything here reads the output of :func:`drivemetrics.posthoc.allseed.extract_all_seeds`
and never an artifact. Before a new number is computed, the released numbers the
new code overlaps with are recomputed from the same extraction and must equal the
released evidence exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from drivemetrics.analysis.aggregate import (
    METRIC_NAMES,
    _mean_or_none,
    _signed_difference_statistic,
    _statistic_for,
)
from drivemetrics.analysis.extended import (
    INSTANCE_CATEGORY_TO_TRAIN_ID,
    _add,
    _finalise,
    _mean_of_blocks,
    _new_bucket,
    _new_tertiles,
)
from drivemetrics.metrics.calibration import (
    ECEBinSufficientStatistics,
    mean_classwise_expected_calibration_error,
    multiclass_brier_score,
)
from drivemetrics.metrics.instances import InstanceCoverage
from drivemetrics.metrics.selective import area_under_risk_coverage, selective_risk_from_histogram
from drivemetrics.posthoc.allseed import (
    COMPONENTS_FILENAME,
    EXTRACT_FILENAME,
    EXTRACT_SCHEMA_VERSION,
    HISTOGRAM_DIRNAME,
    INSTANCES_FILENAME,
    STATES,
)
from drivemetrics.posthoc.draws import (
    TwoStageDraws,
    draw_two_stage,
    interval_from_replicates,
    run_replicates,
)
from drivemetrics.protocol.risk_profiles import BDD100K_SEMANTIC_CLASS_NAMES

Int64Array = npt.NDArray[np.int64]
Float64Array = npt.NDArray[np.float64]

RELEASED_CONFIDENCE = 0.95
SELECTIVE_DEFINED_AT = "confidence_bin_boundaries"


@dataclass(frozen=True)
class Extraction:
    """A finished extraction: its record, its component arrays and its instance lines."""

    directory: Path
    document: dict[str, Any]
    arrays: dict[str, npt.NDArray[Any]]
    lines: tuple[dict[str, Any], ...]

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(str(entry["run_id"]) for entry in self.document["runs"])

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(entry["model"]) for entry in self.document["runs"]))

    @property
    def model_ids(self) -> tuple[int, ...]:
        return tuple(self.models.index(str(entry["model"])) for entry in self.document["runs"])

    def positions(self, model: str) -> list[int]:
        return [index for index, label in enumerate(self.model_ids) if self.models[label] == model]

    def histograms(self, run_id: str, state: str) -> npt.NDArray[np.int32]:
        loaded: npt.NDArray[np.int32] = np.load(
            self.directory / HISTOGRAM_DIRNAME / f"{run_id}.{state}.npy", mmap_mode="r"
        )
        return loaded


def load_extraction(extract_dir: Path) -> Extraction:
    """Load a finished extraction; one whose integrity gate did not pass is refused."""

    document = json.loads((extract_dir / EXTRACT_FILENAME).read_text(encoding="utf-8"))
    if document.get("schema_version") != EXTRACT_SCHEMA_VERSION:
        raise ValueError(f"{extract_dir} does not hold an all-seed extraction")
    if document.get("gates") != {"integrity": "passed"}:
        raise ValueError("the extraction's integrity gate did not pass")
    with np.load(extract_dir / COMPONENTS_FILENAME) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    lines = tuple(
        json.loads(line)
        for line in (extract_dir / INSTANCES_FILENAME).read_text(encoding="utf-8").splitlines()
    )
    return Extraction(directory=extract_dir, document=document, arrays=arrays, lines=lines)


def instance_block(extraction: Extraction, run_id: str) -> dict[str, Any]:
    """One run's instance block, in exactly the released structure and summation order."""

    overall = _new_bucket()
    by_tertile = _new_tertiles()
    by_class: dict[str, dict[str, Any]] = {
        BDD100K_SEMANTIC_CLASS_NAMES[train_id]: {**_new_bucket(), "by_tertile": _new_tertiles()}
        for _category, train_id in sorted(INSTANCE_CATEGORY_TO_TRAIN_ID.items())
    }
    without_semantic_pixels = 0
    fractions: list[float] = []
    for line in extraction.lines:
        without_semantic_pixels += int(line["without_semantic_pixels"])
        fractions.extend(line["corroborated_fractions"])
        for meta, fraction in zip(line["instances"], line["correct_fraction"][run_id], strict=True):
            coverage = InstanceCoverage(
                instance_id=int(meta["instance_id"]),
                class_id=int(meta["class_id"]),
                area_pixels=int(meta["area_pixels"]),
                correct_fraction=float(fraction),
                is_critical_miss=float(fraction) < 0.5,
                area_tertile=meta["area_tertile"],
            )
            _add(overall, coverage)
            _add(by_tertile[coverage.area_tertile], coverage)
            class_block = by_class[BDD100K_SEMANTIC_CLASS_NAMES[coverage.class_id]]
            _add(class_block, coverage)
            _add(class_block["by_tertile"][coverage.area_tertile], coverage)
    return {
        "tertile_edges_sha256": extraction.document["tertiles_sha256"],
        "instance_count": overall["instance_count"],
        "excluded_without_semantic_pixels": without_semantic_pixels,
        "mean_corroborated_fraction": float(np.mean(fractions)) if fractions else None,
        "by_tertile": {name: _finalise(bucket) for name, bucket in by_tertile.items()},
        "by_class": {
            name: {
                **_finalise(block),
                "by_tertile": {
                    tertile: _finalise(bucket) for tertile, bucket in block["by_tertile"].items()
                },
            }
            for name, block in by_class.items()
        },
    }


def pooled_histogram(
    extraction: Extraction, run_id: str, state: str
) -> tuple[Int64Array, Int64Array]:
    """One run's cohort confidence histogram and correct counts, summed exactly."""

    histograms = extraction.histograms(run_id, state)
    counts = np.asarray(histograms[:, 0, :], dtype=np.int64).sum(axis=0)
    correct = np.asarray(histograms[:, 1, :], dtype=np.int64).sum(axis=0)
    return counts, correct


def aurc_of(counts: Int64Array, correct: Int64Array) -> float | None:
    """The released AURC of one pooled histogram: ``None`` for a one-point curve."""

    coverage, risk = selective_risk_from_histogram(counts, correct)
    return area_under_risk_coverage(coverage, risk) if coverage.size > 1 else None


def _selective_block(extraction: Extraction, run_id: str, state: str) -> dict[str, Any]:
    counts, correct = pooled_histogram(extraction, run_id, state)
    coverage, _ = selective_risk_from_histogram(counts, correct)
    return {
        "aurc": aurc_of(counts, correct),
        "coverage_points": int(coverage.size),
        "defined_at": SELECTIVE_DEFINED_AT,
    }


def calibration_values(
    extraction: Extraction, state_index: int, position: int
) -> dict[str, float | None]:
    """One run's ECE and Brier, summed image by image in the released order."""

    arrays = extraction.arrays
    counts = arrays["ece_counts"][state_index, position, 0].copy()
    confidence_sums = arrays["ece_confidence_sums"][state_index, position, 0].copy()
    positive_counts = arrays["ece_positive_counts"][state_index, position, 0].copy()
    brier = arrays["brier"][state_index, position, 0].copy()
    pixels = int(arrays["valid_pixels"][position, 0])
    for image in range(1, arrays["valid_pixels"].shape[1]):
        counts = counts + arrays["ece_counts"][state_index, position, image]
        confidence_sums = (
            confidence_sums + arrays["ece_confidence_sums"][state_index, position, image]
        )
        positive_counts = (
            positive_counts + arrays["ece_positive_counts"][state_index, position, image]
        )
        brier = brier + arrays["brier"][state_index, position, image]
        pixels += int(arrays["valid_pixels"][position, image])
    return {
        "ece": mean_classwise_expected_calibration_error(
            ECEBinSufficientStatistics(
                counts=counts, confidence_sums=confidence_sums, positive_counts=positive_counts
            )
        ),
        "brier": multiclass_brier_score(brier, pixels),
    }


def pair_labels(extraction: Extraction, left: int, right: int) -> tuple[list[int], tuple[int, ...]]:
    """The runs of two models, and the 0/1 labels the released pairing gives them."""

    selected = [
        position for position, label in enumerate(extraction.model_ids) if label in (left, right)
    ]
    labels = tuple(0 if extraction.model_ids[position] == left else 1 for position in selected)
    return selected, labels


@dataclass(frozen=True)
class Reproduction:
    """Which released numbers were recomputed, and which did not come out equal."""

    passed: bool
    checked: tuple[str, ...]
    mismatches: tuple[str, ...]


def _load_evidence(evidence_dir: Path, extraction: Extraction) -> dict[str, dict[str, Any]]:
    documents = {
        name: json.loads((evidence_dir / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("intervals", "metrics", "extended-metrics")
    }
    for name, document in documents.items():
        if (
            document.get("protocol_hash") != extraction.document["protocol_sha256"]
            or document.get("dataset_manifest_hash")
            != extraction.document["dataset_manifest_sha256"]
        ):
            raise ValueError(f"the released {name}.json belongs to another study")
    return documents


def reproduce_released(
    extraction: Extraction,
    evidence_dir: Path,
    *,
    pair_draws: TwoStageDraws | None = None,
) -> Reproduction:
    """Recompute every released number the analysis overlaps with, and compare exactly."""

    evidence = _load_evidence(evidence_dir, extraction)
    checked: list[str] = []
    mismatches: list[str] = []

    def check(name: str, reproduced: object, released: object) -> None:
        checked.append(name)
        if reproduced != released:
            mismatches.append(name)

    num_classes = int(extraction.document["num_classes"])
    critical = tuple(int(value) for value in extraction.document["critical_class_ids"])
    models = extraction.models
    confusion = extraction.arrays["confusion"].astype(np.float64)
    image_count = confusion.shape[1]
    cache: dict[tuple[int, int], TwoStageDraws] = {}
    for name in METRIC_NAMES:
        statistic = _statistic_for(name, num_classes, critical)
        for left in range(len(models)):
            for right in range(left + 1, len(models)):
                key = f"{models[left]} minus {models[right]} ({name})"
                released = evidence["intervals"]["intervals"][key]
                selected, labels = pair_labels(extraction, left, right)
                settings = (int(released["resamples"]), int(released["seed"]))
                if settings not in cache:
                    cache[settings] = pair_draws or draw_two_stage(
                        image_count, labels, resamples=settings[0], seed=settings[1]
                    )
                draws = cache[settings]
                signed = _signed_difference_statistic(statistic, labels)
                paired = confusion[selected]
                interval = interval_from_replicates(
                    run_replicates(draws, paired, signed),
                    signed(np.sum(paired, axis=1)),
                    labels,
                    draws,
                    combine="sum",
                    confidence=RELEASED_CONFIDENCE,
                )
                check(
                    f"interval: {key}",
                    (interval.estimate, interval.low, interval.high),
                    (released["estimate"], released["low"], released["high"]),
                )

    runs = extraction.document["runs"]
    for model in models:
        first = extraction.positions(model)[0]
        check(
            f"instances: {model}",
            instance_block(extraction, str(runs[first]["run_id"])),
            evidence["extended-metrics"]["instances"][model],
        )
    for model in models:
        for state in STATES:
            blocks = [
                _selective_block(extraction, str(runs[position]["run_id"]), state)
                for position in extraction.positions(model)
            ]
            check(
                f"aurc: {model} {state}",
                _mean_of_blocks(blocks),
                evidence["extended-metrics"]["selective_risk"][model][state],
            )
    for model in models:
        for state_index, state in enumerate(STATES):
            positions = extraction.positions(model)
            finalised = [calibration_values(extraction, state_index, p) for p in positions]
            reproduced = {
                **{
                    field: _mean_or_none([row[field] for row in finalised])
                    for field in ("ece", "brier")
                },
                "per_seed": {
                    str(runs[p]["seed"]): {"ece": row["ece"], "brier": row["brier"]}
                    for p, row in zip(positions, finalised, strict=True)
                },
            }
            check(
                f"calibration: {model} {state}",
                reproduced,
                evidence["metrics"]["calibration"][model][state],
            )
    return Reproduction(passed=not mismatches, checked=tuple(checked), mismatches=tuple(mismatches))
