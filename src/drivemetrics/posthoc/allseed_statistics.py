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
    DEFAULT_RISK_PROFILES_DIR,
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
from drivemetrics.metrics.risk import compute_cost_risk
from drivemetrics.metrics.selective import area_under_risk_coverage, selective_risk_from_histogram
from drivemetrics.posthoc.allseed import (
    COMPONENTS_FILENAME,
    ECE_BINS,
    EXTRACT_FILENAME,
    EXTRACT_SCHEMA_VERSION,
    HISTOGRAM_DIRNAME,
    INSTANCES_FILENAME,
    STATES,
    TOPLABEL_BINS,
    toplabel_ece,
)
from drivemetrics.posthoc.draws import (
    TwoStageDraws,
    draw_two_stage,
    histogram_run_replicates,
    interval_from_replicates,
    run_replicates,
    run_replicates_many,
)
from drivemetrics.protocol.risk_profiles import BDD100K_SEMANTIC_CLASS_NAMES, load_risk_profile

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


# ---------------------------------------------------------------------------
# The pre-registered analysis
# ---------------------------------------------------------------------------

SUMMARY_SCHEMA_VERSION = "driving-risk-allseed-summary/v1"
SUMMARY_FILENAME = "summary.json"
REPRODUCTION_FILENAME = "reproduction.json"
RESAMPLES = 5000
BOOTSTRAP_SEED = 20260831
PRIMARY_CONFIDENCE = 1.0 - 0.05 / 3.0
DESCRIPTIVE_CONFIDENCE = 0.95
PERSON = BDD100K_SEMANTIC_CLASS_NAMES.index("person")
VRU_CLASSES: tuple[str, ...] = ("person", "rider", "motorcycle", "bicycle")
COST_PROFILES: tuple[str, ...] = ("vru_priority", "drivable_boundary")
#: Critical-miss thresholds of the descriptive sweep, as tenths: 0.1 ... 0.9.
THRESHOLD_TENTHS: tuple[int, ...] = tuple(range(1, 10))
STATUS = (
    "post-release analysis of the released locked-cohort predictions; pre-registered decision rules"
)
NO_PIXELS = "no pixel of this class in the cohort"
ECE_CLASSES = 19


@dataclass(frozen=True)
class AnalysisResult:
    """Where the summary was written and what the pre-registered rules concluded."""

    summary_path: Path
    categories: dict[str, str]
    separable: dict[str, bool]


def _ratio(summed: Float64Array) -> Float64Array:
    ratio: Float64Array = summed[:, 0] / summed[:, 1]
    return ratio


def _small_person_components(
    extraction: Extraction, tenths: int | None = None
) -> tuple[Int64Array, Int64Array]:
    """Per image, the small-tertile person count and, per run, how many are critical misses.

    With ``tenths`` unset the released rule applies (``correct_fraction < 0.5``);
    otherwise an instance is a miss when ``10 * correct < tenths * area``,
    compared as integers.
    """

    run_ids = extraction.run_ids
    counts = np.zeros(len(extraction.lines), dtype=np.int64)
    misses = np.zeros((len(run_ids), len(extraction.lines)), dtype=np.int64)
    for image, line in enumerate(extraction.lines):
        chosen = [
            index
            for index, meta in enumerate(line["instances"])
            if meta["class_id"] == PERSON and meta["area_tertile"] == "small"
        ]
        counts[image] = len(chosen)
        for position, run_id in enumerate(run_ids):
            if tenths is None:
                fractions = line["correct_fraction"][run_id]
                misses[position, image] = sum(fractions[index] < 0.5 for index in chosen)
            else:
                correct = line["correct_pixels"][run_id]
                misses[position, image] = sum(
                    10 * correct[index] < tenths * line["instances"][index]["area_pixels"]
                    for index in chosen
                )
    return counts, misses


def _interval_documents(
    replicates: Float64Array,
    estimate_runs: Float64Array,
    labels: tuple[int, ...],
    draws: TwoStageDraws,
    combine: str,
) -> tuple[float, list[dict[str, float]]]:
    """The estimate and the primary and descriptive intervals of one quantity."""

    documents: list[dict[str, float]] = []
    estimate = 0.0
    for confidence in (PRIMARY_CONFIDENCE, DESCRIPTIVE_CONFIDENCE):
        interval = interval_from_replicates(
            replicates, estimate_runs, labels, draws, combine=combine, confidence=confidence
        )
        estimate = interval.estimate
        documents.append({"confidence": confidence, "low": interval.low, "high": interval.high})
    return estimate, documents


def _category(low: float, high: float) -> str:
    if low > 0.5:
        return "majority missed"
    if high < 0.5:
        return "minority missed"
    return "undetermined"


def _excludes_zero(low: float, high: float) -> bool:
    return low > 0.0 or high < 0.0


def _pairs(extraction: Extraction) -> list[tuple[str, list[int], tuple[int, ...]]]:
    models = extraction.models
    pairs = []
    for left in range(len(models)):
        for right in range(left + 1, len(models)):
            selected, labels = pair_labels(extraction, left, right)
            pairs.append((f"{models[left]} minus {models[right]}", selected, labels))
    return pairs


def _signs(labels: tuple[int, ...]) -> Float64Array:
    return np.array([1.0 if label == 0 else -1.0 for label in labels], dtype=np.float64)


def _primary(
    extraction: Extraction, level_draws: TwoStageDraws, pair_draws: TwoStageDraws
) -> tuple[dict[str, Any], dict[str, str], dict[str, bool]]:
    """Plan 5.1 and 5.2: each model's small-person miss rate, and the paired differences."""

    counts, misses = _small_person_components(extraction)
    components = np.stack(
        [np.stack([misses[run], counts], axis=1) for run in range(misses.shape[0])]
    ).astype(np.float64)
    runs = extraction.document["runs"]
    rates: dict[str, Any] = {}
    categories: dict[str, str] = {}
    for model in extraction.models:
        positions = extraction.positions(model)
        selected = components[positions]
        estimate_runs = _ratio(np.sum(selected, axis=1))
        estimate, intervals = _interval_documents(
            run_replicates(level_draws, selected, _ratio),
            estimate_runs,
            (0,) * len(positions),
            level_draws,
            "mean",
        )
        per_seed = {
            str(runs[p]["seed"]): float(value)
            for p, value in zip(positions, estimate_runs, strict=True)
        }
        categories[model] = _category(intervals[0]["low"], intervals[0]["high"])
        rates[model] = {
            "estimate": estimate,
            "per_seed": per_seed,
            "seed_17_minus_mean": per_seed["17"] - estimate,
            "intervals": intervals,
            "category": categories[model],
        }
    pairs: dict[str, Any] = {}
    separable: dict[str, bool] = {}
    for key, selected_runs, labels in _pairs(extraction):
        signed = _signed_difference_statistic(_ratio, labels)
        paired = components[selected_runs]
        estimate, intervals = _interval_documents(
            run_replicates(pair_draws, paired, signed),
            signed(np.sum(paired, axis=1)),
            labels,
            pair_draws,
            "sum",
        )
        separable[key] = _excludes_zero(intervals[0]["low"], intervals[0]["high"])
        pairs[key] = {"estimate": estimate, "intervals": intervals, "separable": separable[key]}
    return {"small_person_miss_rate": rates, "pairs": pairs}, categories, separable


def _described(
    extraction: Extraction,
    run_level: Float64Array,
    run_pair: Float64Array,
    estimate_runs: Float64Array,
    level_draws: TwoStageDraws,
    pair_draws: TwoStageDraws,
) -> dict[str, Any]:
    """Levels and paired differences at the descriptive confidence, from run replicates."""

    levels: dict[str, Any] = {}
    for model in extraction.models:
        positions = extraction.positions(model)
        interval = interval_from_replicates(
            run_level[:, positions],
            estimate_runs[positions],
            (0,) * len(positions),
            level_draws,
            combine="mean",
            confidence=DESCRIPTIVE_CONFIDENCE,
        )
        levels[model] = {
            "estimate": interval.estimate,
            "low": interval.low,
            "high": interval.high,
            "confidence": interval.confidence,
        }
    pairs: dict[str, Any] = {}
    for key, selected, labels in _pairs(extraction):
        signs = _signs(labels)
        interval = interval_from_replicates(
            run_pair[:, selected] * signs,
            estimate_runs[selected] * signs,
            labels,
            pair_draws,
            combine="sum",
            confidence=DESCRIPTIVE_CONFIDENCE,
        )
        pairs[key] = {
            "estimate": interval.estimate,
            "low": interval.low,
            "high": interval.high,
            "confidence": interval.confidence,
            "excludes_zero": _excludes_zero(interval.low, interval.high),
        }
    return {"levels": levels, "pairs": pairs}


def _all_run_replicates(
    draws: TwoStageDraws, components: Float64Array, statistics: dict[str, Any]
) -> dict[str, Float64Array]:
    """Every run's replicates under one draw, one run at a time.

    A run's replicate depends only on the image weights, never on the seed stage,
    so computing the runs separately gives exactly the columns a joint call would.
    """

    columns: dict[str, list[Float64Array]] = {name: [] for name in statistics}
    for run in range(components.shape[0]):
        produced = run_replicates_many(draws, components[run : run + 1], statistics)
        for name, values in produced.items():
            columns[name].append(values[:, 0])
    return {name: np.stack(values, axis=1) for name, values in columns.items()}


def _ece_statistic(summed: Float64Array) -> Float64Array:
    cells = ECE_CLASSES * ECE_BINS
    shape = (-1, ECE_CLASSES, ECE_BINS)
    counts = np.rint(summed[:, :cells]).astype(np.int64).reshape(shape)
    sums = summed[:, cells : 2 * cells].reshape(shape)
    positives = np.rint(summed[:, 2 * cells :]).astype(np.int64).reshape(shape)
    return np.array(
        [
            mean_classwise_expected_calibration_error(
                ECEBinSufficientStatistics(
                    counts=counts[row], confidence_sums=sums[row], positive_counts=positives[row]
                )
            )
            for row in range(summed.shape[0])
        ],
        dtype=np.float64,
    )


def _toplabel_statistic(summed: Float64Array) -> Float64Array:
    return np.array(
        [toplabel_ece(row.reshape(TOPLABEL_BINS, 3)) for row in summed], dtype=np.float64
    )


def _brier_statistic(summed: Float64Array) -> Float64Array:
    return np.array(
        [multiclass_brier_score(row[:-1], int(np.rint(row[-1]))) for row in summed],
        dtype=np.float64,
    )


def _iou_statistic(class_id: int, num_classes: int) -> Any:
    def statistic(summed: Float64Array) -> Float64Array:
        matrices = np.rint(summed).astype(np.int64).reshape(-1, num_classes, num_classes)
        hit = matrices[:, class_id, class_id]
        union = matrices[:, class_id, :].sum(axis=1) + matrices[:, :, class_id].sum(axis=1) - hit
        iou: Float64Array = hit / union
        return iou

    return statistic


def _cost_statistic(profile: Any, num_classes: int) -> Any:
    def statistic(summed: Float64Array) -> Float64Array:
        matrices = np.rint(summed).astype(np.int64).reshape(-1, num_classes, num_classes)
        return np.array([compute_cost_risk(matrix, profile) for matrix in matrices])

    return statistic


def _finite_aurc(counts: Int64Array, correct: Int64Array) -> float:
    value = aurc_of(counts, correct)
    if value is None:
        raise ValueError("a one-point selective-risk curve has no area to bootstrap")
    return value


def _state_intervals(
    extraction: Extraction,
    state_index: int,
    state: str,
    level_draws: TwoStageDraws,
    pair_draws: TwoStageDraws,
) -> dict[str, Any]:
    """Plan 5.4 items 1 to 4 for one calibration state."""

    arrays = extraction.arrays
    run_ids = extraction.run_ids
    run_count = len(run_ids)
    calibration = [
        calibration_values(extraction, state_index, position) for position in range(run_count)
    ]
    ece = np.concatenate(
        [
            arrays[name][state_index].reshape(run_count, -1, ECE_CLASSES * ECE_BINS)
            for name in ("ece_counts", "ece_confidence_sums", "ece_positive_counts")
        ],
        axis=2,
    ).astype(np.float64)
    brier = np.concatenate(
        [arrays["brier"][state_index], arrays["valid_pixels"][:, :, None]], axis=2
    ).astype(np.float64)
    toplabel = arrays["toplabel"][state_index].reshape(run_count, -1, TOPLABEL_BINS * 3)
    specs: tuple[tuple[str, Float64Array, Any, Float64Array], ...] = (
        (
            "classwise_ece",
            ece,
            _ece_statistic,
            np.array([row["ece"] for row in calibration], dtype=np.float64),
        ),
        (
            "brier",
            brier,
            _brier_statistic,
            np.array([row["brier"] for row in calibration], dtype=np.float64),
        ),
        ("toplabel_ece", toplabel, _toplabel_statistic, _toplabel_statistic(toplabel.sum(axis=1))),
    )
    results: dict[str, Any] = {}
    for name, components, statistic, estimate_runs in specs:
        results[f"{name} {state}"] = _described(
            extraction,
            _all_run_replicates(level_draws, components, {name: statistic})[name],
            _all_run_replicates(pair_draws, components, {name: statistic})[name],
            estimate_runs,
            level_draws,
            pair_draws,
        )
    level_columns, pair_columns, estimates = [], [], []
    for run_id in run_ids:
        histograms = extraction.histograms(run_id, state)
        counts = np.asarray(histograms[:, 0, :], dtype=np.int64)
        correct = np.asarray(histograms[:, 1, :], dtype=np.int64)
        level_columns.append(histogram_run_replicates(level_draws, counts, correct, _finite_aurc))
        pair_columns.append(histogram_run_replicates(pair_draws, counts, correct, _finite_aurc))
        estimates.append(_finite_aurc(counts.sum(axis=0), correct.sum(axis=0)))
    results[f"aurc {state}"] = _described(
        extraction,
        np.stack(level_columns, axis=1),
        np.stack(pair_columns, axis=1),
        np.array(estimates, dtype=np.float64),
        level_draws,
        pair_draws,
    )
    return results


def _confusion_intervals(
    extraction: Extraction, level_draws: TwoStageDraws, pair_draws: TwoStageDraws
) -> tuple[dict[str, Any], dict[str, str]]:
    """Plan 5.4 items 5 and 6: VRU class IoU and risk-weighted cost, from the confusions."""

    num_classes = int(extraction.document["num_classes"])
    confusion = extraction.arrays["confusion"].astype(np.float64)
    support = confusion.sum(axis=(0, 1)).reshape(num_classes, num_classes).sum(axis=1)
    statistics: dict[str, Any] = {}
    not_computed: dict[str, str] = {}
    for name in VRU_CLASSES:
        class_id = BDD100K_SEMANTIC_CLASS_NAMES.index(name)
        if support[class_id] == 0:
            not_computed[f"iou {name}"] = NO_PIXELS
        else:
            statistics[f"iou {name}"] = _iou_statistic(class_id, num_classes)
    for profile_name in COST_PROFILES:
        profile = load_risk_profile(DEFAULT_RISK_PROFILES_DIR / f"{profile_name}.yaml")
        statistics[f"cost {profile_name}"] = _cost_statistic(profile, num_classes)
    level_replicates = _all_run_replicates(level_draws, confusion, statistics)
    pair_replicates = _all_run_replicates(pair_draws, confusion, statistics)
    totals = np.sum(confusion, axis=1)
    results = {
        name: _described(
            extraction,
            level_replicates[name],
            pair_replicates[name],
            statistic(totals),
            level_draws,
            pair_draws,
        )
        for name, statistic in statistics.items()
    }
    return results, not_computed


def _mean_bucket(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "instance_count": buckets[0]["instance_count"],
        "critical_misses": float(np.mean([bucket["critical_misses"] for bucket in buckets])),
        "mean_correct_fraction": _mean_or_none(
            [bucket["mean_correct_fraction"] for bucket in buckets]
        ),
    }


def mean_instance_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """The seed mean of one model's instance blocks, bucket by bucket (plan 5.3)."""

    first = blocks[0]
    return {
        "instance_count": first["instance_count"],
        "excluded_without_semantic_pixels": first["excluded_without_semantic_pixels"],
        "mean_corroborated_fraction": first["mean_corroborated_fraction"],
        "by_tertile": {
            tertile: _mean_bucket([block["by_tertile"][tertile] for block in blocks])
            for tertile in first["by_tertile"]
        },
        "by_class": {
            name: {
                **_mean_bucket([block["by_class"][name] for block in blocks]),
                "by_tertile": {
                    tertile: _mean_bucket(
                        [block["by_class"][name]["by_tertile"][tertile] for block in blocks]
                    )
                    for tertile in first["by_class"][name]["by_tertile"]
                },
            }
            for name in first["by_class"]
        },
    }


def _threshold_sweep(extraction: Extraction) -> list[dict[str, Any]]:
    """Plan 5.5: the seed-mean small-person miss rate at every tenth, and the model order."""

    orders: dict[int, list[str]] = {}
    entries: list[dict[str, Any]] = []
    for tenths in THRESHOLD_TENTHS:
        counts, misses = _small_person_components(extraction, tenths)
        run_rates = misses.sum(axis=1) / counts.sum()
        rates = {
            model: float(np.mean(run_rates[extraction.positions(model)]))
            for model in extraction.models
        }
        orders[tenths] = sorted(
            extraction.models, key=lambda model: (rates[model], extraction.models.index(model))
        )
        entries.append({"threshold": tenths / 10, "rates": rates, "order": orders[tenths]})
    for entry, tenths in zip(entries, THRESHOLD_TENTHS, strict=True):
        entry["order_equals_threshold_05"] = orders[tenths] == orders[5]
    return entries


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def analyse_all_seeds(
    extract_dir: Path,
    evidence_dir: Path,
    output_dir: Path,
    *,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> AnalysisResult:
    """Run the reproduction gate, then the pre-registered statistics, and write ``summary.json``.

    The gate's report is written whether or not it passes. When it fails the
    analysis stops there, and no new number is computed or written.
    """

    summary_path = output_dir / SUMMARY_FILENAME
    if summary_path.exists():
        raise FileExistsError(f"an all-seed summary already exists: {summary_path}")
    extraction = load_extraction(extract_dir)
    reproduction = reproduce_released(extraction, evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(
        output_dir / REPRODUCTION_FILENAME,
        {
            "passed": reproduction.passed,
            "checked": list(reproduction.checked),
            "mismatches": list(reproduction.mismatches),
        },
    )
    if not reproduction.passed:
        raise ValueError(
            "the reproduction gate failed; no result is computed: "
            + ", ".join(reproduction.mismatches)
        )

    image_count = len(extraction.lines)
    level_draws = draw_two_stage(image_count, (0, 0, 0), resamples=resamples, seed=seed)
    pair_draws = draw_two_stage(image_count, (0, 0, 0, 1, 1, 1), resamples=resamples, seed=seed)
    primary, categories, separable = _primary(extraction, level_draws, pair_draws)
    intervals: dict[str, Any] = {}
    for state_index, state in enumerate(STATES):
        intervals.update(_state_intervals(extraction, state_index, state, level_draws, pair_draws))
    confusion_intervals, not_computed = _confusion_intervals(extraction, level_draws, pair_draws)
    intervals.update(confusion_intervals)
    blocks = {run_id: instance_block(extraction, run_id) for run_id in extraction.run_ids}
    counts, _ = _small_person_components(extraction)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "analysis_plan": extraction.document["analysis_plan"],
        "status": STATUS,
        "protocol_hash": extraction.document["protocol_sha256"],
        "dataset_manifest_hash": extraction.document["dataset_manifest_sha256"],
        "settings": {
            "resamples": resamples,
            "seed": seed,
            "primary_confidence": PRIMARY_CONFIDENCE,
            "descriptive_confidence": DESCRIPTIVE_CONFIDENCE,
            "threshold_tenths": list(THRESHOLD_TENTHS),
        },
        "gates": {
            "integrity": extraction.document["gates"]["integrity"],
            "reproduction": "passed",
        },
        "counts": {
            "images": image_count,
            "small_person_instances": int(counts.sum()),
            "images_with_small_person": int(np.count_nonzero(counts)),
        },
        "primary": primary,
        "secondary": {
            "intervals": intervals,
            "not_computed": not_computed,
            "instance_blocks": blocks,
            "instance_seed_means": {
                model: mean_instance_blocks(
                    [blocks[extraction.run_ids[p]] for p in extraction.positions(model)]
                )
                for model in extraction.models
            },
            "threshold_sweep": _threshold_sweep(extraction),
        },
    }
    _write(summary_path, summary)
    return AnalysisResult(summary_path=summary_path, categories=categories, separable=separable)
