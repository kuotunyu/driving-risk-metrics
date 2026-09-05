"""Turn validated per-image artifacts into the three published documents.

Every metric this project reports is a function of the summed confusion matrix.
That is what makes one component array enough: the per-image confusions are
resampled once, summed, and each metric is then recomputed from those sums the
way the cohort metric is actually defined. A mean of per-image mIoU would be a
different number, so the estimator used is written next to every interval rather
than left for a reader to assume.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from drivemetrics.analysis.bootstrap import (
    two_stage_paired_bootstrap_statistic,
)
from drivemetrics.analysis.rankings import compare_rankings
from drivemetrics.artifacts.documents import validated_document
from drivemetrics.artifacts.formal_set import (
    APPROVED_MODELS,
    APPROVED_SEEDS,
    validate_formal_run_index,
)
from drivemetrics.artifacts.predictions import read_prediction_artifact
from drivemetrics.metrics.calibration import (
    ECEBinSufficientStatistics,
    mean_classwise_expected_calibration_error,
    multiclass_brier_score,
)
from drivemetrics.metrics.confusion import summarize_confusion
from drivemetrics.metrics.risk import compute_cost_risk, critical_false_negative_rate
from drivemetrics.protocol.risk_profiles import BDD100K_SEMANTIC_CLASS_NAMES, load_risk_profile

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

OUTPUT_NAMES: tuple[str, ...] = ("metrics", "intervals", "rankings")
BASELINE_METRIC = "miou"
#: Every declared cost profile is published, not one. The study's question is
#: whether the ranking depends on how cost is assigned, and a single profile
#: cannot answer it; `balanced`, whose critical set is empty, is the comparison
#: the other two are read against.
DEFAULT_RISK_PROFILES_DIR = Path(__file__).resolve().parents[3] / "configs" / "risk_profiles"
RATIO_ESTIMATOR = "ratio_of_sums"
BOOTSTRAP_SEED = 20260831

#: Every reported metric, oriented so that larger is always better. Ranking
#: comparison depends on that orientation, and a metric left in its natural
#: lower-is-better form would invert its own ranking silently.
METRIC_NAMES: tuple[str, ...] = ("miou", "pixel_accuracy", "critical_recall")
METRICS_SCHEMA_VERSION = "driving-risk-metrics-table/v1"
INTERVALS_SCHEMA_VERSION = "driving-risk-intervals/v1"
RANKINGS_SCHEMA_VERSION = "driving-risk-rankings/v1"


@dataclass(frozen=True)
class AggregateResult:
    """Where the three published documents were written."""

    metrics_path: Path
    intervals_path: Path
    rankings_path: Path
    models: tuple[str, ...]
    sample_count: int


def summed_confusion(directory: Path, sample_ids: Sequence[str]) -> Int64Array:
    """Sum one run's per-image confusion matrices in cohort order."""

    total: Int64Array | None = None
    for sample_id in sample_ids:
        _, record, _ = read_prediction_artifact(directory / f"{sample_id}.json")
        total = record.confusion.copy() if total is None else total + record.confusion
    if total is None:
        raise ValueError(f"no artifacts found for the cohort under {directory}")
    return total


def _metric_from_confusion(
    confusion: Int64Array,
    name: str,
    critical_class_ids: tuple[int, ...],
) -> float:
    summary = summarize_confusion(confusion)
    if name == "miou":
        return summary.mean_iou
    if name == "pixel_accuracy":
        return summary.pixel_accuracy
    rate = critical_false_negative_rate(confusion, critical_class_ids)
    if rate is None:
        raise ValueError(
            "the critical classes have no ground-truth support in this cohort, so "
            "critical recall is undefined and must not be reported as zero"
        )
    return 1.0 - rate


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    """Average the values that exist, or report that none did.

    A class no cohort pixel ever carried has no score, and averaging it in as
    zero would flatter the published number by an amount that grows with how
    many classes were absent.
    """

    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(np.mean(present))


def _excludes_zero(low: float, high: float) -> bool:
    """Whether a paired interval excludes zero — the plain reading of its two bounds.

    Strict on both sides: an interval whose bound IS zero does not exclude it, and
    is reported as not separating the pair. The helper exists so that this boundary
    is tested on its own rather than only through whatever the bootstrap happens to
    produce.
    """

    return bool(low > 0.0 or high < 0.0)


def _ground_truth_support(
    components: Float64Array,
    runs: Sequence[dict[str, Any]],
    image_count: int,
    num_classes: int,
) -> tuple[list[int], list[int]]:
    """Read the cohort's per-class support from the confusions, and prove the runs share it.

    The true-row sums of a confusion are the ground truth's class counts, and the
    ground truth is the same for every run of one study. The index validator proves
    the runs share a SET of sample IDs; it cannot see whether the confusions behind
    those IDs were built from the same masks. A run evaluated against different
    masks would pass every hash check while its numbers described another cohort,
    so the agreement is checked here, once, before anything is published.
    """

    per_image = components.reshape(len(runs), image_count, num_classes, num_classes)
    true_rows = per_image.sum(axis=3)
    for position in range(1, len(runs)):
        if not np.array_equal(true_rows[position], true_rows[0]):
            raise ValueError(
                "the runs do not share one ground truth: "
                f"{runs[position]['run_id']} has different per-class support "
                f"than {runs[0]['run_id']}"
            )
    support = true_rows[0].sum(axis=0)
    images = (true_rows[0] > 0).sum(axis=0)
    return [int(value) for value in support], [int(value) for value in images]


def _mean_over_seeds(rows: Sequence[Sequence[float | None]]) -> list[float | None]:
    """Mean each class position across the seeds that measured it."""

    return [_mean_or_none([row[position] for row in rows]) for position in range(len(rows[0]))]


def _statistic_for(name: str, num_classes: int, critical_class_ids: tuple[int, ...]):
    """Build the run-wise statistic that recomputes one metric from summed components."""

    def statistic(summed: Float64Array) -> Float64Array:
        matrices = np.rint(summed).astype(np.int64).reshape(-1, num_classes, num_classes)
        return np.array(
            [_metric_from_confusion(matrix, name, critical_class_ids) for matrix in matrices],
            dtype=np.float64,
        )

    return statistic


@dataclass(frozen=True)
class _CalibrationSums:
    """One run's summed calibration statistics over the cohort."""

    ece: ECEBinSufficientStatistics
    brier_sum_by_class: Float64Array
    valid_pixel_count: int

    def finalised(self) -> dict[str, float | None]:
        return {
            "ece": mean_classwise_expected_calibration_error(self.ece),
            "brier": multiclass_brier_score(self.brier_sum_by_class, self.valid_pixel_count),
        }


def _accumulate_calibration(directory: Path, sample_ids: Sequence[str]) -> _CalibrationSums:
    """Sum one evaluation's calibration statistics, which is the only way they combine.

    ECE and Brier are stored per image as sufficient statistics rather than as
    scores, because statistics add exactly and scores do not. Averaging per-image
    ECEs would weight a small image like a large one and would not be the cohort
    number at all.

    The sums are seeded from the first artifact rather than from zeros of a
    guessed shape: the artifact is what knows how many classes and bins there are.
    An empty cohort is refused before anything is read, because a sum over nothing
    is zero and zero calibration error is a perfect score.
    """

    if not sample_ids:
        raise ValueError(f"no artifacts found for the cohort under {directory}")
    _, first_record, first_ece = read_prediction_artifact(directory / f"{sample_ids[0]}.json")
    counts = first_ece.counts.copy()
    confidence_sums = first_ece.confidence_sums.copy()
    positive_counts = first_ece.positive_counts.copy()
    brier = first_record.brier_sum_by_class.copy()
    pixels = int(first_record.valid_pixel_count)
    for sample_id in sample_ids[1:]:
        _, record, ece = read_prediction_artifact(directory / f"{sample_id}.json")
        counts = counts + ece.counts
        confidence_sums = confidence_sums + ece.confidence_sums
        positive_counts = positive_counts + ece.positive_counts
        brier = brier + record.brier_sum_by_class
        pixels += int(record.valid_pixel_count)
    return _CalibrationSums(
        ece=ECEBinSufficientStatistics(
            counts=counts, confidence_sums=confidence_sums, positive_counts=positive_counts
        ),
        brier_sum_by_class=brier,
        valid_pixel_count=pixels,
    )


def _load_components(
    index_dir: Path,
    runs: Sequence[dict[str, Any]],
    sample_ids: Sequence[str],
    *,
    protocol_sha256: str,
    dataset_manifest_sha256: str,
) -> tuple[Float64Array, tuple[Int64Array, ...]]:
    """Read every per-image confusion, verifying each artifact belongs to this study."""

    per_run: list[list[Int64Array]] = []
    totals: list[Int64Array] = []
    for entry in runs:
        directory = index_dir / str(entry["artifacts_dir"])
        images: list[Int64Array] = []
        for sample_id in sample_ids:
            manifest, record, _ = read_prediction_artifact(directory / f"{sample_id}.json")
            if manifest.protocol_sha256 != protocol_sha256:
                raise ValueError(
                    f"artifact {sample_id} in {entry['run_id']} carries a different "
                    "protocol hash than the run index"
                )
            if manifest.dataset_manifest_sha256 != dataset_manifest_sha256:
                raise ValueError(
                    f"artifact {sample_id} in {entry['run_id']} carries a different "
                    "dataset manifest hash than the run index"
                )
            images.append(record.confusion)
        per_run.append(images)
        totals.append(np.sum(np.stack(images), axis=0))

    components = np.stack(
        [np.stack([matrix.reshape(-1) for matrix in images]) for images in per_run]
    ).astype(np.float64)
    return components, tuple(totals)


def aggregate_runs(
    index_path: Path,
    output_dir: Path,
    *,
    resamples: int = 5000,
    seed: int = BOOTSTRAP_SEED,
    risk_profiles_dir: Path = DEFAULT_RISK_PROFILES_DIR,
) -> AggregateResult:
    """Compute the metric table, paired intervals and ranking comparison.

    The run index is re-validated here rather than trusted, because analysing an
    incomplete or drifted matrix produces a narrowed interval that looks exactly
    like a confident result.
    """

    document = json.loads(index_path.read_text(encoding="utf-8"))
    violations = validate_formal_run_index(document)
    if violations:
        raise ValueError("formal run index is not valid: " + "; ".join(violations))

    for name in OUTPUT_NAMES:
        candidate = output_dir / f"{name}.json"
        if candidate.exists():
            raise FileExistsError(f"an analysis already exists: {candidate}")

    # SORTED into the declared order, not the order the index happens to list.
    # The run order reaches two published things if it is left alone. The order
    # the models first appear becomes the orientation of every pair, so the same
    # study would publish `A minus B` from one index and `B minus A` from
    # another; and the order of the seeds inside a model is the axis the seed
    # resample draws POSITIONS on, so the bounds move with it. The validator
    # requires all nine combinations, so this ordering is always total.
    runs = sorted(
        document["runs"],
        key=lambda entry: (
            APPROVED_MODELS.index(str(entry["model"])),
            APPROVED_SEEDS.index(int(entry["seed"])),
        ),
    )
    # SORTED, not the order the first run happens to list. Every replicate draws
    # POSITIONS on the image axis, so the axis order decides which images a given
    # seed selects; taking it from `runs[0]` would make the published bound depend
    # on how the index was assembled, while the index validator proves only that
    # the runs share the same SET of samples. Two indexes describing the identical
    # study would then publish different intervals.
    sample_ids = tuple(sorted(runs[0]["uncalibrated_sample_ids"]))
    num_classes = int(document["num_classes"])
    if num_classes > len(BDD100K_SEMANTIC_CLASS_NAMES):
        raise ValueError(
            f"the index declares {num_classes} classes but the BDD100K class table names "
            f"{len(BDD100K_SEMANTIC_CLASS_NAMES)}; a class without a name cannot be published"
        )
    class_names = list(BDD100K_SEMANTIC_CLASS_NAMES[:num_classes])
    critical_class_ids = tuple(int(value) for value in document["critical_class_ids"])
    protocol_sha256 = str(document["protocol_sha256"])
    dataset_manifest_sha256 = str(document["dataset_manifest_sha256"])

    models = tuple(dict.fromkeys(str(entry["model"]) for entry in runs))
    model_ids = tuple(models.index(str(entry["model"])) for entry in runs)
    index_dir = index_path.parent
    components, totals = _load_components(
        index_dir,
        runs,
        sample_ids,
        protocol_sha256=protocol_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    support_pixels, images_with_class = _ground_truth_support(
        components, runs, len(sample_ids), num_classes
    )

    table: dict[str, dict[str, float]] = {model: {} for model in models}
    for name in METRIC_NAMES:
        for model in models:
            values = [
                _metric_from_confusion(totals[position], name, critical_class_ids)
                for position, entry in enumerate(runs)
                if entry["model"] == model
            ]
            table[model][name] = float(np.mean(values))

    # Per-class values travel with their names and their support. A class the
    # cohort holds in seven images can legitimately score zero, and a reader who
    # cannot see the seven cannot tell that from a model that failed. The names
    # are the table the risk-profile schema pins classes to, so the two documents
    # can never disagree about what class 2 is called.
    by_model: dict[str, dict[str, list[float | None]]] = {}
    calibration: dict[str, dict[str, dict[str, Any]]] = {}
    for model in models:
        positions = [index for index, label in enumerate(model_ids) if models[label] == model]
        summaries = [summarize_confusion(totals[index]) for index in positions]
        by_model[model] = {
            "iou": _mean_over_seeds([summary.class_iou for summary in summaries]),
            "recall": _mean_over_seeds([summary.class_recall for summary in summaries]),
        }
        calibration[model] = {}
        for kind, key in (
            ("uncalibrated", "artifacts_dir"),
            ("calibrated", "calibrated_artifacts_dir"),
        ):
            finalised = [
                _accumulate_calibration(index_dir / str(runs[index][key]), sample_ids).finalised()
                for index in positions
            ]
            # The mean is published beside the values it is a mean of. Three seeds
            # that agree and three that disagree give the same mean, and whether a
            # calibration effect is one seed or all of them is the whole question.
            calibration[model][kind] = {
                **{
                    field: _mean_or_none([row[field] for row in finalised])
                    for field in ("ece", "brier")
                },
                "per_seed": {
                    str(runs[index]["seed"]): {"ece": row["ece"], "brier": row["brier"]}
                    for index, row in zip(positions, finalised, strict=True)
                },
            }
    per_class: dict[str, Any] = {
        "class_names": class_names,
        "support_pixels": support_pixels,
        "images_with_class": images_with_class,
        "by_model": by_model,
    }

    risk_profiles: dict[str, dict[str, Any]] = {}
    for profile_path in sorted(Path(risk_profiles_dir).glob("*.yaml")):
        profile = load_risk_profile(profile_path)
        risk_profiles[profile_path.stem] = {
            "sensitivity": profile.sensitivity,
            "critical_class_ids": list(profile.critical_class_ids),
            "cost_risk": {
                model: float(
                    np.mean(
                        [
                            compute_cost_risk(totals[index], profile)
                            for index, label in enumerate(model_ids)
                            if models[label] == model
                        ]
                    )
                )
                for model in models
            },
        }

    intervals: dict[str, dict[str, Any]] = {}
    for name in METRIC_NAMES:
        statistic = _statistic_for(name, num_classes, critical_class_ids)
        for left in range(len(models)):
            for right in range(left + 1, len(models)):
                selected = [
                    position for position, label in enumerate(model_ids) if label in (left, right)
                ]
                paired = components[selected]
                labels = tuple(0 if model_ids[p] == left else 1 for p in selected)
                signed = _signed_difference_statistic(statistic, labels)
                interval = two_stage_paired_bootstrap_statistic(
                    paired, labels, signed, combine="sum", resamples=resamples, seed=seed
                )
                key = f"{models[left]} minus {models[right]} ({name})"
                intervals[key] = {
                    "estimate": interval.estimate,
                    "low": interval.low,
                    "high": interval.high,
                    "confidence": interval.confidence,
                    "resamples": interval.resamples,
                    "seed": interval.seed,
                    "estimator": RATIO_ESTIMATOR,
                }

    # The report presents model-major, the ranking comparison consumes
    # metric-major. Both are derived from the same values.
    by_metric: dict[str, dict[str, float]] = {
        name: {model: table[model][name] for model in models} for name in METRIC_NAMES
    }
    comparisons = compare_rankings(by_metric, BASELINE_METRIC)
    # Whether two models can be told apart at all is a different question from
    # which is ahead, and the ranking comparison answers only the second. Each
    # pair's interval is copied from the intervals document, never recomputed, and
    # the flag is a plain reading of it. On the real cohort the two best models
    # are inseparable on mIoU and separable on VRU recall; a document that said
    # only "no reversal" would hide that.
    separability: dict[str, list[dict[str, Any]]] = {
        name: [
            {
                "left": models[left],
                "right": models[right],
                "estimate": entry["estimate"],
                "low": entry["low"],
                "high": entry["high"],
                "excludes_zero": _excludes_zero(entry["low"], entry["high"]),
            }
            for left in range(len(models))
            for right in range(left + 1, len(models))
            for entry in (intervals[f"{models[left]} minus {models[right]} ({name})"],)
        ]
        for name in METRIC_NAMES
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol_hash": protocol_sha256,
        "dataset_manifest_hash": dataset_manifest_sha256,
    }
    _write(
        output_dir / "metrics.json",
        {
            "schema_version": METRICS_SCHEMA_VERSION,
            **common,
            "cohort": str(document.get("cohort", "locked_validation")),
            "sample_count": len(sample_ids),
            "seed_count": len(runs) // len(models),
            "interval_method": (
                f"two-stage paired bootstrap over summed confusions, {resamples} "
                f"resamples, seed {seed}"
            ),
            "metrics": table,
            "per_class": per_class,
            "calibration": calibration,
            "risk_profiles": risk_profiles,
        },
    )
    _write(
        output_dir / "intervals.json",
        {"schema_version": INTERVALS_SCHEMA_VERSION, **common, "intervals": intervals},
    )
    _write(
        output_dir / "rankings.json",
        {
            "schema_version": RANKINGS_SCHEMA_VERSION,
            **common,
            "baseline_metric": BASELINE_METRIC,
            "separability": separability,
            "comparisons": [
                {
                    "metric_name": comparison.metric_name,
                    "baseline_order": list(comparison.baseline_order),
                    "comparison_order": list(comparison.comparison_order),
                    "reversal_observed": comparison.reversal_observed,
                }
                for comparison in comparisons
            ],
        },
    )
    return AggregateResult(
        metrics_path=output_dir / "metrics.json",
        intervals_path=output_dir / "intervals.json",
        rankings_path=output_dir / "rankings.json",
        models=models,
        sample_count=len(sample_ids),
    )


def _signed_difference_statistic(statistic, labels: tuple[int, ...]):
    """Orient the paired statistic so the interval is left-minus-right.

    The sign lives here and the combination lives in the bootstrap, and both
    halves are needed. This function makes the right model's runs negative; the
    bootstrap is then asked to SUM the two group means rather than average them,
    which is what turns `+A` and `-B` into `A - B`. Averaging them reports half
    the difference, which is the estimand this pairing published until P1-17.
    """

    signs = np.array([1.0 if label == 0 else -1.0 for label in labels], dtype=np.float64)

    def signed(summed: Float64Array) -> Float64Array:
        return statistic(summed) * signs

    return signed


def _serialised(document: dict[str, Any]) -> str:
    """The exact text a document is cited by: two-space indent, sorted keys, one trailing newline."""

    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _write(path: Path, document: dict[str, Any]) -> None:
    # Validated through the contract its version names BEFORE it is written, so a
    # shape regression fails here, in the producer, and never reaches a reader.
    path.write_text(_serialised(validated_document(document)), encoding="utf-8", newline="\n")
