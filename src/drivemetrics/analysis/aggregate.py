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
from drivemetrics.artifacts.formal_set import (
    APPROVED_MODELS,
    APPROVED_SEEDS,
    validate_formal_run_index,
)
from drivemetrics.artifacts.predictions import read_prediction_artifact
from drivemetrics.metrics.confusion import summarize_confusion
from drivemetrics.metrics.risk import critical_false_negative_rate

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

OUTPUT_NAMES: tuple[str, ...] = ("metrics", "intervals", "rankings")
BASELINE_METRIC = "miou"
RATIO_ESTIMATOR = "ratio_of_sums"
BOOTSTRAP_SEED = 20260831

#: Every reported metric, oriented so that larger is always better. Ranking
#: comparison depends on that orientation, and a metric left in its natural
#: lower-is-better form would invert its own ranking silently.
METRIC_NAMES: tuple[str, ...] = ("miou", "pixel_accuracy", "critical_recall")


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


def _statistic_for(name: str, num_classes: int, critical_class_ids: tuple[int, ...]):
    """Build the run-wise statistic that recomputes one metric from summed components."""

    def statistic(summed: Float64Array) -> Float64Array:
        matrices = np.rint(summed).astype(np.int64).reshape(-1, num_classes, num_classes)
        return np.array(
            [_metric_from_confusion(matrix, name, critical_class_ids) for matrix in matrices],
            dtype=np.float64,
        )

    return statistic


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
    critical_class_ids = tuple(int(value) for value in document["critical_class_ids"])
    protocol_sha256 = str(document["protocol_sha256"])
    dataset_manifest_sha256 = str(document["dataset_manifest_sha256"])

    models = tuple(dict.fromkeys(str(entry["model"]) for entry in runs))
    model_ids = tuple(models.index(str(entry["model"])) for entry in runs)
    components, totals = _load_components(
        index_path.parent,
        runs,
        sample_ids,
        protocol_sha256=protocol_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol_hash": protocol_sha256,
        "dataset_manifest_hash": dataset_manifest_sha256,
    }
    _write(
        output_dir / "metrics.json",
        {
            **common,
            "cohort": str(document.get("cohort", "locked_validation")),
            "sample_count": len(sample_ids),
            "seed_count": len(runs) // len(models),
            "interval_method": (
                f"two-stage paired bootstrap over summed confusions, {resamples} "
                f"resamples, seed {seed}"
            ),
            "metrics": table,
        },
    )
    _write(output_dir / "intervals.json", {**common, "intervals": intervals})
    _write(
        output_dir / "rankings.json",
        {
            **common,
            "baseline_metric": BASELINE_METRIC,
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


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
