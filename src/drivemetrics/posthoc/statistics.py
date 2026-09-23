"""The pre-registered statistics of the post-release resolution analysis.

Every level, threshold and seed in this module is fixed by
``docs/posthoc/resolution-v1/analysis-plan.md``, which was committed before any
result existed. None of them is a parameter of :func:`analyse_sweep`, so the
analysis cannot be tuned after the sweep has been seen.

The primary contrast per model is the seed-mean excess critical-miss rate of
small persons at the formal geometry over the native geometry, a ratio of sums
over the whole calibration cohort. Its interval comes from a paired bootstrap
that resamples images and holds the seeds fixed; the three models are corrected
with Bonferroni. Everything else in the summary is descriptive and decides
nothing.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from drivemetrics.artifacts.envelope import canonical_json_bytes
from drivemetrics.metrics.confusion import summarize_confusion
from drivemetrics.posthoc.resolution import (
    ANALYSIS_PLAN,
    PER_IMAGE_FILENAME,
    RUN_FILENAME,
    SESSIONS_FILENAME,
    SWEEP_SCHEMA_VERSION,
)
from drivemetrics.protocol.risk_profiles import BDD100K_SEMANTIC_CLASS_NAMES

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

SUMMARY_SCHEMA_VERSION = "driving-risk-resolution-summary/v1"
SUMMARY_FILENAME = "summary.json"
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260831
PRIMARY_CONFIDENCE = 1.0 - 0.05 / 3.0
DESCRIPTIVE_CONFIDENCE = 0.95
SUPPORT_THRESHOLD = 0.10
EQUIVALENCE_BOUND = 0.05
MIOU_DROP_LIMIT = 0.02
PERSON = BDD100K_SEMANTIC_CLASS_NAMES.index("person")
RIDER = BDD100K_SEMANTIC_CLASS_NAMES.index("rider")
MOTORCYCLE = BDD100K_SEMANTIC_CLASS_NAMES.index("motorcycle")
BICYCLE = BDD100K_SEMANTIC_CLASS_NAMES.index("bicycle")
#: The arm that isolates padding shares the formal scale, so it is not a step
#: on the dose-response axis.
NOT_ON_SCALE_AXIS = ("formal_unpadded",)


# ---------------------------------------------------------------------------
# Bootstrap, decisions and tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RatioBootstrap:
    """A ratio of sums with percentile intervals from one set of image draws."""

    estimate: float | None
    intervals: dict[float, tuple[float, float] | None]
    resamples: int
    seed: int
    undefined_resamples: int


def paired_ratio_bootstrap(
    numerator: Float64Array,
    denominator: Float64Array,
    *,
    confidences: Sequence[float] = (PRIMARY_CONFIDENCE, DESCRIPTIVE_CONFIDENCE),
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> RatioBootstrap:
    """Resample images with replacement and recompute sum(numerator) / sum(denominator).

    Both arrays hold one value per image. Every image takes part, so an image
    without an eligible instance adds zero to both sums. A resample whose
    denominator is zero has no ratio; it is dropped from the percentiles and
    counted, never treated as zero.
    """

    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise ValueError("numerator and denominator must have the same shape")
    if numerator.ndim != 1:
        raise ValueError("numerator and denominator must be one-dimensional, one value per image")
    if numerator.size == 0:
        raise ValueError("the bootstrap needs at least one image")
    if not (np.all(np.isfinite(numerator)) and np.all(np.isfinite(denominator))):
        raise ValueError("bootstrap inputs must be finite")
    if np.any(denominator < 0):
        raise ValueError("denominators must be nonnegative")

    images = numerator.size
    generator = np.random.default_rng(seed)
    weights = np.empty((resamples, images), dtype=np.float64)
    for index in range(resamples):
        weights[index] = np.bincount(generator.integers(0, images, size=images), minlength=images)
    sampled_numerator = weights @ numerator
    sampled_denominator = weights @ denominator
    defined = sampled_denominator > 0
    ratios = sampled_numerator[defined] / sampled_denominator[defined]

    intervals: dict[float, tuple[float, float] | None] = {}
    for confidence in confidences:
        if ratios.size == 0:
            intervals[confidence] = None
            continue
        tail = (1.0 - confidence) / 2.0
        low, high = np.quantile(ratios, [tail, 1.0 - tail])
        intervals[confidence] = (float(low), float(high))
    total = float(denominator.sum())
    return RatioBootstrap(
        estimate=float(numerator.sum()) / total if total > 0 else None,
        intervals=intervals,
        resamples=resamples,
        seed=seed,
        undefined_resamples=int(np.count_nonzero(~defined)),
    )


def classify_contrast(
    estimate: float | None,
    primary_interval: tuple[float, float] | None,
    *,
    confounded: bool,
) -> tuple[str, str | None]:
    """Apply the decision rules of §4.3 in the order they are listed.

    Returns the category and, when the scale-mismatch diagnostic of §4.4 turned
    a C into a D, the category it was downgraded from. The rules are checked in
    the listed order A, B, C, D and the first that holds applies.
    """

    if estimate is None or primary_interval is None:
        return "D", None
    low, high = primary_interval
    if low > 0 and estimate >= SUPPORT_THRESHOLD:
        return "A", None
    if low > 0:
        return "B", None
    if low > -EQUIVALENCE_BOUND and high < EQUIVALENCE_BOUND:
        return ("D", "C") if confounded else ("C", None)
    return "D", None


def overall_conclusion(categories: Mapping[str, str]) -> str:
    """At least two models in one category make an overall conclusion."""

    category, count = Counter(categories.values()).most_common(1)[0]
    return category if count >= 2 else "model-dependent"


def exact_mcnemar(formal_only: int, native_only: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs."""

    from scipy.stats import binomtest

    discordant = formal_only + native_only
    if discordant == 0:
        return 1.0
    return float(binomtest(formal_only, discordant, 0.5).pvalue)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""

    count = len(p_values)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(sorted(range(count), key=lambda i: p_values[i])):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


# ---------------------------------------------------------------------------
# Reading a finished sweep
# ---------------------------------------------------------------------------


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@dataclass(frozen=True)
class _Sweep:
    configuration: dict[str, Any]
    records: list[dict[str, Any]]
    sessions: list[dict[str, Any]]
    arms: list[str]
    scales: dict[str, float]
    models: dict[str, list[str]]


def _load_sweep(output_dir: Path) -> _Sweep:
    configuration = json.loads((output_dir / RUN_FILENAME).read_text(encoding="utf-8"))
    if configuration.get("schema_version") != SWEEP_SCHEMA_VERSION:
        raise ValueError(f"{output_dir} does not hold a resolution sweep")
    records = _read_json_lines(output_dir / PER_IMAGE_FILENAME)
    expected = list(configuration["sample_ids"])
    scored = [record["sample_id"] for record in records]
    if sorted(scored) != sorted(expected):
        raise ValueError(
            f"the sweep is incomplete: {len(set(scored))} distinct of {len(expected)} images "
            f"in {len(scored)} lines; the pre-registered analysis needs the whole cohort once"
        )
    order = {sample_id: position for position, sample_id in enumerate(expected)}
    records.sort(key=lambda record: order[record["sample_id"]])

    sessions = _read_json_lines(output_dir / SESSIONS_FILENAME)
    commits = {session["provenance"]["commit"] for session in sessions}
    locks = {session["provenance"]["lock_sha256"] for session in sessions}
    if len(commits) != 1 or len(locks) != 1:
        raise ValueError(
            "every sweep session must share one commit and one lock file; "
            f"found commits {sorted(commits)} and locks {sorted(locks)}"
        )

    runs = sorted(configuration["runs"], key=lambda run: (run["model"], int(run["seed"])))
    models: dict[str, list[str]] = {}
    for run in runs:
        models.setdefault(run["model"], []).append(run["run_id"])
    return _Sweep(
        configuration=configuration,
        records=records,
        sessions=sessions,
        arms=[arm["name"] for arm in configuration["arms"]],
        scales={arm["name"]: float(arm["scale"]) for arm in configuration["arms"]},
        models=models,
    )


class _Cohort:
    """Per-image counts over one instance selection."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = records

    def select(self, class_ids: Iterable[int], tertile: str | None) -> list[BoolArray]:
        wanted = set(class_ids)
        return [
            np.array(
                [
                    entry["class_id"] in wanted
                    and (tertile is None or entry["area_tertile"] == tertile)
                    for entry in record["instances"]
                ],
                dtype=bool,
            )
            for record in self.records
        ]

    def instances(self, selection: Sequence[BoolArray]) -> Int64Array:
        return np.array([int(mask.sum()) for mask in selection], dtype=np.int64)

    def missed(self, selection: Sequence[BoolArray], arm: str, run_id: str) -> list[BoolArray]:
        return [
            np.array(record["results"][arm][run_id]["critical_miss"], dtype=bool)[mask]
            for record, mask in zip(self.records, selection, strict=True)
        ]

    def misses(
        self, selection: Sequence[BoolArray], arm: str, run_ids: Sequence[str]
    ) -> Int64Array:
        """Critical misses per image (rows) and run (columns)."""

        return np.array(
            [
                [int(missed.sum()) for missed in self.missed(selection, arm, run_id)]
                for run_id in run_ids
            ],
            dtype=np.int64,
        ).T.reshape(len(self.records), len(run_ids))

    def confusion(self, arm: str, run_id: str) -> Int64Array:
        total = np.zeros(len(BDD100K_SEMANTIC_CLASS_NAMES) ** 2, dtype=np.int64)
        for record in self.records:
            total += np.asarray(record["results"][arm][run_id]["confusion"], dtype=np.int64)
        size = len(BDD100K_SEMANTIC_CLASS_NAMES)
        return total.reshape(size, size)


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def _interval_document(result: RatioBootstrap) -> dict[str, Any]:
    return {
        "estimate": result.estimate,
        "intervals": [
            None
            if interval is None
            else {"confidence": confidence, "low": interval[0], "high": interval[1]}
            for confidence, interval in result.intervals.items()
        ],
        "resamples": result.resamples,
        "seed": result.seed,
        "undefined_resamples": result.undefined_resamples,
    }


def _contrast(
    cohort: _Cohort,
    selection: Sequence[BoolArray],
    first: str,
    second: str,
    run_ids: Sequence[str],
) -> tuple[RatioBootstrap, dict[str, Any]]:
    """Seed-mean miss-rate difference ``first - second`` over one instance selection."""

    instances = cohort.instances(selection)
    first_misses = cohort.misses(selection, first, run_ids)
    second_misses = cohort.misses(selection, second, run_ids)
    seeds = len(run_ids)
    result = paired_ratio_bootstrap(
        (first_misses - second_misses).sum(axis=1).astype(np.float64),
        instances.astype(np.float64) * seeds,
    )
    total = int(instances.sum())
    document = _interval_document(result)
    document["per_seed"] = [
        (int(first_misses[:, column].sum()) - int(second_misses[:, column].sum())) / total
        if total
        else None
        for column in range(seeds)
    ]
    document["instances"] = total
    return result, document


def _rate(misses: int, instances: int) -> float | None:
    return misses / instances if instances else None


def _iou_block(cohort: _Cohort, arm: str, run_ids: Sequence[str]) -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    for run_id in run_ids:
        metrics = summarize_confusion(cohort.confusion(arm, run_id))
        per_run[run_id] = {
            "mean_iou": metrics.mean_iou,
            "class_iou": dict(zip(BDD100K_SEMANTIC_CLASS_NAMES, metrics.class_iou, strict=True)),
        }
    class_means: dict[str, float | None] = {}
    for name in BDD100K_SEMANTIC_CLASS_NAMES:
        values = [per_run[run_id]["class_iou"][name] for run_id in run_ids]
        defined = [value for value in values if value is not None]
        class_means[name] = float(np.mean(defined)) if defined else None
    return {
        "seed_mean_miou": float(np.mean([per_run[run_id]["mean_iou"] for run_id in run_ids])),
        "seed_mean_class_iou": class_means,
        "per_run": per_run,
    }


def _input_area_bins(
    cohort: _Cohort, arm: str, scale: float, run_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Person instances in power-of-two bins of model-input area, pooled over seeds."""

    tallies: dict[int, list[int]] = {}
    for record in cohort.records:
        for position, entry in enumerate(record["instances"]):
            if entry["class_id"] != PERSON:
                continue
            exponent = math.floor(math.log2(entry["area_pixels"] * scale * scale))
            tally = tallies.setdefault(exponent, [0, 0])
            for run_id in run_ids:
                tally[0] += 1
                tally[1] += int(record["results"][arm][run_id]["critical_miss"][position])
    return [
        {
            "low": 2.0**exponent,
            "high": 2.0 ** (exponent + 1),
            "instances": instances,
            "critical_misses": misses,
            "miss_rate": misses / instances,
        }
        for exponent, (instances, misses) in sorted(tallies.items())
    ]


def _small_counts(cohort: _Cohort, class_id: int, sweep: _Sweep) -> dict[str, Any]:
    selection = cohort.select([class_id], "small")
    run_ids = [run_id for ids in sweep.models.values() for run_id in ids]
    return {
        "instances": int(cohort.instances(selection).sum()),
        "critical_misses": {
            arm: {run_id: int(cohort.misses(selection, arm, [run_id]).sum()) for run_id in run_ids}
            for arm in sweep.arms
        },
    }


@dataclass(frozen=True)
class AnalysisResult:
    """Where the summary was written and what the decision rules concluded."""

    summary_path: Path
    categories: dict[str, str]
    overall: str


def analyse_sweep(output_dir: Path) -> AnalysisResult:
    """Compute the pre-registered statistics from a finished sweep and write ``summary.json``."""

    sweep = _load_sweep(output_dir)
    cohort = _Cohort(sweep.records)
    small_person = cohort.select([PERSON], "small")
    small_person_counts = cohort.instances(small_person)
    if int(small_person_counts.sum()) == 0:
        raise ValueError("the cohort holds no small-tertile person; there is nothing to analyse")
    large_person = cohort.select([PERSON], "large")
    small_rider = cohort.select([RIDER], "small")
    small_person_and_rider = cohort.select([PERSON, RIDER], "small")
    total_small_persons = int(small_person_counts.sum())

    primary: dict[str, Any] = {}
    categories: dict[str, str] = {}
    mcnemar: list[dict[str, Any]] = []
    secondary: dict[str, Any] = {
        "dose_response": {},
        "small_rider": {},
        "small_person_and_rider": {},
        "input_area_bins": {},
        "residual_native_miss_rate": {},
        "iou": {},
    }
    scale_axis = sorted(
        (arm for arm in sweep.arms if arm not in NOT_ON_SCALE_AXIS),
        key=lambda arm: sweep.scales[arm],
    )
    for model, run_ids in sweep.models.items():
        seeds = len(run_ids)
        main, main_document = _contrast(cohort, small_person, "formal", "native", run_ids)
        large, large_document = _contrast(cohort, large_person, "formal", "native", run_ids)
        iou = {arm: _iou_block(cohort, arm, run_ids) for arm in sweep.arms}
        miou_drop = iou["formal"]["seed_mean_miou"] - iou["native"]["seed_mean_miou"]
        large_interval = large.intervals[DESCRIPTIVE_CONFIDENCE]
        drop_exceeds = miou_drop > MIOU_DROP_LIMIT
        large_worse = large_interval is not None and large_interval[1] < 0
        category, downgraded_from = classify_contrast(
            main.estimate,
            main.intervals[PRIMARY_CONFIDENCE],
            confounded=drop_exceeds or large_worse,
        )
        categories[model] = category
        misses_by_arm = {arm: cohort.misses(small_person, arm, run_ids) for arm in sweep.arms}
        primary[model] = {
            "runs": list(run_ids),
            "category": category,
            "downgraded_from": downgraded_from,
            "formal_minus_native": main_document,
            "large_person_formal_minus_native": large_document,
            "scale_mismatch": {
                "miou_drop_formal_minus_native": miou_drop,
                "miou_drop_exceeds_limit": drop_exceeds,
                "large_person_native_worse": large_worse,
                "confounded": drop_exceeds or large_worse,
            },
            "formal_minus_s085": (
                _contrast(cohort, small_person, "formal", "s085", run_ids)[1]
                if "s085" in sweep.arms
                else None
            ),
            "critical_misses": {
                arm: {run_id: int(misses[:, column].sum()) for column, run_id in enumerate(run_ids)}
                for arm, misses in misses_by_arm.items()
            },
        }

        for run_id in run_ids:
            formal = np.concatenate(cohort.missed(small_person, "formal", run_id))
            native = np.concatenate(cohort.missed(small_person, "native", run_id))
            formal_only = int(np.count_nonzero(formal & ~native))
            native_only = int(np.count_nonzero(native & ~formal))
            mcnemar.append(
                {
                    "run_id": run_id,
                    "formal_only_misses": formal_only,
                    "native_only_misses": native_only,
                    "p_value": exact_mcnemar(formal_only, native_only),
                }
            )

        steps: list[dict[str, Any]] = [
            {
                "arm": arm,
                "scale": sweep.scales[arm],
                "miss_rate": int(misses_by_arm[arm].sum()) / (seeds * total_small_persons),
            }
            for arm in scale_axis
        ]
        secondary["dose_response"][model] = {
            "steps": steps,
            "monotone_non_increasing": all(
                later["miss_rate"] <= earlier["miss_rate"] for earlier, later in pairwise(steps)
            ),
        }
        secondary["small_rider"][model] = _contrast(
            cohort, small_rider, "formal", "native", run_ids
        )[1]
        secondary["small_person_and_rider"][model] = _contrast(
            cohort, small_person_and_rider, "formal", "native", run_ids
        )[1]
        secondary["input_area_bins"][model] = {
            arm: _input_area_bins(cohort, arm, sweep.scales[arm], run_ids) for arm in sweep.arms
        }
        native_misses = misses_by_arm["native"]
        secondary["residual_native_miss_rate"][model] = {
            "seed_mean": int(native_misses.sum()) / (seeds * total_small_persons),
            "per_seed": [
                _rate(int(native_misses[:, column].sum()), total_small_persons)
                for column in range(seeds)
            ],
        }
        secondary["iou"][model] = iou

    adjusted = holm_adjust([entry["p_value"] for entry in mcnemar])
    for entry, value in zip(mcnemar, adjusted, strict=True):
        entry["holm_adjusted_p_value"] = value
    secondary["mcnemar_formal_vs_native"] = mcnemar
    secondary["small_counts"] = {
        "motorcycle": _small_counts(cohort, MOTORCYCLE, sweep),
        "bicycle": _small_counts(cohort, BICYCLE, sweep),
    }

    overall = overall_conclusion(categories)
    configuration = sweep.configuration
    pairs = sorted([record["sample_id"], record["bitmask_sha256"]] for record in sweep.records)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "analysis_plan": ANALYSIS_PLAN,
        "status": (
            "post-release analysis on the calibration split, not the locked cohort; "
            "pre-registered decision rules"
        ),
        "settings": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_confidence": PRIMARY_CONFIDENCE,
            "descriptive_confidence": DESCRIPTIVE_CONFIDENCE,
            "support_threshold": SUPPORT_THRESHOLD,
            "equivalence_bound": EQUIVALENCE_BOUND,
            "miou_drop_limit": MIOU_DROP_LIMIT,
        },
        "sweep": {
            "protocol_sha256": configuration["protocol_sha256"],
            "dataset_manifest_sha256": configuration["dataset_manifest_sha256"],
            "tertiles_sha256": configuration["tertiles_sha256"],
            "arms": sweep.arms,
            "runs": configuration["runs"],
            "commit": sweep.sessions[0]["provenance"]["commit"],
            "lock_sha256": sweep.sessions[0]["provenance"]["lock_sha256"],
            "hardware": [
                json.loads(entry)
                for entry in sorted(
                    {
                        json.dumps(s["provenance"]["hardware"], sort_keys=True)
                        for s in sweep.sessions
                    }
                )
            ],
            "parity_criteria": sorted({session["parity_criterion"] for session in sweep.sessions}),
            "bitmask_set_sha256": _sha256_hex(canonical_json_bytes(pairs)),
        },
        "counts": {
            "images": len(sweep.records),
            "small_person_instances": total_small_persons,
            "images_with_small_person": int(np.count_nonzero(small_person_counts)),
        },
        "primary": primary,
        "overall": overall,
        "secondary": secondary,
    }
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return AnalysisResult(summary_path=summary_path, categories=categories, overall=overall)


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
