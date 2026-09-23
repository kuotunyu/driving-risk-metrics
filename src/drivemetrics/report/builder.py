"""Claim-safe static report generation from verified claims and frozen artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from drivemetrics import __version__
from drivemetrics.analysis.claims import ClaimV1, audit_claims, claim_value, verified_claims
from drivemetrics.artifacts.formal_set import APPROVED_SEEDS
from drivemetrics.report.figures import bar_figure, interval_figure

REPORT_INPUTS: tuple[str, ...] = (
    "metrics",
    "intervals",
    "rankings",
    "extended-metrics",
    "gallery-manifest",
    "formal_run_index",
)

#: A class seen in fewer images than this is marked thin in the per-class table. The
#: mark states the count rather than a judgement: the reader is told how little the
#: row rests on and draws their own conclusion.
THIN_CLASS_IMAGE_COUNT: int = 50

BAND_ORDER: tuple[str, ...] = ("top", "middle", "bottom")
TERTILE_ORDER: tuple[str, ...] = ("small", "medium", "large")
#: Indexed by ``calibrated > uncalibrated``, so the direction is read from the numbers
#: rather than asserted, and no branch decides it.
CALIBRATION_DIRECTION: tuple[str, str] = ("lowered", "raised")

LIMITATIONS: tuple[str, ...] = (
    "Every number on this page comes from one frozen protocol, one frozen cohort,"
    " and the exact artifact hashes shown above.",
    "Results describe the models trained under this protocol only. They are not a"
    " statement about these architectures in general.",
    "A ranking reversal is an observation about this cohort and these metrics. It is"
    " never a success criterion, and a stable ranking is reported just as plainly.",
    "Image-band results describe normalized image regions, not physical distance or depth.",
    "Confidence intervals come from the paired bootstrap named above. They are not"
    " hypothesis tests and are not a substitute for effect sizes.",
)

#: How the page names each approved model: the names the claims and the README use.
#: The committed SVG figures keep their own shorter labels, which are frozen bytes.
MODEL_NAMES: dict[str, str] = {
    "segformer_b2": "SegFormer-B2",
    "upernet_convnextv2_tiny": "UperNet-ConvNeXtV2-Tiny",
    "upernet_dinov2_small": "UperNet-DINOv2-Small",
}
#: How the page names each metric, in the order the headline table lists them.
METRIC_LABELS: dict[str, str] = {
    "miou": "mean IoU",
    "critical_recall": "critical-class recall",
    "pixel_accuracy": "pixel accuracy",
}

#: The paired intervals the page opens with, as (claim, metric). They are the
#: intervals the README's "At a glance" list quotes, rounded the same way.
HEADLINE_INTERVAL_CLAIMS: tuple[tuple[str, str], ...] = (
    ("p1.interval.miou.segformer-minus-convnextv2", "miou"),
    ("p1.interval.critical-recall.segformer-minus-convnextv2", "critical_recall"),
)
#: The instance counts the page opens with, quoted as their verified claim text.
HEADLINE_COUNT_CLAIMS: tuple[str, ...] = (
    "p1.instances.convnextv2.person-small",
    "p1.instances.convnextv2.rider-small",
    "p1.instances.convnextv2.car-small",
)
#: Every displayed rounding uses the claims validator's rule, so a value on this page
#: reads exactly as the same value in the README.
DISPLAY_QUANTUM = Decimal("0.001")

#: The evidence figures the page opens with, in reading order, each with the text a
#: screen reader announces. All are drawn from the evidence by ``svg.write_figures``.
CURATED_FIGURES: tuple[tuple[str, str], ...] = (
    (
        "headline-top-two",
        "Top two models: paired differences with bootstrap intervals on mean IoU and"
        " critical-class recall, drawn from rankings.json",
    ),
    (
        "small-tertile-critical-misses",
        "Critical misses on the smallest-tertile instances by class, one training seed per"
        " model, drawn from extended-metrics.json",
    ),
    (
        "miou-gap-by-class",
        "Per-class contribution to the mean IoU difference between the top two models,"
        " critical classes highlighted, drawn from metrics.json",
    ),
)
CURATED_FIGURE_DIR = "figures-svg"
NOT_DRAWN = (
    "The curated figures are not drawn: they need the instance coverage and the paired"
    " intervals between the top two models, and this analysis run did not publish both."
)

_INTERVAL_KEY = re.compile(r"(?P<left>\S+) minus (?P<right>\S+) \((?P<metric>[^)]+)\)")


@dataclass(frozen=True)
class ReportResult:
    """Where the published page, its machine-readable figures and its SVGs were written."""

    index_path: Path
    figure_paths: tuple[Path, ...]
    claim_count: int
    svg_paths: tuple[Path, ...] = ()


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"report input is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"report input must be a JSON object: {path}")
    return document


def model_name(model: str) -> str:
    """Return the page name of a model, or the model's own name when it has none."""

    return MODEL_NAMES.get(model, model)


def metric_label(metric: str) -> str:
    """Return the page name of a metric, or the metric's own name when it has none."""

    return METRIC_LABELS.get(metric, metric)


def _leading(label: str) -> str:
    """Capitalise only the first letter, so "mean IoU" leads a sentence as "Mean IoU"."""

    return label[:1].upper() + label[1:]


def display_value(value: float) -> dict[str, str]:
    """Return a value rounded to three decimals beside the full value it was rounded from."""

    return {
        "text": f"{Decimal(str(value)).quantize(DISPLAY_QUANTUM):.3f}",
        "full": str(value),
    }


def _model_order(rankings: dict[str, Any], models: Iterable[str]) -> tuple[list[str], str]:
    """Return the models best first under the baseline metric, and how they were ordered.

    Without a ranking there is no best, so the models are listed by name.
    """

    comparisons = rankings["comparisons"]
    if comparisons:
        return (
            [str(model) for model in comparisons[0]["baseline_order"]],
            f"ordered by {metric_label(str(rankings['baseline_metric']))}, best first",
        )
    return sorted(models), "ordered by name"


def _interval_label(key: str) -> str:
    """Rename the models and the metric in an interval key such as ``a minus b (miou)``."""

    match = _INTERVAL_KEY.fullmatch(key)
    if match is None:
        return key
    return (
        f"{model_name(match['left'])} minus {model_name(match['right'])}"
        f" ({metric_label(match['metric'])})"
    )


def _headline_view(claims: Iterable[ClaimV1], repository_root: Path) -> dict[str, list[Any]]:
    """Return the opening statements whose claims this registry has verified.

    An interval is stated from the values its claim points at, rounded to three
    decimals, and says whether it excludes zero because the artifact says so. A
    count is quoted as its claim text. A claim this registry does not hold, or has
    not verified, is left out rather than stated without its evidence.
    """

    by_id = {claim.claim_id: claim for claim in claims}
    intervals: list[dict[str, Any]] = []
    for claim_id, metric in HEADLINE_INTERVAL_CLAIMS:
        if claim_id not in by_id:
            continue
        entry: Any = claim_value(by_id[claim_id], repository_root)
        label = metric_label(metric)
        intervals.append(
            {
                "label": _leading(label),
                "left": model_name(str(entry["left"])),
                "right": model_name(str(entry["right"])),
                "estimate": display_value(entry["estimate"]),
                "low": display_value(entry["low"]),
                "high": display_value(entry["high"]),
                "verdict": "excludes zero" if bool(entry["excludes_zero"]) else "includes zero",
            }
        )
    counts = [by_id[claim_id].text for claim_id in HEADLINE_COUNT_CLAIMS if claim_id in by_id]
    return {"intervals": intervals, "counts": counts}


def _headline_table(
    metric_table: Mapping[str, Mapping[str, float]], order: list[str]
) -> dict[str, Any]:
    """Return every model's metrics rounded to three decimals, best model first."""

    positions = {name: index for index, name in enumerate(METRIC_LABELS)}
    names = sorted(
        {name for scores in metric_table.values() for name in scores},
        key=lambda name: (positions.get(name, len(positions)), name),
    )
    return {
        "columns": [metric_label(name) for name in names],
        "rows": [
            {
                "model": model_name(model),
                "cells": [display_value(metric_table[model][name]) for name in names],
            }
            for model in order
        ],
    }


def _curated_view(
    artifacts_dir: Path,
    output_dir: Path,
    *,
    metrics: dict[str, Any],
    rankings: dict[str, Any],
    extended: dict[str, Any],
) -> tuple[list[dict[str, str]], tuple[Path, ...]]:
    """Draw the evidence SVGs beside the page when the evidence supports every one.

    The figures need the instance coverage and the paired intervals between the top
    two models. When either is missing none is drawn and the page says so, rather
    than failing or publishing a figure of nothing.
    """

    supported = (
        "not_computed" not in extended["instances"]
        and "separability" in rankings
        and bool(rankings["comparisons"])
    )
    if not supported:
        return [], ()

    # Imported here because the SVG module reads its inputs through this module.
    from drivemetrics.report import svg

    written = svg.write_figures(artifacts_dir, output_dir / CURATED_FIGURE_DIR)
    first, second = (
        model_name(str(model)) for model in rankings["comparisons"][0]["baseline_order"][:2]
    )
    captions = {
        "headline-top-two": (
            f"Paired differences between the top two models, {first} and {second}, on mean"
            " IoU and critical-class recall, with intervals from the"
            f" {metrics['interval_method']}. A filled marker means the interval excludes zero."
        ),
        "small-tertile-critical-misses": (
            "The share of each class's smallest-tertile instances that each model critically"
            " misses, labelled with the exact counts. Instance counts come from one training"
            f" seed per model (seed {APPROVED_SEEDS[0]}); this chart draws no interval."
        ),
        "miou-gap-by-class": (
            f"The mean IoU difference between {first} and {second}, split into one"
            " contribution per class, with the vulnerable-road-user classes highlighted."
            " Seed-averaged point estimates from metrics.json; no per-class interval is drawn."
        ),
    }
    figures: list[dict[str, str]] = []
    for name, alt in CURATED_FIGURES:
        path = output_dir / CURATED_FIGURE_DIR / f"{name}.svg"
        root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
        figures.append(
            {
                "src": f"{CURATED_FIGURE_DIR}/{name}.svg",
                "alt": alt,
                "caption": captions[name],
                "width": root.attrib["width"],
                "height": root.attrib["height"],
            }
        )
    return figures, written.figure_paths


def _figure_html(name: str, figure: dict[str, Any], *, include_library: bool) -> str:
    import plotly.io

    return str(
        plotly.io.to_html(
            figure,
            include_plotlyjs="cdn" if include_library else False,
            full_html=False,
            div_id=f"figure-{name}",
        )
    )


def _per_class_view(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return every class beside the support it rests on.

    A per-class score published without its support invites a conclusion the data
    cannot carry: an IoU of 0.0 on a class present in seven images is a statement
    about the cohort, not about the model.
    """

    block = metrics["per_class"]
    models = sorted(block["by_model"])
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(block["class_names"]):
        images = block["images_with_class"][index]
        rows.append(
            {
                "name": name,
                "support_pixels": block["support_pixels"][index],
                "images_with_class": images,
                "thin": images < THIN_CLASS_IMAGE_COUNT,
                "scores": [
                    {
                        "model": model,
                        "iou": block["by_model"][model]["iou"][index],
                        "recall": block["by_model"][model]["recall"][index],
                    }
                    for model in models
                ],
            }
        )
    return {"models": models, "rows": rows}


def _calibration_view(metrics: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return pooled and per-seed calibration, because the pooled value can mislead."""

    rows: list[dict[str, Any]] = []
    for model in sorted(metrics["calibration"]):
        block = metrics["calibration"][model]
        seeds = sorted(block["calibrated"]["per_seed"], key=int)
        rows.append(
            {
                "model": model,
                "uncalibrated": block["uncalibrated"],
                "calibrated": block["calibrated"],
                "direction": CALIBRATION_DIRECTION[
                    block["calibrated"]["ece"] > block["uncalibrated"]["ece"]
                ],
                "per_seed": [
                    {
                        "seed": seed,
                        "uncalibrated_ece": block["uncalibrated"]["per_seed"][seed]["ece"],
                        "calibrated_ece": block["calibrated"]["per_seed"][seed]["ece"],
                    }
                    for seed in seeds
                ],
            }
        )
    return tuple(rows)


def _risk_profile_view(metrics: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return each profile beside the classes it weights; the weights are the meaning."""

    rows: list[dict[str, Any]] = []
    for name in sorted(metrics["risk_profiles"]):
        profile = metrics["risk_profiles"][name]
        rows.append(
            {
                "name": name,
                "critical_class_ids": profile["critical_class_ids"],
                "sensitivity": profile["sensitivity"],
                "cost_risk": [
                    {"model": model, "value": profile["cost_risk"][model]}
                    for model in sorted(profile["cost_risk"])
                ],
            }
        )
    return tuple(rows)


def _band_view(extended: dict[str, Any]) -> dict[str, Any]:
    """Return the image bands with the definition that stops them reading as depth."""

    block = extended["normalized_image_bands"]
    if "not_computed" in block:
        return {"not_computed": block["not_computed"]}
    return {
        "definition": block["definition"],
        "rows": [
            {
                "model": model,
                "bands": [{"band": band, **block["by_model"][model][band]} for band in BAND_ORDER],
            }
            for model in sorted(block["by_model"])
        ],
    }


def _instance_view(extended: dict[str, Any]) -> dict[str, Any]:
    """Return instance coverage by class and by tertile, never pooled alone.

    Pooled coverage cannot answer the question this study exists to ask, because
    most instances are cars and the classes that matter for safety are rare.

    When the analysis had no instance ground truth it records why, and that reason
    is passed through so the page states it rather than rendering an empty table.
    """

    blocks = extended["instances"]
    if "not_computed" in blocks:
        return {"not_computed": blocks["not_computed"], "rows": []}
    rows: list[dict[str, Any]] = []
    for model in sorted(blocks):
        block = blocks[model]
        rows.append(
            {
                "model": model,
                "instance_count": block["instance_count"],
                "excluded_without_semantic_pixels": block["excluded_without_semantic_pixels"],
                "mean_corroborated_fraction": block["mean_corroborated_fraction"],
                "tertile_edges_sha256": block["tertile_edges_sha256"],
                "by_tertile": [
                    {"tertile": tertile, **block["by_tertile"][tertile]}
                    for tertile in TERTILE_ORDER
                ],
                "by_class": [
                    {
                        "class_name": class_name,
                        "instance_count": entry["instance_count"],
                        "critical_misses": entry["critical_misses"],
                        "mean_correct_fraction": entry["mean_correct_fraction"],
                        "by_tertile": [
                            {"tertile": tertile, **entry["by_tertile"][tertile]}
                            for tertile in TERTILE_ORDER
                        ],
                    }
                    for class_name, entry in sorted(block["by_class"].items())
                ],
            }
        )
    return {"rows": rows}


def _selective_risk_view(extended: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return selective risk carrying where its curve is defined.

    An area under a curve evaluated only at confidence-bin boundaries is not the
    area under a continuous curve, and publishing the number without that fact
    would invite comparison with figures computed another way.
    """

    return tuple(
        {
            "model": model,
            "uncalibrated": extended["selective_risk"][model]["uncalibrated"],
            "calibrated": extended["selective_risk"][model]["calibrated"],
        }
        for model in sorted(extended["selective_risk"])
    )


def _gallery_view(gallery: dict[str, Any]) -> dict[str, Any]:
    """Return the gallery as sample identifiers and per-seed scores, never as images.

    The release redistributes no BDD100K pixels. A reader who holds the dataset can
    resolve every identifier; one who does not still sees which samples were chosen
    and by what rule.
    """

    rows: list[dict[str, Any]] = []
    for model in sorted(gallery["per_model"]):
        block = gallery["per_model"][model]
        seeds = sorted(
            {
                seed
                for group in ("best", "worst")
                for entry in block[group]
                for seed in entry["per_seed"]
            },
            key=int,
        )
        rows.append(
            {"model": model, "seeds": seeds, "best": block["best"], "worst": block["worst"]}
        )
    return {"rule": gallery["rule"], "evaluation": gallery["evaluation"], "rows": rows}


def _run_view(index: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return the runs behind every aggregate, without their per-sample identifier lists."""

    fields = (
        "run_id",
        "model",
        "seed",
        "status",
        "final_step",
        "temperature",
        "checkpoint_sha256",
    )
    return tuple(
        {field: run[field] for field in fields}
        for run in sorted(index["runs"], key=lambda run: str(run["run_id"]))
    )


def build_report(
    claims_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    repository_root: Path,
) -> ReportResult:
    """Render the static evidence report from verified claims and frozen artifacts.

    The claim audit runs first and a single violation refuses to publish, because
    rendering a number that cannot be traced to its artifact is exactly the
    failure this project exists to prevent. Only claims marked verified reach the
    page, every claim is shown beside its evidence type, and the cohort, seed
    count and interval method are stated at the top so no chart can be read out of
    context. The page then leads with the findings and the evidence figures; the
    protocol and dataset manifest hashes and the full claims table follow the
    detailed results, collapsed but complete.
    """

    violations = audit_claims(claims_path, repository_root)
    if violations:
        raise ValueError("claim audit failed: " + "; ".join(violations))

    documents = {name: load_json_object(artifacts_dir / f"{name}.json") for name in REPORT_INPUTS}
    claims = verified_claims(claims_path)
    metrics = documents["metrics"]
    rankings = documents["rankings"]
    extended = documents["extended-metrics"]
    metric_table: dict[str, dict[str, float]] = metrics["metrics"]
    metric_names = sorted({name for scores in metric_table.values() for name in scores})
    order, order_note = _model_order(rankings, metric_table)
    category_order = [model_name(model) for model in order]

    figures: dict[str, dict[str, Any]] = {}
    captions: dict[str, str] = {}
    for name in metric_names:
        label = metric_label(name)
        figure = bar_figure(
            f"{_leading(label)} by model",
            label,
            {model_name(model): scores[name] for model, scores in metric_table.items()},
        )
        figure["layout"]["xaxis"].update(categoryorder="array", categoryarray=category_order)
        figures[name] = figure
        captions[name] = (
            f"{_leading(label)} of each model on the {metrics['cohort']} cohort,"
            f" {metrics['sample_count']} samples, the mean over {metrics['seed_count']} seeds;"
            f" models {order_note}. Each bar is a point estimate and this chart draws no"
            " interval."
        )
    figures["paired-differences"] = interval_figure(
        "Paired differences with 95 percent intervals",
        {_interval_label(key): entry for key, entry in documents["intervals"]["intervals"].items()},
    )
    captions["paired-differences"] = (
        f"Every paired difference in intervals.json on the {metrics['cohort']} cohort,"
        f" {metrics['sample_count']} samples, {metrics['seed_count']} seeds per model, with its"
        f" interval from the {metrics['interval_method']}."
    )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    rendered: list[dict[str, str]] = []
    for position, name in enumerate(sorted(figures)):
        figure_path = figure_dir / f"{name}.json"
        figure_path.write_text(
            json.dumps(figures[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        figure_paths.append(figure_path)
        rendered.append(
            {
                "name": name,
                "caption": captions[name],
                "html": _figure_html(name, figures[name], include_library=position == 0),
            }
        )
    curated, svg_paths = _curated_view(
        artifacts_dir, output_dir, metrics=metrics, rankings=rankings, extended=extended
    )

    from jinja2 import Environment, FileSystemLoader, select_autoescape

    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        keep_trailing_newline=True,
    )
    page = environment.get_template("index.html.j2").render(
        version=__version__,
        headlines=_headline_view(claims, repository_root),
        instance_seed=APPROVED_SEEDS[0],
        curated=curated,
        not_drawn=NOT_DRAWN,
        headline_table=_headline_table(metric_table, order),
        claims=claims,
        provenance=metrics,
        figures=rendered,
        rankings=rankings,
        runs=_run_view(documents["formal_run_index"]),
        per_class=_per_class_view(metrics),
        calibration=_calibration_view(metrics),
        risk_profiles=_risk_profile_view(metrics),
        bands=_band_view(extended),
        instances=_instance_view(extended),
        selective_risk=_selective_risk_view(extended),
        ground_truth=extended["ground_truth"],
        gallery=_gallery_view(documents["gallery-manifest"]),
        thin_class_image_count=THIN_CLASS_IMAGE_COUNT,
        limitations=LIMITATIONS,
    )
    index_path = output_dir / "index.html"
    index_path.write_text(
        page,
        encoding="utf-8",
        newline="\n",
    )
    return ReportResult(
        index_path=index_path,
        figure_paths=tuple(figure_paths),
        claim_count=len(claims),
        svg_paths=svg_paths,
    )
