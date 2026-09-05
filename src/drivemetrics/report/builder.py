"""Claim-safe static report generation from verified claims and frozen artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drivemetrics.analysis.claims import audit_claims, verified_claims
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


@dataclass(frozen=True)
class ReportResult:
    """Where the published page and its machine-readable figures were written."""

    index_path: Path
    figure_paths: tuple[Path, ...]
    claim_count: int


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"report input is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"report input must be a JSON object: {path}")
    return document


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
    count, interval method, and artifact hashes are stated once at the top so no
    chart can be read out of context.
    """

    violations = audit_claims(claims_path, repository_root)
    if violations:
        raise ValueError("claim audit failed: " + "; ".join(violations))

    documents = {name: _load_json_object(artifacts_dir / f"{name}.json") for name in REPORT_INPUTS}
    claims = verified_claims(claims_path)
    metrics = documents["metrics"]
    metric_table: dict[str, dict[str, float]] = metrics["metrics"]
    metric_names = sorted({name for scores in metric_table.values() for name in scores})

    figures: dict[str, dict[str, Any]] = {
        name: bar_figure(
            name, name, {model: scores[name] for model, scores in metric_table.items()}
        )
        for name in metric_names
    }
    figures["paired-differences"] = interval_figure(
        "Paired differences with 95 percent intervals",
        documents["intervals"]["intervals"],
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
                "html": _figure_html(name, figures[name], include_library=position == 0),
            }
        )

    from jinja2 import Environment, FileSystemLoader, select_autoescape

    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        keep_trailing_newline=True,
    )
    extended = documents["extended-metrics"]
    page = environment.get_template("index.html.j2").render(
        claims=claims,
        provenance=metrics,
        figures=rendered,
        rankings=documents["rankings"],
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
    )
