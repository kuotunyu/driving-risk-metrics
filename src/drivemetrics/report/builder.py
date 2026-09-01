"""Claim-safe static report generation from verified claims and frozen artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drivemetrics.analysis.claims import audit_claims, verified_claims
from drivemetrics.report.figures import bar_figure, interval_figure

REPORT_INPUTS: tuple[str, ...] = ("metrics", "intervals", "rankings")

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
    page = environment.get_template("index.html.j2").render(
        claims=claims,
        provenance=metrics,
        figures=rendered,
        rankings=documents["rankings"],
        limitations=LIMITATIONS,
    )
    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return ReportResult(
        index_path=index_path,
        figure_paths=tuple(figure_paths),
        claim_count=len(claims),
    )
