"""The published report must open with the committed evidence, drawn by the build itself.

The Pages workflow publishes exactly what `driving-risk report` writes. The page
opens with evidence figures that the build draws beside it, so those must be the
committed figures byte for byte; and the opening values are rounded with the
claims validator's rule, so they read exactly as the README states them.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from drivemetrics.analysis.claims import claim_value, verified_claims
from drivemetrics.report.builder import (
    HEADLINE_COUNT_CLAIMS,
    HEADLINE_INTERVAL_CLAIMS,
    ReportResult,
    build_report,
)
from drivemetrics.report.svg import FIGURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS = REPO_ROOT / "docs/claims.yaml"
EVIDENCE = REPO_ROOT / "docs/evidence/bdd100k_semseg_v1"
FIGURES = REPO_ROOT / "docs/figures"


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> ReportResult:
    return build_report(
        CLAIMS, EVIDENCE, tmp_path_factory.mktemp("site"), repository_root=REPO_ROOT
    )


def rounded(value: Any) -> str:
    """The claims validator's rule: round half to even on the decimal text."""

    return f"{Decimal(str(value)).quantize(Decimal('0.001')):.3f}"


def test_the_published_figures_are_the_committed_figures(report: ReportResult) -> None:
    """Pages no longer copies docs/figures, so the build must write the same bytes."""

    page = report.index_path.read_text(encoding="utf-8")

    assert [path.name for path in report.svg_paths] == [f"{name}.svg" for name in FIGURE_NAMES]
    for path in report.svg_paths:
        assert path.read_bytes() == (FIGURES / path.name).read_bytes(), path.name
    assert re.findall(r'<img src="figures-svg/([a-z0-9-]+)\.svg"', page) == [
        "headline-top-two",
        "small-tertile-critical-misses",
        "miou-gap-by-class",
    ]


def test_every_opening_statement_is_its_claim_rounded_by_the_validator_rule(
    report: ReportResult,
) -> None:
    """The lede states the README's headline values, each traced to its verified claim."""

    page = report.index_path.read_text(encoding="utf-8")
    findings = re.search(r'<ul class="findings">(.*?)</ul>', page, re.DOTALL)
    claims = {claim.claim_id: claim for claim in verified_claims(CLAIMS)}

    assert findings is not None
    items = [
        " ".join(item.split())
        for item in re.findall(r"<li>(.*?)</li>", findings.group(1), re.DOTALL)
    ]
    assert len(items) == len(HEADLINE_INTERVAL_CLAIMS) + len(HEADLINE_COUNT_CLAIMS)
    for item, (claim_id, _) in zip(items, HEADLINE_INTERVAL_CLAIMS, strict=False):
        entry: Any = claim_value(claims[claim_id], REPO_ROOT)
        assert re.findall(r'<span title="([^"]+)">([^<]+)</span>', item) == [
            (str(entry[field]), rounded(entry[field])) for field in ("estimate", "low", "high")
        ]
        verdict = "excludes zero" if entry["excludes_zero"] else "includes zero"
        assert item.endswith(f"which {verdict}.")
    # The same values the README's "At a glance" list states.
    assert [re.findall(r">(-?\d\.\d{3})<", item) for item in items[:2]] == [
        ["-0.010", "-0.022", "0.002"],
        ["-0.023", "-0.040", "-0.008"],
    ]
    assert items[len(HEADLINE_INTERVAL_CLAIMS) :] == [
        claims[claim_id].text for claim_id in HEADLINE_COUNT_CLAIMS
    ]


def test_the_headline_table_rounds_the_released_metrics(report: ReportResult) -> None:
    """The compact table reads as the README's headline table, full values in tooltips."""

    page = report.index_path.read_text(encoding="utf-8")
    body = re.search(r'<section id="headline">(.*?)</section>', page, re.DOTALL)
    metrics = json.loads((EVIDENCE / "metrics.json").read_text(encoding="utf-8"))["metrics"]

    assert body is not None
    rows = re.findall(r"<tr><td>([^<]+)</td>(.*?)</tr>", body.group(1))
    cells = [(model, re.findall(r'<td title="([^"]+)">([^<]+)</td>', row)) for model, row in rows]
    assert [(model, [text for _, text in values]) for model, values in cells] == [
        ("UperNet-ConvNeXtV2-Tiny", ["0.632", "0.811", "0.939"]),
        ("SegFormer-B2", ["0.622", "0.787", "0.939"]),
        ("UperNet-DINOv2-Small", ["0.474", "0.520", "0.914"]),
    ]
    keys = ("upernet_convnextv2_tiny", "segformer_b2", "upernet_dinov2_small")
    for key, (_, values) in zip(keys, cells, strict=True):
        expected = [metrics[key][name] for name in ("miou", "critical_recall", "pixel_accuracy")]
        assert values == [(str(value), rounded(value)) for value in expected]


def test_the_curated_captions_read_in_the_published_orientation(report: ReportResult) -> None:
    """The captions name the pair as the figures and the README do: left minus right."""

    page = report.index_path.read_text(encoding="utf-8")
    body = re.search(r'<section id="key-figures">(.*?)</section>', page, re.DOTALL)

    assert body is not None
    captions = [
        " ".join(caption.split())
        for caption in re.findall(r"<figcaption>(.*?)</figcaption>", body.group(1), re.DOTALL)
    ]
    pair = "SegFormer-B2 minus UperNet-ConvNeXtV2-Tiny"
    assert pair in captions[0]
    assert pair in captions[2]
    assert "SegFormer-B2 minus ConvNeXtV2-Tiny" in (FIGURES / "headline-top-two.svg").read_text(
        encoding="utf-8"
    )
