"""Contracts for the claim-safe static report builder and its figures."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

PROTOCOL_HASH = "a" * 64
MANIFEST_HASH = "b" * 64

METRICS = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "cohort": "locked_validation",
    "sample_count": 1000,
    "seed_count": 3,
    "interval_method": "two-stage paired bootstrap, 5000 resamples, seed 20260831",
    "metrics": {
        "upernet_dinov2_small": {"miou": 0.58, "critical_recall": 0.71},
        "upernet_convnextv2_tiny": {"miou": 0.61, "critical_recall": 0.64},
    },
}

INTERVALS = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "intervals": {
        "upernet_convnextv2_tiny minus upernet_dinov2_small (miou)": {
            "estimate": 0.03,
            "low": 0.01,
            "high": 0.05,
            "confidence": 0.95,
            "resamples": 5000,
            "seed": 20260831,
        }
    },
}

RANKINGS = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "baseline_metric": "miou",
    "comparisons": [
        {
            "metric_name": "critical_recall",
            "baseline_order": ["upernet_convnextv2_tiny", "upernet_dinov2_small"],
            "comparison_order": ["upernet_dinov2_small", "upernet_convnextv2_tiny"],
            "reversal_observed": True,
        }
    ],
}


def load_builder_module() -> ModuleType:
    try:
        from drivemetrics.report import builder
    except ImportError:
        pytest.fail("drivemetrics.report.builder is missing", pytrace=False)
    return builder


def load_figures_module() -> ModuleType:
    try:
        from drivemetrics.report import figures
    except ImportError:
        pytest.fail("drivemetrics.report.figures is missing", pytrace=False)
    return figures


def write_workspace(
    tmp_path: Path,
    *,
    claims: list[dict[str, Any]] | None = None,
    drop: str | None = None,
) -> tuple[Path, Path, Path]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    documents = {"metrics": METRICS, "intervals": INTERVALS, "rankings": RANKINGS}
    for name, document in documents.items():
        if name == drop:
            continue
        (artifacts / f"{name}.json").write_text(
            json.dumps(document, sort_keys=True),
            encoding="utf-8",
        )

    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(
        yaml.safe_dump(
            {
                "allowed_evidence_types": ["observed", "derived", "synthetic", "illustrative"],
                "claim_required_fields": [
                    "claim_id",
                    "text",
                    "evidence_type",
                    "protocol_hash",
                    "dataset_manifest_hash",
                    "artifact_path",
                    "metric_path",
                    "status",
                ],
                "allowed_statuses": ["draft", "verified", "rejected", "superseded"],
                "claims": claims if claims is not None else default_claims(),
            }
        ),
        encoding="utf-8",
    )
    return claims_path, artifacts, tmp_path / "site"


def default_claims() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "miou-fcn",
            "text": "FCN reaches 0.61 mIoU on the locked cohort.",
            "evidence_type": "observed",
            "protocol_hash": PROTOCOL_HASH,
            "dataset_manifest_hash": MANIFEST_HASH,
            "artifact_path": "artifacts/metrics.json",
            "metric_path": "/metrics/upernet_convnextv2_tiny/miou",
            "status": "verified",
        },
        {
            "claim_id": "draft-note",
            "text": "A draft sentence with no number.",
            "evidence_type": "illustrative",
            "protocol_hash": PROTOCOL_HASH,
            "dataset_manifest_hash": MANIFEST_HASH,
            "artifact_path": "artifacts/metrics.json",
            "metric_path": "/metrics/upernet_convnextv2_tiny/miou",
            "status": "draft",
        },
    ]


def build(tmp_path: Path, **kwargs: Any) -> Any:
    builder = load_builder_module()
    claims_path, artifacts, output_dir = write_workspace(tmp_path, **kwargs)
    return builder.build_report(claims_path, artifacts, output_dir, repository_root=tmp_path)


def test_only_verified_claims_reach_the_published_page(tmp_path: Path) -> None:
    """Publishing a draft or rejected claim would present unreviewed evidence as a result."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert result.claim_count == 1
    assert "FCN reaches 0.61 mIoU" in page
    assert "A draft sentence" not in page


def test_every_published_claim_shows_its_evidence_type(tmp_path: Path) -> None:
    """A number without its evidence label could be read as a measurement it is not."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "observed" in page


def test_the_page_states_the_cohort_seed_count_and_interval_method(tmp_path: Path) -> None:
    """A chart without its cohort, seed count, and interval method is not interpretable."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "locked_validation" in page
    assert "1000" in page
    assert "two-stage paired bootstrap" in page
    assert PROTOCOL_HASH in page


def test_the_limitations_section_is_always_present(tmp_path: Path) -> None:
    """Omitting the limitations would let a controlled study read as a general claim."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "Limitations" in page


def test_a_failing_claim_audit_refuses_to_publish(tmp_path: Path) -> None:
    """Rendering an unverifiable number is exactly the failure this project exists to prevent."""

    broken = default_claims()
    broken[0]["text"] = "FCN reaches 0.99 mIoU on the locked cohort."

    with pytest.raises(ValueError, match="claim"):
        build(tmp_path, claims=broken)


def test_a_missing_report_input_fails_closed(tmp_path: Path) -> None:
    """A page built from a partial artifact set would silently omit a whole result."""

    with pytest.raises(FileNotFoundError, match=r"rankings\.json"):
        build(tmp_path, drop="rankings")


def test_two_builds_produce_byte_identical_output(tmp_path: Path) -> None:
    """A timestamp or random identifier would make every published page unverifiable."""

    first = build(tmp_path / "one")
    second = build(tmp_path / "two")

    assert first.index_path.read_bytes() == second.index_path.read_bytes()
    assert [path.name for path in first.figure_paths] == [path.name for path in second.figure_paths]
    for left, right in zip(first.figure_paths, second.figure_paths, strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_claim_text_is_escaped_rather_than_injected(tmp_path: Path) -> None:
    """Unescaped claim text would turn a registry entry into script on the public page."""

    injected = default_claims()
    injected[0]["text"] = "FCN reaches 0.61 mIoU <script>steal()</script>"

    result = build(tmp_path, claims=injected)
    page = result.index_path.read_text(encoding="utf-8")

    assert "<script>steal()</script>" not in page
    assert "&lt;script&gt;steal()&lt;/script&gt;" in page


def test_a_ranking_reversal_is_reported_as_an_observation(tmp_path: Path) -> None:
    """A reversal is an observed outcome, never a success criterion the page celebrates."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "critical_recall" in page
    assert "observation" in page.lower()


def test_bar_figures_order_their_categories_deterministically() -> None:
    """Dictionary iteration order must never change a published chart."""

    figures = load_figures_module()

    figure = figures.bar_figure("mIoU", "mIoU", {"zeta": 0.2, "alpha": 0.9})

    assert figure["data"][0]["x"] == ["alpha", "zeta"]
    assert figure["data"][0]["y"] == [0.9, 0.2]
    assert figure["data"][0]["type"] == "bar"
    assert figure["layout"]["title"]["text"] == "mIoU"


def test_interval_figures_carry_the_effect_size_and_its_bounds() -> None:
    """An interval drawn without its estimate would misrepresent the uncertainty."""

    figures = load_figures_module()

    figure = figures.interval_figure(
        "Paired differences",
        {"a minus b": {"estimate": 0.03, "low": 0.01, "high": 0.05}},
    )

    trace = figure["data"][0]
    assert trace["x"] == [0.03]
    assert trace["y"] == ["a minus b"]
    assert trace["error_x"]["array"] == [pytest.approx(0.02)]
    assert trace["error_x"]["arrayminus"] == [pytest.approx(0.02)]


def test_report_package_exports_the_public_entry_points() -> None:
    """The CLI consumes the report builder through the package entry point."""

    import drivemetrics.report as report

    builder = load_builder_module()
    assert report.build_report is builder.build_report
    assert report.ReportResult is builder.ReportResult


def test_a_report_input_that_is_not_an_object_fails_closed(tmp_path: Path) -> None:
    """A JSON list would reach the renderer as positional garbage and fail obscurely."""

    builder = load_builder_module()
    claims_path, artifacts, output_dir = write_workspace(tmp_path)
    (artifacts / "rankings.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(TypeError, match="JSON object"):
        builder.build_report(claims_path, artifacts, output_dir, repository_root=tmp_path)
