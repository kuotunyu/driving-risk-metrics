"""Contracts for the claim-safe static report builder and its figures."""

from __future__ import annotations

import json
import re
from decimal import Decimal
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
    "per_class": {
        "class_names": ["road", "person", "train"],
        "support_pixels": [900000, 40000, 109005],
        "images_with_class": [998, 640, 7],
        "by_model": {
            # Each model's class IoUs average to its mean IoU above, as the per-class
            # decomposition figure requires.
            "upernet_convnextv2_tiny": {"iou": [0.95, 0.88, 0.0], "recall": [0.98, 0.83, 0.0]},
            "upernet_dinov2_small": {"iou": [0.93, 0.81, 0.0], "recall": [0.97, 0.57, 0.0]},
        },
    },
    "calibration": {
        "upernet_convnextv2_tiny": {
            "calibrated": {
                "brier": 0.0987,
                "ece": 0.0032,
                "per_seed": {
                    "17": {"brier": 0.0983, "ece": 0.0033},
                    "42": {"brier": 0.0985, "ece": 0.0032},
                },
            },
            "uncalibrated": {
                "brier": 0.1051,
                "ece": 0.0046,
                "per_seed": {
                    "17": {"brier": 0.1049, "ece": 0.0045},
                    "42": {"brier": 0.1052, "ece": 0.0046},
                },
            },
        },
        "upernet_dinov2_small": {
            "calibrated": {
                "brier": 0.1345,
                "ece": 0.0039,
                "per_seed": {
                    "17": {"brier": 0.1341, "ece": 0.0038},
                    "42": {"brier": 0.1348, "ece": 0.0040},
                },
            },
            "uncalibrated": {
                "brier": 0.1409,
                "ece": 0.0054,
                "per_seed": {
                    "17": {"brier": 0.1405, "ece": 0.0054},
                    "42": {"brier": 0.1412, "ece": 0.0053},
                },
            },
        },
    },
    "risk_profiles": {
        "balanced": {
            "cost_risk": {"upernet_convnextv2_tiny": 0.0612, "upernet_dinov2_small": 0.0858},
            "critical_class_ids": [],
            "sensitivity": 1.0,
        },
        "vru_priority": {
            "cost_risk": {"upernet_convnextv2_tiny": 0.0511, "upernet_dinov2_small": 0.0722},
            "critical_class_ids": [11, 12, 17, 18],
            "sensitivity": 1.0,
        },
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
    "separability": {
        "miou": [
            {
                "left": "upernet_convnextv2_tiny",
                "right": "upernet_dinov2_small",
                "estimate": 0.03,
                "low": 0.01,
                "high": 0.05,
                "excludes_zero": True,
            }
        ],
        "critical_recall": [
            {
                "left": "upernet_convnextv2_tiny",
                "right": "upernet_dinov2_small",
                "estimate": -0.07,
                "low": -0.1,
                "high": 0.004,
                "excludes_zero": False,
            }
        ],
    },
}

#: The shape `rank` writes when no metric was compared: no order and no intervals.
RANKINGS_WITHOUT_COMPARISONS = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "baseline_metric": "miou",
    "comparisons": [],
}


BANDS = ("top", "middle", "bottom")

EXTENDED = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "evaluation_for_ground_truth_metrics": "eval_calibrated",
    "ground_truth": {
        "cohort_manifest_sha256": MANIFEST_HASH,
        "instance_bitmasks": {"count": 1000, "set_sha256": "c" * 64},
        "masks_verified": 1000,
        "split_name": "locked_validation",
    },
    "normalized_image_bands": {
        "definition": "normalized image rows only; not physical depth or metric distance",
        "by_model": {
            "upernet_convnextv2_tiny": {
                "top": {"pixel_accuracy": 0.91, "pixels": 11},
                "middle": {"pixel_accuracy": 0.82, "pixels": 12},
                "bottom": {"pixel_accuracy": 0.93, "pixels": 13},
            },
            "upernet_dinov2_small": {
                "top": {"pixel_accuracy": 0.71, "pixels": 11},
                "middle": {"pixel_accuracy": 0.62, "pixels": 12},
                "bottom": {"pixel_accuracy": 0.73, "pixels": 13},
            },
        },
    },
    "instances": {
        "upernet_convnextv2_tiny": {
            "instance_count": 120,
            "excluded_without_semantic_pixels": 4,
            "mean_corroborated_fraction": 0.94,
            "tertile_edges_sha256": "d" * 64,
            "by_tertile": {
                "small": {
                    "critical_misses": 30,
                    "instance_count": 40,
                    "mean_correct_fraction": 0.31,
                },
                "medium": {
                    "critical_misses": 10,
                    "instance_count": 40,
                    "mean_correct_fraction": 0.62,
                },
                "large": {
                    "critical_misses": 2,
                    "instance_count": 40,
                    "mean_correct_fraction": 0.83,
                },
            },
            "by_class": {
                "person": {
                    "critical_misses": 21,
                    "instance_count": 30,
                    "mean_correct_fraction": 0.41,
                    "by_tertile": {
                        "small": {
                            "critical_misses": 17,
                            "instance_count": 18,
                            "mean_correct_fraction": 0.22,
                        },
                        "medium": {
                            "critical_misses": 3,
                            "instance_count": 7,
                            "mean_correct_fraction": 0.55,
                        },
                        "large": {
                            "critical_misses": 1,
                            "instance_count": 5,
                            "mean_correct_fraction": 0.77,
                        },
                    },
                }
            },
        },
        "upernet_dinov2_small": {
            "instance_count": 120,
            "excluded_without_semantic_pixels": 4,
            "mean_corroborated_fraction": 0.94,
            "tertile_edges_sha256": "d" * 64,
            "by_tertile": {
                "small": {
                    "critical_misses": 36,
                    "instance_count": 40,
                    "mean_correct_fraction": 0.21,
                },
                "medium": {
                    "critical_misses": 14,
                    "instance_count": 40,
                    "mean_correct_fraction": 0.52,
                },
                "large": {
                    "critical_misses": 5,
                    "instance_count": 40,
                    "mean_correct_fraction": 0.73,
                },
            },
            "by_class": {
                "person": {
                    "critical_misses": 25,
                    "instance_count": 30,
                    "mean_correct_fraction": 0.31,
                    "by_tertile": {
                        "small": {
                            "critical_misses": 18,
                            "instance_count": 18,
                            "mean_correct_fraction": 0.12,
                        },
                        "medium": {
                            "critical_misses": 5,
                            "instance_count": 7,
                            "mean_correct_fraction": 0.45,
                        },
                        "large": {
                            "critical_misses": 2,
                            "instance_count": 5,
                            "mean_correct_fraction": 0.67,
                        },
                    },
                }
            },
        },
    },
    "selective_risk": {
        "upernet_convnextv2_tiny": {
            "calibrated": {
                "aurc": 0.0201,
                "coverage_points": 58707,
                "defined_at": "confidence_bin_boundaries",
            },
            "uncalibrated": {
                "aurc": 0.0198,
                "coverage_points": 52121,
                "defined_at": "confidence_bin_boundaries",
            },
        },
        "upernet_dinov2_small": {
            "calibrated": {
                "aurc": 0.0233,
                "coverage_points": 60184,
                "defined_at": "confidence_bin_boundaries",
            },
            "uncalibrated": {
                "aurc": 0.0228,
                "coverage_points": 56871,
                "defined_at": "confidence_bin_boundaries",
            },
        },
    },
}

#: The exact shape `extended-metrics` writes when the analysis had no ground truth:
#: it states what was missing instead of publishing an empty table.
EXTENDED_WITHOUT_GROUND_TRUTH = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "evaluation_for_ground_truth_metrics": "eval_calibrated",
    "ground_truth": {"not_computed": "ground truth was not available to this analysis run"},
    "normalized_image_bands": {
        "not_computed": "ground truth was not available to this analysis run"
    },
    "instances": {"not_computed": "the frozen area tertiles were not supplied"},
    "selective_risk": EXTENDED["selective_risk"],
}

GALLERY = {
    "protocol_hash": PROTOCOL_HASH,
    "dataset_manifest_hash": MANIFEST_HASH,
    "evaluation": "eval_calibrated",
    "rule": {
        "aggregate": "mean_over_seeds",
        "metric": "image_mean_iou",
        "per_model": 2,
        "tie_break": "sample_id",
    },
    "per_model": {
        "upernet_convnextv2_tiny": {
            "best": [
                {
                    "sample_id": "aaa-001",
                    "mean_iou_over_seeds": 0.87,
                    "per_seed": {"17": 0.88, "42": 0.86},
                }
            ],
            "worst": [
                {
                    "sample_id": "zzz-009",
                    "mean_iou_over_seeds": 0.14,
                    "per_seed": {"17": 0.13, "42": 0.15},
                }
            ],
        },
        "upernet_dinov2_small": {
            "best": [
                {
                    "sample_id": "aaa-002",
                    "mean_iou_over_seeds": 0.77,
                    "per_seed": {"17": 0.78, "42": 0.76},
                }
            ],
            "worst": [
                {
                    "sample_id": "zzz-010",
                    "mean_iou_over_seeds": 0.11,
                    "per_seed": {"17": 0.10, "42": 0.12},
                }
            ],
        },
    },
}

RUN_INDEX = {
    "cohort": "locked_validation",
    "critical_class_ids": [11, 12, 17, 18],
    "dataset_manifest_sha256": MANIFEST_HASH,
    "expected_steps": 30000,
    "num_classes": 19,
    "protocol_sha256": PROTOCOL_HASH,
    "schema_version": "drivemetrics-formal-set/v1",
    "runs": [
        {
            "run_id": "upernet_convnextv2_tiny-seed-17",
            "model": "upernet_convnextv2_tiny",
            "seed": 17,
            "status": "succeeded",
            "final_step": 30000,
            "temperature": 2.51,
            "checkpoint_sha256": "e" * 64,
            "artifacts_dir": "upernet_convnextv2_tiny/seed-17/eval",
            "calibrated_artifacts_dir": "upernet_convnextv2_tiny/seed-17/eval_calibrated",
            "uncalibrated_sample_ids": ["aaa-001"],
            "calibrated_sample_ids": ["aaa-001"],
        },
        {
            "run_id": "upernet_dinov2_small-seed-42",
            "model": "upernet_dinov2_small",
            "seed": 42,
            "status": "succeeded",
            "final_step": 30000,
            "temperature": 2.14,
            "checkpoint_sha256": "f" * 64,
            "artifacts_dir": "upernet_dinov2_small/seed-42/eval",
            "calibrated_artifacts_dir": "upernet_dinov2_small/seed-42/eval_calibrated",
            "uncalibrated_sample_ids": ["aaa-002"],
            "calibrated_sample_ids": ["aaa-002"],
        },
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
    extended: dict[str, Any] | None = None,
    rankings: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    documents = {
        "metrics": METRICS,
        "intervals": INTERVALS,
        "rankings": rankings if rankings is not None else RANKINGS,
        "extended-metrics": extended if extended is not None else EXTENDED,
        "gallery-manifest": GALLERY,
        "formal_run_index": RUN_INDEX,
    }
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

    with pytest.raises(ValueError, match=r"^claim audit failed: miou-fcn: claim number is absent"):
        build(tmp_path, claims=broken)


def test_a_missing_report_input_fails_closed(tmp_path: Path) -> None:
    """A page built from a partial artifact set would silently omit a whole result."""

    with pytest.raises(FileNotFoundError, match=r"^report input is missing:"):
        build(tmp_path, drop="rankings")


def test_two_builds_produce_byte_identical_output(tmp_path: Path) -> None:
    """A timestamp or random identifier would make every published page unverifiable."""

    first = build(tmp_path / "one")
    second = build(tmp_path / "two")

    assert first.index_path.read_bytes() == second.index_path.read_bytes()
    assert [path.name for path in first.figure_paths] == [path.name for path in second.figure_paths]
    for left, right in zip(first.figure_paths, second.figure_paths, strict=True):
        assert left.read_bytes() == right.read_bytes()
    assert [path.name for path in first.svg_paths] == [path.name for path in second.svg_paths]
    assert first.svg_paths
    for left, right in zip(first.svg_paths, second.svg_paths, strict=True):
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

    with pytest.raises(TypeError, match=r"^report input must be a JSON object:"):
        builder.build_report(claims_path, artifacts, output_dir, repository_root=tmp_path)


def test_every_per_class_row_carries_its_support_and_image_count(tmp_path: Path) -> None:
    """A per-class score without its support invites a conclusion the data cannot carry.

    Class `train` scores 0.0 IoU for every model in this study. Read alone that
    looks like a model failure; read beside 109005 pixels in 7 images it is a
    statement about the cohort. The page must never print the first without the
    second.
    """

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "109005" in page
    assert ">7<" in page or "> 7 <" in page


def test_a_class_seen_in_too_few_images_is_marked_thin(tmp_path: Path) -> None:
    """The reader is told which rows are thin by a count, not by anyone's judgement."""

    builder = load_builder_module()
    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert builder.THIN_CLASS_IMAGE_COUNT == 50
    assert "thin" in page.lower()


def test_calibration_is_published_for_every_seed(tmp_path: Path) -> None:
    """A seed mean hides whether temperature scaling helped every run or only on average."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "0.0033" in page
    assert "0.0045" in page


def test_all_three_risk_profiles_reach_the_page_with_their_critical_classes(
    tmp_path: Path,
) -> None:
    """A cost-risk number means nothing without the profile that weighted it."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "vru_priority" in page
    assert "balanced" in page
    assert "0.0511" in page


def test_the_image_bands_carry_their_definition(tmp_path: Path) -> None:
    """A band read as distance would turn an image region into a depth claim."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "0.82" in page
    assert "not physical depth" in page


def test_instance_coverage_is_published_by_class_and_by_tertile(tmp_path: Path) -> None:
    """Pooled instance coverage cannot answer a question about pedestrians."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "person" in page
    assert "0.94" in page


def test_selective_risk_states_where_its_curve_is_defined(tmp_path: Path) -> None:
    """An AURC over a quantized curve is not the AURC of a continuous one."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "0.0201" in page
    assert "confidence_bin_boundaries" in page


def test_the_gallery_is_a_table_of_sample_ids_and_never_ships_an_image(
    tmp_path: Path,
) -> None:
    """The release redistributes no BDD100K pixels, so the gallery names samples only."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "zzz-009" in page
    # The only images are the evidence figures drawn beside the page, never pixels.
    sources = re.findall(r'<img src="([^"]*)"', page)
    assert sources
    assert all(re.fullmatch(r"figures-svg/[a-z0-9-]+\.svg", source) for source in sources)
    assert page.count("<img") == len(sources)
    assert not re.search(r"""(?:src|href)\s*=\s*["']?data:""", page)


def test_the_nine_formal_runs_are_listed_with_their_checkpoints(tmp_path: Path) -> None:
    """A published aggregate that cannot name the runs behind it cannot be checked."""

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")

    assert "upernet_convnextv2_tiny-seed-17" in page
    assert "e" * 64 in page
    assert "2.51" in page


def test_a_block_the_analysis_could_not_compute_is_stated_rather_than_faked(
    tmp_path: Path,
) -> None:
    """`extended-metrics` records why a block is missing; the page must say so, not crash.

    Without ground truth the analysis writes `{"not_computed": <reason>}` for the
    bands, the instances and the ground-truth record. A page that crashed on that
    shape could only ever be built from a complete run, and a page that rendered an
    empty table would hide that a result was never measured.
    """

    result = build(tmp_path, extended=EXTENDED_WITHOUT_GROUND_TRUTH)
    page = result.index_path.read_text(encoding="utf-8")

    assert "ground truth was not available to this analysis run" in page
    assert "the frozen area tertiles were not supplied" in page
    assert "Critical misses" not in page
    assert "<img" not in page
    assert load_builder_module().NOT_DRAWN in page
    assert result.svg_paths == ()
    assert not (tmp_path / "site" / "figures-svg").exists()


def headline_claims() -> list[dict[str, Any]]:
    """The registry entries the page opens with, pointing at this workspace's fixtures."""

    def claim(claim_id: str, text: str, artifact: str, pointer: str) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "text": text,
            "evidence_type": "observed",
            "protocol_hash": PROTOCOL_HASH,
            "dataset_manifest_hash": MANIFEST_HASH,
            "artifact_path": f"artifacts/{artifact}",
            "metric_path": pointer,
            "status": "verified",
        }

    return [
        *default_claims(),
        claim(
            "p1.interval.miou.segformer-minus-convnextv2",
            "The paired difference in mean IoU is 0.03, and its bootstrap interval from 0.01"
            " to 0.05 excludes zero.",
            "rankings.json",
            "/separability/miou/0",
        ),
        claim(
            "p1.interval.critical-recall.segformer-minus-convnextv2",
            "The paired difference in critical-class recall is -0.07, and its bootstrap"
            " interval from -0.1 to 0.004 includes zero.",
            "rankings.json",
            "/separability/critical_recall/0",
        ),
        claim(
            "p1.instances.convnextv2.person-small",
            "UperNet-ConvNeXtV2-Tiny recovers less than half the pixels of 17 of the 18"
            " smallest-tertile person instances.",
            "extended-metrics.json",
            "/instances/upernet_convnextv2_tiny/by_class/person/by_tertile/small",
        ),
    ]


def section_body(page: str, section_id: str) -> str:
    match = re.search(rf'<section id="{section_id}">(.*?)</section>', page, re.DOTALL)
    assert match is not None, section_id
    return match.group(1)


def figure_blocks(page: str) -> list[tuple[str, str]]:
    """Return every figure on the page as (caption, body)."""

    return re.findall(
        r"<figure>\s*<figcaption>(.*?)</figcaption>(.*?)</figure>", page, flags=re.DOTALL
    )


def test_the_page_leads_with_findings_and_ends_with_its_provenance(tmp_path: Path) -> None:
    """A reviewer meets the findings first; the hashes and the claims table close the page."""

    page = build(tmp_path).index_path.read_text(encoding="utf-8")

    assert re.findall(r'<section id="([^"]+)"', page) == [
        "summary",
        "key-figures",
        "headline",
        "figures",
        "runs",
        "per-class",
        "calibration",
        "risk-profiles",
        "bands",
        "instances",
        "selective-risk",
        "gallery",
        "rankings",
        "provenance",
        "claims",
        "limitations",
    ]


def test_the_hashes_and_the_claims_table_are_collapsed_but_complete(tmp_path: Path) -> None:
    """Collapsing the provenance must not drop a hash or a claim from the page."""

    page = build(tmp_path).index_path.read_text(encoding="utf-8")

    assert page.count("<details>") == 2
    for section_id, heading in (
        ("provenance", "What these numbers describe"),
        ("claims", "Verified claims"),
    ):
        body = section_body(page, section_id).strip()
        assert body.startswith(f"<details>\n<summary><h2>{heading}</h2></summary>")
        assert body.endswith("</details>")
    assert PROTOCOL_HASH in section_body(page, "provenance")
    assert MANIFEST_HASH in section_body(page, "provenance")
    assert "FCN reaches 0.61 mIoU" in section_body(page, "claims")


def test_the_cohort_seeds_and_interval_method_stay_visible_at_the_top(tmp_path: Path) -> None:
    """Collapsing the hashes must not hide what every number describes."""

    summary = section_body(build(tmp_path).index_path.read_text(encoding="utf-8"), "summary")

    assert "<code>locked_validation</code> cohort" in summary
    assert " ".join(summary.split()).count("1000 evaluated samples, 3 seeds per model") == 1
    assert str(METRICS["interval_method"]) in summary


def test_the_page_links_the_repository_readmes_release_and_sibling_projects(
    tmp_path: Path,
) -> None:
    """The Pages homepage is the entry point, so it must lead back to everything else."""

    from drivemetrics import __version__

    page = build(tmp_path).index_path.read_text(encoding="utf-8")
    nav = re.search(r'<nav class="links"[^>]*>(.*?)</nav>', page, re.DOTALL)

    assert nav is not None
    repository = "https://github.com/kuotunyu/driving-risk-metrics"
    assert re.findall(r'href="([^"]+)"', nav.group(1)) == [
        repository,
        f"{repository}/blob/main/README.en.md",
        f"{repository}/blob/main/README.md",
        f"{repository}/releases/tag/v{__version__}",
        "https://github.com/kuotunyu/bev-calibration-lab",
        "https://github.com/kuotunyu/perception-error-to-aeb",
    ]
    assert f">Release v{__version__}</a>" in nav.group(1)


def test_the_page_describes_itself_to_link_previews(tmp_path: Path) -> None:
    """A shared link must preview as the study, not as a bare file name."""

    page = build(tmp_path).index_path.read_text(encoding="utf-8")
    title = "driving-risk-metrics: safety-oriented segmentation evaluation (BDD100K)"

    assert f"<title>{title}</title>" in page
    assert f'<meta property="og:title" content="{title}">' in page
    assert '<meta property="og:type" content="website">' in page
    assert (
        '<meta property="og:url" content="https://kuotunyu.github.io/driving-risk-metrics/">'
        in page
    )
    for name in ('name="description"', 'property="og:description"'):
        content = re.search(rf'<meta {name} content="([^"]+)">', page)
        assert content is not None, name
        assert "locked_validation cohort, 1000 samples, 3 seeds per model" in content.group(1)


def test_the_lede_states_the_headline_claims_rounded_like_the_readme(tmp_path: Path) -> None:
    """The opening values use the validator's rounding and keep the full value beside them."""

    page = build(tmp_path, claims=headline_claims()).index_path.read_text(encoding="utf-8")
    findings = re.search(r'<ul class="findings">(.*?)</ul>', page, re.DOTALL)

    assert findings is not None
    items = [
        " ".join(item.split())
        for item in re.findall(r"<li>(.*?)</li>", findings.group(1), re.DOTALL)
    ]
    assert items == [
        "Mean IoU, UperNet-ConvNeXtV2-Tiny minus UperNet-DINOv2-Small:"
        ' <span title="0.03">0.030</span>, paired bootstrap interval'
        ' <span title="0.01">0.010</span> to <span title="0.05">0.050</span>,'
        " which excludes zero.",
        "Critical-class recall, UperNet-ConvNeXtV2-Tiny minus UperNet-DINOv2-Small:"
        ' <span title="-0.07">-0.070</span>, paired bootstrap interval'
        ' <span title="-0.1">-0.100</span> to <span title="0.004">0.004</span>,'
        " which includes zero.",
        "UperNet-ConvNeXtV2-Tiny recovers less than half the pixels of 17 of the 18"
        " smallest-tertile person instances.",
    ]
    assert "one training seed per model (seed 17, the first" in " ".join(page.split())


def test_a_headline_claim_the_registry_does_not_verify_is_left_out(tmp_path: Path) -> None:
    """A statement whose claim is missing or unverified must not open the page."""

    claims = headline_claims()
    assert claims[3]["claim_id"] == "p1.interval.critical-recall.segformer-minus-convnextv2"
    claims[3]["status"] = "draft"
    page = build(tmp_path, claims=claims).index_path.read_text(encoding="utf-8")
    items = re.findall(r"<li>(.*?)</li>", section_body(page, "summary"), re.DOTALL)

    assert len(items) == 2
    assert items[0].startswith("Mean IoU")
    assert "smallest-tertile person instances" in items[1]
    assert "Critical-class recall" not in section_body(page, "summary")
    assert "one training seed per model" in section_body(page, "summary")

    bare = build(tmp_path / "bare").index_path.read_text(encoding="utf-8")
    assert '<ul class="findings">' not in bare
    assert "one training seed per model" not in section_body(bare, "summary")


def test_displayed_values_use_the_validator_rounding_rule() -> None:
    """Round half to even on the decimal text, as the claims validator does, never on the float."""

    builder = load_builder_module()

    for value in (0.6320100232208011, -0.010027276977824351, 0.520379600009604, 0.0005, 0.0015):
        expected = f"{Decimal(str(value)).quantize(Decimal('0.001')):.3f}"
        assert builder.display_value(value) == {"text": expected, "full": str(value)}
    assert builder.display_value(0.0005)["text"] == "0.000"
    assert builder.display_value(0.520379600009604)["text"] == "0.520"


def test_the_headline_table_rounds_every_metric_best_model_first(tmp_path: Path) -> None:
    """The compact table follows the baseline ranking and names models as the claims do."""

    body = section_body(build(tmp_path).index_path.read_text(encoding="utf-8"), "headline")
    rows = re.findall(r"<tr>(.*?)</tr>", body, re.DOTALL)

    assert rows[0] == "<th>Model</th><th>mean IoU</th><th>critical-class recall</th>"
    assert rows[1:] == [
        '<td>UperNet-ConvNeXtV2-Tiny</td><td title="0.61">0.610</td><td title="0.64">0.640</td>',
        '<td>UperNet-DINOv2-Small</td><td title="0.58">0.580</td><td title="0.71">0.710</td>',
    ]


def test_only_figures_that_draw_intervals_mention_the_bootstrap(tmp_path: Path) -> None:
    """A caption must not promise intervals on a chart that draws none."""

    page = build(tmp_path).index_path.read_text(encoding="utf-8")
    blocks = figure_blocks(page)

    assert len(blocks) == 6
    for caption, body in blocks:
        draws_intervals = (
            'id="figure-paired-differences"' in body or "figures-svg/headline-top-two.svg" in body
        )
        assert ("bootstrap" in caption) == draws_intervals, caption
        states_none = "draws no interval" in caption or "no per-class interval" in caption
        assert states_none != draws_intervals, caption


def test_each_chart_has_its_own_caption_and_readable_labels(tmp_path: Path) -> None:
    """Charts name models and metrics as the claims do, best model first."""

    result = build(tmp_path)
    figures = result.index_path.parent / "figures"
    page = result.index_path.read_text(encoding="utf-8")
    captions = [caption for caption, body in figure_blocks(page) if "plotly-graph-div" in body]

    assert len(set(captions)) == len(captions) == 3
    assert captions[0].startswith("Critical-class recall of each model")
    assert "models ordered by mean IoU, best first" in captions[0]
    miou = json.loads((figures / "miou.json").read_text(encoding="utf-8"))
    assert miou["layout"]["title"]["text"] == "Mean IoU by model"
    assert miou["layout"]["xaxis"]["categoryarray"] == [
        "UperNet-ConvNeXtV2-Tiny",
        "UperNet-DINOv2-Small",
    ]
    intervals = json.loads((figures / "paired-differences.json").read_text(encoding="utf-8"))
    assert intervals["data"][0]["y"] == [
        "UperNet-ConvNeXtV2-Tiny minus UperNet-DINOv2-Small (mean IoU)"
    ]


def test_models_are_ordered_by_name_when_nothing_was_ranked(tmp_path: Path) -> None:
    """Without a ranking there is no best model, and no curated figure can be drawn."""

    result = build(tmp_path, rankings=RANKINGS_WITHOUT_COMPARISONS)
    page = result.index_path.read_text(encoding="utf-8")

    assert "models ordered by name." in page
    assert result.svg_paths == ()
    assert "<img" not in page


def test_an_interval_key_in_another_shape_keeps_its_own_label() -> None:
    """Only a ``left minus right (metric)`` key is renamed; any other passes through."""

    builder = load_builder_module()

    assert builder._interval_label("upernet_convnextv2_tiny (miou)") == (
        "upernet_convnextv2_tiny (miou)"
    )
    assert builder._interval_label("segformer_b2 minus other (pixel_accuracy)") == (
        "SegFormer-B2 minus other (pixel accuracy)"
    )


def test_the_curated_figures_are_drawn_beside_the_page_and_shown_first(tmp_path: Path) -> None:
    """A local build must be self-contained: the page's SVGs are written by the build."""

    from drivemetrics.report.svg import FIGURE_NAMES

    result = build(tmp_path)
    page = result.index_path.read_text(encoding="utf-8")
    site = result.index_path.parent

    assert [path.name for path in result.svg_paths] == [f"{name}.svg" for name in FIGURE_NAMES]
    assert all(path.parent == site / "figures-svg" and path.is_file() for path in result.svg_paths)
    images = re.findall(
        r'<img src="([^"]+)" width="(\d+)" height="(\d+)" alt="([^"]+)">',
        section_body(page, "key-figures"),
    )
    assert [source for source, _, _, _ in images] == [
        "figures-svg/headline-top-two.svg",
        "figures-svg/small-tertile-critical-misses.svg",
        "figures-svg/miou-gap-by-class.svg",
    ]
    for source, width, height, alt in images:
        svg = (site / source).read_text(encoding="utf-8")
        assert f'width="{width}" height="{height}"' in svg
        assert alt
    assert page.index("figures-svg/") < page.index('id="figure-')


def test_rankings_without_intervals_draw_no_curated_figure(tmp_path: Path) -> None:
    """An older rankings document has no separability block; the page says so."""

    rankings = {key: value for key, value in RANKINGS.items() if key != "separability"}
    result = build(tmp_path, rankings=rankings)
    page = result.index_path.read_text(encoding="utf-8")

    assert result.svg_paths == ()
    assert load_builder_module().NOT_DRAWN in section_body(page, "key-figures")
