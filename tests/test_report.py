"""Tests for the report layer.

The report is where numbers become claims, so the properties pinned here are
mostly about honesty rather than rendering: that a synthetic run cannot be
mistaken for a real one, that the assumptions travel with the figures, and that
the page cannot silently blend two evaluation runs.

Matplotlib is required for these; they skip cleanly without it so the metric
suite still runs on a bare install.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from drivemetrics import PROFILES, degrade, evaluate_pairs, sweep_harm_model  # noqa: E402
from drivemetrics.metrics.confusion import ConfusionMatrix  # noqa: E402
from drivemetrics.report.html import build_report  # noqa: E402
from drivemetrics.taxonomy import class_index  # noqa: E402

PED = class_index("Pedestrian")
ROAD = class_index("Road")
SKY = class_index("Sky")
BUILDING = class_index("Building")


def _scene(h=48, w=64):
    t = np.full((h, w), ROAD, dtype=np.int64)
    t[: h // 3] = SKY
    t[h // 3 : h // 2] = BUILDING
    t[h // 2 : h // 2 + 12, 8:20] = PED
    return t


def _run(n=4):
    """Evaluate every synthetic profile and build the comparison dict."""
    bundles, matrices = {}, {}
    base = _scene()
    for name, profile in PROFILES.items():
        rng = np.random.default_rng(0)
        pairs, cm = [], ConfusionMatrix()
        for i in range(n):
            pred = degrade(base, profile, rng)
            cm.update(base, pred)
            pairs.append((f"img{i}", base, pred))
        b = evaluate_pairs(pairs, model=name, split="val",
                           provenance={"synthetic": True})
        bundles[name] = b.as_dict()
        bundles[name]["protocol_gap"] = b.protocol_gap
        matrices[name] = cm

    sweep = sweep_harm_model(matrices)
    comparison = {
        "split": "val",
        "synthetic": True,
        "synthetic_warning": "SYNTHETIC pipeline fixture",
        "models": {
            k: dict(v["headline"], protocol_gap=v["protocol_gap"])
            for k, v in bundles.items()
        },
        "harm_sweep": sweep.as_dict(),
        "rank_stability": {},
        "harm_model": {"name": "default-v1", "notes": "ordinal tiers"},
        "camera": bundles[next(iter(bundles))]["provenance"]["camera"],
    }
    return comparison, bundles


def test_report_renders_and_is_self_contained(tmp_path):
    comparison, bundles = _run()
    out = build_report(comparison, bundles, tmp_path / "r.html")
    doc = out.read_text(encoding="utf-8")

    assert doc.startswith("<!doctype html>")
    assert doc.rstrip().endswith("</html>")
    # Every image must be inlined; a linked asset would break when the file moves.
    assert "data:image/png;base64," in doc
    assert 'src="http' not in doc and "src='http" not in doc
    assert "<figure>" in doc


def test_synthetic_run_is_banner_marked(tmp_path):
    comparison, bundles = _run()
    doc = build_report(comparison, bundles, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Synthetic" in doc
    assert "not model results" in doc.lower()
    assert "banner" in doc


def test_real_run_has_no_synthetic_banner(tmp_path):
    comparison, bundles = _run()
    comparison["synthetic"] = False
    comparison.pop("synthetic_warning", None)
    doc = build_report(comparison, bundles, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "not model results" not in doc.lower()


def test_assumptions_are_printed_not_merely_referenced(tmp_path):
    """A reader must be able to see the harm model and camera without leaving."""
    comparison, bundles = _run()
    doc = build_report(comparison, bundles, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "default-v1" in doc
    assert "horizon row" in doc
    assert "assumed" in doc


def test_class_count_is_shown_next_to_every_miou(tmp_path):
    """The three-versus-eleven-class trap must be visible in the table itself."""
    comparison, bundles = _run()
    doc = build_report(comparison, bundles, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<th>classes</th>" in doc
    assert "must not be compared" in doc


def test_both_protocols_are_disclosed(tmp_path):
    comparison, bundles = _run()
    doc = build_report(comparison, bundles, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "protocol gap" in doc
    assert "nanmean" in doc


def test_no_unrendered_placeholders(tmp_path):
    comparison, bundles = _run()
    doc = build_report(comparison, bundles, tmp_path / "r.html").read_text(encoding="utf-8")
    body = doc.split("</style>")[-1]
    # Template braces or a stringified None reaching the page means a formatting bug.
    assert "{" not in body and "}" not in body
    assert ">None<" not in body


def test_html_escaping_of_model_names(tmp_path):
    """Model names come from the command line and must not inject markup."""
    comparison, bundles = _run()
    name = next(iter(bundles))
    evil = "<script>alert(1)</script>"
    bundles[evil] = bundles.pop(name)
    bundles[evil]["model"] = evil
    comparison["models"][evil] = comparison["models"].pop(name)

    doc = build_report(comparison, bundles, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc


def test_missing_stratification_does_not_break_the_report(tmp_path):
    comparison, bundles = _run()
    for b in bundles.values():
        b["stratified"] = {}
    comparison["camera"] = None
    out = build_report(comparison, bundles, tmp_path / "r.html")
    assert out.exists()


def test_report_matches_the_json_it_was_built_from(tmp_path):
    """The page and the machine-readable evidence must not drift apart."""
    comparison, bundles = _run()
    doc = build_report(comparison, bundles, tmp_path / "r.html").read_text(encoding="utf-8")
    for name, bundle in bundles.items():
        miou = bundle["headline"]["mean_iou"]
        assert f"{miou:.4f}" in doc, f"{name}'s mIoU is missing from the report"
