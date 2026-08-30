"""Self-contained HTML evaluation report.

One file, no external assets, figures embedded as data URIs — so it can be
attached to an email or dropped in a PR without a directory of PNGs going
missing. The report is generated from the JSON bundles rather than from live
objects, which means the page and the machine-readable evidence can never drift
apart: if a number appears here, it is in ``reports/`` too.

Design rules that are not negotiable in this report:

* A synthetic run is banner-marked at the top and on every table.
* The harm model and camera assumptions are printed, not linked.
* Both IoU protocols are shown wherever mIoU is shown.
* Blind-spot rates are shown as curves, never as a single chosen threshold.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..taxonomy import CLASS_NAMES
from .figures import (
    figure_to_data_uri,
    plot_blind_spot_curves,
    plot_distance_bands,
    plot_harm_sweep,
    plot_per_class_comparison,
    plot_risk_contributions,
    plot_share_vs_iou,
)

__all__ = ["build_report"]

_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fbfafc;color:#1a1720;
 font:16px/1.7 "Noto Sans TC",-apple-system,"Segoe UI",system-ui,sans-serif;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:0 26px 90px}
h1,h2,h3{line-height:1.25;margin:0;text-wrap:balance;font-weight:600}
h1{font-size:34px;letter-spacing:-.02em;margin-bottom:10px}
h2{font-size:23px;margin:56px 0 6px;letter-spacing:-.015em}
h3{font-size:17px;margin:30px 0 8px}
p{margin:0 0 16px}
code{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.87em;
 background:#f2f0f6;border:1px solid #e3dfe9;padding:.08em .36em;border-radius:4px}
header{padding:56px 0 26px;border-bottom:1px solid #e3dfe9}
.eyebrow{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.15em;
 text-transform:uppercase;color:#7c7688;margin:0 0 16px}
.deck{font-size:17px;color:#4a4553;max-width:62ch}
.banner{background:#fbe8e5;border:1.5px solid #b33a2b;border-radius:9px;
 padding:16px 20px;margin:24px 0;color:#7d2419}
.banner b{display:block;font-size:14px;letter-spacing:.04em;text-transform:uppercase;
 margin-bottom:6px}
.note{background:#fff;border:1px solid #e3dfe9;border-left:3px solid #7b3f8f;
 border-radius:0 8px 8px 0;padding:15px 20px;margin:20px 0;font-size:14.5px;color:#4a4553}
.scroll{overflow-x:auto;margin:20px 0;border:1px solid #e3dfe9;border-radius:9px;background:#fff}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid #e3dfe9}
thead th{background:#f4f2f7;font-weight:600;font-size:11.5px;letter-spacing:.03em;
 color:#4a4553;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
td.n{font-family:ui-monospace,monospace;text-align:right;font-variant-numeric:tabular-nums;
 white-space:nowrap}
tr.best td{background:#e2f1ea}
.bad{color:#b33a2b;font-weight:600}
.good{color:#26714f;font-weight:600}
figure{margin:26px 0;background:#fff;border:1px solid #e3dfe9;border-radius:10px;padding:18px}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:13px;color:#7c7688;margin-top:12px;padding-top:12px;
 border-top:1px solid #e3dfe9}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:20px 0}
.kv div{background:#fff;border:1px solid #e3dfe9;border-radius:8px;padding:14px 16px}
.kv .k{font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.1em;
 text-transform:uppercase;color:#7c7688;margin-bottom:5px}
.kv .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums}
.kv .s{font-size:12px;color:#7c7688;margin-top:3px}
footer{margin-top:70px;padding-top:26px;border-top:1px solid #e3dfe9;
 font-size:13px;color:#7c7688}
@media(prefers-color-scheme:dark){
 body{background:#131118;color:#efecf4}
 header,.scroll,figure,.kv div,.note{border-color:#2e2939}
 figure,.kv div,.note{background:#1b1823}
 .scroll{background:#1b1823}
 thead th{background:#221e2c;color:#b7b1c2}
 th,td{border-color:#2e2939}
 code{background:#221e2c;border-color:#2e2939}
 .deck,.note{color:#b7b1c2}
 .banner{background:#331915;border-color:#e87766;color:#f0b6ac}
 tr.best td{background:#132a21}
 .bad{color:#e87766}.good{color:#5fbf93}
 footer{border-color:#2e2939}
 figure img{filter:invert(.92) hue-rotate(180deg)}
}
"""


def _e(x) -> str:
    return _html.escape(str(x))


def _fmt(v, nd=4, dash="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return dash
    return f"{v:.{nd}f}"


def build_report(
    comparison: dict,
    bundles: dict,
    out_path: Path,
    dataset_stats: dict | None = None,
    title: str = "Driving segmentation risk report",
) -> Path:
    """Render the report. ``bundles`` maps model name to its bundle dict."""
    synthetic = bool(comparison.get("synthetic"))
    models = list(bundles)
    parts: list[str] = []

    # -- header ------------------------------------------------------------
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        f"<header><p class='eyebrow'>drivemetrics · {_e(comparison.get('split', 'val'))} split "
        f"· {_e(generated)}</p><h1>{_e(title)}</h1>"
        "<p class='deck'>Segmentation scored by what its mistakes would cost, "
        "not by how many pixels it got right.</p></header>"
    )

    if synthetic:
        parts.append(
            "<div class='banner'><b>Synthetic — not model results</b>"
            f"{_e(comparison.get('synthetic_warning', ''))} "
            "These figures validate the pipeline; they say nothing about any "
            "segmentation architecture.</div>"
        )

    # -- headline ----------------------------------------------------------
    parts.append("<h2>Headline</h2>")
    rows = []
    best_iou = max(
        (bundles[m]["headline"]["mean_iou"] or 0) for m in models
    )
    best_skill = max((bundles[m]["headline"]["risk_skill"] or -9e9) for m in models)
    for m in models:
        h = bundles[m]["headline"]
        cls_iou = "good" if h["mean_iou"] == best_iou else ""
        cls_sk = "good" if h["risk_skill"] == best_skill else ""
        vru = h["vru_recall"]
        vru_cls = "bad" if (vru is not None and vru < 0.5) else ""
        rows.append(
            f"<tr><td><code>{_e(m)}</code></td>"
            f"<td class='n'><span class='{cls_iou}'>{_fmt(h['mean_iou'])}</span></td>"
            f"<td class='n'>{h['n_classes_counted']}</td>"
            f"<td class='n'>{_fmt(h['pixel_accuracy'])}</td>"
            f"<td class='n'><span class='{cls_sk}'>{_fmt(h['risk_skill'])}</span></td>"
            f"<td class='n'><span class='{vru_cls}'>{_fmt(vru, 3)}</span></td>"
            f"<td class='n'>{_fmt(comparison.get('models', {}).get(m, {}).get('protocol_gap'), 4)}</td>"
            "</tr>"
        )
    parts.append(
        "<div class='scroll'><table><thead><tr>"
        "<th>model</th><th>mIoU</th><th>classes</th><th>pixel acc</th>"
        "<th>risk skill</th><th>VRU recall</th><th>protocol gap</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    parts.append(
        "<div class='note'><b>Reading this table.</b> <code>classes</code> is how many "
        "classes the mIoU actually averages over — a three-class mean and an "
        "eleven-class mean are different quantities and must not be compared. "
        "<code>protocol gap</code> is what the per-image <code>nanmean</code> "
        "aggregation would report <i>minus</i> the dataset-level figure, so a "
        "positive value means that protocol flatters the model and a negative one "
        "means it penalises it. Its sign is not fixed: per-image averaging weights "
        "every image equally regardless of size, which helps a model on some splits "
        "and hurts it on others. It is shown because on the source material this "
        "gap exceeded the spread between the architectures being compared. "
        "<code>risk skill</code> is 1 − risk ÷ (risk of predicting the majority "
        "class everywhere).</div>"
    )

    # -- the disagreement --------------------------------------------------
    sweep = comparison.get("harm_sweep", {})
    disagree = sweep.get("disagreement_fraction")
    if disagree is not None:
        stability = comparison.get("rank_stability", {})
        moved = [m for m, s in stability.items() if s.get("rank_changed_vs_miou")]
        parts.append("<h2>Does mIoU rank these models correctly?</h2>")
        if disagree > 0:
            parts.append(
                f"<p>Across harm models spanning a 100× range in how much a missed "
                f"vulnerable road user costs, the risk ranking differs from the mIoU "
                f"ranking at <b>{disagree * 100:.0f}%</b> of settings. "
                + (
                    f"{len(moved)} of {len(stability)} models change position: "
                    f"<code>{'</code>, <code>'.join(_e(m) for m in moved)}</code>."
                    if moved
                    else ""
                )
                + "</p>"
            )
        else:
            parts.append(
                "<p>The two rankings agree at every harm model swept. On this set, "
                "weighting errors by harm does not change which model you would "
                "pick — a negative result, and worth stating plainly.</p>"
            )
        parts.append(_figure(plot_harm_sweep(sweep),
            "Each line is one model's risk skill as the cost of missing a person is "
            "scaled from a tenth of the default to ten times it. Lines that cross "
            "mean the ranking depends on that choice."))

    # -- per-class ---------------------------------------------------------
    parts.append("<h2>Per class</h2>")
    parts.append(_figure(plot_per_class_comparison(bundles),
        "IoU by class. Safety-critical class labels are marked in red. A model can "
        "carry a respectable mean while scoring near zero on every one of them."))
    parts.append(_figure(plot_per_class_comparison(bundles, "per_class_recall"),
        "Recall by class — the share of each class's pixels actually recovered. For "
        "hazards this is the number that matters, because it isolates misses from "
        "false alarms."))

    if dataset_stats:
        split = comparison.get("split", "val")
        per_class = (
            dataset_stats.get("splits", {}).get(split, {}).get("per_class", {})
        )
        shares = {k: v.get("pixel_share") for k, v in per_class.items()}
        if shares and models:
            m0 = models[0]
            iou_list = bundles[m0]["iou"]["dataset_protocol"]["per_class_iou"]
            iou_map = {CLASS_NAMES[i]: iou_list[i] for i in range(len(CLASS_NAMES))}
            parts.append(_figure(plot_share_vs_iou(shares, iou_map, m0),
                "Competence plotted against how much of the image a class occupies. "
                "The classes in the bottom-left are the ones a car must not hit."))

    # -- blind spots -------------------------------------------------------
    parts.append("<h2>Blind spots</h2>")
    parts.append(
        "<p>How often a hazard that is present is not recovered at all. This "
        "cannot be raised by performing well on images that do not contain the "
        "hazard, which is what makes it hard to game.</p>"
    )
    bs_rows = []
    for m in models:
        for cls, st in bundles[m].get("blind_spot", {}).items():
            if not st["present_images"]:
                continue
            rate = st["blind_rate"]
            cls_attr = "bad" if rate and rate > 0.25 else ""
            bs_rows.append(
                f"<tr><td><code>{_e(m)}</code></td><td>{_e(cls)}</td>"
                f"<td class='n'>{st['present_images']}</td>"
                f"<td class='n'>{st['blind_images']}</td>"
                f"<td class='n'><span class='{cls_attr}'>{_fmt(rate, 3)}</span></td></tr>"
            )
    if bs_rows:
        parts.append(
            "<div class='scroll'><table><thead><tr><th>model</th><th>class</th>"
            "<th>images containing it</th><th>images blind</th><th>blind rate</th>"
            "</tr></thead><tbody>" + "".join(bs_rows) + "</tbody></table></div>"
        )
    for cls in ("Pedestrian", "Bicyclist"):
        parts.append(_figure(plot_blind_spot_curves(bundles, cls),
            f"Blind-spot rate for {cls} against the threshold that counts as "
            "'recovered'. A curve that leaps off zero means the model emits a few "
            "token pixels rather than finding the hazard."))

    # -- risk sources ------------------------------------------------------
    parts.append("<h2>Where the risk comes from</h2>")
    for m in models:
        top = bundles[m].get("risk_top_confusions", [])
        parts.append(_figure(plot_risk_contributions(top, m),
            f"The confusions carrying the most total risk for <code>{_e(m)}</code>, "
            "as a share of its total. This is what turns a low score into "
            "something actionable."))

    # -- distance ----------------------------------------------------------
    if any(bundles[m].get("stratified") for m in models):
        parts.append("<h2>By distance</h2>")
        parts.append(
            "<p>Distances come from a flat-ground pinhole model under assumed "
            "camera parameters — CamVid publishes no calibration. They are good "
            "enough to separate near from far, and not good enough to certify a "
            "range.</p>"
        )
        for cls in ("Pedestrian", "Car"):
            parts.append(_figure(plot_distance_bands(bundles, cls),
                f"{cls} IoU by approximate ground distance."))

    # -- assumptions -------------------------------------------------------
    parts.append("<h2>Assumptions this report rests on</h2>")
    harm = comparison.get("harm_model", {})
    cam = comparison.get("camera")
    parts.append("<div class='kv'>")
    parts.append(
        f"<div><div class='k'>harm model</div><div class='v'>{_e(harm.get('name', '—'))}</div>"
        "<div class='s'>ordinal tiers, not calibrated to injury data</div></div>"
    )
    if cam:
        parts.append(
            f"<div><div class='k'>horizon row</div><div class='v'>{_fmt(cam['horizon_row'], 1)}</div>"
            "<div class='s'>assumed</div></div>"
            f"<div><div class='k'>focal length</div><div class='v'>{_fmt(cam['focal_px'], 0)} px</div>"
            "<div class='s'>assumed</div></div>"
            f"<div><div class='k'>camera height</div><div class='v'>{_fmt(cam['height_m'], 2)} m</div>"
            "<div class='s'>assumed</div></div>"
        )
    parts.append("</div>")
    if harm.get("notes"):
        parts.append(f"<div class='note'>{_e(harm['notes'])}</div>")

    parts.append(
        "<footer><p>Generated by <code>drivemetrics</code>. Every number here is "
        "also in <code>reports/</code> as JSON, derived from a single traversal of "
        "the split. This is a research and teaching artefact; it is not a safety "
        "case and does not constitute validation of any vehicle.</p></footer>"
    )

    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
        f"<div class='wrap'>{''.join(parts)}</div></body></html>"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def _figure(fig, caption: str) -> str:
    uri = figure_to_data_uri(fig)
    return f"<figure><img src='{uri}' alt=''><figcaption>{caption}</figcaption></figure>"
