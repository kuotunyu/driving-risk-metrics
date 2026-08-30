"""Figures for the evaluation report.

Every figure here answers one question, and each is designed so that the honest
reading is the easy reading — the safety-critical classes are always drawn in a
distinct colour, and any axis that could flatter a model by omission (a class
with no support, a threshold chosen after the fact) is drawn explicitly rather
than dropped.

Matplotlib is imported lazily inside each function so the evaluation core stays
numpy-only and CI can run the metric tests without a plotting stack.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence

import numpy as np

from ..taxonomy import CAMVID_11, CLASS_NAMES, VRU_CLASSES

__all__ = [
    "figure_to_data_uri",
    "plot_share_vs_iou",
    "plot_per_class_comparison",
    "plot_blind_spot_curves",
    "plot_risk_contributions",
    "plot_harm_sweep",
    "plot_distance_bands",
]

# A restrained palette. Hazard classes are always the warm colour so that the
# eye lands on them first; everything else stays quiet.
INK = "#1a1720"
MUTED = "#7c7688"
RULE = "#d9d4e0"
ACCENT = "#7b3f8f"
HAZARD = "#b33a2b"
OK = "#26714f"
SERIES = ["#7b3f8f", "#2f6f9f", "#b8860b", "#26714f", "#a0446f", "#5a5a6e"]

SAFETY_CRITICAL = set(VRU_CLASSES) | {
    i for i, c in enumerate(CAMVID_11) if c.name in ("Pole", "SignSymbol", "Car")
}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight="600", pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def figure_to_data_uri(fig, dpi: int = 150) -> str:
    """Render a figure to a base64 PNG data URI and close it.

    The report embeds figures rather than linking them so a single HTML file is
    self-contained and can be sent to someone without a directory of assets.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=True)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def plot_share_vs_iou(pixel_shares: dict, per_class_iou: dict, model: str):
    """The premise of the package, in one frame.

    Pixel share on one axis, achieved IoU on the other. If a model's competence
    tracks how much of the image a class occupies, the points line up — and the
    classes that matter for safety sit in the bottom-left corner where nobody
    looks.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, name in enumerate(CLASS_NAMES):
        share = pixel_shares.get(name)
        iou = per_class_iou.get(name)
        if share is None or iou is None:
            continue
        hazard = i in SAFETY_CRITICAL
        ax.scatter(
            share * 100,
            iou,
            s=90 if hazard else 60,
            color=HAZARD if hazard else ACCENT,
            alpha=0.9 if hazard else 0.55,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            name,
            (share * 100, iou),
            textcoords="offset points",
            xytext=(7, 3),
            fontsize=8,
            color=HAZARD if hazard else MUTED,
            fontweight="600" if hazard else "normal",
        )

    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    _style(
        ax,
        title=f"Competence tracks pixel share — {model}",
        xlabel="share of labelled pixels (%, log scale)",
        ylabel="IoU",
    )
    ax.axhline(0, color=RULE, linewidth=0.8)
    return fig


def plot_per_class_comparison(bundles: dict, metric: str = "per_class_iou"):
    """Grouped bars of a per-class metric across models.

    Ordered by training-set rarity so the eye travels from the classes that
    dominate the loss to the ones that dominate the risk.
    """
    import matplotlib.pyplot as plt

    names = list(CLASS_NAMES)
    models = list(bundles)
    n = len(models)
    x = np.arange(len(names))
    width = min(0.8 / max(n, 1), 0.28)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for k, model in enumerate(models):
        src = bundles[model]
        if metric == "per_class_iou":
            values = src["iou"]["dataset_protocol"]["per_class_iou"]
            vals = [values[i] if values[i] is not None else 0.0 for i in range(len(names))]
        else:
            table = src[metric]
            vals = [table.get(nm) or 0.0 for nm in names]
        ax.bar(
            x + (k - (n - 1) / 2) * width,
            vals,
            width,
            label=model,
            color=SERIES[k % len(SERIES)],
            alpha=0.9,
        )

    ax.set_xticks(x)
    labels = ax.set_xticklabels(names, rotation=35, ha="right")
    for i, lbl in enumerate(labels):
        if i in SAFETY_CRITICAL:
            lbl.set_color(HAZARD)
            lbl.set_fontweight("600")
    ax.set_ylim(0, 1.0)
    _style(ax, title=f"{metric.replace('_', ' ')} by class", ylabel="score")
    ax.legend(frameon=False, fontsize=8.5, ncol=min(n, 4), loc="upper right")
    return fig


def plot_blind_spot_curves(bundles: dict, class_name: str = "Pedestrian"):
    """Blind-spot rate against the recall threshold that defines 'recovered'.

    Publishing the curve rather than a point is the whole discipline here: a
    model whose curve leaps off zero is emitting a few token pixels, not
    finding the hazard.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    plotted = False
    for k, (model, bundle) in enumerate(bundles.items()):
        curve = bundle.get("blind_spot_curve", {}).get(class_name)
        if not curve:
            continue
        xs = [p[0] for p in curve]
        ys = [np.nan if p[1] is None else p[1] for p in curve]
        ax.plot(xs, ys, label=model, color=SERIES[k % len(SERIES)], linewidth=2, alpha=0.9)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, f"no {class_name} present", ha="center", color=MUTED, fontsize=10)

    ax.axvline(0.10, color=MUTED, linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate(
        "default operating point",
        (0.10, 0.02),
        textcoords="offset points",
        xytext=(6, 0),
        fontsize=8,
        color=MUTED,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, 1)
    _style(
        ax,
        title=f"Blind-spot rate — {class_name}",
        xlabel="recall threshold counted as 'recovered'",
        ylabel="fraction of images where it was missed",
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    return fig


def plot_risk_contributions(top: Sequence[dict], model: str, k: int = 8):
    """Which specific confusions carry the risk."""
    import matplotlib.pyplot as plt

    rows = list(top)[:k][::-1]
    if not rows:
        fig, ax = plt.subplots(figsize=(6.4, 2.0))
        ax.text(0.5, 0.5, "no errors", ha="center", color=OK, fontsize=11)
        ax.axis("off")
        return fig

    labels = [f"{r['true']} → {r['pred']}" for r in rows]
    shares = [r["risk_share"] * 100 for r in rows]
    colors = [
        HAZARD if r["true"] in ("Pedestrian", "Bicyclist", "Pole", "SignSymbol", "Car") else ACCENT
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(rows) + 1.2))
    ax.barh(np.arange(len(rows)), shares, color=colors, alpha=0.9, height=0.66)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8.5)
    for i, s in enumerate(shares):
        ax.text(s + 0.6, i, f"{s:.1f}%", va="center", fontsize=8, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    _style(ax, title=f"Where the risk comes from — {model}", xlabel="share of total risk (%)")
    ax.set_xlim(0, max(shares) * 1.18)
    return fig


def plot_harm_sweep(sweep: dict):
    """Risk-skill against the VRU weight multiplier, with mIoU rank annotated.

    Lines that cross are the finding: the ordering of these models depends on
    how much you decide a person is worth, which is a fact about the metric and
    not about the models.
    """
    import matplotlib.pyplot as plt

    factors = sweep["vru_weight_factors"]
    skills = sweep["risk_skill_by_model"]
    miou_rank = sweep.get("miou_rank", {})

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for k, (model, vals) in enumerate(skills.items()):
        ys = [np.nan if v is None else v for v in vals]
        rank = miou_rank.get(model)
        label = f"{model}  (mIoU rank {rank})" if rank else model
        ax.plot(
            factors, ys, marker="o", markersize=4.5, linewidth=2,
            label=label, color=SERIES[k % len(SERIES)], alpha=0.9,
        )

    ax.set_xscale("log")
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate("default harm model", (1.0, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(6, 8), fontsize=8, color=MUTED)
    _style(
        ax,
        title="Does the ranking survive a different harm model?",
        xlabel="multiplier on the VRU miss cost (log scale)",
        ylabel="risk skill (higher is better)",
    )
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    return fig


def plot_distance_bands(bundles: dict, class_name: str = "Pedestrian"):
    """Per-class IoU split by ground distance band."""
    import matplotlib.pyplot as plt

    models = list(bundles)
    first = bundles[models[0]].get("stratified", {})
    bands = first.get("bands", [])
    if not bands:
        fig, ax = plt.subplots(figsize=(6.4, 2.0))
        ax.text(0.5, 0.5, "distance stratification disabled", ha="center", color=MUTED)
        ax.axis("off")
        return fig

    x = np.arange(len(bands))
    width = min(0.8 / max(len(models), 1), 0.3)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for k, model in enumerate(models):
        per_band = bundles[model].get("stratified", {}).get("per_band", {})
        vals = []
        for b in bands:
            v = per_band.get(b, {}).get("per_class_iou", {}).get(class_name)
            vals.append(0.0 if v is None else v)
        ax.bar(
            x + (k - (len(models) - 1) / 2) * width, vals, width,
            label=model, color=SERIES[k % len(SERIES)], alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylim(0, 1.0)
    _style(
        ax,
        title=f"{class_name} IoU by ground distance",
        xlabel="approximate distance band (assumed camera geometry)",
        ylabel="IoU",
    )
    ax.legend(frameon=False, fontsize=8.5)
    return fig
