#!/usr/bin/env python
"""Evaluate one or more models on a split and write the full evidence bundle.

Two modes.

**Real predictions** — point at directories of index-mask PNGs::

    python scripts/evaluate.py --root /path/to/CamVid --split val \
        --predictions fcn8s=preds/fcn8s deeplabv3=preds/deeplabv3

**Synthetic fixtures** — no model needed, for validating the pipeline::

    python scripts/evaluate.py --root /path/to/CamVid --split val --synthetic

Synthetic runs are stamped ``synthetic: true`` throughout and must never be
quoted as segmentation results. They exist so the pipeline can be debugged
before GPU time is spent, and so CI can exercise the report without a
checkpoint.

Output goes to ``reports/eval/<model>.json``, plus ``reports/comparison.json``
holding the cross-model table and the harm-model sensitivity sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drivemetrics import (  # noqa: E402
    CAMVID_DEFAULT_CAMERA,
    DEFAULT_HARM,
    PROFILES,
    SYNTHETIC_WARNING,
    ConfusionMatrix,
    degrade,
    evaluate_pairs,
    rank_stability,
    sweep_harm_model,
)
from drivemetrics.data.camvid import load_split, read_mask  # noqa: E402
from drivemetrics.synthetic import profile_summary  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, required=True, help="CamVid root")
    p.add_argument("--split", default="val")
    p.add_argument(
        "--predictions",
        nargs="*",
        default=[],
        metavar="NAME=DIR",
        help="model name and its prediction directory of index-mask PNGs",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="generate synthetic predictions instead of loading real ones",
    )
    p.add_argument("--seed", type=int, default=0, help="seed for synthetic generation")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    p.add_argument(
        "--no-geometry",
        action="store_true",
        help="skip distance stratification (also skips the camera assumptions)",
    )
    return p.parse_args(argv)


def _synthetic_pairs(split, profile, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    for sample in split:
        target = read_mask(sample.label)
        yield sample.sample_id, target, degrade(target, profile, rng)


def _real_pairs(root, split_name, prediction_dir):
    from drivemetrics.evaluate import pairs_from_directory

    return pairs_from_directory(root, split_name, prediction_dir)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.synthetic and not args.predictions:
        print("error: pass --predictions NAME=DIR ... or --synthetic", file=sys.stderr)
        return 2
    if not args.root.exists():
        print(f"error: CamVid root not found: {args.root}", file=sys.stderr)
        return 2

    camera = None if args.no_geometry else CAMVID_DEFAULT_CAMERA
    eval_dir = args.out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.root, args.split)
    bundles = {}
    matrices = {}

    if args.synthetic:
        print(f"!! {SYNTHETIC_WARNING}\n")
        jobs = [
            (name, lambda p=profile: _synthetic_pairs(split, p, args.seed))
            for name, profile in PROFILES.items()
        ]
    else:
        jobs = []
        for spec in args.predictions:
            if "=" not in spec:
                print(f"error: expected NAME=DIR, got {spec!r}", file=sys.stderr)
                return 2
            name, _, d = spec.partition("=")
            jobs.append((name, lambda d=d: _real_pairs(args.root, args.split, Path(d))))

    for name, make_pairs in jobs:
        print(f"evaluating {name} on {args.split} ({len(split)} images) ...")
        provenance = {"synthetic": args.synthetic, "split_size": len(split)}
        if args.synthetic:
            provenance["synthetic_warning"] = SYNTHETIC_WARNING
            provenance["profile"] = PROFILES[name].description

        # The confusion matrix is rebuilt here so the sweep can re-rank models
        # without another traversal of the images.
        cm = ConfusionMatrix()
        pairs = []
        for sid, t, pr in make_pairs():
            cm.update(t, pr)
            pairs.append((sid, t, pr))

        bundle = evaluate_pairs(
            pairs,
            model=name,
            split=args.split,
            harm=DEFAULT_HARM,
            camera=camera,
            provenance=provenance,
        )
        bundles[name] = bundle
        matrices[name] = cm
        bundle.write(eval_dir / f"{name}.json")

        h = bundle.as_dict()["headline"]
        gap = bundle.protocol_gap
        print(
            f"  mIoU {h['mean_iou']:.4f} ({h['n_classes_counted']} classes)"
            f"   pixAcc {h['pixel_accuracy']:.4f}"
            f"   risk-skill {h['risk_skill']:.4f}"
            f"   VRU recall {h['vru_recall']:.4f}"
        )
        if gap is not None:
            print(f"  per-image protocol would report {gap:+.4f} mIoU vs dataset protocol")
        for cls, st in bundle.blind_spot.items():
            if st["present_images"]:
                print(
                    f"  blind-spot {cls:<11} {st['blind_images']}/{st['present_images']} "
                    f"images ({100 * st['blind_rate']:.1f}%)"
                )

    # -- cross-model comparison + sensitivity ------------------------------
    print("\nsweeping harm model over VRU weight 0.1x .. 10x ...")
    sweep = sweep_harm_model(matrices)
    stability = rank_stability(sweep)

    print(f"\n{'model':<26}{'mIoU':>9}{'rank':>6}{'risk rank':>11}  stability")
    for name in sorted(bundles, key=lambda n: sweep.miou_rank[n]):
        st = stability[name]
        rng = f"{st['best_rank']}" if st["stable"] else f"{st['best_rank']}-{st['worst_rank']}"
        flag = "" if not st["rank_changed_vs_miou"] else "  <-- differs from mIoU rank"
        print(
            f"{name:<26}{bundles[name].dataset_iou['mean_iou']:>9.4f}"
            f"{st['miou_rank']:>6}{rng:>11}{flag}"
        )

    print(
        f"\nrisk ranking differs from mIoU ranking at "
        f"{100 * sweep.disagreement_fraction:.0f}% of swept harm models"
    )
    if sweep.disagreement_fraction == 0:
        print("  -> on this set, weighting by harm does not change the ordering")
    elif sweep.disagreement_fraction == 1:
        print("  -> the two metrics disagree at every setting swept")

    comparison = {
        "split": args.split,
        "synthetic": args.synthetic,
        "models": {
            name: b.as_dict()["headline"] | {"protocol_gap": b.protocol_gap}
            for name, b in bundles.items()
        },
        "harm_sweep": sweep.as_dict(),
        "rank_stability": stability,
        "harm_model": {"name": DEFAULT_HARM.name, "notes": DEFAULT_HARM.notes},
        "camera": camera.as_dict() if camera else None,
    }
    if args.synthetic:
        comparison["synthetic_warning"] = SYNTHETIC_WARNING
        comparison["synthetic_profiles"] = profile_summary()

    out = args.out / "comparison.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)} and {len(bundles)} bundle(s) in {eval_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
