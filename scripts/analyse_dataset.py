#!/usr/bin/env python
"""Measure CamVid's class prior and write it to reports/dataset_stats.json.

This is the evidence behind the first claim in the README — that the classes
which can get someone killed are a rounding error in the pixel budget, and that
any pixel-averaged objective is therefore right to ignore them.

Run::

    python scripts/analyse_dataset.py --root /path/to/CamVid

The script writes machine-readable output and prints a table. Nothing here needs
a GPU, a model, or a network connection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drivemetrics.data.camvid import (  # noqa: E402
    build_manifest,
    class_pixel_distribution,
    load_split,
)

SAFETY_CRITICAL = ("Pedestrian", "Bicyclist", "Pole")
BACKGROUND = ("Road", "Building", "Sky")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="CamVid root (holds train.csv, val.csv)")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "dataset_stats.json")
    p.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "reports" / "split_manifest.json",
        help="where to write the frozen split manifest (SHA-256 per file)",
    )
    p.add_argument("--no-manifest", action="store_true", help="skip hashing (much faster)")
    return p.parse_args(argv)


def _fmt_row(name: str, block: dict) -> str:
    share = block["pixel_share"]
    img_share = block["image_share"]
    bar = "█" * max(0, int(round((share or 0) * 100)))
    return (
        f"  {name:<12} {block['pixels']:>12,}  {100 * (share or 0):>6.3f}%   "
        f"{100 * (img_share or 0):>5.1f}%  {bar}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.root.exists():
        print(f"error: CamVid root not found: {args.root}", file=sys.stderr)
        return 2

    report = {"dataset": "CamVid-11", "root": str(args.root), "splits": {}}

    for split_name in args.splits:
        split = load_split(args.root, split_name)
        stats = class_pixel_distribution(split)
        report["splits"][split_name] = stats

        print(f"\n=== {split_name}: {stats['n_images']} images, "
              f"{stats['total_labelled_pixels']:,} labelled pixels "
              f"({stats['void_pixels']:,} void) ===")
        print(f"  {'class':<12} {'pixels':>12}  {'share':>7}   {'in imgs':>6}")
        ordered = sorted(
            stats["per_class"].items(), key=lambda kv: kv[1]["pixel_share"] or 0, reverse=True
        )
        for name, block in ordered:
            print(_fmt_row(name, block))

        crit = sum(stats["per_class"][c]["pixel_share"] or 0 for c in SAFETY_CRITICAL)
        bg = sum(stats["per_class"][c]["pixel_share"] or 0 for c in BACKGROUND)
        stats["summary"] = {
            "safety_critical_classes": list(SAFETY_CRITICAL),
            "safety_critical_pixel_share": crit,
            "background_classes": list(BACKGROUND),
            "background_pixel_share": bg,
            "ratio_background_to_safety_critical": (bg / crit) if crit else None,
        }
        print(f"\n  {'/'.join(SAFETY_CRITICAL)}: {100 * crit:.2f}% of labelled pixels")
        print(f"  {'/'.join(BACKGROUND)}: {100 * bg:.2f}%")
        if crit:
            print(f"  -> background outweighs safety-critical by {bg / crit:.0f}x")

    if not args.no_manifest:
        print("\nhashing files for the split manifest ...")
        manifest = build_manifest(
            args.root,
            splits=args.splits,
            notes=(
                "CamVid is not redistributed by this repository. This manifest records "
                "the exact bytes used for the reported results."
            ),
        )
        path = manifest.write(args.manifest)
        total = sum(len(v) for v in manifest.splits.values())
        print(f"  wrote {path.relative_to(REPO_ROOT)} ({total} samples, no cross-split duplicates)")
        report["split_manifest"] = str(path.relative_to(REPO_ROOT)).replace("\\", "/")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
