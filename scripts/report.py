#!/usr/bin/env python
"""Render the HTML evaluation report from the JSON written by evaluate.py.

    python scripts/evaluate.py --root /path/to/CamVid --split val --synthetic
    python scripts/report.py

Reads ``reports/comparison.json`` and ``reports/eval/*.json`` and writes a
single self-contained ``reports/report.html``. The report is built from the JSON
rather than from live objects on purpose: the page and the machine-readable
evidence are then guaranteed to agree, because they are the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drivemetrics.report.html import build_report  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--reports", type=Path, default=REPO_ROOT / "reports")
    p.add_argument("--out", type=Path, default=None, help="defaults to <reports>/report.html")
    p.add_argument("--title", default="Driving segmentation risk report")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    comparison_path = args.reports / "comparison.json"
    if not comparison_path.exists():
        print(
            f"error: {comparison_path} not found. Run scripts/evaluate.py first.",
            file=sys.stderr,
        )
        return 2

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    # comparison.json defines what the current run contained. Bundles left over
    # from an earlier run are ignored rather than silently mixed in — a report
    # that blends two runs is exactly the kind of untraceable artefact this
    # repository exists to argue against.
    wanted = set(comparison.get("models", {}))
    eval_dir = args.reports / "eval"
    bundles = {}
    stale = []
    for path in sorted(eval_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data["model"]
        if wanted and name not in wanted:
            stale.append(path.name)
            continue
        bundles[name] = data

    if stale:
        print(
            f"ignoring {len(stale)} bundle(s) not in this run: {', '.join(stale)}",
            file=sys.stderr,
        )
    missing = wanted - set(bundles)
    if missing:
        print(
            f"error: comparison.json lists {sorted(missing)} but no bundle was found "
            f"in {eval_dir}. Re-run scripts/evaluate.py.",
            file=sys.stderr,
        )
        return 2
    if not bundles:
        print(f"error: no bundles in {eval_dir}", file=sys.stderr)
        return 2

    stats_path = args.reports / "dataset_stats.json"
    dataset_stats = (
        json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else None
    )

    out = args.out or (args.reports / "report.html")
    build_report(comparison, bundles, out, dataset_stats=dataset_stats, title=args.title)

    size_kb = out.stat().st_size / 1024
    print(f"wrote {out.relative_to(REPO_ROOT)} ({size_kb:.0f} KB, {len(bundles)} models)")
    if comparison.get("synthetic"):
        print("NOTE: this report is built from synthetic fixtures, not model outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
