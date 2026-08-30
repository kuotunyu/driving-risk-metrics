#!/usr/bin/env python
"""Audit training notebooks for results that cannot be compared to each other.

Motivation: four CamVid notebooks sharing one ``iou()`` implementation reported
mIoU values of 0.8157, 0.6780, 0.5657 and 0.3808, and those values were compared
directly. They were not comparable — one of them was a 3-class run whose stored
output happened to look like the others.

This script re-derives that finding from the notebook files rather than trusting
anybody's summary of them. For each notebook it reports:

* every ``pix_acc / meanIoU / IoUs`` line found in the *stored cell outputs*,
* how many per-class values each of those lines actually contains,
* the number of classes the *source* declares, and
* any numeric claim in a markdown cell that disagrees with the outputs.

Run::

    python scripts/audit_notebooks.py path/to/notebooks/ --out reports/notebook_audit.json

Nothing is executed. Notebooks are parsed as JSON, so this is safe to run on
material of unknown provenance.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# "epoch7, pix_acc: 0.88, meanIoU: 0.57, IoUs: [0.9 0.78 ...]"
EVAL_RE = re.compile(
    r"epoch\s*(\d+)\s*,\s*pix_acc:\s*([0-9.eE+-]+)\s*,\s*"
    r"meanIoU:\s*([0-9.eE+-]+)\s*,\s*IoUs:\s*\[([^\]]*)\]",
    re.S,
)
NUM_CLASS_RE = re.compile(r"^\s*num_class(?:es)?\s*=\s*(\d+)", re.M)
CLAIM_RE = re.compile(
    r"(?:highest\s+mIOU\s+is\s+([0-9.]+)"
    r"|highest\s+pixel\s+accuracy\s+is\s+([0-9.]+)"
    r"|pixel\s+acc\s*=\s*([0-9.]+)\s+and\s+([0-9.]+)\s+mIoU)",
    re.I,
)


def _cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _output_text(cell: dict) -> str:
    chunks: list[str] = []
    for out in cell.get("outputs", []) or []:
        txt = out.get("text")
        if txt is None:
            txt = (out.get("data") or {}).get("text/plain")
        if txt:
            chunks.append("".join(txt) if isinstance(txt, list) else str(txt))
    return "".join(chunks)


def audit_notebook(path: Path) -> dict | None:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"notebook": path.name, "error": f"unreadable: {exc}"}

    cells = nb.get("cells", [])
    source = "\n".join(_cell_text(c) for c in cells if c.get("cell_type") == "code")
    outputs = "".join(_output_text(c) for c in cells if c.get("cell_type") == "code")
    markdown = "\n".join(_cell_text(c) for c in cells if c.get("cell_type") == "markdown")

    evals = []
    for epoch, pix, miou, ious in EVAL_RE.findall(outputs):
        values = ious.replace("\n", " ").split()
        evals.append(
            {
                "epoch": int(epoch),
                "pixel_accuracy": float(pix),
                "mean_iou": float(miou),
                "n_class_values": len(values),
            }
        )
    if not evals:
        return None  # not an evaluation notebook; nothing to audit

    declared = NUM_CLASS_RE.search(source)
    declared_n = int(declared.group(1)) if declared else None
    observed = sorted({e["n_class_values"] for e in evals})
    best = max(evals, key=lambda e: e["mean_iou"])
    final = evals[-1]

    claims = []
    for m in CLAIM_RE.finditer(markdown):
        miou_claim = m.group(1) or m.group(4)
        acc_claim = m.group(2) or m.group(3)
        claims.append(
            {
                "text": m.group(0).strip(),
                "claimed_miou": float(miou_claim) if miou_claim else None,
                "claimed_pixel_accuracy": float(acc_claim) if acc_claim else None,
            }
        )

    flags = []
    if declared_n is not None and any(n != declared_n for n in observed):
        flags.append(
            f"source declares num_class={declared_n} but stored output has "
            f"{observed} per-class values — results are from a different run "
            f"than the code, or from a different class count"
        )
    if len(observed) > 1:
        flags.append(f"inconsistent class counts within one notebook: {observed}")
    for c in claims:
        if c["claimed_miou"] is not None and not any(
            abs(c["claimed_miou"] - e["mean_iou"]) < 1e-6 for e in evals
        ):
            flags.append(
                f"markdown claims mIoU {c['claimed_miou']} which appears in no cell output "
                f"(best observed {best['mean_iou']:.4f})"
            )

    return {
        "notebook": path.name,
        "n_evaluations": len(evals),
        "declared_num_class": declared_n,
        "class_values_in_output": observed,
        "final": final,
        "best_by_miou": best,
        "markdown_claims": claims,
        "flags": flags,
        "comparable": not flags,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("paths", nargs="+", type=Path, help="notebook files or directories")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "notebook_audit.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.ipynb")))
        elif p.suffix == ".ipynb":
            files.append(p)
    if not files:
        print("no notebooks found", file=sys.stderr)
        return 2

    results = [r for r in (audit_notebook(f) for f in files) if r]
    if not results:
        print(f"scanned {len(files)} notebook(s); none contained evaluation output")
        return 0

    print(f"{'notebook':<42}{'classes':>8}{'declared':>9}{'final mIoU':>12}  flags")
    print("-" * 88)
    for r in sorted(results, key=lambda r: r.get("final", {}).get("mean_iou", 0), reverse=True):
        if "error" in r:
            print(f"{r['notebook']:<42}  {r['error']}")
            continue
        obs = ",".join(str(n) for n in r["class_values_in_output"])
        dec = r["declared_num_class"] if r["declared_num_class"] is not None else "?"
        mark = "  <-- " + r["flags"][0][:60] if r["flags"] else ""
        print(
            f"{r['notebook']:<42}{obs:>8}{str(dec):>9}"
            f"{r['final']['mean_iou']:>12.4f}{mark}"
        )

    incomparable = [r for r in results if r.get("flags")]
    print()
    if incomparable:
        print(f"{len(incomparable)} of {len(results)} notebooks carry results that cannot be")
        print("compared with the others as reported:")
        for r in incomparable:
            print(f"\n  {r['notebook']}")
            for f in r["flags"]:
                print(f"    - {f}")
    else:
        print("all scanned notebooks report mutually comparable results")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"n_notebooks": len(results), "results": results}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
