"""Learn the classwise area tertiles from a cohort's instance bitmasks, and freeze them.

The frozen `area_tertiles.json` the extended metrics consume was produced at P1-14
from `instance_areas` over the training cohort, and the script that produced it
was not kept. The file could be cited but not regenerated. This module is that
producer, written so that running it over the same cohort reproduces the frozen
file byte for byte — the only acceptable proof that it implements the same
definition rather than a plausible one.

Areas are whole-instance bitmask areas, exactly as `instance_areas` counts them,
with no semantic-mask filter. That is what the frozen edges were learned over. A
producer that quietly changed the definition would be minting new edges under the
old file's name, and every tertile in the published evidence would move.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from drivemetrics.data.bdd100k import instance_areas
from drivemetrics.data.manifest import load_manifest
from drivemetrics.metrics.instances import learn_area_tertiles


@dataclass(frozen=True)
class TertilesResult:
    """Where the edges were written and what they were learned from."""

    output_path: Path
    eligible_images: int
    missing_bitmasks: tuple[str, ...]
    total_instances: int


def learn_tertiles_from_bitmasks(
    manifest_path: Path, instance_root: Path, output_path: Path
) -> TertilesResult:
    """Walk one frozen cohort's bitmasks and write the classwise area tertiles.

    The cohort is the manifest's, never a directory listing: a bitmask that is not
    in the manifest is not in the study, and a manifest image without a bitmask is
    counted as missing rather than invented. Edges are keyed by BDD100K instance
    category, the space the bitmask itself uses, which is why the extended metrics
    translate them before comparing anything to a semantic train ID.
    """

    if output_path.exists():
        raise FileExistsError(f"frozen area tertiles already exist: {output_path}")

    manifest = load_manifest(manifest_path)
    observations: list[tuple[int, int]] = []
    per_category: dict[int, int] = {}
    missing: list[str] = []
    eligible_images = 0
    for sample_id in manifest.sample_ids:
        path = instance_root / f"{sample_id}.png"
        if not path.exists():
            missing.append(sample_id)
            continue
        with Image.open(path) as image:
            bitmask = np.asarray(image, dtype=np.uint8)
        eligible_images += 1
        for category, area in instance_areas(bitmask):
            observations.append((category, area))
            per_category[category] = per_category.get(category, 0) + 1

    if not observations:
        raise ValueError(f"no instances were found for this cohort under {instance_root}")

    edges = learn_area_tertiles(observations)
    document = {
        "eligible_images": eligible_images,
        "instances_per_category": {
            str(category): count for category, count in sorted(per_category.items())
        },
        "learned_from": manifest.split_name,
        "tertile_edges": {
            str(category): [low, high] for category, (low, high) in sorted(edges.items())
        },
        "total_instances": len(observations),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return TertilesResult(
        output_path=output_path,
        eligible_images=eligible_images,
        missing_bitmasks=tuple(missing),
        total_instances=len(observations),
    )
