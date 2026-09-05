"""Contracts for the producer of the frozen area tertiles.

The frozen edges were cited by the published evidence and could not be
regenerated: the script that made them was not kept. This producer exists so the
file can be reproduced byte for byte, which is the only proof that it implements
the definition the edges were actually learned under — whole-instance bitmask
areas, no semantic filter, keyed by BDD100K instance category.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from drivemetrics.data.manifest import build_paired_manifest, save_manifest
from drivemetrics.metrics.instances import learn_area_tertiles

SAMPLES = ("v0001", "v0002", "v0003")
HEIGHT, WIDTH = 6, 6


def load_tertiles() -> ModuleType:
    try:
        from drivemetrics.data import tertiles
    except ImportError:
        pytest.fail("drivemetrics.data.tertiles is missing", pytrace=False)
    return tertiles


def place(
    mask: np.ndarray, rows: slice, columns: slice, *, category: int, annotation_id: int
) -> None:
    mask[rows, columns, 0] = category
    mask[rows, columns, 2] = annotation_id >> 8
    mask[rows, columns, 3] = annotation_id & 0xFF


def write_cohort(root: Path) -> Path:
    """A three-image manifest built the way the pipeline builds one."""

    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for sample_id in SAMPLES:
        Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).save(
            images / f"{sample_id}.jpg"
        )
        Image.fromarray(np.zeros((HEIGHT, WIDTH), dtype=np.uint8)).save(
            labels / f"{sample_id}_train_id.png"
        )
    manifest_path = root / "train.json"
    save_manifest(build_paired_manifest(images, labels, "train"), manifest_path)
    return manifest_path


#: Category 1 gets areas 4, 1 and 9; category 3 gets areas 2 and 6. Two of the
#: three images carry a bitmask; the third does not, and must be counted as
#: missing rather than invented.
def write_bitmasks(root: Path) -> Path:
    root.mkdir(parents=True)
    first = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    place(first, slice(0, 2), slice(0, 2), category=1, annotation_id=1)  # area 4
    place(first, slice(3, 4), slice(0, 1), category=1, annotation_id=2)  # area 1
    place(first, slice(4, 6), slice(0, 1), category=3, annotation_id=300)  # area 2
    Image.fromarray(first, mode="RGBA").save(root / "v0001.png")
    second = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    place(second, slice(0, 3), slice(0, 3), category=1, annotation_id=7)  # area 9
    place(second, slice(3, 6), slice(3, 5), category=3, annotation_id=8)  # area 6
    Image.fromarray(second, mode="RGBA").save(root / "v0002.png")
    return root


def test_the_frozen_document_shape_is_reproduced_from_the_bitmasks(tmp_path: Path) -> None:
    """The document has exactly the five fields the frozen file has, with the same meanings.

    Edges are hand-derivable: category 1 areas sorted are [1, 4, 9], so with three
    observations the rank indices (3-1)//3 and (6-1)//3 select 1 and 4; category 3
    areas [2, 6] select 2 and 6, because with two observations the rank indices are 0 and 1. The counts are per category and in total, and
    `eligible_images` is the number of manifest images that HAD a bitmask.
    """

    tertiles = load_tertiles()
    output = tmp_path / "out" / "area_tertiles.json"

    result = tertiles.learn_tertiles_from_bitmasks(
        write_cohort(tmp_path / "cohort"), write_bitmasks(tmp_path / "bitmasks"), output
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document == {
        "eligible_images": 2,
        "instances_per_category": {"1": 3, "3": 2},
        "learned_from": "train",
        "tertile_edges": {"1": [1, 4], "3": [2, 6]},
        "total_instances": 5,
    }
    assert document["tertile_edges"] == {
        str(k): list(v)
        for k, v in learn_area_tertiles([(1, 4), (1, 1), (3, 2), (1, 9), (3, 6)]).items()
    }
    assert result.eligible_images == 2
    assert result.missing_bitmasks == ("v0003",)
    assert result.total_instances == 5
    assert output.read_bytes() == (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def test_the_cohort_is_the_manifest_not_the_directory(tmp_path: Path) -> None:
    """A bitmask that is not in the manifest is not in the study."""

    tertiles = load_tertiles()
    bitmasks = write_bitmasks(tmp_path / "bitmasks")
    extra = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    place(extra, slice(0, 6), slice(0, 6), category=1, annotation_id=99)
    Image.fromarray(extra, mode="RGBA").save(bitmasks / "not-in-cohort.png")

    tertiles.learn_tertiles_from_bitmasks(
        write_cohort(tmp_path / "cohort"), bitmasks, tmp_path / "area_tertiles.json"
    )
    document = json.loads((tmp_path / "area_tertiles.json").read_text(encoding="utf-8"))

    assert document["total_instances"] == 5


def test_a_cohort_with_no_instances_is_refused(tmp_path: Path) -> None:
    """Edges learned from nothing would be a file that looks frozen and means nothing."""

    tertiles = load_tertiles()
    empty = tmp_path / "bitmasks"
    empty.mkdir()

    with pytest.raises(ValueError, match=r"^no instances were found for this cohort under "):
        tertiles.learn_tertiles_from_bitmasks(
            write_cohort(tmp_path / "cohort"), empty, tmp_path / "area_tertiles.json"
        )


def test_the_frozen_file_is_never_overwritten(tmp_path: Path) -> None:
    """Frozen means frozen; regenerating writes beside it, or fails."""

    tertiles = load_tertiles()
    output = tmp_path / "area_tertiles.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"^frozen area tertiles already exist: "):
        tertiles.learn_tertiles_from_bitmasks(
            write_cohort(tmp_path / "cohort"), write_bitmasks(tmp_path / "bitmasks"), output
        )
