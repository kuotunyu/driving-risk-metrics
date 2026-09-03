"""Contracts for reading instance areas out of BDD100K instance bitmasks.

The official bitmask packs four fields into an RGBA image: the category in red,
attribute flags in green, and a 16-bit annotation id split across blue and
alpha. Area tertiles are learned from these areas, and a tertile learned from a
misread bitmask would silently reclassify every instance in the study.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest


def load_bdd100k() -> ModuleType:
    from drivemetrics.data import bdd100k

    return bdd100k


def bitmask(height: int = 4, width: int = 4) -> np.ndarray:
    return np.zeros((height, width, 4), dtype=np.uint8)


def place(
    mask: np.ndarray,
    rows: slice,
    columns: slice,
    *,
    category: int,
    annotation_id: int,
) -> None:
    mask[rows, columns, 0] = category
    mask[rows, columns, 2] = annotation_id >> 8
    mask[rows, columns, 3] = annotation_id & 0xFF


def test_each_annotation_id_becomes_one_area_record() -> None:
    """Instance coverage weights every instance equally, so each needs its own area."""

    bdd100k = load_bdd100k()
    mask = bitmask()
    place(mask, slice(0, 2), slice(0, 2), category=3, annotation_id=1)
    place(mask, slice(2, 4), slice(0, 1), category=3, annotation_id=2)

    assert sorted(bdd100k.instance_areas(mask)) == [(3, 2), (3, 4)]


def test_background_pixels_produce_no_instance() -> None:
    """Annotation id zero is unlabeled background, not an instance of area N."""

    bdd100k = load_bdd100k()
    mask = bitmask()
    place(mask, slice(0, 1), slice(0, 1), category=5, annotation_id=7)

    assert bdd100k.instance_areas(mask) == ((5, 1),)


def test_the_annotation_id_spans_both_low_and_high_bytes() -> None:
    """Reading only the alpha channel would merge instances 1 and 257 into one."""

    bdd100k = load_bdd100k()
    mask = bitmask()
    place(mask, slice(0, 1), slice(0, 2), category=2, annotation_id=1)
    place(mask, slice(1, 2), slice(0, 1), category=2, annotation_id=257)

    assert sorted(bdd100k.instance_areas(mask)) == [(2, 1), (2, 2)]


def test_one_annotation_id_may_not_carry_two_categories() -> None:
    """A split category means the bitmask was misread or the file is corrupt."""

    bdd100k = load_bdd100k()
    mask = bitmask()
    place(mask, slice(0, 1), slice(0, 1), category=2, annotation_id=9)
    place(mask, slice(1, 2), slice(0, 1), category=6, annotation_id=9)

    with pytest.raises(ValueError, match=r"^annotation id"):
        bdd100k.instance_areas(mask)


def test_a_non_rgba_image_is_rejected() -> None:
    """A silently accepted RGB image would read the annotation id off by one channel."""

    bdd100k = load_bdd100k()

    with pytest.raises(ValueError, match=r"^instance bitmask must be RGBA, got shape"):
        bdd100k.instance_areas(np.zeros((4, 4, 3), dtype=np.uint8))


def test_a_non_uint8_image_is_rejected() -> None:
    """Byte packing is only meaningful over the raw uint8 channels."""

    bdd100k = load_bdd100k()

    with pytest.raises(ValueError, match=r"^instance bitmask must be"):
        bdd100k.instance_areas(np.zeros((4, 4, 4), dtype=np.uint16))


def test_an_empty_bitmask_yields_no_instances() -> None:
    """An unlabeled frame is legitimate and must not raise."""

    bdd100k = load_bdd100k()

    assert bdd100k.instance_areas(bitmask()) == ()
