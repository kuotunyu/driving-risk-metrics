"""Synthetic license-safe CamVid adapter integration test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from drivemetrics.data.manifest import DatasetManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "smoke" / "camvid.yaml"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "camvid_tiny"


def test_synthetic_camvid_manifest_loads_and_prepares_without_randomness() -> None:
    try:
        from drivemetrics.data.camvid import CamVidAdapter, load_camvid_config
    except ImportError:
        pytest.fail("drivemetrics.data.camvid is missing", pytrace=False)

    image_sha = hashlib.sha256((FIXTURE_ROOT / "image.ppm").read_bytes()).hexdigest()
    mask_sha = hashlib.sha256((FIXTURE_ROOT / "mask.pgm").read_bytes()).hexdigest()
    values: dict[str, object] = {
        "dataset_name": "camvid",
        "dataset_version": "synthetic-test-v1",
        "split_name": "smoke",
        "sample_ids": ("synthetic-0001",),
        "relative_image_paths": ("image.ppm",),
        "relative_label_paths": ("mask.pgm",),
        "file_sha256": (image_sha, mask_sha),
        "ineligible_sample_ids": (),
        "ineligibility_reasons": (),
    }
    canonical = json.dumps(
        values,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    manifest = DatasetManifest(
        **values,  # type: ignore[arg-type]
        manifest_sha256=hashlib.sha256(canonical).hexdigest(),
    )
    adapter = CamVidAdapter(
        manifest,
        FIXTURE_ROOT,
        FIXTURE_ROOT,
        load_camvid_config(CONFIG_PATH),
    )

    assert len(adapter) == 1
    first = adapter.prepare(0, training=True, flip_draw=0.25)
    second = adapter.prepare(0, training=True, flip_draw=0.25)
    unflipped = adapter.prepare(0, training=True, flip_draw=0.75)

    assert first.image_chw.shape == (3, 512, 1024)
    assert first.mask_hw.shape == (512, 1024)
    np.testing.assert_array_equal(first.image_chw, second.image_chw)
    np.testing.assert_array_equal(first.mask_hw, second.mask_hw)
    assert first.mask_hw[0, 256] == 1
    assert first.mask_hw[0, 767] == 0
    assert unflipped.mask_hw[0, 256] == 0
    assert unflipped.mask_hw[0, 767] == 1
