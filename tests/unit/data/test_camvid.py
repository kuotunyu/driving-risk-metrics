"""Tests for strict smoke-only CamVid configuration and adapter boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from drivemetrics.data.manifest import DatasetManifest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "smoke" / "camvid.yaml"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "camvid_tiny"


def load_camvid_module() -> ModuleType:
    try:
        from drivemetrics.data import camvid
    except ImportError:
        pytest.fail("drivemetrics.data.camvid is missing", pytrace=False)
    return camvid


def canonical_sha256(value: dict[str, object]) -> str:
    document = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(document).hexdigest()


def tiny_manifest(
    *,
    dataset_name: str = "camvid",
    split_name: str = "smoke",
) -> DatasetManifest:
    image_sha = hashlib.sha256((FIXTURE_ROOT / "image.ppm").read_bytes()).hexdigest()
    mask_sha = hashlib.sha256((FIXTURE_ROOT / "mask.pgm").read_bytes()).hexdigest()
    values: dict[str, object] = {
        "dataset_name": dataset_name,
        "dataset_version": "synthetic-test-v1",
        "split_name": split_name,
        "sample_ids": ("synthetic-0001",),
        "relative_image_paths": ("image.ppm",),
        "relative_label_paths": ("mask.pgm",),
        "file_sha256": (image_sha, mask_sha),
        "ineligible_sample_ids": (),
        "ineligibility_reasons": (),
    }
    return DatasetManifest(**values, manifest_sha256=canonical_sha256(values))  # type: ignore[arg-type]


def test_committed_camvid_config_is_explicitly_smoke_only() -> None:
    camvid = load_camvid_module()

    config = camvid.load_camvid_config(CONFIG_PATH)

    assert config.dataset.name == "camvid"
    assert config.dataset.purpose == "smoke"
    assert config.dataset.smoke_only is True


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "purpose: smoke",
            "purpose: formal",
            r"^1 validation error for CamVidSmokeConfig\ndataset\.purpose\n  Input should be 'smoke'",
        ),
        (
            "smoke_only: true",
            "smoke_only: false",
            r"^1 validation error for CamVidSmokeConfig\ndataset\.smoke_only\n  Input should be True",
        ),
        (
            "smoke_only: true",
            "smoke_only: true\n  unexpected: value",
            r"^1 validation error for CamVidSmokeConfig\ndataset\.unexpected\n  Extra inputs are not permitted",
        ),
    ],
)
def test_camvid_config_rejects_formal_label_or_unknown_key(
    tmp_path: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    camvid = load_camvid_module()
    path = tmp_path / "camvid.yaml"
    path.write_text(
        """schema_version: camvid-smoke/v1
dataset:
  name: camvid
  purpose: smoke
  smoke_only: true
input:
  target_height: 512
  canvas_width: 1024
  image_pad_value_after_normalization: 0.0
  mask_pad_value: 255
  horizontal_flip_probability: 0.5
""".replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=expected):
        camvid.load_camvid_config(path)


def test_camvid_adapter_rejects_non_camvid_manifest() -> None:
    camvid = load_camvid_module()
    config = camvid.load_camvid_config(CONFIG_PATH)

    with pytest.raises(ValueError, match=r"^CamVid manifest must be explicitly marked as a smoke"):
        camvid.CamVidAdapter(
            tiny_manifest(dataset_name="bdd100k"),
            FIXTURE_ROOT,
            FIXTURE_ROOT,
            config,
        )


def test_camvid_adapter_rejects_non_smoke_split() -> None:
    camvid = load_camvid_module()
    config = camvid.load_camvid_config(CONFIG_PATH)

    with pytest.raises(ValueError, match=r"^CamVid manifest must be explicitly marked as a smoke"):
        camvid.CamVidAdapter(
            tiny_manifest(split_name="formal"),
            FIXTURE_ROOT,
            FIXTURE_ROOT,
            config,
        )


def test_camvid_file_resolution_rejects_root_escape() -> None:
    camvid = load_camvid_module()

    with pytest.raises(ValueError, match=r"^CamVid path escapes its root:"):
        camvid._resolve_file(FIXTURE_ROOT / "nested", "../image.ppm")


def test_camvid_config_loader_rejects_non_mapping(tmp_path: Path) -> None:
    camvid = load_camvid_module()
    path = tmp_path / "camvid.yaml"
    path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(TypeError, match=r"^CamVid config document must be a mapping"):
        camvid.load_camvid_config(path)
