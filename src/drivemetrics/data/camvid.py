"""Strict smoke-only CamVid configuration and manifest adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from drivemetrics.data.manifest import DatasetManifest
from drivemetrics.data.transforms import PreparedSample, prepare_sample


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CamVidDatasetConfig(_StrictModel):
    name: Literal["camvid"]
    purpose: Literal["smoke"]
    smoke_only: Literal[True]


class CamVidInputConfig(_StrictModel):
    target_height: Literal[512]
    canvas_width: Literal[1024]
    image_pad_value_after_normalization: float = Field(ge=0.0, le=0.0)
    mask_pad_value: Literal[255]
    horizontal_flip_probability: float = Field(ge=0.5, le=0.5)


class CamVidSmokeConfig(_StrictModel):
    schema_version: Literal["camvid-smoke/v1"]
    dataset: CamVidDatasetConfig
    input: CamVidInputConfig


def load_camvid_config(path: Path) -> CamVidSmokeConfig:
    """Load a fail-closed config that cannot label CamVid as formal evidence."""

    raw_value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise TypeError("CamVid config document must be a mapping")
    return CamVidSmokeConfig.model_validate(raw_value)


def _resolve_file(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*PurePosixPath(relative_path).parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"CamVid path escapes its root: {relative_path!r}") from exc
    return candidate


@dataclass(frozen=True)
class CamVidAdapter:
    """Resolve a CamVid smoke manifest and delegate geometry to the pure transform."""

    manifest: DatasetManifest
    image_root: Path
    label_root: Path
    config: CamVidSmokeConfig

    def __post_init__(self) -> None:
        if self.manifest.dataset_name != "camvid" or self.manifest.split_name != "smoke":
            raise ValueError("CamVid manifest must be explicitly marked as a smoke split")

    def __len__(self) -> int:
        return len(self.manifest.sample_ids)

    def prepare(self, index: int, *, training: bool, flip_draw: float) -> PreparedSample:
        """Read one manifest pair and apply deterministic caller-controlled preprocessing."""

        image_path = _resolve_file(self.image_root, self.manifest.relative_image_paths[index])
        label_path = _resolve_file(self.label_root, self.manifest.relative_label_paths[index])
        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
        with Image.open(label_path) as label_file:
            mask = np.asarray(label_file, dtype=np.uint8)
        return prepare_sample(image, mask, training=training, flip_draw=flip_draw)
