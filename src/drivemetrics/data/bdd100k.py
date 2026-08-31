"""BDD100K semantic filename conventions."""

from __future__ import annotations

from pathlib import Path

DATASET_NAME = "bdd100k"
DATASET_VERSION = "10k-semantic-v1"
IMAGE_SUFFIX = ".jpg"
TRAIN_ID_LABEL_SUFFIX = "_train_id.png"


def image_sample_id(path: Path) -> str:
    """Extract a sample ID from one official BDD100K image filename."""

    if path.suffix.lower() != IMAGE_SUFFIX:
        raise ValueError(f"unsupported BDD100K image filename: {path.name}")
    return path.stem


def label_sample_id(path: Path) -> str:
    """Extract a sample ID from one official semantic train-ID mask filename."""

    if not path.name.lower().endswith(TRAIN_ID_LABEL_SUFFIX):
        raise ValueError(f"unsupported BDD100K train-ID label filename: {path.name}")
    return path.name[: -len(TRAIN_ID_LABEL_SUFFIX)]
