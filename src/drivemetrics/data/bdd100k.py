"""BDD100K semantic filename conventions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DATASET_NAME = "bdd100k"
DATASET_VERSION = "10k-semantic-v1"
IMAGE_SUFFIX = ".jpg"
TRAIN_ID_LABEL_SUFFIX = "_train_id.png"
NUM_TRAIN_CLASSES = 19

IntersectionDropReason = Literal[
    "missing_instance_annotation",
    "missing_semantic_annotation",
]


@dataclass(frozen=True)
class SemanticInstancePair:
    """One sample with both semantic and instance annotation paths."""

    sample_id: str
    semantic_label_path: Path
    instance_label_path: Path


@dataclass(frozen=True)
class IntersectionDrop:
    """One annotation excluded from an explicitly audited intersection."""

    sample_id: str
    reason: IntersectionDropReason
    annotation_path: Path


@dataclass(frozen=True)
class SemanticInstanceIntersection:
    """Deterministically paired annotations plus every exclusion reason."""

    pairs: tuple[SemanticInstancePair, ...]
    dropped: tuple[IntersectionDrop, ...]

    @property
    def retained_count(self) -> int:
        return len(self.pairs)


class SemanticInstanceMismatchError(ValueError):
    """Raised unless a cohort mismatch receives explicit audited opt-in."""

    def __init__(self, dropped: tuple[IntersectionDrop, ...]) -> None:
        self.dropped = dropped
        details = ", ".join(f"{item.sample_id}:{item.reason}" for item in dropped)
        super().__init__(
            "semantic/instance sample IDs differ; review exclusions and pass "
            f"allow_audited_intersection=True to proceed ({details})"
        )


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


def instance_label_sample_id(path: Path) -> str:
    """Extract a sample ID from one BDD100K instance bitmask filename."""

    if path.suffix.lower() != ".png":
        raise ValueError(f"unsupported BDD100K instance label filename: {path.name}")
    return path.stem


def _index_unique_paths(
    paths: Sequence[Path],
    *,
    annotation_kind: str,
    sample_id_from_path: Callable[[Path], str],
) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        sample_id = sample_id_from_path(path)
        if sample_id in indexed:
            raise ValueError(f"duplicate {annotation_kind} sample ID: {sample_id}")
        indexed[sample_id] = path
    return indexed


def semantic_instance_intersection(
    semantic_label_paths: Sequence[Path],
    instance_label_paths: Sequence[Path],
    *,
    allow_audited_intersection: bool = False,
) -> SemanticInstanceIntersection:
    """Pair semantic and instance paths, failing closed on ID mismatch by default."""

    if not isinstance(allow_audited_intersection, bool):
        raise TypeError("allow_audited_intersection must be a boolean")
    semantic_by_id = _index_unique_paths(
        semantic_label_paths,
        annotation_kind="semantic",
        sample_id_from_path=label_sample_id,
    )
    instance_by_id = _index_unique_paths(
        instance_label_paths,
        annotation_kind="instance",
        sample_id_from_path=instance_label_sample_id,
    )
    shared_ids = sorted(semantic_by_id.keys() & instance_by_id.keys())
    pairs = tuple(
        SemanticInstancePair(sample_id, semantic_by_id[sample_id], instance_by_id[sample_id])
        for sample_id in shared_ids
    )
    dropped = tuple(
        sorted(
            (
                *(
                    IntersectionDrop(sample_id, "missing_instance_annotation", path)
                    for sample_id, path in semantic_by_id.items()
                    if sample_id not in instance_by_id
                ),
                *(
                    IntersectionDrop(sample_id, "missing_semantic_annotation", path)
                    for sample_id, path in instance_by_id.items()
                    if sample_id not in semantic_by_id
                ),
            ),
            key=lambda item: item.sample_id,
        )
    )
    if dropped and not allow_audited_intersection:
        raise SemanticInstanceMismatchError(dropped)
    if not pairs:
        raise ValueError("semantic/instance intersection must retain at least one sample")
    return SemanticInstanceIntersection(pairs=pairs, dropped=dropped)
