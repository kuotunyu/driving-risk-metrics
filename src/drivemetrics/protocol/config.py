"""Strict, formatting-independent BDD100K semantic protocol loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from drivemetrics.artifacts.envelope import canonical_json_bytes


class StrictProtocolModel(BaseModel):
    """Shared fail-closed model settings for every nested protocol section."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetProtocol(StrictProtocolModel):
    name: Literal["bdd100k"]
    version: Literal["10k-semantic-v1"]


class SplitProtocol(StrictProtocolModel):
    source_train: Literal[7000]
    train: Literal[6300]
    calibration: Literal[700]
    locked_validation: Literal[1000]
    unlabeled_test: Literal[2000]


class DatasetPaths(StrictProtocolModel):
    train_images: str = Field(min_length=1)
    train_labels: str = Field(min_length=1)
    validation_images: str = Field(min_length=1)
    validation_labels: str = Field(min_length=1)

    @field_validator("*")
    @classmethod
    def validate_relative_posix_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        has_drive_prefix = bool(path.parts and path.parts[0].endswith(":"))
        has_noncanonical_segment = any(part in {"", ".", ".."} for part in value.split("/"))
        if "\\" in value or path.is_absolute() or has_drive_prefix or has_noncanonical_segment:
            raise ValueError("dataset paths must be safe relative POSIX paths")
        return value


class InputProtocol(StrictProtocolModel):
    resize_height: Literal[512]
    resize_width: Literal[910]
    padded_height: Literal[512]
    padded_width: Literal[1024]
    image_pad_value_after_normalization: float = Field(ge=0.0, le=0.0)
    mask_pad_value: Literal[255]


class TrainingProtocol(StrictProtocolModel):
    steps: Literal[30000]
    warmup_steps: Literal[1000]
    effective_batch_size: Literal[16]
    horizontal_flip_probability: float = Field(ge=0.5, le=0.5)
    checkpoint_selection: Literal["final_step_only"]


class SegformerOptimizerProtocol(StrictProtocolModel):
    """The published SegFormer recipe, pinned so a run cannot drift off it."""

    optimizer: Literal["adamw"]
    learning_rate: float = Field(ge=0.00006, le=0.00006)
    weight_decay: float = Field(ge=0.01, le=0.01)


class UperNetOptimizerProtocol(StrictProtocolModel):
    """The standard UPerNet recipe, shared by both backbones under that decoder."""

    optimizer: Literal["adamw"]
    learning_rate: float = Field(ge=0.0001, le=0.0001)
    weight_decay: float = Field(ge=0.05, le=0.05)


class ModelProtocols(StrictProtocolModel):
    segformer_b2: SegformerOptimizerProtocol
    upernet_convnextv2_tiny: UperNetOptimizerProtocol
    upernet_dinov2_small: UperNetOptimizerProtocol


class CalibrationProtocol(StrictProtocolModel):
    method: Literal["scalar_temperature"]
    objective: Literal["multiclass_nll"]


class StatisticsProtocol(StrictProtocolModel):
    bootstrap_resamples: Literal[5000]
    bootstrap_seed: Literal[20260831]
    confidence: float = Field(ge=0.95, le=0.95)


class BDD100KSemanticProtocolV1(StrictProtocolModel):
    """The immutable semantic meaning of the formal P1 experiment protocol."""

    schema_version: Literal["bdd100k-semseg-protocol/v1"]
    dataset: DatasetProtocol
    splits: SplitProtocol
    paths: DatasetPaths
    input: InputProtocol
    training: TrainingProtocol
    models: ModelProtocols
    calibration: CalibrationProtocol
    statistics: StatisticsProtocol


@dataclass(frozen=True)
class LoadedProtocol:
    """A validated protocol together with its canonical semantic hash."""

    protocol: BDD100KSemanticProtocolV1
    protocol_sha256: str


def load_protocol(path: Path) -> LoadedProtocol:
    """Load strict YAML and hash validated content rather than its formatting."""

    raw_value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise TypeError("protocol document must be a mapping")
    protocol = BDD100KSemanticProtocolV1.model_validate(raw_value)
    semantic_content = protocol.model_dump(mode="json")
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(semantic_content)).hexdigest()
    return LoadedProtocol(protocol=protocol, protocol_sha256=protocol_sha256)


LOCKED_VALIDATION_SPLIT = "locked_validation"


def split_paths(protocol: BDD100KSemanticProtocolV1, split_name: str) -> tuple[str, str]:
    """Return the image and label directories that hold one cohort.

    Only the locked validation cohort lives in the official validation tree. The
    source-train, train, and calibration cohorts are all subsets of the training
    tree, so reading any of them from the validation directories would score
    entirely different files.
    """

    if split_name == LOCKED_VALIDATION_SPLIT:
        return protocol.paths.validation_images, protocol.paths.validation_labels
    return protocol.paths.train_images, protocol.paths.train_labels
