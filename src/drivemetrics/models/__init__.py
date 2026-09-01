"""Approved segmentation architectures behind one framework-independent adapter."""

from .adapters import SegmentationAdapter, SegmentationModel
from .registry import (
    APPROVED_MODEL_NAMES,
    SEGFORMER_ENCODER_CHECKPOINT,
    ModelName,
    create_model,
)

__all__ = [
    "APPROVED_MODEL_NAMES",
    "SEGFORMER_ENCODER_CHECKPOINT",
    "ModelName",
    "SegmentationAdapter",
    "SegmentationModel",
    "create_model",
]
