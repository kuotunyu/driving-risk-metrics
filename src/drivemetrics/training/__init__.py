"""Locked deterministic training schedule, loss policy, and orchestration engine."""

from .backends import (
    PreparedBatch,
    TorchTrainingBackend,
    TorchTrainingState,
    build_training_backend,
)
from .engine import TrainingBackend, TrainingResult, train
from .losses import IGNORE_INDEX, cross_entropy_spec
from .schedule import optimizer_spec, polynomial_learning_rate

__all__ = [
    "IGNORE_INDEX",
    "PreparedBatch",
    "TorchTrainingBackend",
    "TorchTrainingState",
    "TrainingBackend",
    "TrainingResult",
    "build_training_backend",
    "cross_entropy_spec",
    "optimizer_spec",
    "polynomial_learning_rate",
    "train",
]
