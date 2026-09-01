"""Locked-cohort evaluation over a frozen manifest and one final checkpoint."""

from .backends import TorchEvaluationBackend
from .engine import EvaluationBackend, EvaluationResult, evaluate_checkpoint

__all__ = [
    "EvaluationBackend",
    "EvaluationResult",
    "TorchEvaluationBackend",
    "evaluate_checkpoint",
]
