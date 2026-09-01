"""Locked-cohort evaluation over a frozen manifest and one final checkpoint."""

from .engine import EvaluationBackend, EvaluationResult, evaluate_checkpoint

__all__ = ["EvaluationBackend", "EvaluationResult", "evaluate_checkpoint"]
