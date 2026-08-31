"""Deterministic formal BDD100K split freezing and contamination checks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

SOURCE_TRAIN_COUNT = 7000
TRAIN_COUNT = 6300
CALIBRATION_COUNT = 700
LOCKED_VALIDATION_COUNT = 1000


def _sha256_order_key(sample_id: str) -> tuple[str, str]:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), sample_id


def freeze_bdd100k_split(
    train_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Freeze exactly 7,000 source IDs by canonical SHA-256 filename order."""

    values = tuple(train_ids)
    if len(values) != SOURCE_TRAIN_COUNT:
        raise ValueError("formal BDD100K source train split must contain exactly 7000 IDs")
    if len(set(values)) != len(values):
        raise ValueError("formal BDD100K source train split contains duplicate IDs")
    ordered = tuple(sorted(values, key=_sha256_order_key))
    return ordered[:TRAIN_COUNT], ordered[TRAIN_COUNT:]


def _validate_cohort(name: str, values: Iterable[str], expected_count: int) -> tuple[str, ...]:
    cohort = tuple(values)
    if len(cohort) != expected_count:
        raise ValueError(f"{name} count must be exactly {expected_count}")
    if len(set(cohort)) != len(cohort):
        raise ValueError(f"{name} contains duplicate IDs")
    return cohort


def validate_locked_split(
    train_ids: Iterable[str],
    calibration_ids: Iterable[str],
    validation_ids: Iterable[str],
) -> None:
    """Reject wrong counts, duplicates, overlap, and locked-validation contamination."""

    train = _validate_cohort("train", train_ids, TRAIN_COUNT)
    calibration = _validate_cohort("calibration", calibration_ids, CALIBRATION_COUNT)
    validation = _validate_cohort("locked validation", validation_ids, LOCKED_VALIDATION_COUNT)
    if set(train) & set(calibration):
        raise ValueError("train and calibration cohorts overlap")
    if set(train) & set(validation):
        raise ValueError("train and locked validation cohorts overlap")
    if set(calibration) & set(validation):
        raise ValueError("calibration and locked validation cohorts overlap")
    expected_train, expected_calibration = freeze_bdd100k_split((*train, *calibration))
    if train != expected_train or calibration != expected_calibration:
        raise ValueError("train/calibration cohorts do not match the deterministic SHA-256 split")
