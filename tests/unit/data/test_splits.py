"""Tests for the immutable BDD100K development/calibration/validation cohorts."""

from __future__ import annotations

import hashlib
import random
from types import ModuleType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def load_splits_module() -> ModuleType:
    try:
        from drivemetrics.data import splits
    except ImportError:
        pytest.fail("drivemetrics.data.splits is missing", pytrace=False)
    return splits


def formal_source_ids() -> list[str]:
    return [f"bdd-{index:04d}" for index in range(7000)]


def expected_order(ids: list[str]) -> list[str]:
    return sorted(ids, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))


def test_freeze_split_uses_exact_sha256_order_and_counts() -> None:
    splits = load_splits_module()
    ids = formal_source_ids()

    train, calibration = splits.freeze_bdd100k_split(list(reversed(ids)))

    expected = expected_order(ids)
    assert train == tuple(expected[:6300])
    assert calibration == tuple(expected[6300:])
    assert len(train) == 6300
    assert len(calibration) == 700
    assert set(train).isdisjoint(calibration)


@given(st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=8, deadline=None)
def test_freeze_split_is_permutation_invariant(seed: int) -> None:
    splits = load_splits_module()
    ids = formal_source_ids()
    expected = splits.freeze_bdd100k_split(ids)
    random.Random(seed).shuffle(ids)

    assert splits.freeze_bdd100k_split(ids) == expected


@pytest.mark.parametrize("size", [6999, 7001])
def test_freeze_split_rejects_wrong_formal_count(size: int) -> None:
    splits = load_splits_module()

    with pytest.raises(
        ValueError, match=r"^formal BDD100K source train split must contain exactly"
    ):
        splits.freeze_bdd100k_split(formal_source_ids()[:size] + (["extra"] if size > 7000 else []))


def test_freeze_split_rejects_duplicate_id() -> None:
    splits = load_splits_module()
    ids = formal_source_ids()
    ids[-1] = ids[0]

    with pytest.raises(ValueError, match=r"^formal BDD100K source train split contains duplicate"):
        splits.freeze_bdd100k_split(ids)


def test_validate_locked_split_accepts_exact_disjoint_cohorts() -> None:
    splits = load_splits_module()
    train, calibration = splits.freeze_bdd100k_split(formal_source_ids())
    validation = [f"locked-{index:04d}" for index in range(1000)]

    splits.validate_locked_split(train, calibration, validation)


@pytest.mark.parametrize("mutation", ["swap", "reorder"])
def test_validate_locked_split_rejects_wrong_sha256_partition_or_order(mutation: str) -> None:
    splits = load_splits_module()
    train, calibration = splits.freeze_bdd100k_split(formal_source_ids())
    validation = tuple(f"locked-{index:04d}" for index in range(1000))
    if mutation == "swap":
        train_tail = train[-1]
        calibration_tail = calibration[-1]
        train = (*train[:-1], calibration_tail)
        calibration = (*calibration[:-1], train_tail)
    else:
        train = tuple(reversed(train))

    with pytest.raises(ValueError, match="deterministic SHA-256 split"):
        splits.validate_locked_split(train, calibration, validation)


@pytest.mark.parametrize(
    ("cohort", "expected"),
    [("train", "duplicate"), ("calibration", "duplicate"), ("validation", "duplicate")],
)
def test_validate_locked_split_rejects_duplicates(cohort: str, expected: str) -> None:
    splits = load_splits_module()
    train, calibration = splits.freeze_bdd100k_split(formal_source_ids())
    validation = tuple(f"locked-{index:04d}" for index in range(1000))
    values = {"train": train, "calibration": calibration, "validation": validation}
    selected = values[cohort]
    values[cohort] = (*selected[:-1], selected[0])

    with pytest.raises(ValueError, match=expected):
        splits.validate_locked_split(values["train"], values["calibration"], values["validation"])


@pytest.mark.parametrize(
    ("left", "right"),
    [("train", "calibration"), ("train", "validation"), ("calibration", "validation")],
)
def test_validate_locked_split_rejects_overlap(left: str, right: str) -> None:
    splits = load_splits_module()
    train, calibration = splits.freeze_bdd100k_split(formal_source_ids())
    validation = tuple(f"locked-{index:04d}" for index in range(1000))
    values = {"train": train, "calibration": calibration, "validation": validation}
    replacement = values[left][0]
    target = values[right]
    values[right] = (*target[:-1], replacement)

    with pytest.raises(ValueError, match="overlap"):
        splits.validate_locked_split(values["train"], values["calibration"], values["validation"])


@pytest.mark.parametrize(
    ("cohort", "replacement"),
    [("train", "new-train"), ("calibration", "new-calibration"), ("validation", "new-validation")],
)
def test_validate_locked_split_rejects_wrong_cohort_count(cohort: str, replacement: str) -> None:
    splits = load_splits_module()
    train, calibration = splits.freeze_bdd100k_split(formal_source_ids())
    validation = tuple(f"locked-{index:04d}" for index in range(1000))
    values = {"train": train, "calibration": calibration, "validation": validation}
    values[cohort] = (*values[cohort], replacement)

    with pytest.raises(ValueError, match="count"):
        splits.validate_locked_split(values["train"], values["calibration"], values["validation"])


def test_cohort_membership_hash_ignores_paths_and_file_bytes() -> None:
    """Membership and manifest hashes answer different questions and must not be confused.

    A cohort's membership is which samples are in it. A manifest hash additionally
    binds the paths and the file bytes. Recording one where the other is expected
    reads as dataset drift when nothing drifted, so the two are named separately.
    """

    splits = load_splits_module()

    first = splits.cohort_membership_sha256("train", ("b", "a"))
    same = splits.cohort_membership_sha256("train", ("b", "a"))
    reordered = splits.cohort_membership_sha256("train", ("a", "b"))
    other_split = splits.cohort_membership_sha256("calibration", ("b", "a"))

    assert first == same
    assert first != reordered
    assert first != other_split


def test_cohort_membership_hash_rejects_a_duplicated_sample() -> None:
    """A duplicate would make two different cohorts share one membership hash."""

    splits = load_splits_module()

    with pytest.raises(ValueError, match=r"^train membership contains duplicate IDs"):
        splits.cohort_membership_sha256("train", ("a", "a"))
