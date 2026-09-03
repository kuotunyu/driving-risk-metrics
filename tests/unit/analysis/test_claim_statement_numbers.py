"""Behavior tests for comparing a published statement against its claim's metric."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

ARTIFACT_PATH = "evidence/metrics.json"


def load_claims_module() -> ModuleType:
    """Import inside tests so a missing module is an intentional RED failure."""

    try:
        from drivemetrics.analysis import claims
    except ImportError:
        pytest.fail("drivemetrics.analysis.claims is missing", pytrace=False)
    return claims


def verified_claim(claims: ModuleType) -> object:
    return claims.ClaimV1.model_validate(
        {
            "claim_id": "p1-miou-example",
            "text": "Measured mIoU is 0.8125.",
            "evidence_type": "observed",
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "artifact_path": ARTIFACT_PATH,
            "metric_path": "/metrics/miou",
            "status": "verified",
        }
    )


def write_artifact(repository_root: Path, value: object) -> None:
    artifact = repository_root / ARTIFACT_PATH
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(value), encoding="utf-8")


def test_text_numbers_extracts_every_number_in_document_order() -> None:
    """The README audit compares the numbers a sentence states, not the sentence."""

    claims = load_claims_module()

    numbers = claims.text_numbers("0.712 mIoU, 91.3% critical recall, 5000 resamples")

    assert numbers == (Decimal("0.712"), Decimal("91.3"), Decimal("5000"))


def test_metric_numbers_returns_the_numbers_at_the_claim_pointer(tmp_path: Path) -> None:
    """A published statement is checked against the exact value its claim cites."""

    claims = load_claims_module()
    write_artifact(tmp_path, {"metrics": {"miou": 0.8125, "other": 0.5}})

    assert claims.metric_numbers(verified_claim(claims), tmp_path) == {Decimal("0.8125")}


def test_metric_numbers_fails_closed_on_a_missing_pointer(tmp_path: Path) -> None:
    """A pointer that resolves to nothing must never read as an empty number set."""

    claims = load_claims_module()
    write_artifact(tmp_path, {"metrics": {"other": 0.5}})

    with pytest.raises(LookupError):
        claims.metric_numbers(verified_claim(claims), tmp_path)


def test_metric_numbers_fails_closed_on_a_missing_artifact(tmp_path: Path) -> None:
    """An absent artifact is the clearest case of an unbacked number."""

    claims = load_claims_module()

    with pytest.raises(FileNotFoundError):
        claims.metric_numbers(verified_claim(claims), tmp_path)


def test_a_missing_pointer_failure_names_the_pointer_that_was_not_found(
    tmp_path: Path,
) -> None:
    """The audit reports which pointer failed, so the pointer has to be in the error.

    A `LookupError` carrying nothing tells an operator that some claim in the
    registry does not resolve, without saying which field of which artifact.
    Raising the wrong exception type is just as bad: written `or` instead of
    `and`, the dict branch is entered for a key that is absent and the caller
    gets a `KeyError` naming the token but not the pointer that produced it.
    """

    claims = load_claims_module()
    write_artifact(tmp_path, {"metrics": {"other": 0.5}})

    with pytest.raises(LookupError, match=r"^/metrics/miou$"):
        claims.metric_numbers(verified_claim(claims), tmp_path)


@pytest.mark.parametrize(
    "pointer",
    ["/metrics/series/2", "/metrics/series/9"],
    ids=["exactly the array length", "past the array length"],
)
def test_an_array_index_at_or_past_the_end_names_the_pointer(
    tmp_path: Path,
    pointer: str,
) -> None:
    """The bound is exclusive, and the boundary case must fail like any other.

    A two-element array has no index 2. Written `index > len(current)` the
    boundary case falls through to the subscript and numpy-free Python raises
    `IndexError`, which is not a `LookupError` subclass the audit reports on
    and which never mentions the pointer.
    """

    claims = load_claims_module()
    claim = claims.ClaimV1.model_validate(
        {
            "claim_id": "p1-series-example",
            "text": "Measured mIoU is 0.8125.",
            "evidence_type": "observed",
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "artifact_path": ARTIFACT_PATH,
            "metric_path": pointer,
            "status": "verified",
        }
    )
    write_artifact(tmp_path, {"metrics": {"series": [0.5, 0.8125]}})

    with pytest.raises(LookupError, match=rf"^{re.escape(pointer)}$"):
        claims.metric_numbers(claim, tmp_path)


def test_an_array_pointer_resolves_to_the_element_it_names(tmp_path: Path) -> None:
    """Walking into an array must carry the element forward, not discard it.

    If the step that descends into a list dropped its result, every array
    pointer would resolve to nothing and the audit would report an empty set of
    numbers, which is exactly the "no evidence looks like no disagreement"
    failure the module is built to prevent.
    """

    claims = load_claims_module()
    claim = claims.ClaimV1.model_validate(
        {
            "claim_id": "p1-series-example",
            "text": "Measured mIoU is 0.8125.",
            "evidence_type": "observed",
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "artifact_path": ARTIFACT_PATH,
            "metric_path": "/metrics/series/1",
            "status": "verified",
        }
    )
    write_artifact(tmp_path, {"metrics": {"series": [0.5, 0.8125]}})

    assert claims.metric_numbers(claim, tmp_path) == {Decimal("0.8125")}


def test_a_non_finite_json_constant_is_refused_and_named(tmp_path: Path) -> None:
    """`NaN` is not JSON, and a parser that accepts it publishes an unusable number.

    Python's `json` accepts `NaN`, `Infinity` and `-Infinity` by default, so the
    rejection is opt-in through `parse_constant`. Dropping that argument makes
    a non-finite value parse silently into a float that then flows into a
    published comparison. The message names the constant so an operator can
    find it in a large artifact.
    """

    claims = load_claims_module()
    artifact = tmp_path / ARTIFACT_PATH
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"metrics": {"miou": NaN}}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"^non-finite JSON constant: NaN$"):
        claims.metric_numbers(verified_claim(claims), tmp_path)
