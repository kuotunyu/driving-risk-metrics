"""Fail-closed validator for statements proposed for publication.

`driving-risk audit-claims` proves that every claim in the registry reproduces
from its own artifact. It never sees the sentence that reaches a README, a slide
or a report, and that sentence is where a number gets retyped, rounded, carried
over from a superseded run, or typed from memory. This validator closes that
gap: every proposed statement is traced to a claim ID, an artifact path and a
JSON pointer, and the numbers it states are compared with the numbers held at
that pointer.

Two inputs are audited. A proposal is a YAML list of statements that each cite
a claim ID, for text that is not in a document yet. A document is Markdown in
which every result sentence carries a `<!-- claim: <id> -->` marker on its own
line; a line that names a metric and a number but carries no marker is reported,
because a number nobody can trace is the failure this project exists to prevent.

Every violation is collected and reported rather than stopping at the first, so
one run tells you everything that is wrong instead of one thing at a time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from drivemetrics.analysis.claims import (
    ClaimsRegistryV1,
    ClaimV1,
    audit_claims,
    metric_numbers,
    text_numbers,
)

MARKER = re.compile(r"<!--\s*claim:\s*([a-z0-9][a-z0-9._-]*)\s*-->")
#: Words that mark a sentence as stating a result. A line that contains one of
#: these and a number is a claim, whether or not the author thought of it as one.
METRIC_TERM = re.compile(
    r"\b(?:m?iou|pixel[ _]accuracy|critical[ _-](?:class[ _])?recall|ece|brier|fnr"
    r"|false[ _]negative[ _]rate|aurc|selective[ _]risk|risk[ _-]weighted)\b",
    re.IGNORECASE,
)
#: Evidence types that must be named in the sentence that states the number,
#: so a synthetic or illustrative value can never be read as a measurement.
LABELLED_EVIDENCE_TYPES: tuple[str, ...] = ("synthetic", "illustrative")
FENCE_PREFIXES: tuple[str, ...] = ("```", "~~~")


def load_registry(claims_path: Path) -> dict[str, ClaimV1]:
    """Return every registry claim by ID, verified or not, so status can be reported."""

    registry = ClaimsRegistryV1.model_validate(
        yaml.safe_load(claims_path.read_text(encoding="utf-8"))
    )
    return {claim.claim_id: claim for claim in registry.claims}


def check_statement(
    source: str,
    claim_id: str,
    text: str,
    registry: dict[str, ClaimV1],
    repository_root: Path,
) -> tuple[list[str], dict[str, Any] | None]:
    """Trace one statement to its claim and compare the numbers it states."""

    claim = registry.get(claim_id)
    if claim is None:
        return (
            [
                f"{source} {claim_id}: no registry claim backs it; nothing may be "
                "published without a verified claim"
            ],
            None,
        )

    where = f"{claim.artifact_path} at {claim.metric_path}"
    violations: list[str] = []
    if claim.status != "verified":
        violations.append(
            f"{source} {claim_id}: registry status is {claim.status!r}, not 'verified' ({where})"
        )
    if claim.evidence_type in LABELLED_EVIDENCE_TYPES and claim.evidence_type not in text.lower():
        violations.append(
            f"{source} {claim_id}: evidence type is {claim.evidence_type!r} but the "
            f"statement does not say so ({where})"
        )

    stated = text_numbers(text)
    trace: dict[str, Any] = {
        "source": source,
        "claim_id": claim_id,
        "status": claim.status,
        "evidence_type": claim.evidence_type,
        "artifact_path": claim.artifact_path,
        "metric_path": claim.metric_path,
        "numbers": [str(number) for number in stated],
    }
    try:
        held = metric_numbers(claim, repository_root)
    except (OSError, ValueError, LookupError) as error:
        violations.append(f"{source} {claim_id}: metric could not be read from {where}: {error}")
    else:
        held_text = ", ".join(sorted(str(number) for number in held)) or "no number"
        for number in stated:
            if number not in held:
                violations.append(
                    f"{source} {claim_id}: statement says {number} but {where} holds {held_text}"
                )
    trace["verdict"] = "pass" if not violations else "fail"
    return violations, trace


def audit_proposal(
    proposal_path: Path,
    registry: dict[str, ClaimV1],
    repository_root: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Audit a YAML list of statements, each citing the claim it publishes."""

    document = yaml.safe_load(proposal_path.read_text(encoding="utf-8"))
    entries = document.get("proposals") if isinstance(document, dict) else document
    if not isinstance(entries, list):
        raise ValueError(f"{proposal_path}: expected a list under 'proposals'")

    violations: list[str] = []
    traces: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        source = f"proposal[{position}]"
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("claim_id"), str)
            or not isinstance(entry.get("text"), str)
        ):
            violations.append(f"{source}: every proposal needs a claim_id and a text")
            continue
        found, trace = check_statement(
            source, entry["claim_id"], entry["text"], registry, repository_root
        )
        violations.extend(found)
        if trace is not None:
            traces.append(trace)
    return violations, traces


def audit_document(
    document_path: Path,
    registry: dict[str, ClaimV1],
    repository_root: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Audit every marked line of a Markdown document and report unmarked results."""

    violations: list[str] = []
    traces: list[dict[str, Any]] = []
    in_fence = False
    for line_number, raw in enumerate(
        document_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if raw.strip().startswith(FENCE_PREFIXES):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        source = f"{document_path.name}:{line_number}"
        claim_ids = MARKER.findall(raw)
        text = MARKER.sub("", raw)
        if claim_ids:
            for claim_id in claim_ids:
                found, trace = check_statement(source, claim_id, text, registry, repository_root)
                violations.extend(found)
                if trace is not None:
                    traces.append(trace)
            continue

        term = METRIC_TERM.search(text)
        numbers = text_numbers(text)
        if term is not None and numbers:
            stated = ", ".join(str(number) for number in numbers)
            violations.append(
                f"{source}: number {stated} beside metric term {term.group()!r} but no "
                "<!-- claim: ... --> marker; an untraceable result cannot be published"
            )
    return violations, traces


def validate(
    claims_path: Path,
    repository_root: Path,
    proposal_path: Path | None,
    document_paths: list[Path],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Return every violation, plus the trace of an audit that has none."""

    violations: list[str] = [
        f"registry: {violation}" for violation in audit_claims(claims_path, repository_root)
    ]
    try:
        registry = load_registry(claims_path)
    except (ValidationError, ValueError) as error:
        violations.append(f"registry: claims registry failed its own schema: {error}")
        return tuple(violations), {}

    traces: list[dict[str, Any]] = []
    if proposal_path is not None:
        found, traced = audit_proposal(proposal_path, registry, repository_root)
        violations.extend(found)
        traces.extend(traced)
    for document_path in document_paths:
        found, traced = audit_document(document_path, registry, repository_root)
        violations.extend(found)
        traces.extend(traced)

    status: dict[str, Any] = {
        "validator": "validate_claims",
        "claims_registry": str(claims_path),
        "registry_claims": len(registry),
        "statements": traces,
    }
    return tuple(violations), status


def main(argv: list[str] | None = None) -> int:
    """Print a JSON trace and exit zero, or print every violation and exit one."""

    parser = argparse.ArgumentParser(
        description=(
            "Trace every statement proposed for publication to a verified claim, its "
            "artifact and its metric pointer, and refuse any number the artifact does not hold."
        )
    )
    parser.add_argument("--claims", type=Path, required=True, help="Claims registry YAML.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(),
        help="Directory that claim artifact paths are relative to.",
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        help="YAML list of statements to publish, each with a claim_id and a text.",
    )
    parser.add_argument(
        "--document",
        type=Path,
        action="append",
        default=[],
        help="Markdown document to audit; repeatable.",
    )
    arguments = parser.parse_args(argv)

    if arguments.proposal is None and not arguments.document:
        print("nothing to audit: pass --proposal and/or --document", file=sys.stderr)
        return 2

    try:
        violations, status = validate(
            arguments.claims, arguments.repo_root, arguments.proposal, arguments.document
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"validation could not run: {error}", file=sys.stderr)
        return 2

    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
