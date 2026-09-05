"""Behavior tests for the deterministic repository verification runner."""

from __future__ import annotations

import json
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import pytest

StageRunner = Callable[[str, Sequence[str], Path], int]
EXPECTED_STAGES = [
    "private_guard",
    "format_check",
    "lint",
    "typecheck",
    "unit_and_integration_tests",
    "branch_coverage_100",
    "schema_contracts",
    "docs_links",
]


def load_dev() -> ModuleType:
    try:
        from drivemetrics import dev
    except ImportError:
        pytest.fail("drivemetrics.dev is missing", pytrace=False)
    return dev


def test_verify_repository_runs_every_stage_in_fixed_order(tmp_path: Path) -> None:
    """Reordering or skipping a stage must change the observable runner trace."""

    dev = load_dev()
    observed: list[tuple[str, tuple[str, ...], Path]] = []

    def runner(stage: str, command: Sequence[str], cwd: Path) -> int:
        observed.append((stage, tuple(command), cwd))
        return 0

    assert dev.verify_repository(tmp_path, runner=runner) == 0
    assert [stage for stage, _, _ in observed] == EXPECTED_STAGES
    assert all(command for _, command, _ in observed)
    assert all(cwd == tmp_path for _, _, cwd in observed)


def test_verify_repository_stops_at_first_failure_and_returns_its_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Continuing after a failed lint stage could hide its exit code."""

    dev = load_dev()
    observed: list[str] = []

    def runner(stage: str, command: Sequence[str], cwd: Path) -> int:
        del command, cwd
        observed.append(stage)
        return 7 if stage == "lint" else 0

    assert dev.verify_repository(tmp_path, runner=runner) == 7
    assert observed == ["private_guard", "format_check", "lint"]
    captured = capsys.readouterr()
    assert "lint failed with exit code 7" in captured.err


def test_subprocess_runner_returns_child_exit_code(tmp_path: Path) -> None:
    """Collapsing non-zero child exits to success would invalidate every gate."""

    dev = load_dev()

    assert (
        dev.subprocess_runner(
            "probe",
            [sys.executable, "-c", "raise SystemExit(9)"],
            tmp_path,
        )
        == 9
    )


def test_schema_contract_check_accepts_valid_json_and_reports_every_invalid_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stopping at one malformed schema would hide other invalid tracked contracts."""

    dev = load_dev()
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    (schemas / "valid.json").write_text('{"type":"object"}\n', encoding="utf-8")
    (schemas / "bad-a.json").write_text("{", encoding="utf-8")
    (schemas / "bad-b.json").write_text("not json", encoding="utf-8")

    assert dev.verify_schema_contracts(tmp_path) == 1
    assert capsys.readouterr().err.splitlines() == [
        "invalid JSON schema: schemas/bad-a.json",
        "invalid JSON schema: schemas/bad-b.json",
    ]


def test_schema_contract_check_allows_repository_without_schemas(tmp_path: Path) -> None:
    """The packaging phase has no schemas yet and must not invent a failure."""

    dev = load_dev()

    assert dev.verify_schema_contracts(tmp_path) == 0


def test_docs_link_check_accepts_external_anchor_and_existing_local_target(
    tmp_path: Path,
) -> None:
    """A local-link checker must not reject web URLs, anchors, or valid files."""

    dev = load_dev()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("Guide.\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[guide](docs/guide.md) [/guide](/docs/guide.md) "
        "[web](https://example.com) [section](#section)\n",
        encoding="utf-8",
    )

    assert dev.verify_docs_links(tmp_path) == 0


def test_docs_link_check_reports_every_missing_local_target(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Returning after one broken link would leave later documentation rot hidden."""

    dev = load_dev()
    (tmp_path / "README.md").write_text(
        "[missing](docs/missing.md)\n[also missing](other.md#part)\n",
        encoding="utf-8",
    )

    assert dev.verify_docs_links(tmp_path) == 1
    assert capsys.readouterr().err.splitlines() == [
        "broken local link: README.md -> docs/missing.md",
        "broken local link: README.md -> other.md#part",
    ]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("schema-contracts", 0),
        ("docs-links", 0),
    ],
)
def test_dev_cli_dispatches_data_independent_checks(
    tmp_path: Path,
    command: str,
    expected: int,
) -> None:
    """A wrong CLI branch would make the corresponding verification stage unusable."""

    dev = load_dev()

    assert dev.main([command, "--repo-root", str(tmp_path)]) == expected


def test_dev_cli_dispatches_verify_with_injected_runner(tmp_path: Path) -> None:
    """The public verify command must use the same testable first-failure runner."""

    dev = load_dev()
    observed: list[str] = []

    def runner(stage: str, command: Sequence[str], cwd: Path) -> int:
        del command, cwd
        observed.append(stage)
        return 0

    assert (
        dev.main(
            ["verify", "--repo-root", str(tmp_path)],
            runner=runner,
        )
        == 0
    )
    assert observed == EXPECTED_STAGES


def test_dev_module_entrypoint_dispatches_schema_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module command used by verification stages must dispatch successfully."""

    dev = load_dev()
    assert dev.__file__ is not None
    monkeypatch.setattr(
        sys,
        "argv",
        ["dev.py", "schema-contracts", "--repo-root", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(Path(dev.__file__)), run_name="__main__")

    assert raised.value.code == 0


def test_dev_cli_generates_model_derived_contract_schemas(tmp_path: Path) -> None:
    """The checked-in command must regenerate all contract schemas deterministically."""

    dev = load_dev()

    assert dev.main(["generate-schemas", "--repo-root", str(tmp_path)]) == 0
    assert tuple(path.name for path in sorted((tmp_path / "schemas").glob("*.json"))) == (
        "area_tertiles_v1.json",
        "extended_metrics_v1.json",
        "gallery_manifest_v1.json",
        "intervals_v1.json",
        "metrics_table_v1.json",
        "portfolio_artifact_envelope_v1.json",
        "prediction_artifact_v1.json",
        "rankings_v1.json",
        "run_record_v1.json",
    )


def test_the_typecheck_stage_covers_shipped_skill_scripts(tmp_path: Path) -> None:
    """A validator outside the gate would rot silently while agents still run it."""

    dev = load_dev()
    observed: dict[str, tuple[str, ...]] = {}

    def runner(stage: str, command: Sequence[str], cwd: Path) -> int:
        del cwd
        observed[stage] = tuple(command)
        return 0

    dev.verify_repository(tmp_path, runner=runner)

    assert ".agents" in observed["typecheck"]


def write_evidence(root: Path, name: str, text: str) -> Path:
    directory = root / "docs" / "evidence" / "bdd100k_semseg_v1"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


VALID_INTERVALS = json.dumps(
    {
        "schema_version": "driving-risk-intervals/v1",
        "protocol_hash": "a" * 64,
        "dataset_manifest_hash": "b" * 64,
        "intervals": {
            "x minus y (miou)": {
                "estimate": 0.1,
                "low": 0.0,
                "high": 0.2,
                "confidence": 0.95,
                "resamples": 60,
                "seed": 1,
                "estimator": "ratio_of_sums",
            }
        },
    }
)


def test_schema_contract_check_validates_a_published_document_against_its_contract(
    tmp_path: Path,
) -> None:
    """A version string is a label; the stage makes it a contract the file must meet."""

    dev = load_dev()
    write_evidence(tmp_path, "intervals.json", VALID_INTERVALS)

    assert dev.verify_schema_contracts(tmp_path) == 0


def test_schema_contract_check_reports_a_document_its_contract_rejects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A hand-edited or stale evidence file must fail the build, naming the file and the field."""

    dev = load_dev()
    broken = json.loads(VALID_INTERVALS)
    del broken["protocol_hash"]
    broken["intervals"]["x minus y (miou)"]["confidence"] = 1.5
    write_evidence(tmp_path, "intervals.json", json.dumps(broken))

    assert dev.verify_schema_contracts(tmp_path) == 1
    (line,) = capsys.readouterr().err.splitlines()
    assert line.startswith(
        "docs/evidence/bdd100k_semseg_v1/intervals.json: 2 contract violation(s)"
    )
    assert "first at" in line


def test_schema_contract_check_reports_a_document_no_contract_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validating against a guessed contract could pass; refusing cannot."""

    dev = load_dev()
    write_evidence(tmp_path, "future.json", json.dumps({"schema_version": "driving-risk-x/v9"}))
    write_evidence(tmp_path, "mystery.json", json.dumps({"a": 1}))

    assert dev.verify_schema_contracts(tmp_path) == 1
    assert capsys.readouterr().err.splitlines() == [
        "no contract names this document: docs/evidence/bdd100k_semseg_v1/future.json "
        "(schema_version='driving-risk-x/v9')",
        "no contract names this document: docs/evidence/bdd100k_semseg_v1/mystery.json "
        "(schema_version=None)",
    ]


def test_schema_contract_check_accepts_the_frozen_tertiles_by_filename(tmp_path: Path) -> None:
    """The one document frozen before versioning is matched by its fixed name, not guessed."""

    dev = load_dev()
    write_evidence(
        tmp_path,
        "area_tertiles.json",
        json.dumps(
            {
                "eligible_images": 6296,
                "instances_per_category": {"1": 8946, "3": 64751},
                "learned_from": "train",
                "tertile_edges": {"1": [349, 987], "3": [390, 2110]},
                "total_instances": 80249,
            }
        ),
    )

    assert dev.verify_schema_contracts(tmp_path) == 0


def test_schema_contract_check_routes_the_run_index_to_its_own_validator(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The formal run index already has a gate; the stage applies that gate, not a new one."""

    dev = load_dev()
    write_evidence(
        tmp_path,
        "formal_run_index.json",
        json.dumps({"schema_version": "drivemetrics-formal-set/v1", "runs": []}),
    )

    assert dev.verify_schema_contracts(tmp_path) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines
    assert all(
        line.startswith("docs/evidence/bdd100k_semseg_v1/formal_run_index.json: ") for line in lines
    )


def test_schema_contract_check_reports_evidence_that_is_not_a_json_object(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every file under docs/evidence is a claim's target; none may be unreadable."""

    dev = load_dev()
    write_evidence(tmp_path, "broken.json", "{")
    write_evidence(tmp_path, "list.json", "[1, 2]")

    assert dev.verify_schema_contracts(tmp_path) == 1
    assert capsys.readouterr().err.splitlines() == [
        "invalid JSON document: docs/evidence/bdd100k_semseg_v1/broken.json",
        "document is not an object: docs/evidence/bdd100k_semseg_v1/list.json",
    ]
