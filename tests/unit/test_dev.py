"""Behavior tests for the deterministic repository verification runner."""

from __future__ import annotations

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
        "portfolio_artifact_envelope_v1.json",
        "prediction_artifact_v1.json",
        "run_record_v1.json",
    )
