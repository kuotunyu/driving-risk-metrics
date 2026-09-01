"""Contracts for the validated command entry points."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from drivemetrics.cli.app import app

runner = CliRunner()


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def status_of(output: str) -> dict[str, Any]:
    """Parse the single JSON status object a successful command must print."""

    document = json.loads(output.strip())
    assert isinstance(document, dict)
    return document


def test_the_root_help_lists_every_declared_command() -> None:
    """A command that is documented but unreachable would be a broken published interface."""

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("data", "train", "evaluate", "report", "audit-claims"):
        assert command in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["data", "preflight"],
        ["train"],
        ["evaluate"],
        ["report"],
        ["audit-claims"],
    ],
)
def test_every_command_requires_its_declared_options(arguments: list[str]) -> None:
    """Running with defaults would silently target the wrong protocol or cohort."""

    result = runner.invoke(app, arguments)

    assert result.exit_code != 0


def test_a_missing_input_file_is_rejected_before_any_service_runs(tmp_path: Path) -> None:
    """Reaching a service with a nonexistent path would produce a confusing internal error."""

    result = runner.invoke(
        app,
        [
            "audit-claims",
            "--claims",
            str(tmp_path / "absent.yaml"),
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0


def test_preflight_prints_one_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A human-only message would make the command unusable inside a Colab notebook."""

    import drivemetrics.cli.data as data_cli

    result_stub = SimpleNamespace(
        protocol_sha256="a" * 64,
        counts={"train": 6300},
        manifest_sha256={"train": "b" * 64},
        manifest_paths={"train": tmp_path / "train.json"},
    )

    monkeypatch.setattr(data_cli, "PREFLIGHT_SERVICE", lambda *_: result_stub)
    config = touch(tmp_path / "protocol.yaml")
    data_root = tmp_path / "data"
    data_root.mkdir()

    result = runner.invoke(
        app,
        [
            "data",
            "preflight",
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--output",
            str(tmp_path / "manifests"),
        ],
    )

    assert result.exit_code == 0
    status = status_of(result.stdout)
    assert status["command"] == "data preflight"
    assert status["counts"] == {"train": 6300}


def test_train_prints_one_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nine formal jobs are launched from a notebook and parsed, not read by eye."""

    import drivemetrics.cli.train as train_cli

    class Result:
        final_step = 30000
        checkpoint_path = tmp_path / "final_checkpoint.pt"
        checkpoint_sha256 = "c" * 64
        run_record_path = tmp_path / "run_record.json"

    monkeypatch.setattr(train_cli, "BACKEND_FACTORY", lambda *_, **__: object())
    monkeypatch.setattr(train_cli, "TRAIN_SERVICE", lambda *_, **__: Result())
    config = touch(tmp_path / "run.yaml")
    manifest = touch(tmp_path / "manifest.json")
    data_root = tmp_path / "data"
    data_root.mkdir()

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--data-root",
            str(data_root),
            "--seed",
            "17",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    status = status_of(result.stdout)
    assert status["command"] == "train"
    assert status["final_step"] == 30000
    assert status["checkpoint_sha256"] == "c" * 64


def test_evaluate_prints_one_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation status is consumed by the formal run index, not by a person."""

    import drivemetrics.cli.evaluate as evaluate_cli

    class Result:
        evaluated_samples = 1000
        artifact_paths = (tmp_path / "a.json",)
        run_record_path = tmp_path / "run_record.json"

    monkeypatch.setattr(evaluate_cli, "BACKEND_FACTORY", lambda *_, **__: object())
    monkeypatch.setattr(evaluate_cli, "EVALUATE_SERVICE", lambda *_, **__: Result())
    config = touch(tmp_path / "protocol.yaml")
    manifest = touch(tmp_path / "manifest.json")
    checkpoint = touch(tmp_path / "final_checkpoint.pt")
    data_root = tmp_path / "data"
    data_root.mkdir()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    status = status_of(result.stdout)
    assert status["command"] == "evaluate"
    assert status["evaluated_samples"] == 1000


def test_report_prints_one_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release workflow needs the generated page path without scraping prose."""

    import drivemetrics.cli.report as report_cli

    class Result:
        index_path = tmp_path / "site" / "index.html"
        figure_paths = (tmp_path / "site" / "figures" / "miou.json",)
        claim_count = 2

    monkeypatch.setattr(report_cli, "REPORT_SERVICE", lambda *_, **__: Result())
    claims = touch(tmp_path / "claims.yaml")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    result = runner.invoke(
        app,
        [
            "report",
            "--claims",
            str(claims),
            "--artifacts-dir",
            str(artifacts),
            "--output-dir",
            str(tmp_path / "site"),
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    status = status_of(result.stdout)
    assert status["command"] == "report"
    assert status["claim_count"] == 2


def test_audit_claims_reports_a_clean_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent audit would give no evidence that the registry was actually checked."""

    import drivemetrics.cli.report as report_cli

    monkeypatch.setattr(report_cli, "AUDIT_SERVICE", lambda *_: ())
    claims = touch(tmp_path / "claims.yaml")

    result = runner.invoke(
        app,
        ["audit-claims", "--claims", str(claims), "--repo-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    status = status_of(result.stdout)
    assert status["command"] == "audit-claims"
    assert status["violations"] == 0


def test_audit_claims_exits_nonzero_when_a_claim_cannot_be_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exiting zero on a failed audit would let an unbacked number reach a release."""

    import drivemetrics.cli.report as report_cli

    monkeypatch.setattr(
        report_cli,
        "AUDIT_SERVICE",
        lambda *_: ("miou-fcn: artifact does not exist",),
    )
    claims = touch(tmp_path / "claims.yaml")

    result = runner.invoke(
        app,
        ["audit-claims", "--claims", str(claims), "--repo-root", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "artifact does not exist" in result.output


def test_a_service_failure_becomes_a_diagnostic_and_a_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swallowing a service error would report a broken run as a completed one."""

    import drivemetrics.cli.data as data_cli

    def explode(*_: object) -> None:
        raise ValueError("cohort count mismatch")

    monkeypatch.setattr(data_cli, "PREFLIGHT_SERVICE", explode)
    config = touch(tmp_path / "protocol.yaml")
    data_root = tmp_path / "data"
    data_root.mkdir()

    result = runner.invoke(
        app,
        [
            "data",
            "preflight",
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--output",
            str(tmp_path / "manifests"),
        ],
    )

    assert result.exit_code != 0
    assert "cohort count mismatch" in result.output


def test_an_unapproved_seed_is_rejected_by_the_real_training_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the three approved seeds may produce a run inside the nine-job matrix."""

    import drivemetrics.cli.train as train_cli

    monkeypatch.setattr(train_cli, "BACKEND_FACTORY", lambda *_, **__: object())
    config = touch(tmp_path / "run.yaml")
    manifest = touch(tmp_path / "manifest.json")
    data_root = tmp_path / "data"
    data_root.mkdir()

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--data-root",
            str(data_root),
            "--seed",
            "5",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code != 0
    assert "seed" in result.output
