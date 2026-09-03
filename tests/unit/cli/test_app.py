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
    for command in (
        "data",
        "train",
        "calibrate",
        "evaluate",
        "index",
        "aggregate",
        "report",
        "audit-claims",
    ):
        assert command in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["data", "preflight"],
        ["train"],
        ["calibrate"],
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
        ineligible={},
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
            "--device",
            "cpu",
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
            "--device",
            "cpu",
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
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert "seed" in result.output


def test_calibrate_prints_one_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nine calibration fits are launched from a notebook and parsed, not read."""

    import drivemetrics.cli.calibrate as calibrate_cli

    result_stub = SimpleNamespace(
        temperature=1.37,
        artifact_path=tmp_path / "temperature.json",
        run_record_path=tmp_path / "run_record.json",
        sampled_images=700,
        pixels_per_image=2048,
    )
    monkeypatch.setattr(calibrate_cli, "BACKEND_FACTORY", lambda *_, **__: object())
    monkeypatch.setattr(calibrate_cli, "CALIBRATE_SERVICE", lambda *_, **__: result_stub)
    config = touch(tmp_path / "protocol.yaml")
    manifest = touch(tmp_path / "calibration.json")
    checkpoint = touch(tmp_path / "final_checkpoint.pt")
    data_root = tmp_path / "data"
    data_root.mkdir()

    result = runner.invoke(
        app,
        [
            "calibrate",
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
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    status = status_of(result.stdout)
    assert status["command"] == "calibrate"
    assert status["temperature"] == 1.37
    assert status["sampled_images"] == 700


def test_evaluate_reports_whether_it_applied_a_temperature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibrated and uncalibrated artifacts must be distinguishable after the fact."""

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
    temperature = touch(tmp_path / "temperature.json")
    data_root = tmp_path / "data"
    data_root.mkdir()

    arguments = [
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
        "--device",
        "cpu",
    ]

    uncalibrated = status_of(runner.invoke(app, arguments).stdout)
    calibrated = status_of(
        runner.invoke(app, [*arguments, "--temperature", str(temperature)]).stdout
    )

    assert uncalibrated["calibrated"] is False
    assert calibrated["calibrated"] is True


def compute_arguments(command: str, tmp_path: Path) -> list[str]:
    """A complete, valid argument list for one of the three device-bound commands."""

    config = touch(tmp_path / "config.yaml")
    manifest = touch(tmp_path / "manifest.json")
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    common = [
        command,
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    if command == "train":
        return [*common, "--seed", "17"]
    return [*common, "--checkpoint", str(touch(tmp_path / "final_checkpoint.pt"))]


def stub_services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Replace every backend factory with one that records how it was called."""

    import drivemetrics.cli.calibrate as calibrate_cli
    import drivemetrics.cli.evaluate as evaluate_cli
    import drivemetrics.cli.train as train_cli

    received: dict[str, dict[str, Any]] = {}

    def factory_for(command: str) -> Any:
        def factory(*_: Any, **kwargs: Any) -> object:
            received[command] = dict(kwargs)
            return object()

        return factory

    train_result = SimpleNamespace(
        final_step=30000,
        checkpoint_path=tmp_path / "final_checkpoint.pt",
        checkpoint_sha256="c" * 64,
        run_record_path=tmp_path / "run_record.json",
    )
    calibrate_result = SimpleNamespace(
        temperature=1.0,
        artifact_path=tmp_path / "temperature.json",
        run_record_path=tmp_path / "run_record.json",
        sampled_images=1,
        pixels_per_image=1,
    )
    evaluate_result = SimpleNamespace(
        evaluated_samples=1, artifact_paths=(), run_record_path=tmp_path / "run_record.json"
    )
    monkeypatch.setattr(train_cli, "BACKEND_FACTORY", factory_for("train"))
    monkeypatch.setattr(train_cli, "TRAIN_SERVICE", lambda *_, **__: train_result)
    monkeypatch.setattr(calibrate_cli, "BACKEND_FACTORY", factory_for("calibrate"))
    monkeypatch.setattr(calibrate_cli, "CALIBRATE_SERVICE", lambda *_, **__: calibrate_result)
    monkeypatch.setattr(evaluate_cli, "BACKEND_FACTORY", factory_for("evaluate"))
    monkeypatch.setattr(evaluate_cli, "EVALUATE_SERVICE", lambda *_, **__: evaluate_result)
    return received


@pytest.mark.parametrize("command", ["train", "calibrate", "evaluate"])
def test_the_requested_device_reaches_the_backend_unchanged(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first formal run trained on the CPU for 7.4 hours beside an idle A100.

    The backends had a device parameter and a CPU default, and no command ever
    set it. Every unit test passed on the CPU. This is the test that was missing:
    the device named on the command line must be the one the backend is built
    with, and it is checked with a name no default could ever produce.
    """

    received = stub_services(monkeypatch, tmp_path)

    result = runner.invoke(app, [*compute_arguments(command, tmp_path), "--device", "meta"])

    assert result.exit_code == 0, result.output
    assert received[command]["device"] == "meta"


@pytest.mark.parametrize("command", ["train", "calibrate", "evaluate"])
def test_a_compute_command_without_a_device_is_refused(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No default, ever. A silent fallback to the CPU is exactly what cost the first run."""

    received = stub_services(monkeypatch, tmp_path)

    result = runner.invoke(app, compute_arguments(command, tmp_path))

    assert result.exit_code != 0
    assert "device" in result.output.lower()
    assert received == {}


def test_aggregate_prints_one_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release workflow needs the three document paths without scraping prose."""

    import drivemetrics.cli.aggregate as aggregate_cli

    result_stub = SimpleNamespace(
        metrics_path=tmp_path / "metrics.json",
        intervals_path=tmp_path / "intervals.json",
        rankings_path=tmp_path / "rankings.json",
        models=("upernet_convnextv2_tiny", "upernet_dinov2_small", "segformer_b2"),
        sample_count=1000,
    )
    monkeypatch.setattr(aggregate_cli, "AGGREGATE_SERVICE", lambda *_, **__: result_stub)
    index = touch(tmp_path / "formal_run_index.json")

    result = runner.invoke(
        app,
        ["aggregate", "--index", str(index), "--output-dir", str(tmp_path / "analysis")],
    )

    assert result.exit_code == 0
    status = status_of(result.stdout)
    assert status["command"] == "aggregate"
    assert status["sample_count"] == 1000
    assert len(status["models"]) == 3


def test_index_prints_one_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The formal-set gate and aggregation both start from this path, parsed not read."""

    import drivemetrics.cli.index as index_cli

    result_stub = SimpleNamespace(
        index_path=tmp_path / "formal_run_index.json",
        run_count=9,
        protocol_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
    )
    profile_stub = SimpleNamespace(critical_class_ids=(11, 12, 17, 18))
    monkeypatch.setattr(index_cli, "PROFILE_LOADER", lambda *_: profile_stub)
    monkeypatch.setattr(index_cli, "INDEX_SERVICE", lambda *_, **__: result_stub)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    config = touch(tmp_path / "protocol.yaml")
    manifest = touch(tmp_path / "locked_validation.json")
    profile = touch(tmp_path / "vru_priority.yaml")

    result = runner.invoke(
        app,
        [
            "index",
            "--runs-root",
            str(runs_root),
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--risk-profile",
            str(profile),
            "--output",
            str(tmp_path / "formal_run_index.json"),
        ],
    )

    assert result.exit_code == 0
    status = status_of(result.stdout)
    assert status["command"] == "index"
    assert status["run_count"] == 9


def test_train_passes_the_resume_directory_through_to_the_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag the notebook sets but the engine never receives protects nothing."""

    import drivemetrics.cli.train as train_cli

    class Result:
        final_step = 30000
        checkpoint_path = tmp_path / "final_checkpoint.pt"
        checkpoint_sha256 = "c" * 64
        run_record_path = tmp_path / "run_record.json"

    seen: dict[str, Any] = {}

    def record(*args: Any, **kwargs: Any) -> Result:
        seen.update(kwargs)
        return Result()

    monkeypatch.setattr(train_cli, "BACKEND_FACTORY", lambda *_, **__: object())
    monkeypatch.setattr(train_cli, "TRAIN_SERVICE", record)
    config = touch(tmp_path / "run.yaml")
    manifest = touch(tmp_path / "manifest.json")
    data_root = tmp_path / "data"
    data_root.mkdir()
    resume_dir = tmp_path / "resume"

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
            "--device",
            "cpu",
            "--resume-dir",
            str(resume_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["resume_dir"] == resume_dir


def test_train_defaults_to_no_resume_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs 01 to 03 were produced without it, so absent must stay the default."""

    import drivemetrics.cli.train as train_cli

    class Result:
        final_step = 30000
        checkpoint_path = tmp_path / "final_checkpoint.pt"
        checkpoint_sha256 = "c" * 64
        run_record_path = tmp_path / "run_record.json"

    seen: dict[str, Any] = {}

    def record(*args: Any, **kwargs: Any) -> Result:
        seen.update(kwargs)
        return Result()

    monkeypatch.setattr(train_cli, "BACKEND_FACTORY", lambda *_, **__: object())
    monkeypatch.setattr(train_cli, "TRAIN_SERVICE", record)
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
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["resume_dir"] is None


def test_evaluate_reports_progress_while_it_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent forty-five minute cell is indistinguishable from a hung one.

    The first formal session was interrupted by hand because evaluation printed
    nothing between its start and its end, and a stalled run and a working one
    looked exactly alike.
    """

    import drivemetrics.cli.evaluate as evaluate_cli

    class Result:
        evaluated_samples = 998
        artifact_paths = ()
        run_record_path = tmp_path / "run_record.json"

    seen: dict[str, Any] = {}

    def record(*args: Any, **kwargs: Any) -> Result:
        seen.update(kwargs)
        observer = kwargs["on_sample"]
        observer(1, 998)
        observer(500, 998)
        return Result()

    monkeypatch.setattr(evaluate_cli, "BACKEND_FACTORY", lambda *_, **__: object())
    monkeypatch.setattr(evaluate_cli, "EVALUATE_SERVICE", record)
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
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert callable(seen["on_sample"])
