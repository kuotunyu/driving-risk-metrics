"""Contracts for the `resolution-sweep` command group.

The group is driven from a Colab notebook that parses one JSON status per call,
so each subcommand must print exactly that, and fail with a nonzero exit and a
stderr diagnostic when its service refuses.
"""

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
    path.write_text("{}", encoding="utf-8", newline="\n")
    return path


def common_options(tmp_path: Path) -> list[str]:
    data_root = tmp_path / "data"
    bitmasks = tmp_path / "bitmasks"
    data_root.mkdir(exist_ok=True)
    bitmasks.mkdir(exist_ok=True)
    return [
        "--config",
        str(touch(tmp_path / "protocol.yaml")),
        "--manifest",
        str(touch(tmp_path / "calibration.json")),
        "--checkpoints",
        str(touch(tmp_path / "checkpoints.json")),
        "--data-root",
        str(data_root),
        "--instance-bitmasks",
        str(bitmasks),
        "--tertiles",
        str(touch(tmp_path / "area_tertiles.json")),
        "--device",
        "cpu",
    ]


def test_the_root_help_lists_the_resolution_sweep_group() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "resolution-sweep" in result.output


def test_the_group_offers_parity_run_and_analyse() -> None:
    result = runner.invoke(app, ["resolution-sweep", "--help"])

    assert result.exit_code == 0
    for command in ("parity", "run", "analyse"):
        assert command in result.output


@pytest.mark.parametrize("command", ["parity", "run", "analyse"])
def test_every_subcommand_requires_its_options(command: str) -> None:
    result = runner.invoke(app, ["resolution-sweep", command])

    assert result.exit_code != 0


def test_parity_prints_one_machine_readable_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.resolution as resolution_cli

    calls: dict[str, Any] = {}

    def service(*args: Any, **kwargs: Any) -> Any:
        calls["args"], calls["kwargs"] = args, kwargs
        return SimpleNamespace(
            report_path=tmp_path / "parity.json",
            criterion="decision",
            mismatched_pixels={"r": 3},
            changed_decisions={"r": 0},
        )

    monkeypatch.setattr(resolution_cli, "PARITY_SERVICE", service)
    monkeypatch.setattr(resolution_cli, "BACKEND_FACTORY", lambda device: ("backend", device))

    result = runner.invoke(
        app,
        [
            "resolution-sweep",
            "parity",
            *common_options(tmp_path),
            "--output",
            str(tmp_path / "parity.json"),
            "--images",
            "5",
            "--criterion",
            "decision",
        ],
    )

    assert result.exit_code == 0, result.output
    status = json.loads(result.stdout)
    assert status == {
        "command": "resolution-sweep parity",
        "passed": True,
        "criterion": "decision",
        "report_path": str(tmp_path / "parity.json"),
        "mismatched_pixels": {"r": 3},
        "changed_critical_miss_decisions": {"r": 0},
    }
    assert calls["kwargs"] == {"backend": ("backend", "cpu"), "images": 5, "criterion": "decision"}


def test_run_passes_the_arm_list_and_limit_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.resolution as resolution_cli

    calls: dict[str, Any] = {}

    def service(*args: Any, **kwargs: Any) -> Any:
        calls["args"], calls["kwargs"] = args, kwargs
        kwargs["on_sample"](50, 50)
        return SimpleNamespace(
            results_path=tmp_path / "out" / "per_image.jsonl",
            scored_samples=50,
            completed_samples=700,
            total_samples=700,
        )

    monkeypatch.setattr(resolution_cli, "SWEEP_SERVICE", service)
    monkeypatch.setattr(resolution_cli, "BACKEND_FACTORY", lambda device: ("backend", device))

    result = runner.invoke(
        app,
        [
            "resolution-sweep",
            "run",
            *common_options(tmp_path),
            "--parity-report",
            str(touch(tmp_path / "parity.json")),
            "--output-dir",
            str(tmp_path / "out"),
            "--arms",
            "formal,native",
            "--limit",
            "50",
        ],
    )

    assert result.exit_code == 0, result.output
    status = json.loads(result.stdout)
    assert status == {
        "command": "resolution-sweep run",
        "results_path": str(tmp_path / "out" / "per_image.jsonl"),
        "scored_samples": 50,
        "completed_samples": 700,
        "total_samples": 700,
        "complete": True,
    }
    assert calls["kwargs"]["arms"] == ["formal", "native"]
    assert calls["kwargs"]["limit"] == 50
    assert "scored 50/50" in result.stderr


def test_run_defaults_to_the_pre_registered_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.resolution as resolution_cli

    calls: dict[str, Any] = {}

    def service(*args: Any, **kwargs: Any) -> Any:
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            results_path=tmp_path / "per_image.jsonl",
            scored_samples=1,
            completed_samples=1,
            total_samples=700,
        )

    monkeypatch.setattr(resolution_cli, "SWEEP_SERVICE", service)
    monkeypatch.setattr(resolution_cli, "BACKEND_FACTORY", lambda device: device)

    result = runner.invoke(
        app,
        [
            "resolution-sweep",
            "run",
            *common_options(tmp_path),
            "--parity-report",
            str(touch(tmp_path / "parity.json")),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["kwargs"]["arms"] == ["formal", "s085", "native"]
    assert calls["kwargs"]["limit"] is None
    assert json.loads(result.stdout)["complete"] is False


def test_analyse_prints_the_pre_registered_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.resolution as resolution_cli

    monkeypatch.setattr(
        resolution_cli,
        "ANALYSE_SERVICE",
        lambda output_dir: SimpleNamespace(
            summary_path=output_dir / "summary.json",
            categories={"segformer_b2": "A"},
            overall="model-dependent",
        ),
    )
    (tmp_path / "out").mkdir()

    result = runner.invoke(
        app, ["resolution-sweep", "analyse", "--output-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "command": "resolution-sweep analyse",
        "summary_path": str(tmp_path / "out" / "summary.json"),
        "categories": {"segformer_b2": "A"},
        "overall": "model-dependent",
    }


def test_a_refusing_service_exits_nonzero_with_a_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.resolution as resolution_cli

    def refuse(_output_dir: Path) -> Any:
        raise ValueError("the sweep is incomplete")

    monkeypatch.setattr(resolution_cli, "ANALYSE_SERVICE", refuse)
    (tmp_path / "out").mkdir()

    result = runner.invoke(
        app, ["resolution-sweep", "analyse", "--output-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 1
    assert "the sweep is incomplete" in result.stderr
