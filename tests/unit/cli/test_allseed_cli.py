"""Contracts for the `allseed` command group.

A Colab notebook drives the group and parses one JSON status per call, so each
subcommand prints exactly that, and fails with a nonzero exit and a stderr
diagnostic when its service refuses.
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


def test_the_root_help_lists_the_allseed_group() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "allseed" in result.output


@pytest.mark.parametrize("command", ["extract", "analyse", "evidence"])
def test_every_subcommand_requires_its_options(command: str) -> None:
    result = runner.invoke(app, ["allseed", command])

    assert result.exit_code != 0


def test_extract_reads_the_study_and_prints_one_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.allseed as allseed_cli

    calls: dict[str, Any] = {}

    def extract(*args: Any, **kwargs: Any) -> Any:
        calls["args"], calls["kwargs"] = args, kwargs
        kwargs["on_image"](50, 998)
        return SimpleNamespace(output_dir=args[5], runs=("a-seed-17",), images=998)

    monkeypatch.setattr(allseed_cli, "EXTRACT_SERVICE", extract)
    labels = tmp_path / "labels"
    instances = tmp_path / "instances"
    labels.mkdir()
    instances.mkdir()

    result = runner.invoke(
        app,
        [
            "allseed",
            "extract",
            "--index",
            str(touch(tmp_path / "formal_run_index.json")),
            "--manifest",
            str(touch(tmp_path / "locked_validation.json")),
            "--labels-root",
            str(labels),
            "--instance-root",
            str(instances),
            "--tertiles",
            str(touch(tmp_path / "area_tertiles.json")),
            "--output-dir",
            str(tmp_path / "extract"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["args"][5] == tmp_path / "extract"
    assert json.loads(result.stdout) == {
        "command": "allseed extract",
        "output_dir": str(tmp_path / "extract"),
        "runs": ["a-seed-17"],
        "images": 998,
    }
    assert "50/998" in result.stderr


def test_analyse_prints_the_pre_registered_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.allseed as allseed_cli

    monkeypatch.setattr(
        allseed_cli,
        "ANALYSE_SERVICE",
        lambda extract_dir, evidence_dir, output_dir: SimpleNamespace(
            summary_path=output_dir / "summary.json",
            categories={"segformer_b2": "majority missed"},
            separable={"segformer_b2 minus upernet_dinov2_small": True},
        ),
    )
    (tmp_path / "extract").mkdir()
    (tmp_path / "evidence").mkdir()

    result = runner.invoke(
        app,
        [
            "allseed",
            "analyse",
            "--extract-dir",
            str(tmp_path / "extract"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "command": "allseed analyse",
        "summary_path": str(tmp_path / "analysis" / "summary.json"),
        "categories": {"segformer_b2": "majority missed"},
        "separable": {"segformer_b2 minus upernet_dinov2_small": True},
    }


def test_a_refusing_service_exits_nonzero_with_a_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.allseed as allseed_cli

    def refuse(*_args: Any) -> Any:
        raise ValueError("the reproduction gate failed")

    monkeypatch.setattr(allseed_cli, "ANALYSE_SERVICE", refuse)
    (tmp_path / "extract").mkdir()
    (tmp_path / "evidence").mkdir()

    result = runner.invoke(
        app,
        [
            "allseed",
            "analyse",
            "--extract-dir",
            str(tmp_path / "extract"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
    )

    assert result.exit_code == 1
    assert "the reproduction gate failed" in result.stderr


def test_evidence_derives_the_report_evidence_from_a_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drivemetrics.cli.allseed as allseed_cli

    calls: dict[str, Any] = {}

    def derive(summary: Path, output: Path) -> Any:
        calls["args"] = (summary, output)
        return SimpleNamespace(evidence_path=output)

    monkeypatch.setattr(allseed_cli, "EVIDENCE_SERVICE", derive)
    summary = touch(tmp_path / "summary.json")
    output = tmp_path / "evidence" / "allseed-evidence.json"

    result = runner.invoke(
        app, ["allseed", "evidence", "--summary", str(summary), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert calls["args"] == (summary, output)
    assert json.loads(result.stdout) == {
        "command": "allseed evidence",
        "evidence_path": str(output),
    }
