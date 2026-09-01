"""Contracts for step progress reporting during a formal training run.

A formal run is eight to twelve hours in a notebook cell with no checkpoint
before the end. Without a heartbeat there is no way to tell a healthy run from
a hung one, and no way to know how far a killed run got. The hook exists for
that and for nothing else: it observes, it never influences the result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR
from drivemetrics.data.manifest import build_paired_manifest
from drivemetrics.training.engine import train

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"
TOTAL_STEPS = 30000

PROVENANCE = {
    "commit": "1" * 40,
    "lock_sha256": "2" * 64,
    "hardware": {"gpu": "cpu-only", "runtime": "pytest"},
}


class LossBackend:
    """A framework stand-in whose micro-batch losses are known in advance."""

    def __init__(self, losses: tuple[float, ...] = (0.25, 0.75)) -> None:
        self.losses = losses
        self.calls = 0

    def seed_all(self, seed: int) -> None:
        self.seed = seed

    def create_training_state(self, model_name: str, optimizer: Any) -> dict[str, Any]:
        return {"weight": 0.0}

    def run_step(
        self, state: Any, batch: Any, learning_rate: float, *, apply_update: bool
    ) -> float:
        loss = self.losses[self.calls % len(self.losses)]
        self.calls += 1
        state["weight"] = round(state["weight"] + learning_rate, 12)
        return loss

    def save_checkpoint(self, state: Any, path: Path, metadata: Any) -> str:
        payload = json.dumps({"state": state, "metadata": dict(metadata)}, sort_keys=True)
        path.write_bytes(payload.encode("utf-8"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load_checkpoint(self, path: Path, expected_metadata: Any) -> Any:
        return json.loads(path.read_bytes())


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for index in range(2):
        (images / f"t{index}.jpg").write_bytes(b"i")
        (labels / f"t{index}_train_id.png").write_bytes(b"l")
    manifest = build_paired_manifest(images, labels, "train")
    manifest_path = tmp_path / "train.json"
    manifest_path.write_text(json.dumps(dataclasses.asdict(manifest), sort_keys=True), "utf-8")

    configs = tmp_path / "configs"
    (configs / "protocols").mkdir(parents=True)
    shutil.copyfile(PROTOCOL_SOURCE, configs / "protocols" / "bdd100k_semseg_v1.yaml")
    run_config = configs / "run_segformer_b2.yaml"
    run_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "drivemetrics-training-run/v1",
                "protocol_path": "protocols/bdd100k_semseg_v1.yaml",
                "model": "segformer_b2",
                "micro_batch_size": 8,
            }
        ),
        encoding="utf-8",
    )
    return {"config": run_config, "manifest": manifest_path, "root": tmp_path}


def test_the_hook_sees_every_optimizer_step_once_with_the_window_mean_loss(
    workspace: dict[str, Path],
) -> None:
    """One call per optimizer step, not per micro batch, carrying the mean of the window."""

    seen: list[tuple[int, int, float]] = []

    train(
        workspace["config"],
        workspace["manifest"],
        workspace["root"] / "out",
        17,
        backend=LossBackend(),
        on_step=lambda step, total, loss: seen.append((step, total, loss)),
    )

    assert [step for step, _, _ in seen] == list(range(1, TOTAL_STEPS + 1))
    assert {total for _, total, _ in seen} == {TOTAL_STEPS}
    # Micro batch 8 under effective batch 16 is a window of two micro batches.
    assert all(loss == pytest.approx(0.5) for _, _, loss in seen)


def test_observing_progress_changes_nothing_about_the_checkpoint(
    workspace: dict[str, Path],
) -> None:
    """The hook is a heartbeat, and a heartbeat that altered the weights is a bug."""

    silent = train(
        workspace["config"],
        workspace["manifest"],
        workspace["root"] / "a",
        17,
        backend=LossBackend(),
    )
    observed = train(
        workspace["config"],
        workspace["manifest"],
        workspace["root"] / "b",
        17,
        backend=LossBackend(),
        on_step=lambda *_: None,
    )

    assert silent.checkpoint_sha256 == observed.checkpoint_sha256
    assert silent.checkpoint_path.read_bytes() == observed.checkpoint_path.read_bytes()


def test_the_progress_line_carries_what_a_reader_needs_to_judge_a_run() -> None:
    """Step, total, loss, elapsed and a remaining estimate; nothing that needs a decoder."""

    from drivemetrics.cli.train import format_progress

    line = format_progress(1500, 30000, 0.4321, elapsed_seconds=1800.0)

    assert "1500/30000" in line
    assert "0.4321" in line
    assert "30m" in line or "0.5h" in line
    # 1500 steps in 1800 s leaves 28500 steps at 1.2 s each: 9.5 h remaining.
    assert "9.5h" in line


def test_the_printer_writes_only_on_the_interval_and_at_the_end() -> None:
    """Thirty thousand lines would bury the one that matters; silence would hide it."""

    from drivemetrics.cli.train import progress_printer

    stream = io.StringIO()
    ticks = iter(range(0, 10_000_000))
    printer = progress_printer(every=500, stream=stream, clock=lambda: float(next(ticks)))

    for step in range(1, 1201):
        printer(step, 1200, 0.5)

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert [int(line.split("step ")[1].split("/")[0]) for line in lines] == [500, 1000, 1200]


def test_the_train_command_wires_a_progress_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A formal run launched from the notebook must never be silent for ten hours."""

    import drivemetrics.cli.train as train_cli
    from drivemetrics.cli.app import app

    captured: dict[str, Any] = {}

    class Result:
        final_step = TOTAL_STEPS
        checkpoint_path = tmp_path / "final_checkpoint.pt"
        checkpoint_sha256 = "c" * 64
        run_record_path = tmp_path / "run_record.json"

    def fake_train(*args: Any, **kwargs: Any) -> Result:
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(train_cli, "BACKEND_FACTORY", lambda *_, **__: object())
    monkeypatch.setattr(train_cli, "TRAIN_SERVICE", fake_train)
    for name in ("run.yaml", "manifest.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "data").mkdir()

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            str(tmp_path / "run.yaml"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--data-root",
            str(tmp_path / "data"),
            "--seed",
            "17",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert callable(captured.get("on_step"))
