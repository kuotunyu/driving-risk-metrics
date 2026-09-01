"""End-to-end determinism contract for the locked training engine.

The backend here is a deterministic pure-Python stand-in: it performs arithmetic
"optimizer steps" whose result depends on the exact micro-batch contents and
learning rate, so a change in sampling order, accumulation, or schedule changes
the final checkpoint bytes. Real framework training first runs at P1-15; this
test proves the orchestration around it is reproducible on CPU.

It runs against the committed protocol file, so a protocol edit that breaks the
engine is caught here rather than on a paid GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR
from drivemetrics.data.manifest import build_paired_manifest
from drivemetrics.training import train

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

PROVENANCE = {
    "commit": "c" * 40,
    "lock_sha256": "d" * 64,
    "hardware": {"gpu": "cpu-only", "runtime": "pytest"},
}


class DeterministicBackend:
    """Fold every micro batch and learning rate into one reproducible weight."""

    def __init__(self) -> None:
        self.micro_batches = 0
        self.applied_updates = 0

    def seed_all(self, seed: int) -> None:
        self.seed = seed

    def create_training_state(self, model_name: str, optimizer: Any) -> dict[str, Any]:
        return {"model": model_name, "optimizer": dict(optimizer), "weight": 0.0}

    def run_step(
        self,
        state: Any,
        batch: Any,
        learning_rate: float,
        *,
        apply_update: bool,
    ) -> float:
        self.micro_batches += 1
        self.applied_updates += int(apply_update)
        payload = "|".join(f"{sample_id}:{draw:.17g}" for sample_id, draw in batch)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        contribution = learning_rate * (int(digest[:8], 16) % 97) / 97.0
        state["weight"] = round(state["weight"] + contribution, 12)
        return float(state["weight"])

    def save_checkpoint(self, state: Any, path: Path, metadata: Any) -> str:
        payload = json.dumps(
            {"state": state, "metadata": dict(metadata)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def load_checkpoint(self, path: Path, expected_metadata: Any) -> Any:
        payload = json.loads(path.read_bytes())
        if payload["metadata"] != dict(expected_metadata):
            raise ValueError("checkpoint metadata does not match the expected run")
        return payload["state"]


def build_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny synthetic cohort beside a copy of the committed protocol."""

    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    for index in range(10):
        (image_root / f"tiny{index:02d}.jpg").write_bytes(f"image-{index}".encode())
        (label_root / f"tiny{index:02d}_train_id.png").write_bytes(f"label-{index}".encode())

    manifest = build_paired_manifest(image_root, label_root, "train")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(dataclasses.asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )

    shutil.copyfile(PROTOCOL_SOURCE, tmp_path / "protocol.yaml")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "drivemetrics-training-run/v1",
                "protocol_path": "protocol.yaml",
                "model": "fcn_resnet50",
                "micro_batch_size": 4,
            }
        ),
        encoding="utf-8",
    )
    return config_path, manifest_path


@pytest.fixture
def provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))


def test_rerunning_the_same_job_reproduces_the_checkpoint_byte_for_byte(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Nondeterministic sampling would make a published checkpoint impossible to replay."""

    config_path, manifest_path = build_workspace(tmp_path)

    first = train(
        config_path, manifest_path, tmp_path / "run-a", 17, backend=DeterministicBackend()
    )
    second = train(
        config_path,
        manifest_path,
        tmp_path / "run-b",
        17,
        backend=DeterministicBackend(),
    )

    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert first.checkpoint_path.read_bytes() == second.checkpoint_path.read_bytes()


def test_two_identical_runs_agree_on_every_recorded_field_except_timestamps(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A drifting config, protocol, or manifest hash would silently split one experiment."""

    config_path, manifest_path = build_workspace(tmp_path)

    first = train(
        config_path, manifest_path, tmp_path / "run-a", 17, backend=DeterministicBackend()
    )
    second = train(
        config_path,
        manifest_path,
        tmp_path / "run-b",
        17,
        backend=DeterministicBackend(),
    )

    records = []
    for result in (first, second):
        record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
        del record["started_at_utc"]
        del record["finished_at_utc"]
        records.append(record)

    assert records[0] == records[1]
    assert records[0]["status"] == "succeeded"


def test_the_locked_step_budget_and_effective_batch_are_actually_executed(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Stopping early or skipping accumulation would train a different model than declared."""

    config_path, manifest_path = build_workspace(tmp_path)
    backend = DeterministicBackend()

    result = train(config_path, manifest_path, tmp_path / "run", 17, backend=backend)

    assert result.final_step == 30000
    assert backend.micro_batches == 30000 * 4
    assert backend.applied_updates == 30000


def test_a_different_approved_seed_changes_the_final_checkpoint(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Identical checkpoints across seeds would mean the seed never reached the data order."""

    config_path, manifest_path = build_workspace(tmp_path)

    first = train(
        config_path, manifest_path, tmp_path / "run-17", 17, backend=DeterministicBackend()
    )
    second = train(
        config_path,
        manifest_path,
        tmp_path / "run-42",
        42,
        backend=DeterministicBackend(),
    )

    assert first.checkpoint_sha256 != second.checkpoint_sha256


def test_the_checkpoint_reloads_only_against_its_own_metadata(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Loading a checkpoint from another run would evaluate the wrong model in silence."""

    config_path, manifest_path = build_workspace(tmp_path)
    backend = DeterministicBackend()
    result = train(config_path, manifest_path, tmp_path / "run", 17, backend=backend)
    metadata = json.loads(result.checkpoint_path.read_bytes())["metadata"]

    state = backend.load_checkpoint(result.checkpoint_path, metadata)

    assert state["model"] == "fcn_resnet50"
    with pytest.raises(ValueError, match="metadata"):
        backend.load_checkpoint(result.checkpoint_path, {**metadata, "seed": 42})
