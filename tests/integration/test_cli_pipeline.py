"""End-to-end pipeline contract driven entirely through the command line.

The frameworks are injected as fakes, so the whole chain runs on CPU in seconds:
frozen manifest, training, evaluation, claim audit, and the published page. What
is being tested is that the commands compose into one reproducible workflow.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR
from drivemetrics.cli.app import app
from drivemetrics.data.manifest import build_paired_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

VALIDATION_IMAGES = "images/10k/val"
VALIDATION_LABELS = "labels/sem_seg/masks/val"
SOURCE_HEIGHT = 90
SOURCE_WIDTH = 160
NUM_CLASSES = 4
PREDICTED_CLASS = 1

PROVENANCE = {
    "commit": "1" * 40,
    "lock_sha256": "2" * 64,
    "hardware": {"gpu": "cpu-only", "runtime": "pytest"},
}

runner = CliRunner()


class FakeTrainingBackend:
    """Fold every micro batch into one reproducible weight without a framework."""

    def __init__(self) -> None:
        self.micro_batches = 0

    def seed_all(self, seed: int) -> None:
        self.seed = seed

    def create_training_state(self, model_name: str, optimizer: Any) -> dict[str, Any]:
        return {"model": model_name, "weight": 0.0}

    def run_step(
        self,
        state: Any,
        batch: Any,
        learning_rate: float,
        *,
        apply_update: bool,
    ) -> float:
        self.micro_batches += 1
        state["weight"] = round(state["weight"] + learning_rate, 12)
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
        return json.loads(path.read_bytes())


class FakeModel:
    def logits(self, image_nchw: np.ndarray) -> np.ndarray:
        batch, _, height, width = image_nchw.shape
        values = np.zeros((batch, NUM_CLASSES, height, width), dtype=np.float64)
        values[:, PREDICTED_CLASS] = np.log(3.0)
        return values

    def trainable_parameters(self) -> object:
        return ("weight",)


class FakeEvaluationBackend:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def load_model(self, checkpoint_path: Path) -> tuple[FakeModel, dict[str, Any]]:
        return FakeModel(), dict(self.metadata)


def build_workspace(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    images = data_root / VALIDATION_IMAGES
    labels = data_root / VALIDATION_LABELS
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    mask = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), PREDICTED_CLASS, dtype=np.uint8)
    mask[:, SOURCE_WIDTH // 2 :] = PREDICTED_CLASS + 1
    for sample_id in ("v00001", "v00002"):
        Image.fromarray(np.zeros((SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)).save(
            images / f"{sample_id}.jpg"
        )
        Image.fromarray(mask).save(labels / f"{sample_id}_train_id.png")

    manifest = build_paired_manifest(images, labels, "locked_validation")
    manifest_path = tmp_path / "locked_validation.json"
    manifest_path.write_text(
        json.dumps(dataclasses.asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )

    protocol_dir = tmp_path / "configs" / "protocols"
    protocol_dir.mkdir(parents=True)
    shutil.copyfile(PROTOCOL_SOURCE, protocol_dir / "bdd100k_semseg_v1.yaml")
    run_config = tmp_path / "configs" / "run_fcn_resnet50.yaml"
    run_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "drivemetrics-training-run/v1",
                "protocol_path": "protocols/bdd100k_semseg_v1.yaml",
                "model": "fcn_resnet50",
                "micro_batch_size": 4,
            }
        ),
        encoding="utf-8",
    )
    return {
        "data_root": data_root,
        "manifest": manifest_path,
        "protocol": protocol_dir / "bdd100k_semseg_v1.yaml",
        "run_config": run_config,
    }


def write_analysis_artifacts(tmp_path: Path, protocol_hash: str, manifest_hash: str) -> Path:
    """Stand in for the P1-17 statistics stage with a synthetic, self-consistent set."""

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    common = {"protocol_hash": protocol_hash, "dataset_manifest_hash": manifest_hash}
    (artifacts / "metrics.json").write_text(
        json.dumps(
            {
                **common,
                "cohort": "locked_validation",
                "sample_count": 2,
                "seed_count": 3,
                "interval_method": "two-stage paired bootstrap, 5000 resamples, seed 20260831",
                "metrics": {"fcn_resnet50": {"miou": 0.5}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (artifacts / "intervals.json").write_text(
        json.dumps(
            {
                **common,
                "intervals": {
                    "fcn_resnet50 (miou)": {
                        "estimate": 0.5,
                        "low": 0.4,
                        "high": 0.6,
                        "confidence": 0.95,
                        "resamples": 5000,
                        "seed": 20260831,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (artifacts / "rankings.json").write_text(
        json.dumps(
            {
                **common,
                "baseline_metric": "miou",
                "comparisons": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return artifacts


def write_claims(tmp_path: Path, protocol_hash: str, manifest_hash: str) -> Path:
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(
        yaml.safe_dump(
            {
                "allowed_evidence_types": ["observed", "derived", "synthetic", "illustrative"],
                "claim_required_fields": [
                    "claim_id",
                    "text",
                    "evidence_type",
                    "protocol_hash",
                    "dataset_manifest_hash",
                    "artifact_path",
                    "metric_path",
                    "status",
                ],
                "allowed_statuses": ["draft", "verified", "rejected", "superseded"],
                "claims": [
                    {
                        "claim_id": "synthetic-miou",
                        "text": "The synthetic pipeline model reaches 0.5 mIoU.",
                        "evidence_type": "synthetic",
                        "protocol_hash": protocol_hash,
                        "dataset_manifest_hash": manifest_hash,
                        "artifact_path": "artifacts/metrics.json",
                        "metric_path": "/metrics/fcn_resnet50/miou",
                        "status": "verified",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return claims_path


def invoke(arguments: list[str]) -> dict[str, Any]:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    return dict(json.loads(result.stdout.strip()))


def test_the_whole_pipeline_runs_through_the_command_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each command is proven in isolation; this proves they compose into one workflow."""

    import drivemetrics.cli.evaluate as evaluate_cli
    import drivemetrics.cli.train as train_cli
    from drivemetrics.data.manifest import load_manifest
    from drivemetrics.protocol.config import load_protocol

    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))
    paths = build_workspace(tmp_path)
    protocol_hash = load_protocol(paths["protocol"]).protocol_sha256
    manifest_hash = load_manifest(paths["manifest"]).manifest_sha256

    training_backend = FakeTrainingBackend()
    monkeypatch.setattr(train_cli, "BACKEND_FACTORY", lambda *_, **__: training_backend)
    trained = invoke(
        [
            "train",
            "--config",
            str(paths["run_config"]),
            "--manifest",
            str(paths["manifest"]),
            "--data-root",
            str(paths["data_root"]),
            "--seed",
            "17",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert trained["final_step"] == 30000
    assert training_backend.micro_batches == 30000 * 4

    checkpoint_metadata = json.loads(Path(trained["checkpoint_path"]).read_bytes())["metadata"]
    monkeypatch.setattr(
        evaluate_cli,
        "BACKEND_FACTORY",
        lambda *_, **__: FakeEvaluationBackend(checkpoint_metadata),
    )
    evaluated = invoke(
        [
            "evaluate",
            "--config",
            str(paths["protocol"]),
            "--manifest",
            str(paths["manifest"]),
            "--checkpoint",
            trained["checkpoint_path"],
            "--data-root",
            str(paths["data_root"]),
            "--output-dir",
            str(tmp_path / "predictions"),
        ]
    )
    assert evaluated["evaluated_samples"] == 2
    assert evaluated["artifact_count"] == 2

    write_analysis_artifacts(tmp_path, protocol_hash, manifest_hash)
    claims_path = write_claims(tmp_path, protocol_hash, manifest_hash)

    audited = invoke(["audit-claims", "--claims", str(claims_path), "--repo-root", str(tmp_path)])
    assert audited["violations"] == 0

    published = invoke(
        [
            "report",
            "--claims",
            str(claims_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-dir",
            str(tmp_path / "site"),
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert published["claim_count"] == 1
    page = Path(published["index_path"]).read_text(encoding="utf-8")
    assert "synthetic" in page
    assert "Limitations" in page
