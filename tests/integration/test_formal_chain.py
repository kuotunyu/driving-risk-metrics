"""The whole formal chain, nine runs, driven through the command line on CPU.

Each stage is proven in isolation elsewhere. This proves they compose into the
exact sequence the paid GPU sessions will execute, on the exact directory
convention the index builder reads, so the first time the chain runs end to end
is not the first time it costs money.

Frameworks are injected as fakes; everything else is the real code path.
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

from drivemetrics.artifacts.formal_set import validate_formal_run_index
from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR
from drivemetrics.cli.app import app
from drivemetrics.data.manifest import build_paired_manifest
from drivemetrics.models.registry import APPROVED_MODEL_NAMES
from drivemetrics.training.engine import APPROVED_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"
RISK_PROFILE = REPO_ROOT / "configs" / "risk_profiles" / "vru_priority.yaml"

SOURCE_HEIGHT = 48
SOURCE_WIDTH = 96
NUM_CLASSES = 19
PREDICTED_CLASS = 1
# Critical classes must have ground-truth support, or critical recall is
# undefined and the aggregation correctly refuses to report it as zero.
CRITICAL_PAINT = (11, 12)

PROVENANCE = {
    "commit": "1" * 40,
    "lock_sha256": "2" * 64,
    "hardware": {"gpu": "cpu-only", "runtime": "pytest"},
}

runner = CliRunner()


class FakeTrainingBackend:
    """Fold every micro batch into one reproducible weight without a framework."""

    def seed_all(self, seed: int) -> None:
        self.seed = seed

    def create_training_state(self, model_name: str, optimizer: Any) -> dict[str, Any]:
        return {"model": model_name, "weight": 0.0}

    def run_step(
        self, state: Any, batch: Any, learning_rate: float, *, apply_update: bool
    ) -> float:
        state["weight"] = round(state["weight"] + learning_rate, 12)
        return float(state["weight"])

    def save_checkpoint(self, state: Any, path: Path, metadata: Any) -> str:
        payload = json.dumps(
            {"state": state, "metadata": dict(metadata)}, sort_keys=True, separators=(",", ":")
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
    """Restore the metadata the fake checkpoint carries, as the real backend would."""

    def load_model(self, checkpoint_path: Path) -> tuple[FakeModel, dict[str, Any]]:
        return FakeModel(), dict(json.loads(checkpoint_path.read_bytes())["metadata"])


def paint_cohort(images: Path, labels: Path, prefix: str) -> None:
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    mask = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), PREDICTED_CLASS, dtype=np.uint8)
    mask[:, SOURCE_WIDTH // 2 :] = PREDICTED_CLASS + 1
    for row, value in enumerate(CRITICAL_PAINT):
        mask[row * 8 : row * 8 + 8, :] = value
    for index in range(2):
        sample_id = f"{prefix}{index:04d}"
        Image.fromarray(np.zeros((SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)).save(
            images / f"{sample_id}.jpg"
        )
        Image.fromarray(mask).save(labels / f"{sample_id}_train_id.png")


def build_workspace(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    paint_cohort(data_root / "images/10k/val", data_root / "labels/sem_seg/masks/val", "v")
    paint_cohort(data_root / "images/10k/train", data_root / "labels/sem_seg/masks/train", "c")

    manifests: dict[str, Path] = {}
    for split, tree in (
        ("locked_validation", "val"),
        ("calibration", "train"),
        ("train", "train"),
    ):
        manifest = build_paired_manifest(
            data_root / f"images/10k/{tree}", data_root / f"labels/sem_seg/masks/{tree}", split
        )
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(dataclasses.asdict(manifest), sort_keys=True), "utf-8")
        manifests[split] = path

    configs = tmp_path / "configs"
    (configs / "protocols").mkdir(parents=True)
    shutil.copyfile(PROTOCOL_SOURCE, configs / "protocols" / "bdd100k_semseg_v1.yaml")
    for model in APPROVED_MODEL_NAMES:
        (configs / f"run_{model}.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": "drivemetrics-training-run/v1",
                    "protocol_path": "protocols/bdd100k_semseg_v1.yaml",
                    "model": model,
                    "micro_batch_size": 8,
                }
            ),
            encoding="utf-8",
        )
    return {
        "data_root": data_root,
        "protocol": configs / "protocols" / "bdd100k_semseg_v1.yaml",
        "configs": configs,
        **manifests,
    }


def invoke(arguments: list[str]) -> dict[str, Any]:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, f"{arguments[:2]} failed:\n{result.output}"
    return dict(json.loads(result.stdout.strip()))


def test_nine_runs_flow_from_training_to_the_published_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One (model, seed) at a time, exactly as the GPU sessions will do it."""

    import drivemetrics.cli.calibrate as calibrate_cli
    import drivemetrics.cli.evaluate as evaluate_cli
    import drivemetrics.cli.train as train_cli

    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))
    monkeypatch.setattr(train_cli, "BACKEND_FACTORY", lambda *_, **__: FakeTrainingBackend())
    monkeypatch.setattr(calibrate_cli, "BACKEND_FACTORY", lambda *_, **__: FakeEvaluationBackend())
    monkeypatch.setattr(evaluate_cli, "BACKEND_FACTORY", lambda *_, **__: FakeEvaluationBackend())

    paths = build_workspace(tmp_path)
    runs_root = tmp_path / "runs"
    common = ["--data-root", str(paths["data_root"])]

    for model in APPROVED_MODEL_NAMES:
        for seed in APPROVED_SEEDS:
            run_dir = runs_root / model / f"seed-{seed}"
            trained = invoke(
                [
                    "train",
                    "--config",
                    str(paths["configs"] / f"run_{model}.yaml"),
                    "--manifest",
                    str(paths["train"]),
                    *common,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(run_dir / "train"),
                ]
            )
            assert trained["final_step"] == 30000
            checkpoint = trained["checkpoint_path"]

            fitted = invoke(
                [
                    "calibrate",
                    "--config",
                    str(paths["protocol"]),
                    "--manifest",
                    str(paths["calibration"]),
                    "--checkpoint",
                    checkpoint,
                    *common,
                    "--output-dir",
                    str(run_dir / "calibration"),
                ]
            )
            assert fitted["temperature"] > 0.0

            evaluate = [
                "evaluate",
                "--config",
                str(paths["protocol"]),
                "--manifest",
                str(paths["locked_validation"]),
                "--checkpoint",
                checkpoint,
                *common,
            ]
            raw = invoke([*evaluate, "--output-dir", str(run_dir / "eval")])
            calibrated = invoke(
                [
                    *evaluate,
                    "--output-dir",
                    str(run_dir / "eval_calibrated"),
                    "--temperature",
                    fitted["artifact_path"],
                ]
            )
            assert raw["calibrated"] is False and calibrated["calibrated"] is True
            assert raw["evaluated_samples"] == calibrated["evaluated_samples"] == 2

    indexed = invoke(
        [
            "index",
            "--runs-root",
            str(runs_root),
            "--config",
            str(paths["protocol"]),
            "--manifest",
            str(paths["locked_validation"]),
            "--risk-profile",
            str(RISK_PROFILE),
            "--output",
            str(runs_root / "formal_run_index.json"),
        ]
    )
    assert indexed["run_count"] == 9
    document = json.loads(Path(indexed["index_path"]).read_text(encoding="utf-8"))
    assert validate_formal_run_index(document) == ()

    aggregated = invoke(
        ["aggregate", "--index", indexed["index_path"], "--output-dir", str(tmp_path / "analysis")]
    )
    assert sorted(aggregated["models"]) == sorted(APPROVED_MODEL_NAMES)
    metrics = json.loads(Path(aggregated["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["seed_count"] == 3 and metrics["sample_count"] == 2
    assert set(metrics["metrics"]) == set(APPROVED_MODEL_NAMES)

    claims = tmp_path / "claims.yaml"
    claims.write_text(
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
                        "claim_id": "synthetic-chain",
                        "text": "The synthetic chain completed for every approved model.",
                        "evidence_type": "synthetic",
                        "protocol_hash": metrics["protocol_hash"],
                        "dataset_manifest_hash": metrics["dataset_manifest_hash"],
                        "artifact_path": "analysis/metrics.json",
                        "metric_path": f"/metrics/{APPROVED_MODEL_NAMES[0]}/miou",
                        "status": "verified",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audited = invoke(["audit-claims", "--claims", str(claims), "--repo-root", str(tmp_path)])
    assert audited["violations"] == 0

    published = invoke(
        [
            "report",
            "--claims",
            str(claims),
            "--artifacts-dir",
            str(tmp_path / "analysis"),
            "--output-dir",
            str(tmp_path / "site"),
            "--repo-root",
            str(tmp_path),
        ]
    )
    page = Path(published["index_path"]).read_text(encoding="utf-8")
    assert published["claim_count"] == 1
    assert "synthetic" in page and "Limitations" in page
