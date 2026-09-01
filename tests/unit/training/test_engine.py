"""Contracts for the injected-backend deterministic training engine."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from drivemetrics.artifacts.run_record import PROVENANCE_ENV_VAR

PROTOCOL_DOCUMENT: dict[str, Any] = {
    "schema_version": "bdd100k-semseg-protocol/v1",
    "dataset": {"name": "bdd100k", "version": "10k-semantic-v1"},
    "splits": {
        "source_train": 7000,
        "train": 6300,
        "calibration": 700,
        "locked_validation": 1000,
        "unlabeled_test": 2000,
    },
    "paths": {
        "train_images": "images/10k/train",
        "train_labels": "labels/sem_seg/masks/train",
        "validation_images": "images/10k/val",
        "validation_labels": "labels/sem_seg/masks/val",
    },
    "input": {
        "resize_height": 512,
        "resize_width": 910,
        "padded_height": 512,
        "padded_width": 1024,
        "image_pad_value_after_normalization": 0.0,
        "mask_pad_value": 255,
    },
    "training": {
        "steps": 30000,
        "warmup_steps": 1000,
        "effective_batch_size": 16,
        "horizontal_flip_probability": 0.5,
        "checkpoint_selection": "final_step_only",
    },
    "models": {
        "fcn_resnet50": {
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        },
        "deeplabv3_resnet50": {
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        },
        "segformer_b0": {
            "optimizer": "adamw",
            "learning_rate": 0.00006,
            "weight_decay": 0.01,
        },
    },
    "calibration": {"method": "scalar_temperature", "objective": "multiclass_nll"},
    "statistics": {
        "bootstrap_resamples": 5000,
        "bootstrap_seed": 20260831,
        "confidence": 0.95,
    },
}

PROVENANCE = {
    "commit": "0" * 39 + "a",
    "lock_sha256": "b" * 64,
    "hardware": {"gpu": "NVIDIA A100-SXM4-40GB", "cuda": "12.4"},
}


def load_engine_module() -> ModuleType:
    try:
        from drivemetrics.training import engine
    except ImportError:
        pytest.fail("drivemetrics.training.engine is missing", pytrace=False)
    return engine


def load_losses_module() -> ModuleType:
    try:
        from drivemetrics.training import losses
    except ImportError:
        pytest.fail("drivemetrics.training.losses is missing", pytrace=False)
    return losses


class FakeBackend:
    """Record every engine call without depending on any training framework."""

    def __init__(self, fail_at_step: int | None = None) -> None:
        self.fail_at_step = fail_at_step
        self.seeded: list[int] = []
        self.created: list[tuple[str, dict[str, Any]]] = []
        self.step_count = 0
        self.first_steps: list[tuple[Any, float]] = []
        self.last_step: tuple[Any, float] | None = None
        self.saves: list[tuple[Path, dict[str, Any]]] = []
        self.update_flags: list[bool] = []
        self.applied_updates = 0

    def seed_all(self, seed: int) -> None:
        self.seeded.append(seed)

    def create_training_state(self, model_name: str, optimizer: Any) -> dict[str, Any]:
        self.created.append((model_name, dict(optimizer)))
        return {"model": model_name, "weight": 0.0}

    def run_step(
        self,
        state: Any,
        batch: Any,
        learning_rate: float,
        *,
        apply_update: bool,
    ) -> float:
        self.step_count += 1
        if len(self.update_flags) < 8:
            self.update_flags.append(apply_update)
        self.applied_updates += int(apply_update)
        if self.fail_at_step is not None and self.step_count == self.fail_at_step:
            raise RuntimeError("backend exploded")
        if len(self.first_steps) < 8:
            self.first_steps.append((batch, learning_rate))
        self.last_step = (batch, learning_rate)
        state["weight"] += learning_rate
        return float(state["weight"])

    def save_checkpoint(self, state: Any, path: Path, metadata: Any) -> str:
        payload = json.dumps(
            {"state": state, "metadata": dict(metadata)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(payload)
        self.saves.append((path, dict(metadata)))
        return hashlib.sha256(payload).hexdigest()

    def load_checkpoint(self, path: Path, expected_metadata: Any) -> Any:
        payload = json.loads(path.read_bytes())
        if payload["metadata"] != dict(expected_metadata):
            raise ValueError("checkpoint metadata does not match the expected run")
        return payload["state"]


def write_manifest(directory: Path, sample_count: int = 12) -> Path:
    from drivemetrics.data.manifest import build_paired_manifest

    image_root = directory / "images"
    label_root = directory / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    for index in range(sample_count):
        (image_root / f"sample{index:03d}.jpg").write_bytes(f"image-{index}".encode())
        (label_root / f"sample{index:03d}_train_id.png").write_bytes(f"label-{index}".encode())

    manifest = build_paired_manifest(image_root, label_root, "train")
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(dataclasses.asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def write_configs(
    directory: Path,
    *,
    model: str = "fcn_resnet50",
    micro_batch_size: int = 4,
    protocol_overrides: dict[str, Any] | None = None,
) -> Path:
    document = json.loads(json.dumps(PROTOCOL_DOCUMENT))
    if protocol_overrides is not None:
        for section, values in protocol_overrides.items():
            document[section].update(values)
    (directory / "protocol.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    run_config_path = directory / "run.yaml"
    run_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "drivemetrics-training-run/v1",
                "protocol_path": "protocol.yaml",
                "model": model,
                "micro_batch_size": micro_batch_size,
            }
        ),
        encoding="utf-8",
    )
    return run_config_path


@pytest.fixture
def provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))


def test_the_backend_is_seeded_before_any_state_or_step_exists(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Creating the model before seeding would randomize initialization outside the seed."""

    engine = load_engine_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    assert backend.seeded == [17]
    assert backend.created == [
        (
            "fcn_resnet50",
            {
                "optimizer": "sgd",
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        )
    ]


def test_effective_batch_sixteen_is_reached_by_gradient_accumulation(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A single micro batch per step would silently train at one quarter of the locked batch."""

    engine = load_engine_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path, micro_batch_size=4)
    manifest_path = write_manifest(tmp_path / "data")

    engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    assert backend.step_count == 30000 * 4
    assert all(len(batch) == 4 for batch, _ in backend.first_steps)
    assert all(
        isinstance(sample_id, str) for batch, _ in backend.first_steps for sample_id, _ in batch
    )


def test_every_micro_batch_in_one_optimizer_step_shares_its_learning_rate(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Advancing the schedule per micro batch would decay four times too fast."""

    engine = load_engine_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    first_rates = [rate for _, rate in backend.first_steps[:4]]
    second_rates = [rate for _, rate in backend.first_steps[4:8]]
    assert first_rates == [0.01 / 1000] * 4
    assert second_rates == [0.01 * 2 / 1000] * 4
    assert backend.last_step is not None
    assert backend.last_step[1] == 0.0


def test_exactly_one_checkpoint_is_written_and_only_at_the_final_step(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Periodic or best-metric checkpoints would reopen the forbidden selection decision."""

    engine = load_engine_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    result = engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    assert len(backend.saves) == 1
    assert result.final_step == 30000
    assert result.checkpoint_path == backend.saves[0][0]
    assert result.checkpoint_path.read_bytes()
    assert (
        result.checkpoint_sha256 == hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest()
    )


def test_checkpoint_metadata_pins_the_ignored_label_and_the_run_identity(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A different ignore index would score padded pixels as real classification errors."""

    engine = load_engine_module()
    losses = load_losses_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    metadata = backend.saves[0][1]
    assert metadata["loss"]["ignore_index"] == 255
    assert metadata["loss"]["ignore_index"] == losses.IGNORE_INDEX
    assert metadata["loss"]["ignore_index"] == PROTOCOL_DOCUMENT["input"]["mask_pad_value"]
    assert metadata["seed"] == 17
    assert metadata["model"] == "fcn_resnet50"
    assert metadata["final_step"] == 30000


def test_a_successful_run_record_carries_every_provenance_hash(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A run without provenance hashes could never be replayed or audited."""

    engine = load_engine_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    result = engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    assert record["schema_version"] == "driving-risk-run/v1"
    assert record["status"] == "succeeded"
    assert record["seed"] == 17
    assert record["commit"] == PROVENANCE["commit"]
    assert record["lock_sha256"] == PROVENANCE["lock_sha256"]
    assert record["hardware"] == PROVENANCE["hardware"]
    assert record["artifacts"]["final_checkpoint"] == result.checkpoint_sha256
    assert record["finished_at_utc"] is not None


def test_a_backend_failure_records_a_failed_run_and_reraises(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A silent failure would leave a partial checkpoint that looks like a finished run."""

    engine = load_engine_module()
    backend = FakeBackend(fail_at_step=3)
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="backend exploded"):
        engine.train(config_path, manifest_path, output_dir, 17, backend=backend)

    record = json.loads((output_dir / "run_record.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["artifacts"] == {}
    assert backend.saves == []


@pytest.mark.parametrize("seed", [1, 0, -17, 18, True])
def test_an_unapproved_seed_fails_closed(
    seed: object,
    tmp_path: Path,
    provenance: None,
) -> None:
    """An unlisted seed would produce a run that the nine-job matrix cannot contain."""

    engine = load_engine_module()
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError, match="seed"):
        engine.train(
            config_path,
            manifest_path,
            tmp_path / "out",
            seed,  # type: ignore[arg-type]
            backend=FakeBackend(),
        )


@pytest.mark.parametrize("micro_batch_size", [5, 32, 0, -4])
def test_a_micro_batch_that_cannot_form_the_effective_batch_fails_closed(
    micro_batch_size: int,
    tmp_path: Path,
    provenance: None,
) -> None:
    """A non-dividing micro batch would train at an undeclared effective batch size."""

    engine = load_engine_module()
    config_path = write_configs(tmp_path, micro_batch_size=micro_batch_size)
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError, match="micro_batch_size"):
        engine.train(
            config_path,
            manifest_path,
            tmp_path / "out",
            17,
            backend=FakeBackend(),
        )


def test_missing_run_provenance_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guessing the commit or hardware would put an unverifiable claim in the run record."""

    engine = load_engine_module()
    monkeypatch.delenv(PROVENANCE_ENV_VAR, raising=False)
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError, match=PROVENANCE_ENV_VAR):
        engine.train(
            config_path,
            manifest_path,
            tmp_path / "out",
            17,
            backend=FakeBackend(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"training": {"early_stopping": True}},
        {"training": {"checkpoint_selection": "best_validation_miou"}},
        {"training": {"lr_scheduler_metric": "validation_miou"}},
        {"training": {"random_scale_range": [0.5, 2.0]}},
    ],
)
def test_a_protocol_with_unapproved_training_keys_fails_closed(
    overrides: dict[str, Any],
    tmp_path: Path,
    provenance: None,
) -> None:
    """Early stopping, best-checkpoint selection, or extra augmentation would break the lock."""

    engine = load_engine_module()
    config_path = write_configs(tmp_path, protocol_overrides=overrides)
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError):
        engine.train(
            config_path,
            manifest_path,
            tmp_path / "out",
            17,
            backend=FakeBackend(),
        )


def test_an_unapproved_model_in_the_run_config_fails_closed(
    tmp_path: Path,
    provenance: None,
) -> None:
    """An unlisted architecture would produce a checkpoint outside the approved comparison."""

    engine = load_engine_module()
    config_path = write_configs(tmp_path, model="setr")
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError):
        engine.train(
            config_path,
            manifest_path,
            tmp_path / "out",
            17,
            backend=FakeBackend(),
        )


def test_the_optimizer_table_covers_exactly_the_approved_models() -> None:
    """A model without a pinned optimizer would silently fall back to library defaults."""

    from drivemetrics.models import APPROVED_MODEL_NAMES
    from drivemetrics.training import schedule

    assert schedule.OPTIMIZER_TABLE_MODELS == APPROVED_MODEL_NAMES


def test_training_package_exports_the_engine_entry_points() -> None:
    """The CLI consumes these through the package entry point."""

    import drivemetrics.training as training

    engine = load_engine_module()
    assert training.train is engine.train
    assert training.TrainingResult is engine.TrainingResult


@pytest.mark.parametrize("raw", ["not json at all", "[1, 2, 3]", '{"commit": "zz"}'])
def test_malformed_run_provenance_fails_closed(
    raw: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-filled provenance block would produce a run record nobody can verify."""

    engine = load_engine_module()
    monkeypatch.setenv(PROVENANCE_ENV_VAR, raw)
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError):
        engine.train(
            config_path,
            manifest_path,
            tmp_path / "out",
            17,
            backend=FakeBackend(),
        )


def test_a_run_configuration_that_is_not_a_mapping_fails_closed(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A YAML list would reach the strict model as positional garbage."""

    engine = load_engine_module()
    config_path = tmp_path / "run.yaml"
    config_path.write_text("- not\n- a mapping\n", encoding="utf-8")
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(TypeError, match="mapping"):
        engine.train(
            config_path,
            manifest_path,
            tmp_path / "out",
            17,
            backend=FakeBackend(),
        )


@pytest.mark.parametrize("protocol_path", ["/etc/protocol.yaml", "../protocol.yaml", "C:/p.yaml"])
def test_an_unsafe_protocol_path_fails_closed(
    protocol_path: str,
    tmp_path: Path,
    provenance: None,
) -> None:
    """A path escaping the configuration directory could load an unaudited protocol."""

    engine = load_engine_module()
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "drivemetrics-training-run/v1",
                "protocol_path": protocol_path,
                "model": "fcn_resnet50",
                "micro_batch_size": 4,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError, match="relative"):
        engine.train(
            config_path,
            manifest_path,
            tmp_path / "out",
            17,
            backend=FakeBackend(),
        )


def test_the_optimizer_update_is_applied_once_per_accumulation_window(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Updating on every micro batch would train at one quarter of the locked batch size."""

    engine = load_engine_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path, micro_batch_size=4)
    manifest_path = write_manifest(tmp_path / "data")

    engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    assert backend.update_flags == [False, False, False, True, False, False, False, True]
    assert backend.applied_updates == 30000


def test_a_single_micro_batch_updates_on_every_call(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Without accumulation every micro batch is already a whole optimizer step."""

    engine = load_engine_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path, micro_batch_size=16)
    manifest_path = write_manifest(tmp_path / "data")

    engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    assert backend.update_flags == [True] * 8
    assert backend.applied_updates == 30000


def test_every_sample_carries_a_deterministic_augmentation_draw(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Drawing the flip inside the backend would make a rerun depend on backend RNG state."""

    engine = load_engine_module()
    backend = FakeBackend()
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    engine.train(config_path, manifest_path, tmp_path / "out", 17, backend=backend)

    draws = [draw for batch, _ in backend.first_steps for _, draw in batch]
    assert len(draws) == 32
    assert all(0.0 <= draw < 1.0 for draw in draws)
    assert len(set(draws)) > 1
