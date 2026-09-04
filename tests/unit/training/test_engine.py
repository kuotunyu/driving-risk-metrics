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
        "upernet_convnextv2_tiny": {
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.05,
        },
        "upernet_dinov2_small": {
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.05,
        },
        "segformer_b2": {
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
        self.resume_saves: list[tuple[Path, dict[str, Any]]] = []
        self.resume_loads: list[int] = []

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

    def save_resume_state(self, state: Any, path: Path, metadata: Any) -> str:
        payload = json.dumps(
            {"state": state, "metadata": dict(metadata)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(payload)
        self.resume_saves.append((path, dict(metadata)))
        return hashlib.sha256(payload).hexdigest()

    def load_resume_state(self, state: Any, path: Path, expected_metadata: Any) -> int:
        payload = json.loads(path.read_bytes())
        recorded = dict(payload["metadata"])
        completed_step = int(recorded.pop("completed_step"))
        if recorded != dict(expected_metadata):
            raise ValueError("resume state does not belong to this run")
        state.update(payload["state"])
        self.resume_loads.append(completed_step)
        return completed_step


def write_manifest(directory: Path, sample_count: int = 12) -> Path:
    from drivemetrics.data.manifest import build_paired_manifest

    image_root = directory / "images"
    label_root = directory / "labels"
    # Resumed runs rebuild the same cohort in the same place, and the manifest
    # hash must come out identical or the resume metadata check refuses it.
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
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
    model: str = "upernet_convnextv2_tiny",
    micro_batch_size: int = 4,
    protocol_overrides: dict[str, Any] | None = None,
) -> Path:
    document = json.loads(json.dumps(PROTOCOL_DOCUMENT))
    if protocol_overrides is not None:
        for section, values in protocol_overrides.items():
            document[section].update(values)
    directory.mkdir(parents=True, exist_ok=True)
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
            "upernet_convnextv2_tiny",
            {
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "weight_decay": 0.05,
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

    # Derived from the pinned table rather than hard-coded, so this keeps
    # testing the schedule and not one model current learning rate.
    from drivemetrics.training.schedule import optimizer_spec

    base = float(optimizer_spec("upernet_convnextv2_tiny")["learning_rate"])
    first_rates = [rate for _, rate in backend.first_steps[:4]]
    second_rates = [rate for _, rate in backend.first_steps[4:8]]
    assert first_rates == [base / 1000] * 4
    assert second_rates == [base * 2 / 1000] * 4
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
    assert metadata["model"] == "upernet_convnextv2_tiny"
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

    with pytest.raises(RuntimeError, match=r"^backend exploded"):
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

    with pytest.raises(ValueError, match=r"^seed must be one of the approved seeds"):
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

    with pytest.raises(ValueError, match=r"^micro_batch_size must be a positive divisor of the"):
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
    ("overrides", "expected"),
    [
        (
            {"training": {"early_stopping": True}},
            r"^1 validation error for BDD100KSemanticProtocolV1\ntraining\.early_stopping\n  Extra inputs are not permitted",
        ),
        (
            {"training": {"checkpoint_selection": "best_validation_miou"}},
            r"^1 validation error for BDD100KSemanticProtocolV1\ntraining\.checkpoint_selection\n  Input should be 'final_step_only'",
        ),
        (
            {"training": {"lr_scheduler_metric": "validation_miou"}},
            r"^1 validation error for BDD100KSemanticProtocolV1\ntraining\.lr_scheduler_metric\n  Extra inputs are not permitted",
        ),
        (
            {"training": {"random_scale_range": [0.5, 2.0]}},
            r"^1 validation error for BDD100KSemanticProtocolV1\ntraining\.random_scale_range\n  Extra inputs are not permitted",
        ),
    ],
)
def test_a_protocol_with_unapproved_training_keys_fails_closed(
    overrides: dict[str, Any],
    expected: str,
    tmp_path: Path,
    provenance: None,
) -> None:
    """Early stopping, best-checkpoint selection, or extra augmentation would break the lock.

    Each row names the refusal it expects, because they are not the same refusal.
    Three of these keys are not in the schema at all; `checkpoint_selection` IS,
    and the protocol allows exactly one value for it. A single assertion covering
    both would pass if an unapproved key were quietly accepted and some unrelated
    field happened to fail instead.
    """

    engine = load_engine_module()
    config_path = write_configs(tmp_path, protocol_overrides=overrides)
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError, match=expected):
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

    with pytest.raises(
        ValueError, match=r"^1 validation error for TrainingRunConfigV1\nmodel\n  Input should be "
    ):
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("not json at all", r"^DRIVEMETRICS_RUN_PROVENANCE must contain a JSON object$"),
        ("[1, 2, 3]", r"^DRIVEMETRICS_RUN_PROVENANCE must contain a JSON object$"),
        (
            '{"commit": "zz"}',
            r"^3 validation errors for RunProvenance\ncommit\n  String should match pattern ",
        ),
    ],
)
def test_malformed_run_provenance_fails_closed(
    raw: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-filled provenance block would produce a run record nobody can verify.

    The last row is a different refusal from the first two: it IS a JSON object,
    and it is the schema that turns it down. Asserting one message for all three
    would let a malformed block through as long as something else complained.
    """

    engine = load_engine_module()
    monkeypatch.setenv(PROVENANCE_ENV_VAR, raw)
    config_path = write_configs(tmp_path)
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(ValueError, match=expected):
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

    with pytest.raises(TypeError, match=r"^training run configuration must be a mapping"):
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
                "model": "upernet_convnextv2_tiny",
                "micro_batch_size": 4,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = write_manifest(tmp_path / "data")

    with pytest.raises(
        ValueError,
        match=r"^1 validation error for TrainingRunConfigV1\nprotocol_path\n  Value error, protocol_path must be a safe relative POSIX path",
    ):
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


def test_every_committed_run_configuration_is_valid() -> None:
    """A broken run configuration would only be discovered on a paid GPU."""

    from drivemetrics.models import APPROVED_MODEL_NAMES
    from drivemetrics.protocol.config import load_protocol
    from drivemetrics.training.engine import load_run_config

    repo_root = Path(__file__).resolve().parents[3]
    committed = sorted((repo_root / "configs").glob("run_*.yaml"))
    assert len(committed) == len(APPROVED_MODEL_NAMES)

    models = set()
    for path in committed:
        config, digest = load_run_config(path)
        assert len(digest) == 64
        protocol = load_protocol(path.parent / config.protocol_path).protocol
        assert protocol.training.effective_batch_size % config.micro_batch_size == 0
        models.add(config.model)

    assert models == set(APPROVED_MODEL_NAMES)


def test_every_committed_config_uses_the_same_safe_micro_batch() -> None:
    """All three models must train under identical conditions, at a size that fits.

    Micro batching is a memory workaround the protocol never mentions, but it
    changes batch-norm statistics and therefore the trained model. A study whose
    whole point is comparing three models on one protocol cannot let one of them
    normalize over 16 while another normalizes over 8, so the value is uniform.

    It is 8 because that is the largest value all three fit at. Measured on a
    39.5 GiB A100 at micro batch 16: FCN 33.2 GiB, SegFormer 11.2 GiB, and
    DeepLabV3 35.3 GiB only after the caching allocator hit OOM three times and
    evicted cache to make room, at one point with 63 MiB free. That is not a
    configuration to leave running unattended for eight hours.
    """

    from drivemetrics.protocol.config import load_protocol
    from drivemetrics.training.engine import load_run_config

    repo_root = Path(__file__).resolve().parents[3]
    observed = {}
    for path in sorted((repo_root / "configs").glob("run_*.yaml")):
        config, _ = load_run_config(path)
        protocol = load_protocol(path.parent / config.protocol_path).protocol
        assert protocol.training.effective_batch_size % config.micro_batch_size == 0
        observed[path.name] = config.micro_batch_size

    assert len(set(observed.values())) == 1, f"micro batch is not uniform: {observed}"
    assert next(iter(observed.values())) == 8, observed


# The protocol pins 30,000 steps as a literal, so a resume test runs the real
# schedule with one micro batch per optimizer step: the fake backend makes that
# cheap, and it keeps the arithmetic below identical to the formal runs'.
RESUME_EVERY = 5000
CRASH_AT_STEP = 12000
LAST_SAVED_STEP = 10000
TOTAL_STEPS = 30000


def train_run(
    engine: ModuleType,
    tmp_path: Path,
    backend: FakeBackend,
    *,
    name: str = "out",
    model: str = "upernet_convnextv2_tiny",
    **overrides: Any,
) -> Any:
    """Run one full job with a single micro batch per optimizer step."""

    config_path = write_configs(tmp_path / name, model=model, micro_batch_size=16)
    manifest_path = write_manifest(tmp_path / f"data-{name}")
    return engine.train(
        config_path,
        manifest_path,
        tmp_path / name / "out",
        17,
        backend=backend,
        **overrides,
    )


def test_without_a_resume_directory_no_resume_state_is_written(
    tmp_path: Path,
    provenance: None,
) -> None:
    """The feature is opt-in, so a run that does not ask for it is untouched."""

    engine = load_engine_module()
    backend = FakeBackend()

    train_run(engine, tmp_path, backend)

    assert backend.resume_saves == []
    assert list((tmp_path / "out" / "out").glob("resume_state*")) == []


def test_resuming_is_byte_identical_to_never_being_interrupted(
    tmp_path: Path,
    provenance: None,
) -> None:
    """This is the whole claim: the safety net must not change the science.

    An uninterrupted run and a run that died and resumed must produce the same
    final checkpoint. If they can differ, the nine formal runs are no longer
    comparable and the feature is worse than the outage it prevents.
    """

    engine = load_engine_module()
    plain = train_run(engine, tmp_path, FakeBackend(), name="plain")

    resume_dir = tmp_path / "resume"
    with pytest.raises(RuntimeError):
        train_run(
            engine,
            tmp_path,
            FakeBackend(fail_at_step=CRASH_AT_STEP),
            name="interrupted",
            resume_dir=resume_dir,
            resume_every=RESUME_EVERY,
        )
    recovered = FakeBackend()
    resumed = train_run(
        engine,
        tmp_path,
        recovered,
        name="interrupted",
        resume_dir=resume_dir,
        resume_every=RESUME_EVERY,
    )

    assert recovered.resume_loads != [], "the second attempt ignored the saved state"
    assert resumed.checkpoint_sha256 == plain.checkpoint_sha256
    assert resumed.final_step == plain.final_step


def test_a_resumed_run_repeats_only_the_steps_after_the_saved_one(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Resuming that silently retrained from zero would save no time at all."""

    engine = load_engine_module()
    resume_dir = tmp_path / "resume"
    with pytest.raises(RuntimeError):
        train_run(
            engine,
            tmp_path,
            FakeBackend(fail_at_step=CRASH_AT_STEP),
            name="interrupted",
            resume_dir=resume_dir,
            resume_every=RESUME_EVERY,
        )
    recovered = FakeBackend()
    train_run(
        engine,
        tmp_path,
        recovered,
        name="interrupted",
        resume_dir=resume_dir,
        resume_every=RESUME_EVERY,
    )

    assert recovered.resume_loads == [LAST_SAVED_STEP]
    assert recovered.step_count == TOTAL_STEPS - LAST_SAVED_STEP


def test_the_resume_state_is_deleted_once_the_run_succeeds(
    tmp_path: Path,
    provenance: None,
) -> None:
    """A stale resume file would make the next run of this pair start mid-way."""

    engine = load_engine_module()
    resume_dir = tmp_path / "resume"

    train_run(
        engine,
        tmp_path,
        FakeBackend(),
        resume_dir=resume_dir,
        resume_every=RESUME_EVERY,
    )

    assert list(resume_dir.glob("*")) == []


def test_the_resume_state_survives_a_failed_run(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Deleting it on failure would throw away the hours it exists to protect."""

    engine = load_engine_module()
    resume_dir = tmp_path / "resume"

    with pytest.raises(RuntimeError):
        train_run(
            engine,
            tmp_path,
            FakeBackend(fail_at_step=CRASH_AT_STEP),
            resume_dir=resume_dir,
            resume_every=RESUME_EVERY,
        )

    assert (resume_dir / engine.RESUME_FILENAME).is_file()


def test_a_resume_state_from_another_run_is_refused(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Silently continuing another run's weights would corrupt this one invisibly."""

    engine = load_engine_module()
    resume_dir = tmp_path / "resume"
    with pytest.raises(RuntimeError):
        train_run(
            engine,
            tmp_path,
            FakeBackend(fail_at_step=CRASH_AT_STEP),
            name="first",
            resume_dir=resume_dir,
            resume_every=RESUME_EVERY,
        )

    with pytest.raises(ValueError, match=r"^resume state does not belong to this run"):
        train_run(
            engine,
            tmp_path,
            FakeBackend(),
            name="second",
            model="segformer_b2",
            resume_dir=resume_dir,
            resume_every=RESUME_EVERY,
        )


def test_the_final_step_is_not_written_as_a_resume_point(
    tmp_path: Path,
    provenance: None,
) -> None:
    """The last step already has a real checkpoint; a second copy is only confusion."""

    engine = load_engine_module()
    backend = FakeBackend()

    train_run(
        engine,
        tmp_path,
        backend,
        resume_dir=tmp_path / "resume",
        resume_every=RESUME_EVERY,
    )

    saved_steps = [metadata["completed_step"] for _, metadata in backend.resume_saves]
    assert saved_steps == [5000, 10000, 15000, 20000, 25000]


class NonResumableBackend(FakeBackend):
    """A perfectly good training backend that simply cannot put a run down."""

    save_resume_state = None  # type: ignore[assignment]
    load_resume_state = None  # type: ignore[assignment]


def test_a_backend_that_cannot_resume_refuses_a_resume_directory(
    tmp_path: Path,
    provenance: None,
) -> None:
    """Accepting the flag and ignoring it would promise a safety net that is not there."""

    engine = load_engine_module()

    with pytest.raises(TypeError, match=r"^this backend cannot save or restore a resume point"):
        train_run(
            engine,
            tmp_path,
            NonResumableBackend(),
            resume_dir=tmp_path / "resume",
            resume_every=RESUME_EVERY,
        )


class UnwritableResumeBackend(FakeBackend):
    """A backend whose resume writes fail, as a full or flaky network drive would."""

    def save_resume_state(self, state: Any, path: Path, metadata: Any) -> str:
        raise OSError("no space left on device")


def test_a_resume_point_that_cannot_be_written_does_not_end_the_run(
    tmp_path: Path,
    provenance: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Losing the safety net must cost the net, not the eight hours it protects.

    Nothing downstream reads a resume point and no published number depends on
    one, so a drive that refuses the write has no business failing the run.
    """

    engine = load_engine_module()
    plain = train_run(engine, tmp_path, FakeBackend(), name="plain")

    result = train_run(
        engine,
        tmp_path,
        UnwritableResumeBackend(),
        name="unwritable",
        resume_dir=tmp_path / "resume",
        resume_every=RESUME_EVERY,
    )

    assert result.checkpoint_sha256 == plain.checkpoint_sha256
    assert "could not write the resume point at step 5000" in capsys.readouterr().err
