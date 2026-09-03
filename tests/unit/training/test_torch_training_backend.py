"""Contracts for the real-framework training backend.

These tests use real Torch autograd, a real optimizer, and a real cross-entropy
loss against a hand-written one-by-one convolution. Faking those would only test
the fakes; a tiny real module keeps the evidence honest and the suite fast.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip(
        "torch",
        reason="the optional train extra is not installed",
    )

from drivemetrics.data.manifest import build_paired_manifest
from drivemetrics.models.adapters import SegmentationAdapter
from drivemetrics.protocol.config import load_protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE = REPO_ROOT / "configs" / "protocols" / "bdd100k_semseg_v1.yaml"

TRAIN_IMAGES = "images/10k/train"
TRAIN_LABELS = "labels/sem_seg/masks/train"
SOURCE_HEIGHT = 90
SOURCE_WIDTH = 160
NUM_CLASSES = 4


def load_backends_module() -> ModuleType:
    try:
        from drivemetrics.training import backends
    except ImportError:
        pytest.fail("drivemetrics.training.backends is missing", pytrace=False)
    return backends


class TinyModule(torch.nn.Module):
    """One trainable one-by-one convolution wrapped in the torchvision layout."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, NUM_CLASSES, kernel_size=1)
        bias = self.conv.bias
        assert bias is not None
        torch.nn.init.zeros_(bias)
        torch.nn.init.constant_(self.conv.weight, 0.01)

    def forward(self, images: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=self.conv(images))


def build_cohort(tmp_path: Path, sample_ids: tuple[str, ...]) -> tuple[Path, Any, Any]:
    images = tmp_path / "data" / TRAIN_IMAGES
    labels = tmp_path / "data" / TRAIN_LABELS
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(11)
    for sample_id in sample_ids:
        rgb = generator.integers(0, 256, (SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)
        mask = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), 1, dtype=np.uint8)
        mask[:, SOURCE_WIDTH // 2 :] = 2
        Image.fromarray(rgb).save(images / f"{sample_id}.jpg")
        Image.fromarray(mask).save(labels / f"{sample_id}_train_id.png")

    manifest = build_paired_manifest(images, labels, "train")
    config_path = tmp_path / "protocol.yaml"
    shutil.copyfile(PROTOCOL_SOURCE, config_path)
    protocol = load_protocol(config_path).protocol
    return tmp_path / "data", manifest, protocol


def make_state(backends: ModuleType, learning_rate: float = 0.5) -> Any:
    adapter = SegmentationAdapter(module=TinyModule())
    optimizer = torch.optim.SGD(adapter.module.parameters(), lr=learning_rate)
    return backends.TorchTrainingState(adapter=adapter, optimizer=optimizer)


def weight_of(state: Any) -> Any:
    return state.adapter.module.conv.weight.detach().clone()


def test_gradients_accumulate_and_the_optimizer_updates_once_per_window(
    tmp_path: Path,
) -> None:
    """Stepping on every micro batch would train at a smaller batch than the protocol declares."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1", "t2", "t3", "t4"))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    state = make_state(backends)
    before = weight_of(state)

    for position, sample_id in enumerate(("t1", "t2", "t3", "t4"), start=1):
        backend.run_step(
            state,
            ((sample_id, 1.0),),
            0.5,
            apply_update=position == 4,
        )
        if position < 4:
            assert torch.equal(weight_of(state), before)

    assert not torch.equal(weight_of(state), before)


def test_accumulated_micro_batches_match_one_larger_batch(tmp_path: Path) -> None:
    """Accumulation must be equivalent to the batch it claims to reproduce.

    Two micro batches of two samples, accumulated and updated once, must move the
    weights exactly like a single batch of the same four samples.
    """

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1", "t2", "t3", "t4"))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    batch = (("t1", 1.0), ("t2", 1.0), ("t3", 1.0), ("t4", 1.0))

    single = make_state(backends)
    backend.run_step(single, batch, 0.5, apply_update=True)

    accumulated = make_state(backends)
    backend.run_step(accumulated, batch[:2], 0.5, apply_update=False)
    backend.run_step(accumulated, batch[2:], 0.5, apply_update=True)

    assert torch.allclose(weight_of(single), weight_of(accumulated), atol=1e-6)


def test_the_engine_learning_rate_reaches_the_optimizer(tmp_path: Path) -> None:
    """Ignoring the scheduled rate would silently train on the optimizer default."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    state = make_state(backends, learning_rate=99.0)

    backend.run_step(state, (("t1", 1.0),), 0.25, apply_update=True)

    assert [group["lr"] for group in state.optimizer.param_groups] == [0.25]


def test_ignored_pixels_never_contribute_to_the_loss(tmp_path: Path) -> None:
    """Counting padded or ignored pixels would train the model to predict the ignore ID."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    labels = data_root / TRAIN_LABELS
    ignored = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), 255, dtype=np.uint8)
    Image.fromarray(ignored).save(labels / "t1_train_id.png")
    state = make_state(backends)

    with pytest.raises(ValueError, match=r"^micro batch has no valid pixel to learn from"):
        backend.run_step(state, (("t1", 1.0),), 0.5, apply_update=True)


def test_the_engine_supplied_draw_decides_the_horizontal_flip(tmp_path: Path) -> None:
    """Drawing the flip inside the backend would break the reproducibility guarantee."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )

    flipped = backend.load_batch((("t1", 0.0),))
    unflipped = backend.load_batch((("t1", 1.0),))

    assert not np.array_equal(flipped.images, unflipped.images)
    assert np.array_equal(flipped.images, unflipped.images[:, :, :, ::-1])


def test_a_checkpoint_round_trips_with_its_own_metadata(tmp_path: Path) -> None:
    """A checkpoint that cannot be reloaded is not evidence of anything."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    state = make_state(backends)
    metadata = {"run_id": "tiny-seed-17", "seed": 17, "final_step": 1}
    path = tmp_path / "final_checkpoint.pt"

    digest = backend.save_checkpoint(state, path, metadata)

    assert len(digest) == 64
    restored = backend.load_checkpoint(path, metadata)
    assert restored["metadata"] == metadata


def test_a_checkpoint_from_another_run_fails_closed(tmp_path: Path) -> None:
    """Loading the wrong checkpoint would evaluate a model nobody selected."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    state = make_state(backends)
    path = tmp_path / "final_checkpoint.pt"
    backend.save_checkpoint(state, path, {"run_id": "tiny-seed-17", "seed": 17})

    with pytest.raises(ValueError, match=r"^checkpoint metadata does not match the expected run"):
        backend.load_checkpoint(path, {"run_id": "tiny-seed-42", "seed": 42})


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [("upernet_convnextv2_tiny", "AdamW"), ("segformer_b2", "AdamW")],
)
def test_the_declared_optimizer_is_the_one_that_is_built(
    model_name: str,
    expected_type: str,
    tmp_path: Path,
) -> None:
    """Building the wrong optimizer would silently change the locked training recipe."""

    from drivemetrics.training.schedule import optimizer_spec

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )

    state = backend.create_training_state(
        model_name,  # type: ignore[arg-type]
        optimizer_spec(model_name),  # type: ignore[arg-type]
    )

    assert type(state.optimizer).__name__ == expected_type
    assert state.optimizer.param_groups[0]["weight_decay"] == (
        0.05 if model_name.startswith("upernet") else 0.01
    )


def test_seeding_makes_model_initialization_reproducible(tmp_path: Path) -> None:
    """Unseeded initialization would make the recorded seed meaningless."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )

    backend.seed_all(17)
    first = torch.nn.Conv2d(3, NUM_CLASSES, kernel_size=1).weight.detach().clone()
    backend.seed_all(17)
    second = torch.nn.Conv2d(3, NUM_CLASSES, kernel_size=1).weight.detach().clone()
    backend.seed_all(42)
    other = torch.nn.Conv2d(3, NUM_CLASSES, kernel_size=1).weight.detach().clone()

    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_an_unknown_sample_fails_closed(tmp_path: Path) -> None:
    """A batch naming a sample outside the frozen manifest would train on unlisted data."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )

    with pytest.raises(KeyError, match="not in the frozen manifest"):
        backend.load_batch((("unknown", 1.0),))


def test_training_package_exports_the_backend() -> None:
    """The CLI constructs the backend through the package entry point."""

    import drivemetrics.training as training

    backends = load_backends_module()
    assert training.TorchTrainingBackend is backends.TorchTrainingBackend
    assert training.TorchTrainingState is backends.TorchTrainingState
    assert dataclasses.is_dataclass(backends.TorchTrainingState)


class TinyModuleWithUnusedParameter(TinyModule):
    """A module whose extra parameter never receives a gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1))


def test_a_parameter_without_a_gradient_does_not_stop_the_update(tmp_path: Path) -> None:
    """A frozen or unused parameter must not crash the accumulation rescale."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    adapter = SegmentationAdapter(
        module=TinyModuleWithUnusedParameter(),
    )
    state = backends.TorchTrainingState(
        adapter=adapter,
        optimizer=torch.optim.SGD(adapter.module.parameters(), lr=0.5),
    )
    before = adapter.module.conv.weight.detach().clone()

    backend.run_step(state, (("t1", 1.0),), 0.5, apply_update=True)

    assert adapter.module.unused.grad is None
    assert not torch.equal(adapter.module.conv.weight.detach(), before)


def test_the_backend_factory_resolves_the_protocol_and_manifest(tmp_path: Path) -> None:
    """The command line must not resolve protocols itself, so the factory owns that step."""

    import yaml

    backends = load_backends_module()
    data_root, manifest, _ = build_cohort(tmp_path, ("t1",))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(dataclasses.asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )
    protocol_dir = tmp_path / "configs" / "protocols"
    protocol_dir.mkdir(parents=True)
    shutil.copyfile(PROTOCOL_SOURCE, protocol_dir / "bdd100k_semseg_v1.yaml")
    config_path = tmp_path / "configs" / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "drivemetrics-training-run/v1",
                "protocol_path": "protocols/bdd100k_semseg_v1.yaml",
                "model": "upernet_convnextv2_tiny",
                "micro_batch_size": 4,
            }
        ),
        encoding="utf-8",
    )

    backend = backends.build_training_backend(
        config_path,
        manifest_path,
        data_root,
        device="cpu",
        pretrained=False,
    )

    assert isinstance(backend, backends.TorchTrainingBackend)
    assert backend.load_batch((("t1", 1.0),)).images.shape == (1, 3, 512, 1024)


def test_the_factory_builds_its_backend_on_the_requested_device(tmp_path: Path) -> None:
    """The command line reaches the backend only through this factory.

    The backend honoured its device from the start; the factory was the link the
    first formal run fell through, because the command never named one and the
    factory quietly supplied the CPU.
    """

    from drivemetrics.training.schedule import optimizer_spec

    backends = load_backends_module()
    _, manifest, _ = build_cohort(tmp_path, ("t1",))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(dataclasses.asdict(manifest)), encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "schema_version: drivemetrics-training-run/v1\n"
        "protocol_path: protocol.yaml\n"
        "model: upernet_convnextv2_tiny\n"
        "micro_batch_size: 8\n",
        encoding="utf-8",
    )

    backend = backends.build_training_backend(
        config_path, manifest_path, tmp_path / "data", device="meta", pretrained=False
    )
    state = backend.create_training_state(
        "upernet_convnextv2_tiny",  # type: ignore[arg-type]
        optimizer_spec("upernet_convnextv2_tiny"),  # type: ignore[arg-type]
    )

    assert next(state.adapter.module.parameters()).device.type == "meta"


def test_no_device_is_ever_implied(tmp_path: Path) -> None:
    """A default device is a decision nobody made. Every constructor refuses to make it."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(dataclasses.asdict(manifest)), encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "schema_version: drivemetrics-training-run/v1\n"
        "protocol_path: protocol.yaml\n"
        "model: upernet_convnextv2_tiny\n"
        "micro_batch_size: 8\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"^TorchTrainingBackend\.__init__\(\) missing"):
        backends.TorchTrainingBackend(  # type: ignore[call-arg]
            data_root, manifest, protocol, pretrained=False
        )
    with pytest.raises(TypeError, match=r"^build_training_backend\(\) missing"):
        backends.build_training_backend(  # type: ignore[call-arg]
            config_path, manifest_path, data_root, pretrained=False
        )


def test_the_training_state_is_built_on_the_requested_device(tmp_path: Path) -> None:
    """Weights left on the CPU crash the moment a batch arrives on an accelerator.

    Every CPU test passes with the model on the CPU, so this defect is invisible
    until a real GPU run. The ``meta`` device reproduces it without one: it is a
    real non-CPU device, so a backend that never moves the module leaves its
    parameters on the CPU and fails here.
    """

    from drivemetrics.training.schedule import optimizer_spec

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="meta", pretrained=False
    )

    state = backend.create_training_state(
        "upernet_convnextv2_tiny",  # type: ignore[arg-type]
        optimizer_spec("upernet_convnextv2_tiny"),  # type: ignore[arg-type]
    )

    assert next(state.adapter.module.parameters()).device.type == "meta"


def test_parallel_loading_is_byte_identical_to_serial_loading(tmp_path: Path) -> None:
    """Decoding in parallel must change the wall clock and nothing else.

    On an A100 the measured step spent 57% of its time decoding images one at a
    time while the GPU idled. Spreading that across threads is worth roughly a
    doubling, but only if every array comes back identical: the samples, their
    order, and their flip draws are fixed by the engine, and the loader must not
    be able to disturb any of them.
    """

    backends = load_backends_module()
    ids = tuple(f"t{index}" for index in range(6))
    data_root, manifest, protocol = build_cohort(tmp_path, ids)
    batch = tuple((sample_id, 0.0 if index % 2 else 1.0) for index, sample_id in enumerate(ids))

    serial = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False, loader_threads=1
    ).load_batch(batch)
    parallel = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False, loader_threads=4
    ).load_batch(batch)

    assert np.array_equal(serial.images, parallel.images)
    assert np.array_equal(serial.masks, parallel.masks)


def test_a_sample_outside_the_manifest_is_still_refused_when_loading_in_parallel(
    tmp_path: Path,
) -> None:
    """A worker thread must not be able to swallow the frozen-cohort check."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1", "t2"))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False, loader_threads=4
    )

    with pytest.raises(KeyError, match="not in the frozen manifest"):
        backend.load_batch((("t1", 0.0), ("intruder", 0.0)))


def test_the_loader_thread_count_must_be_positive(tmp_path: Path) -> None:
    """Zero threads would deadlock rather than fail, hours into a paid run."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))

    with pytest.raises(ValueError, match=r"^loader_threads must be at least"):
        backends.TorchTrainingBackend(
            data_root, manifest, protocol, device="cpu", pretrained=False, loader_threads=0
        )


def test_a_boolean_thread_count_is_refused(tmp_path: Path) -> None:
    """`True` is an int in Python, so it would silently mean one loader thread."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))

    with pytest.raises(ValueError, match=r"^loader_threads must be an integer"):
        backends.TorchTrainingBackend(
            data_root,
            manifest,
            protocol,
            device="cpu",
            pretrained=False,
            loader_threads=True,  # type: ignore[arg-type]
        )


def test_an_optimizer_outside_the_protocol_fails_closed(tmp_path: Path) -> None:
    """Falling back to a default optimizer would train a model the protocol never described."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )

    with pytest.raises(ValueError, match=r"^optimizer must be adamw for every approved model,"):
        backend.create_training_state(
            "segformer_b2",  # type: ignore[arg-type]
            {"optimizer": "sgd", "learning_rate": 0.01, "weight_decay": 0.0001},
        )


def make_adamw_state(backends: ModuleType) -> Any:
    """A state whose optimizer actually carries moments, as the real runs do.

    ``make_state`` uses plain SGD, which keeps no per-parameter state at all, so
    it cannot show whether the optimizer was restored. The formal runs use
    AdamW, and its moments are exactly what a resumed run must not lose.
    """

    adapter = SegmentationAdapter(module=TinyModule())
    optimizer = torch.optim.AdamW(adapter.module.parameters(), lr=0.5)
    return backends.TorchTrainingState(adapter=adapter, optimizer=optimizer)


def optimizer_fingerprint(state: Any) -> str:
    """Hash the optimizer's tensors so a test can see them move and come back."""

    import hashlib

    digest = hashlib.sha256()
    for entry in state.optimizer.state_dict()["state"].values():
        for value in entry.values():
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def test_resume_state_restores_the_weights_and_the_optimizer(tmp_path: Path) -> None:
    """Restoring weights without the optimizer would reset Adam's moments.

    The schedule would then be applied to a cold optimizer and the resumed run
    would follow a different trajectory from an uninterrupted one, which is the
    failure this pair of methods exists to prevent.
    """

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    state = make_adamw_state(backends)
    backend.run_step(state, (("t1", 0.0),), 0.01, apply_update=True)
    # The engine hands the checkpoint metadata in and the resume file records
    # its own step on top, so what is written and what is expected differ by
    # exactly that one key.
    metadata = {"run_id": "tiny-seed-17", "seed": 17}
    path = tmp_path / "resume_state.pt"

    digest = backend.save_resume_state(state, path, {**metadata, "completed_step": 5})

    assert len(digest) == 64
    saved_weights = {
        name: value.clone() for name, value in state.adapter.module.state_dict().items()
    }
    saved_optimizer = optimizer_fingerprint(state)
    assert saved_optimizer != optimizer_fingerprint(make_adamw_state(backends)), (
        "the optimizer carries no state, so this test could not detect losing it"
    )

    backend.run_step(state, (("t1", 1.0),), 0.01, apply_update=True)
    assert optimizer_fingerprint(state) != saved_optimizer

    completed = backend.load_resume_state(state, path, metadata)

    assert completed == 5
    restored = state.adapter.module.state_dict()
    for name, value in saved_weights.items():
        assert torch.equal(restored[name], value), name
    assert optimizer_fingerprint(state) == saved_optimizer


def test_resume_state_from_another_run_fails_closed(tmp_path: Path) -> None:
    """Continuing another pair's weights would corrupt this run invisibly."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    state = make_adamw_state(backends)
    path = tmp_path / "resume_state.pt"
    backend.save_resume_state(state, path, {"run_id": "tiny-seed-17", "completed_step": 5})

    with pytest.raises(ValueError, match=r"^resume state does not belong to this run"):
        backend.load_resume_state(state, path, {"run_id": "tiny-seed-42"})


def test_a_resume_state_without_a_completed_step_is_refused(tmp_path: Path) -> None:
    """A resume file that cannot say where it stopped cannot be continued from."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    state = make_adamw_state(backends)
    path = tmp_path / "resume_state.pt"
    torch.save({"model": {}, "optimizer": {}, "metadata": {"run_id": "tiny-seed-17"}}, path)

    with pytest.raises(ValueError, match=r"^resume state does not record a completed_step"):
        backend.load_resume_state(state, path, {"run_id": "tiny-seed-17"})


def test_a_half_written_resume_file_does_not_replace_the_last_good_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session killed mid-write must not leave an unloadable resume point."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(
        data_root, manifest, protocol, device="cpu", pretrained=False
    )
    state = make_adamw_state(backends)
    backend.run_step(state, (("t1", 0.0),), 0.01, apply_update=True)
    metadata = {"run_id": "tiny-seed-17"}
    path = tmp_path / "resume_state.pt"
    backend.save_resume_state(state, path, {**metadata, "completed_step": 5})
    good_bytes = path.read_bytes()

    def die_while_saving(*args: Any, **kwargs: Any) -> None:
        raise OSError("the session was killed mid-write")

    monkeypatch.setattr(torch, "save", die_while_saving)
    with pytest.raises(OSError):
        backend.save_resume_state(state, path, {**metadata, "completed_step": 10})

    assert path.read_bytes() == good_bytes
