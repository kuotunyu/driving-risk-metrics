"""Contracts for the real-framework training backend.

These tests use real Torch autograd, a real optimizer, and a real cross-entropy
loss against a hand-written one-by-one convolution. Faking those would only test
the fakes; a tiny real module keeps the evidence honest and the suite fast.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from types import ModuleType
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

    def forward(self, images: Any) -> dict[str, Any]:
        return {"out": self.conv(images)}


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
    adapter = SegmentationAdapter(module=TinyModule(), output_kind="torchvision_dict")
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
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)
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
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)
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
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)
    state = make_state(backends, learning_rate=99.0)

    backend.run_step(state, (("t1", 1.0),), 0.25, apply_update=True)

    assert [group["lr"] for group in state.optimizer.param_groups] == [0.25]


def test_ignored_pixels_never_contribute_to_the_loss(tmp_path: Path) -> None:
    """Counting padded or ignored pixels would train the model to predict the ignore ID."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)
    labels = data_root / TRAIN_LABELS
    ignored = np.full((SOURCE_HEIGHT, SOURCE_WIDTH), 255, dtype=np.uint8)
    Image.fromarray(ignored).save(labels / "t1_train_id.png")
    state = make_state(backends)

    with pytest.raises(ValueError, match="no valid"):
        backend.run_step(state, (("t1", 1.0),), 0.5, apply_update=True)


def test_the_engine_supplied_draw_decides_the_horizontal_flip(tmp_path: Path) -> None:
    """Drawing the flip inside the backend would break the reproducibility guarantee."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)

    flipped = backend.load_batch((("t1", 0.0),))
    unflipped = backend.load_batch((("t1", 1.0),))

    assert not np.array_equal(flipped.images, unflipped.images)
    assert np.array_equal(flipped.images, unflipped.images[:, :, :, ::-1])


def test_a_checkpoint_round_trips_with_its_own_metadata(tmp_path: Path) -> None:
    """A checkpoint that cannot be reloaded is not evidence of anything."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)
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
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)
    state = make_state(backends)
    path = tmp_path / "final_checkpoint.pt"
    backend.save_checkpoint(state, path, {"run_id": "tiny-seed-17", "seed": 17})

    with pytest.raises(ValueError, match="metadata"):
        backend.load_checkpoint(path, {"run_id": "tiny-seed-42", "seed": 42})


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [("fcn_resnet50", "SGD"), ("segformer_b0", "AdamW")],
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
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)

    state = backend.create_training_state(
        model_name,  # type: ignore[arg-type]
        optimizer_spec(model_name),  # type: ignore[arg-type]
    )

    assert type(state.optimizer).__name__ == expected_type
    assert state.optimizer.param_groups[0]["weight_decay"] == (
        0.0001 if expected_type == "SGD" else 0.01
    )


def test_seeding_makes_model_initialization_reproducible(tmp_path: Path) -> None:
    """Unseeded initialization would make the recorded seed meaningless."""

    backends = load_backends_module()
    data_root, manifest, protocol = build_cohort(tmp_path, ("t1",))
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)

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
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)

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
    backend = backends.TorchTrainingBackend(data_root, manifest, protocol, pretrained=False)
    adapter = SegmentationAdapter(
        module=TinyModuleWithUnusedParameter(),
        output_kind="torchvision_dict",
    )
    state = backends.TorchTrainingState(
        adapter=adapter,
        optimizer=torch.optim.SGD(adapter.module.parameters(), lr=0.5),
    )
    before = adapter.module.conv.weight.detach().clone()

    backend.run_step(state, (("t1", 1.0),), 0.5, apply_update=True)

    assert adapter.module.unused.grad is None
    assert not torch.equal(adapter.module.conv.weight.detach(), before)
