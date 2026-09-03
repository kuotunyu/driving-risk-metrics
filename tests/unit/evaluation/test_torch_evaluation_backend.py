"""Contracts for the real-framework evaluation backend."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip(
        "torch",
        reason="the optional train extra is not installed",
    )

from drivemetrics.data.bdd100k import NUM_TRAIN_CLASSES
from drivemetrics.models.registry import create_model


def load_backends_module() -> ModuleType:
    try:
        from drivemetrics.evaluation import backends
    except ImportError:
        pytest.fail("drivemetrics.evaluation.backends is missing", pytrace=False)
    return backends


def write_checkpoint(path: Path, metadata: dict[str, Any]) -> Any:
    """Save a real SegFormer state dictionary with one recognizable weight."""

    adapter = create_model("segformer_b2", NUM_TRAIN_CLASSES, False)
    state_dict = adapter.module.state_dict()
    marker_key = next(iter(state_dict))
    state_dict[marker_key] = torch.full_like(state_dict[marker_key], 0.125)
    torch.save({"model": state_dict, "metadata": metadata}, path)
    return marker_key


def test_the_checkpoint_architecture_is_rebuilt_and_its_weights_restored(
    tmp_path: Path,
) -> None:
    """Rebuilding a different architecture would silently evaluate an untrained model."""

    backends = load_backends_module()
    path = tmp_path / "final_checkpoint.pt"
    metadata = {"run_id": "segformer_b2-seed-17", "model": "segformer_b2", "seed": 17}
    marker_key = write_checkpoint(path, metadata)

    model, restored_metadata = backends.TorchEvaluationBackend(device="cpu").load_model(path)

    assert restored_metadata == metadata
    weight = model.module.state_dict()[marker_key]
    assert torch.allclose(weight, torch.full_like(weight, 0.125))


def test_a_checkpoint_that_does_not_name_its_architecture_fails_closed(
    tmp_path: Path,
) -> None:
    """Guessing the architecture would produce a metric table nobody can attribute."""

    backends = load_backends_module()
    path = tmp_path / "final_checkpoint.pt"
    torch.save({"model": {}, "metadata": {"run_id": "anonymous", "seed": 17}}, path)

    with pytest.raises(
        ValueError, match=r"^checkpoint metadata must name the model it was trained"
    ):
        backends.TorchEvaluationBackend(device="cpu").load_model(path)


def test_a_checkpoint_naming_an_unapproved_architecture_fails_closed(
    tmp_path: Path,
) -> None:
    """An unlisted architecture is outside the approved three-model comparison."""

    backends = load_backends_module()
    path = tmp_path / "final_checkpoint.pt"
    torch.save({"model": {}, "metadata": {"model": "setr", "run_id": "x", "seed": 17}}, path)

    with pytest.raises(ValueError, match=r"^model must be one of the approved"):
        backends.TorchEvaluationBackend(device="cpu").load_model(path)


def test_evaluation_package_exports_the_backend() -> None:
    """The CLI constructs the evaluation backend through the package entry point."""

    import drivemetrics.evaluation as evaluation

    backends = load_backends_module()
    assert evaluation.TorchEvaluationBackend is backends.TorchEvaluationBackend


def test_the_restored_model_is_placed_on_the_requested_device(tmp_path: Path) -> None:
    """A checkpoint restored onto the CPU makes an A100 run scoring on the CPU."""

    from drivemetrics.evaluation.backends import TorchEvaluationBackend

    adapter = create_model("upernet_convnextv2_tiny", NUM_TRAIN_CLASSES, False)
    checkpoint = tmp_path / "final_checkpoint.pt"
    torch.save(
        {
            "model": adapter.module.state_dict(),
            "metadata": {"model": "upernet_convnextv2_tiny", "run_id": "r", "seed": 17},
        },
        checkpoint,
    )

    restored, _ = TorchEvaluationBackend(device="meta").load_model(checkpoint)

    assert next(restored.module.parameters()).device.type == "meta"
