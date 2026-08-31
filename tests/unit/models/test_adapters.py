"""Contracts for the single segmentation adapter that normalizes backend output."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


def load_adapters_module() -> ModuleType:
    try:
        from drivemetrics.models import adapters
    except ImportError:
        pytest.fail("drivemetrics.models.adapters is missing", pytrace=False)
    return adapters


class FakeTensor:
    """Carry one NumPy payload through the tensor calls the adapter performs."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class FakeNoGrad:
    def __init__(self, torch: FakeTorch) -> None:
        self.torch = torch

    def __enter__(self) -> FakeNoGrad:
        self.torch.no_grad_depth += 1
        return self

    def __exit__(self, *exception: object) -> None:
        self.torch.no_grad_depth -= 1


class FakeTorch(ModuleType):
    """Minimal Torch stand-in covering exactly the calls the adapter makes."""

    def __init__(self) -> None:
        super().__init__("torch")
        self.no_grad_depth = 0
        self.interpolate_calls: list[dict[str, Any]] = []
        functional = SimpleNamespace(interpolate=self._interpolate)
        self.nn = SimpleNamespace(functional=functional)

    def from_numpy(self, array: np.ndarray) -> FakeTensor:
        return FakeTensor(array)

    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad(self)

    def _interpolate(
        self,
        tensor: FakeTensor,
        *,
        size: tuple[int, int],
        mode: str,
        align_corners: bool,
    ) -> FakeTensor:
        self.interpolate_calls.append(
            {"shape": tensor.array.shape, "size": size, "mode": mode, "align": align_corners}
        )
        batch, channels = tensor.array.shape[:2]
        return FakeTensor(np.zeros((batch, channels, *size), dtype=np.float32))


class FakeModule:
    """Stand in for one Torch module and record the state active at inference time."""

    def __init__(self, torch: FakeTorch, output: Any) -> None:
        self.torch = torch
        self.output = output
        self.training = True
        self.modes_at_call: list[bool] = []
        self.no_grad_depth_at_call: list[int] = []

    def eval(self) -> FakeModule:
        self.training = False
        return self

    def parameters(self) -> tuple[str, ...]:
        return ("weight",)

    def __call__(self, tensor: FakeTensor) -> Any:
        self.modes_at_call.append(self.training)
        self.no_grad_depth_at_call.append(self.torch.no_grad_depth)
        return self.output


def install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> FakeTorch:
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def make_image(height: int = 512, width: int = 1024) -> np.ndarray:
    return np.zeros((2, 3, height, width), dtype=np.float32)


def test_torchvision_dict_output_is_unwrapped_and_returned_as_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning the raw dict would break every downstream NumPy metric kernel."""

    adapters = load_adapters_module()
    torch = install_fake_torch(monkeypatch)
    logits = np.zeros((2, 19, 512, 1024), dtype=np.float32)
    module = FakeModule(torch, {"out": FakeTensor(logits), "aux": FakeTensor(logits)})
    adapter = adapters.SegmentationAdapter(module=module, output_kind="torchvision_dict")

    result = adapter.logits(make_image())

    assert result.shape == (2, 19, 512, 1024)
    assert result.dtype == np.float64
    assert torch.interpolate_calls[0]["shape"] == (2, 19, 512, 1024)


def test_segformer_quarter_resolution_logits_are_upsampled_to_the_input_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoring quarter-resolution logits against full masks would misalign every pixel."""

    adapters = load_adapters_module()
    torch = install_fake_torch(monkeypatch)
    quarter = np.zeros((2, 19, 128, 256), dtype=np.float32)
    module = FakeModule(torch, SimpleNamespace(logits=FakeTensor(quarter)))
    adapter = adapters.SegmentationAdapter(module=module, output_kind="segformer_logits")

    result = adapter.logits(make_image())

    assert result.shape == (2, 19, 512, 1024)
    assert torch.interpolate_calls == [
        {
            "shape": (2, 19, 128, 256),
            "size": (512, 1024),
            "mode": "bilinear",
            "align": False,
        }
    ]


def test_inference_runs_inside_a_no_gradient_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building an autograd graph during evaluation would exhaust VRAM on the locked cohort."""

    adapters = load_adapters_module()
    torch = install_fake_torch(monkeypatch)
    module = FakeModule(torch, {"out": FakeTensor(np.zeros((1, 19, 8, 8), dtype=np.float32))})
    adapter = adapters.SegmentationAdapter(module=module, output_kind="torchvision_dict")

    adapter.logits(np.zeros((1, 3, 8, 8), dtype=np.float32))

    assert module.no_grad_depth_at_call == [1]
    assert torch.no_grad_depth == 0


def test_inference_switches_the_backend_to_evaluation_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch-norm layers left in training mode would use batch statistics during evaluation.

    They would also overwrite the running statistics of a frozen checkpoint, so
    the same locked cohort could score differently depending on evaluation order.
    """

    adapters = load_adapters_module()
    torch = install_fake_torch(monkeypatch)
    module = FakeModule(torch, {"out": FakeTensor(np.zeros((1, 2, 4, 4), dtype=np.float32))})
    adapter = adapters.SegmentationAdapter(module=module, output_kind="torchvision_dict")

    adapter.logits(np.zeros((1, 3, 4, 4), dtype=np.float32))

    assert module.modes_at_call == [False]


def test_an_unknown_output_kind_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guessing the output layout would silently score the wrong tensor."""

    adapters = load_adapters_module()
    torch = install_fake_torch(monkeypatch)
    module = FakeModule(torch, {"out": FakeTensor(np.zeros((1, 2, 4, 4), dtype=np.float32))})
    adapter = adapters.SegmentationAdapter(
        module=module,
        output_kind="mystery_head",  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="output kind"):
        adapter.logits(np.zeros((1, 3, 4, 4), dtype=np.float32))


def test_logits_rejects_a_non_float32_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """A float64 batch would double activation memory and diverge from the training dtype."""

    adapters = load_adapters_module()
    torch = install_fake_torch(monkeypatch)
    module = FakeModule(torch, {"out": FakeTensor(np.zeros((1, 2, 4, 4), dtype=np.float32))})
    adapter = adapters.SegmentationAdapter(module=module, output_kind="torchvision_dict")

    with pytest.raises(ValueError, match="float32"):
        adapter.logits(np.zeros((1, 3, 4, 4), dtype=np.float64))


@pytest.mark.parametrize("shape", [(3, 4, 4), (2, 1, 3, 4, 4)])
def test_logits_rejects_an_image_that_is_not_batched_nchw(
    shape: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing batch axis would make the output shape contract unverifiable."""

    adapters = load_adapters_module()
    torch = install_fake_torch(monkeypatch)
    module = FakeModule(torch, {"out": FakeTensor(np.zeros((1, 2, 4, 4), dtype=np.float32))})
    adapter = adapters.SegmentationAdapter(module=module, output_kind="torchvision_dict")

    with pytest.raises(ValueError, match="four-dimensional"):
        adapter.logits(np.zeros(shape, dtype=np.float32))


def test_trainable_parameters_delegates_to_the_backend_module() -> None:
    """The training engine needs the real backend parameters, not a copy the adapter invents."""

    adapters = load_adapters_module()
    sentinel = object()
    module = SimpleNamespace(parameters=lambda: sentinel)
    adapter = adapters.SegmentationAdapter(module=module, output_kind="torchvision_dict")

    assert adapter.trainable_parameters() is sentinel
