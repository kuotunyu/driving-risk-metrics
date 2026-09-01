"""One adapter that normalizes every approved backend output to float64 NCHW logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


class SegmentationModel(Protocol):
    """The only model surface the training and evaluation stages are allowed to use."""

    def logits(self, image_nchw: Float32Array) -> Float64Array:
        """Return float64 ``[N, C, H, W]`` logits at the input resolution."""

    def trainable_parameters(self) -> object:
        """Return the backend parameter collection the optimizer consumes."""


@dataclass(frozen=True)
class SegmentationAdapter:
    """Wrap one backend module behind the framework-independent model surface.

    Torch is imported lazily inside :meth:`logits`, so the pure metric core stays
    importable in the base CPU environment. Both approved backend layouts are
    normalized here: torchvision segmentation models return a mapping whose
    ``out`` entry holds the logits, and SegFormer returns quarter-resolution
    logits on a ``logits`` attribute. Every layout is resized to the input height
    and width with bilinear interpolation so downstream metrics always compare
    logits and masks on the same pixel grid.
    """

    module: Any

    def extract_logits(self, raw_output: Any) -> Any:
        """Normalize one backend output to a plain logits tensor.

        Every approved architecture exposes its dense predictions on a `logits`
        attribute, so there is one layout rather than a branch per backend. A
        model that does not is not one this project can score: the whole
        calibration stage is defined on logits, and anything else reaching here
        would be silently miscalibrated rather than rejected.

        The training backend reuses this so a gradient-carrying forward pass
        and the inference path can never disagree about output layout.
        """

        logits = getattr(raw_output, "logits", None)
        if logits is None:
            raise ValueError(
                "backend output does not expose logits; every approved model must, "
                "because calibration is defined on logits and not on probabilities"
            )
        return logits

    def logits(self, image_nchw: Float32Array) -> Float64Array:
        """Return float64 ``[N, C, H, W]`` logits for one normalized image batch.

        This is the inference path only. The backend is switched to evaluation
        mode first, so batch-norm layers use their frozen running statistics and
        never update them while a locked cohort is being scored.
        """

        if not isinstance(image_nchw, np.ndarray) or image_nchw.dtype != np.float32:
            raise ValueError("image_nchw must be a float32 array")
        if image_nchw.ndim != 4:
            raise ValueError("image_nchw must be four-dimensional NCHW")

        import torch

        self.module.eval()
        # Follow the weights rather than assuming the CPU. Getting this wrong
        # does not raise: the input and the weights simply agree on the CPU and
        # the accelerator sits idle for the whole cohort.
        parameter = next(iter(self.module.parameters()), None)
        batch = torch.from_numpy(image_nchw)
        if parameter is not None:
            batch = batch.to(parameter.device)
        with torch.no_grad():
            raw_output = self.module(batch)
        tensor = self.extract_logits(raw_output)
        height = int(image_nchw.shape[2])
        width = int(image_nchw.shape[3])
        resized = torch.nn.functional.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return np.asarray(resized.detach().cpu().numpy(), dtype=np.float64)

    def trainable_parameters(self) -> Any:
        """Return the backend parameters without copying or reordering them."""

        return self.module.parameters()
