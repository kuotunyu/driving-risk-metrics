"""Real-library construction contracts for the three approved architectures.

These tests need the optional ``train`` extra. They are skipped in the base CPU
environment, where the injected fake-backend unit tests already cover every
branch. Nothing here may download weights: every model is built with
``pretrained=False`` so the suite stays offline and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivemetrics.models import APPROVED_MODEL_NAMES, create_model

pytestmark = pytest.mark.slow

NUM_CLASSES = 19
SMOKE_HEIGHT = 64
SMOKE_WIDTH = 128


@pytest.mark.parametrize("name", APPROVED_MODEL_NAMES)
def test_each_approved_model_produces_logits_at_the_input_resolution(name: str) -> None:
    """A backend output-layout change would silently misalign logits and masks."""

    pytest.importorskip("torch", reason="the optional train extra is not installed")
    if name == "segformer_b0":
        pytest.importorskip("transformers", reason="the optional train extra is not installed")
    else:
        pytest.importorskip("torchvision", reason="the optional train extra is not installed")

    model = create_model(name, NUM_CLASSES, False)  # type: ignore[arg-type]
    image = np.zeros((1, 3, SMOKE_HEIGHT, SMOKE_WIDTH), dtype=np.float32)

    logits = model.logits(image)

    assert logits.shape == (1, NUM_CLASSES, SMOKE_HEIGHT, SMOKE_WIDTH)
    assert logits.dtype == np.float64
    assert np.all(np.isfinite(logits))


@pytest.mark.parametrize("name", APPROVED_MODEL_NAMES)
def test_each_approved_model_exposes_trainable_parameters(name: str) -> None:
    """The training engine cannot build an optimizer without real backend parameters."""

    pytest.importorskip("torch", reason="the optional train extra is not installed")
    if name == "segformer_b0":
        pytest.importorskip("transformers", reason="the optional train extra is not installed")
    else:
        pytest.importorskip("torchvision", reason="the optional train extra is not installed")

    model = create_model(name, NUM_CLASSES, False)  # type: ignore[arg-type]

    parameters = list(model.trainable_parameters())  # type: ignore[call-overload]

    assert parameters, "the backend reported no trainable parameters"
