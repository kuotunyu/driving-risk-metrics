"""What the pretrained-weight loader copies into the real DINOv2 backbone.

These tests need the optional ``train`` extra and are skipped without it. Nothing
is downloaded: the checkpoint stand-in is a randomly initialised ``Dinov2Model``
built at the geometry the published ``facebook/dinov2-small`` configuration
declares, and a stub loader hands it to the registry's weight loader.
"""

from __future__ import annotations

from typing import Any

import pytest

from drivemetrics.models import registry

pytestmark = pytest.mark.slow

NUM_CLASSES = 19

#: The geometry fields of the published ``facebook/dinov2-small`` config.json.
PUBLISHED_DINOV2_SMALL_GEOMETRY: dict[str, Any] = {
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "image_size": 518,
    "patch_size": 14,
}


def test_the_dinov2_position_table_is_the_only_tensor_skipped_for_its_shape() -> None:
    """A tensor skipped in silence trains from random initialisation under a pretrained label.

    The backbone is configured at the library's default image size, so its
    position-embedding table is smaller than the published checkpoint's. The
    loader copies only same-shape tensors; it must say which one it skipped, and
    every other checkpoint tensor must arrive unchanged.
    """

    torch = pytest.importorskip("torch", reason="the optional train extra is not installed")
    transformers = pytest.importorskip(
        "transformers", reason="the optional train extra is not installed"
    )

    checkpoint = transformers.Dinov2Model(
        transformers.Dinov2Config(**PUBLISHED_DINOV2_SMALL_GEOMETRY)
    )
    requested: list[str] = []

    class StubLoader:
        @classmethod
        def from_pretrained(cls, name: str) -> Any:
            requested.append(name)
            return checkpoint

    model = registry.create_model("upernet_dinov2_small", NUM_CLASSES, False)
    backbone: Any = model.module.backbone

    skipped = registry._load_backbone_weights(
        backbone, registry.DINOV2_BACKBONE_CHECKPOINT, StubLoader
    )

    assert requested == [registry.DINOV2_BACKBONE_CHECKPOINT]
    assert skipped == ("embeddings.position_embeddings",)
    assert tuple(backbone.embeddings.position_embeddings.shape) == (1, 257, 384)
    source = checkpoint.state_dict()
    target = backbone.state_dict()
    copied = [key for key in source if key not in skipped]
    assert copied
    assert all(torch.equal(target[key], source[key]) for key in copied)
