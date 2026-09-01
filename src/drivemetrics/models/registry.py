"""The exactly three approved segmentation architectures and one construction policy.

All three are contemporary and each represents a different pretraining paradigm,
because pretraining is one of the strongest influences on how confident a model
is, and confidence is what half the metrics in this study measure.

- ``segformer_b2``: supervised hierarchical transformer with an all-MLP decoder.
- ``upernet_convnextv2_tiny``: a modern convolutional backbone pretrained with
  fully convolutional masked autoencoding, under a pyramid pooling decoder.
- ``upernet_dinov2_small``: a self-supervised foundation-model vision
  transformer backbone, under the same decoder.

Two share the UPerNet decoder on purpose. Holding the decoder fixed isolates the
backbone and its pretraining, while SegFormer varies the decoder as well, so the
comparison has both a controlled and an unconstrained axis.

Every backbone is initialized from image-classification or self-supervised
weights only, never from a checkpoint already trained for segmentation. Starting
one model from a segmentation checkpoint would hand it a head start the others
never had, and every ranking downstream would measure that instead.
"""

from __future__ import annotations

from typing import Any, Literal

from drivemetrics.models.adapters import SegmentationAdapter

ModelName = Literal["segformer_b2", "upernet_convnextv2_tiny", "upernet_dinov2_small"]

APPROVED_MODEL_NAMES: tuple[ModelName, ...] = (
    "segformer_b2",
    "upernet_convnextv2_tiny",
    "upernet_dinov2_small",
)

SEGFORMER_ENCODER_CHECKPOINT = "nvidia/mit-b2"
CONVNEXTV2_BACKBONE_CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
DINOV2_BACKBONE_CHECKPOINT = "facebook/dinov2-small"

#: Backbone geometry is pinned here rather than fetched, so the architecture a
#: run trains is fixed by this repository and not by a remote configuration that
#: could change under it.
SEGFORMER_B2_GEOMETRY: dict[str, Any] = {
    "depths": [3, 4, 6, 3],
    "hidden_sizes": [64, 128, 320, 512],
    "num_attention_heads": [1, 2, 5, 8],
    "decoder_hidden_size": 768,
}
CONVNEXTV2_TINY_GEOMETRY: dict[str, Any] = {
    "depths": [3, 3, 9, 3],
    "hidden_sizes": [96, 192, 384, 768],
    "out_features": ["stage1", "stage2", "stage3", "stage4"],
}
DINOV2_SMALL_GEOMETRY: dict[str, Any] = {
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "out_features": ["stage9", "stage10", "stage11", "stage12"],
}


def _load_backbone_weights(backbone: Any, checkpoint: str, loader: Any) -> None:
    """Copy classification or self-supervised weights into a segmentation backbone.

    Only parameters that exist on both sides with matching shapes are copied.
    The rest are the per-stage output norms the segmentation wrapper adds, which
    have no counterpart in the pretrained model and must start fresh.
    """

    source = loader.from_pretrained(checkpoint).state_dict()
    target = backbone.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    if not matched:
        raise ValueError(f"no pretrained parameter matched the backbone from {checkpoint}")
    backbone.load_state_dict(matched, strict=False)


def _build_segformer(num_classes: int, pretrained: bool) -> Any:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_ENCODER_CHECKPOINT,
            num_labels=num_classes,
        )
    # ``num_labels`` reaches SegformerConfig only through untyped PretrainedConfig
    # keyword handling, so it is assigned rather than passed to the constructor.
    config = SegformerConfig(**SEGFORMER_B2_GEOMETRY)
    config.num_labels = num_classes
    return SegformerForSemanticSegmentation(config)


def _build_upernet(name: str, num_classes: int, pretrained: bool) -> Any:
    from transformers import (
        ConvNextV2Config,
        ConvNextV2Model,
        Dinov2Config,
        Dinov2Model,
        UperNetConfig,
        UperNetForSemanticSegmentation,
    )

    backbone_config: Any
    checkpoint: str
    loader: Any
    if name == "upernet_convnextv2_tiny":
        backbone_config = ConvNextV2Config(**CONVNEXTV2_TINY_GEOMETRY)
        checkpoint, loader = CONVNEXTV2_BACKBONE_CHECKPOINT, ConvNextV2Model
    else:
        backbone_config = Dinov2Config(**DINOV2_SMALL_GEOMETRY)
        checkpoint, loader = DINOV2_BACKBONE_CHECKPOINT, Dinov2Model

    config = UperNetConfig(backbone_config=backbone_config)
    config.num_labels = num_classes
    model = UperNetForSemanticSegmentation(config)
    if pretrained:
        _load_backbone_weights(model.backbone, checkpoint, loader)
    return model


def create_model(name: ModelName, num_classes: int, pretrained: bool) -> SegmentationAdapter:
    """Build one approved architecture behind the framework-independent adapter.

    Any name outside the approved three fails closed, because the study compares
    exactly these three and a fourth would have no protocol entry, no optimizer,
    and no place in the run matrix.
    """

    if name not in APPROVED_MODEL_NAMES:
        raise ValueError(f"model must be one of the approved {APPROVED_MODEL_NAMES}, got {name!r}")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained must be a boolean")

    if name == "segformer_b2":
        module = _build_segformer(num_classes, pretrained)
    else:
        module = _build_upernet(name, num_classes, pretrained)
    return SegmentationAdapter(module=module)
