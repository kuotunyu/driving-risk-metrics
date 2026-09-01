"""The exactly three approved segmentation architectures and one construction policy."""

from __future__ import annotations

from typing import Any, Literal

from drivemetrics.models.adapters import SegmentationAdapter

ModelName = Literal["fcn_resnet50", "deeplabv3_resnet50", "segformer_b0"]

APPROVED_MODEL_NAMES: tuple[ModelName, ...] = (
    "fcn_resnet50",
    "deeplabv3_resnet50",
    "segformer_b0",
)

SEGFORMER_ENCODER_CHECKPOINT = "nvidia/mit-b0"


def _build_torchvision(name: str, num_classes: int, pretrained: bool) -> Any:
    from torchvision.models import segmentation

    builder = getattr(segmentation, name)
    return builder(
        weights=None,
        weights_backbone="DEFAULT" if pretrained else None,
        num_classes=num_classes,
        aux_loss=False,
    )


def _build_segformer(num_classes: int, pretrained: bool) -> Any:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_ENCODER_CHECKPOINT,
            num_labels=num_classes,
        )
    # ``num_labels`` reaches SegformerConfig only through untyped PretrainedConfig
    # keyword arguments, so set the documented attribute instead of guessing at
    # a constructor signature that changes between Transformers releases.
    config = SegformerConfig()
    config.num_labels = num_classes
    return SegformerForSemanticSegmentation(config)


def create_model(name: ModelName, num_classes: int, pretrained: bool) -> SegmentationAdapter:
    """Build one approved architecture with a freshly initialized classifier head.

    ``pretrained`` initializes the image backbone or encoder from ImageNet
    weights only. No segmentation-pretrained decoder or classifier is ever
    loaded, so the three architectures enter the comparison with the same
    supervision history and the head always has exactly ``num_classes`` outputs.
    Torchvision and Transformers are imported lazily inside the builders, so the
    pure metric core never requires the training extra.

    The declared return type narrows the fixed `SegmentationModel` protocol to
    the one concrete adapter this factory builds, so the training backend can
    reach the framework module it must optimize.
    """

    if name not in APPROVED_MODEL_NAMES:
        raise ValueError(f"{name!r} is not an approved model; use one of {APPROVED_MODEL_NAMES}")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained must be a boolean")

    if name == "segformer_b0":
        return SegmentationAdapter(
            module=_build_segformer(num_classes, pretrained),
            output_kind="segformer_logits",
        )
    return SegmentationAdapter(
        module=_build_torchvision(name, num_classes, pretrained),
        output_kind="torchvision_dict",
    )
