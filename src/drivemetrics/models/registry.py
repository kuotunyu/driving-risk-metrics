"""Segmentation architectures, built behind one interface.

Every model here takes ``(N, 3, H, W)`` float input and returns ``(N, C, H, W)``
logits at the input resolution. That uniformity is the point: the comparison
this repository makes is only meaningful if the four architectures differ in
architecture and in nothing else — same input pipeline, same output contract,
same loss, same budget.

Torch is imported lazily so the evaluation core stays numpy-only and the metric
tests keep running on a machine with no deep-learning stack.

Three of the four correspond to models in the source notebooks (FCN8s,
DeepLabV3, SETR), so their results under this repository's protocol can be set
against the originals. SegFormer is added as a modern baseline.
"""

from __future__ import annotations

__all__ = ["build_model", "available_models", "MODEL_INFO"]


MODEL_INFO = {
    "fcn8s": {
        "paper": "Fully Convolutional Networks for Semantic Segmentation (2015)",
        "backbone": "VGG16 (ImageNet)",
        "note": "Matches the FCN8s in the source notebooks.",
    },
    "deeplabv3_resnet50": {
        "paper": "Rethinking Atrous Convolution for Semantic Image Segmentation (2017)",
        "backbone": "ResNet-50 (ImageNet)",
        "note": "torchvision reference implementation; matches the source notebook.",
    },
    "setr_pup": {
        "paper": "Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective (2021)",
        "backbone": "ViT-B/16 (ImageNet)",
        "note": "Progressive UPsampling decoder, as the source notebook used.",
    },
    "segformer_b0": {
        "paper": "SegFormer (2021)",
        "backbone": "MiT-B0",
        "note": "Modern baseline, not present in the source material. Needs `transformers`.",
    },
}


def available_models() -> list:
    return sorted(MODEL_INFO)


def build_model(name: str, num_classes: int, pretrained: bool = True):
    """Construct one architecture by name.

    ``pretrained`` loads ImageNet backbone weights. It is on by default because
    every model in the comparison must get the same advantage — turning it on
    for one and off for another is the kind of silent asymmetry that makes a
    benchmark meaningless.
    """
    name = name.lower()
    if name not in MODEL_INFO:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        )

    if name == "fcn8s":
        from .fcn import FCN8s

        return FCN8s(num_classes=num_classes, pretrained=pretrained)

    if name == "deeplabv3_resnet50":
        return _deeplabv3(num_classes, pretrained)

    if name == "setr_pup":
        from .setr import SETRPUP

        return SETRPUP(num_classes=num_classes, pretrained=pretrained)

    if name == "segformer_b0":
        return _segformer(num_classes, pretrained)

    raise AssertionError("unreachable")


def _deeplabv3(num_classes: int, pretrained: bool):
    import torch.nn as nn
    from torchvision.models.segmentation import deeplabv3_resnet50

    weights = "DEFAULT" if pretrained else None
    model = deeplabv3_resnet50(weights=None, weights_backbone=weights, aux_loss=False)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    class _Wrap(nn.Module):
        """Unwrap torchvision's OrderedDict output so every model returns a tensor."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)["out"]

    return _Wrap(model)


def _segformer(num_classes: int, pretrained: bool):
    import torch.nn as nn
    import torch.nn.functional as F

    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "segformer_b0 needs `transformers`. Install it, or choose another "
            f"model from: {', '.join(available_models())}"
        ) from exc

    if pretrained:
        inner = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=num_classes, ignore_mismatched_sizes=True
        )
    else:
        inner = SegformerForSemanticSegmentation(
            SegformerConfig(num_labels=num_classes)
        )

    class _Wrap(nn.Module):
        """SegFormer emits logits at 1/4 resolution; upsample to match the rest."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            logits = self.model(pixel_values=x).logits
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(
                    logits, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            return logits

    return _Wrap(inner)
