"""FCN8s with a VGG16 backbone — the architecture from the source notebooks.

Reimplemented rather than copied so the skip-connection structure is explicit
and testable. The 8s variant fuses predictions from pool3, pool4 and the final
layer, which is what lets it recover objects smaller than a 32-pixel stride —
directly relevant here, since poles and distant pedestrians are exactly that.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FCN8s(nn.Module):
    """Fully convolutional VGG16 with 8-stride skip fusion."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.DEFAULT if pretrained else None
        features = vgg16(weights=weights).features

        # VGG16 pooling layers sit at indices 4, 9, 16, 23, 30.
        self.stage1 = features[:17]   # -> 1/8   (pool3)
        self.stage2 = features[17:24]  # -> 1/16  (pool4)
        self.stage3 = features[24:]    # -> 1/32  (pool5)

        self.head = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.score_head = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

        # Skip-connection scorers start at zero so training begins as plain
        # FCN32s and learns to bring in fine detail, as in the original paper.
        for layer in (self.score_pool4, self.score_pool3):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.up2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
        self.up2b = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
        self.up8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]

        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)

        score = self.score_head(self.head(p5))
        score = self.up2(score)
        score = score + _match(self.score_pool4(p4), score)

        score = self.up2b(score)
        score = score + _match(self.score_pool3(p3), score)

        score = self.up8(score)
        return _match(score, None, size)


def _match(x: torch.Tensor, like: torch.Tensor | None, size=None) -> torch.Tensor:
    """Crop or pad `x` to match `like` (or an explicit size).

    Odd input dimensions make transposed convolutions land a pixel off; the
    original implementation cropped, and doing it explicitly here keeps the
    residual additions from failing on non-power-of-two inputs such as
    CamVid's 360x480.
    """
    import torch.nn.functional as F

    target = size if size is not None else like.shape[-2:]
    if x.shape[-2:] == tuple(target):
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=False)
