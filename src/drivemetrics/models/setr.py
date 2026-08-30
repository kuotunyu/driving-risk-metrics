"""SETR with the Progressive UPsampling (PUP) decoder.

The source notebook recorded two findings worth preserving: Adam did not work
for SETR where SGD did, and PUP beat naive upsampling. Both are reproduced in
the default configuration here rather than left as folklore in a markdown cell.

A ViT encoder emits a 1/16-resolution token grid. PUP walks that back to full
resolution in four 2x steps with convolutions between, which avoids the blocky
artefacts of a single large transposed convolution — and blockiness is
disproportionately costly for thin structures like poles.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SETRPUP(nn.Module):
    """ViT-B/16 encoder with a progressive-upsampling decoder."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        embed_dim: int = 768,
        decoder_dim: int = 256,
    ):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = vit_b_16(weights=weights)
        self.patch_size = vit.patch_size
        self.vit_image_size = vit.image_size
        self.conv_proj = vit.conv_proj
        self.encoder = vit.encoder
        self.class_token = vit.class_token
        self.embed_dim = embed_dim

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )

        self.decoder = nn.Sequential(
            block(embed_dim, decoder_dim),
            block(decoder_dim, decoder_dim),
            block(decoder_dim, decoder_dim),
            block(decoder_dim, decoder_dim),
        )
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        out_size = x.shape[-2:]
        # The ViT's positional embedding is fixed to its training resolution, so
        # the input is resized to it rather than interpolating the embeddings —
        # fewer moving parts, and identical treatment across the comparison.
        if x.shape[-2:] != (self.vit_image_size, self.vit_image_size):
            x = F.interpolate(
                x,
                size=(self.vit_image_size, self.vit_image_size),
                mode="bilinear",
                align_corners=False,
            )

        n = x.shape[0]
        grid = self.vit_image_size // self.patch_size

        tokens = self.conv_proj(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.class_token.expand(n, -1, -1), tokens], dim=1)
        tokens = self.encoder(tokens)[:, 1:]  # drop the class token

        feat = tokens.transpose(1, 2).reshape(n, self.embed_dim, grid, grid)
        logits = self.classifier(self.decoder(feat))
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
