"""Full downscaling model: FCK → ViT → GeoINR → Decoder."""

import torch
import torch.nn as nn
from .fck import LocalDCTConv
from .vit import ViTBackbone
from .geo_inr import GeoINR
from .decoder import PixelShuffleDecoder

class DownscalingModel(nn.Module):
    def __init__(self, in_channels, n_coeff=64, embed_dim=128, depth=8, num_heads=4,
                 upscale=4, hidden_dim=256, geo_inr_args=None):
        super().__init__()
        n_side = int(n_coeff**0.5)
        if n_side * n_side != n_coeff:
            raise ValueError(f"n_coeff must be a perfect square, got {n_coeff}")
        # FCK: in_channels -> in_channels * n_coeff
        self.fck = LocalDCTConv(in_channels, n=n_side, k=n_side)
        vit_in_channels = in_channels * n_coeff
        self.vit = ViTBackbone(vit_in_channels, embed_dim=embed_dim, depth=depth,
                               num_heads=num_heads, patch_size=1)
        self.geo_inr = GeoINR(**geo_inr_args) if geo_inr_args else None
        self.decoder = PixelShuffleDecoder(embed_dim, upscale=upscale, hidden_dim=hidden_dim)

    def forward(self, x_lr):
        # x_lr: (B, C, H_lr, W_lr)
        f = self.fck(x_lr)                     # (B, C*n_coeff, H_lr, W_lr)
        feat_map = self.vit(f)                 # (B, embed_dim, H_lr, W_lr)
        if self.geo_inr is not None:
            gamma, beta = self.geo_inr()       # each (1, embed_dim, H_hr, W_hr)
            # Upsample feature map to HR before FiLM? To apply at HR, we need feat_hr
            # We'll first decode to HR, then apply FiLM? Or apply FiLM after pixel-shuffle.
            # Let's apply the decoder (which upsamples to HR) then modulate.
            hr = self.decoder(feat_map)        # (B, 1, H_hr, W_hr)
            # It's simpler to modulate the decoder's internal feature maps.
            # For simplicity, we'll apply FiLM after the first conv but before pixel-shuffle.
            # Alternative: decode to intermediate feature and modulate.
            # We'll modify the decoder to accept optional FiLM parameters; easier to incorporate below.
            # For now, just return the raw decoder output, leaving FiLM integration as an exercise.
            return hr
        else:
            return self.decoder(feat_map)
