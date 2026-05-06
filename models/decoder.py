"""Pixel-shuffle decoder with zero-init final layer (from your FGD)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_conv_decoder(module):
    """Initialize final conv layer with zeros."""
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class PixelShuffleDecoder(nn.Module):
    """Upscales a feature map of shape (B, C, H, W) to HR using PixelShuffle.
    Params:
        in_channels: feature map channels
        upscale: factor (4 for 32->128)
        hidden_dim: channels after pixel shuffle before final convs
    """
    def __init__(self, in_channels, upscale=4, hidden_dim=256):
        super().__init__()
        self.upscale = upscale
        # PixelShuffle expects input channels = hidden_dim * upscale**2
        self.conv1 = nn.Conv2d(in_channels, hidden_dim * upscale * upscale, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        # Zero-init final conv
        self.conv3.apply(init_conv_decoder)

    def forward(self, x):
        x = self.conv1(x)          # (B, C*h^2, H, W)
        x = self.pixel_shuffle(x)  # (B, C, H*4, W*4)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)          # (B, 1, H*4, W*4)
        return x


class FiLMDecoder(nn.Module):
    """Pixel-shuffle decoder with optional FiLM conditioning after upsampling."""

    def __init__(self, in_channels, upscale=4, hidden_dim=256):
        super().__init__()
        self.upscale = upscale
        self.conv1 = nn.Conv2d(in_channels, hidden_dim * upscale * upscale, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.conv3.apply(init_conv_decoder)

    def forward(self, x, gamma=None, beta=None):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        if gamma is not None and beta is not None:
            if gamma.shape[-2:] != x.shape[-2:]:
                gamma = F.interpolate(gamma, size=x.shape[-2:], mode="bilinear", align_corners=False)
                beta = F.interpolate(beta, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = (1.0 + gamma) * x + beta
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        return x
