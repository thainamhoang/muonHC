"""Vision Transformer backbone – token per pixel, no reduction.
Mimics GeoFAR's Mapper_Vit but outputs a full-resolution feature map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, dim)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTBackbone(nn.Module):
    """ViT that takes a feature map (B, C_in, H, W) and outputs a feature map (B, embed_dim, H, W)."""
    def __init__(self, in_channels, img_size=(32, 64), embed_dim=128, depth=8, num_heads=4,
                 patch_size=1, dropout=0.0):
        super().__init__()
        self.img_size = img_size
        H, W = img_size
        # patch_size=1 means no spatial downsampling
        assert H % patch_size == 0 and W % patch_size == 0, "patch_size must divide image dimensions"
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        # Use convolution to embed patches
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Learnable positional embedding as a 2D grid (flattened)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, H // patch_size, W // patch_size) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # x from FCK: (B, C, 32, 64)
        x = self.patch_embed(x)  # (B, embed_dim, H/p, W/p) -> with p=1 same as (H,W)
        # Add positional embedding
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        if self.pos_embed.shape[-2:] == (grid_h, grid_w):
            pos = self.pos_embed
        else:
            pos = F.interpolate(
                self.pos_embed,
                size=(grid_h, grid_w),
                mode="bilinear",
                align_corners=False,
            )
        x = x + pos
        # Flatten to tokens: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Reshape back to spatial feature map
        x = x.transpose(1, 2).reshape(B, -1, H//self.patch_size, W//self.patch_size)
        return x  # (B, embed_dim, H, W) with H,W same as input if patch_size=1
