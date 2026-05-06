"""Full downscaling model: FCK → (Hyperloop-mHC or Standard ViT) → GeoINR → Decoder.

Supports both standard ViT and Hyperloop-mHC backbones.
GeoINR FiLM is applied after the decoder's pixel-shuffle upsampling.
"""

import torch.nn as nn
import torch.nn.functional as F
from .fck import LocalDCTConv
from .vit import ViTBackbone
from .hyperloop_mhc import HyperloopViT
from .decoder import FiLMDecoder


class DownscalingModel(nn.Module):
    """Full downscaling model.

    Args:
        in_channels: input channels (1 for static, 3 for temporal)
        n_coeff: number of DCT coefficients (must be perfect square)
        embed_dim: ViT hidden dimension
        depth: effective depth (only used for standard ViT backbone)
        num_heads: attention heads
        upscale: spatial upscaling factor
        decoder_hidden_dim: decoder hidden dimension
        backbone: 'vit' or 'hyperloop_mhc'
        hyperloop_kwargs: dict with begin_depth, middle_depth, end_depth, K
        geo_inr_args: dict for GeoINR (n_basis, out_dim, hidden_dim)
    """
    def __init__(self, in_channels, n_coeff=64, embed_dim=128, depth=8, num_heads=4,
                 upscale=4, decoder_hidden_dim=256, backbone='vit',
                 hyperloop_kwargs=None, geo_inr_args=None, img_size=(32, 64),
                 patch_size=1, decoder_upscale=None, input_upsample_size=None,
                 output_size=None):
        super().__init__()
        n_side = int(n_coeff ** 0.5)
        if n_side * n_side != n_coeff:
            raise ValueError(f"n_coeff must be a perfect square, got {n_coeff}")

        # FCK: fixed DCT basis (0 trainable params)
        self.fck = LocalDCTConv(in_channels, n=n_side, k=n_side)
        vit_in_channels = in_channels * n_coeff

        # Backbone
        self.backbone_type = backbone
        self.img_size = tuple(img_size)
        self.patch_size = int(patch_size)
        self.input_upsample_size = (
            tuple(input_upsample_size) if input_upsample_size is not None else None
        )
        self.output_size = tuple(output_size) if output_size is not None else None
        decoder_upscale = int(decoder_upscale or (upscale * self.patch_size))
        if backbone == 'hyperloop_mhc':
            if hyperloop_kwargs is None:
                hyperloop_kwargs = {}
            self.vit = HyperloopViT(
                in_channels=vit_in_channels,
                img_size=self.img_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                begin_depth=hyperloop_kwargs.get('begin_depth', 2),
                middle_depth=hyperloop_kwargs.get('middle_depth', 4),
                end_depth=hyperloop_kwargs.get('end_depth', 2),
                K=hyperloop_kwargs.get('K', 3),
                n_streams=hyperloop_kwargs.get('n_streams', 2),
                mlp_ratio=hyperloop_kwargs.get('mlp_ratio', 4),
                dropout=hyperloop_kwargs.get('dropout', 0.0),
                sinkhorn_iters=hyperloop_kwargs.get('sinkhorn_iters', 5),
                use_spatial_gate=hyperloop_kwargs.get('use_spatial_gate', False),
                gate_hidden_ratio=hyperloop_kwargs.get('gate_hidden_ratio', 0.25),
                gate_init_bias=hyperloop_kwargs.get('gate_init_bias', -1.0),
                patch_size=self.patch_size,
            )
        else:
            self.vit = ViTBackbone(
                vit_in_channels, img_size=self.img_size,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads, patch_size=self.patch_size,
            )

        # GeoINR (optional)
        self.geo_inr = None
        if geo_inr_args is not None:
            from .geo_inr import GeoINR
            self.geo_inr = GeoINR(**geo_inr_args)

        # Decoder with FiLM support
        self.decoder = FiLMDecoder(embed_dim, upscale=decoder_upscale, hidden_dim=decoder_hidden_dim)

    def forward(self, x_lr):
        """
        Args:
            x_lr: (B, C, 32, 64) LR input
        Returns:
            (B, 1, 128, 256) HR prediction
        """
        if self.input_upsample_size is not None and x_lr.shape[-2:] != self.input_upsample_size:
            x_lr = F.interpolate(
                x_lr,
                size=self.input_upsample_size,
                mode="bilinear",
                align_corners=False,
            )

        f = self.fck(x_lr)                    # (B, C*n_coeff, H, W)
        feat_map = self.vit(f)                # (B, embed_dim, 32, 64)

        if self.geo_inr is not None:
            gamma, beta = self.geo_inr()      # each (1, embed_dim, 128, 256)
            # Broadcast to batch
            gamma = gamma.expand(feat_map.size(0), -1, -1, -1)
            beta = beta.expand(feat_map.size(0), -1, -1, -1)
            out = self.decoder(feat_map, gamma, beta)
        else:
            out = self.decoder(feat_map)

        if self.output_size is not None and out.shape[-2:] != self.output_size:
            out = F.interpolate(
                out,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
        return out
