"""Hyperloop-mHC Vision Transformer backbone.
Combines manifold-constrained hyper-connections (mHC) with parameter-efficient
Hyperloop recurrence. Implements n=2 parallel streams with doubly stochastic
mixing via Sinkhorn-Knopp, looped Middle block for iterative refinement.

Architecture:
    Begin:  2 standard Transformer layers
    Middle: 4 mHC layers (2 parallel streams) × K=3 loops
    End:    2 standard Transformer layers
Total: 8 unique layers → 16 effective depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block (attention + MLP)."""
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
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class mHCTransformerLayer(nn.Module):
    """Manifold‑Constrained Hyper‑Connection layer (Sinkhorn‑Knopp).
    
    Maintains n parallel residual streams. The mixing matrix M ∈ ℝ^{n×n}
    is a freely learnable weight that is projected onto the Birkhoff polytope
    (doubly stochastic matrices) at every forward pass using the Sinkhorn‑Knopp
    algorithm. This guarantees ||M||₂ = 1 and stable signal propagation.
    
    For n=2 the result is equivalent to M = [[α, 1-α], [1-α, α]],
    but the Sinkhorn‑Knopp implementation generalises to any n.
    """
    def __init__(self, dim, num_heads, n_streams=2, mlp_ratio=4, dropout=0.0,
                 sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        
        # One attention + MLP per stream
        self.norms1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_streams)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_streams)])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(n_streams)
        ])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(dim * mlp_ratio), dim),
                nn.Dropout(dropout),
            ) for _ in range(n_streams)
        ])
        
        # Learnable mixing matrix (will be projected to doubly stochastic)
        self.W = nn.Parameter(torch.randn(n_streams, n_streams) * 0.1)

    def sinkhorn(self, M):
        """Project onto doubly stochastic matrices via Sinkhorn‑Knopp."""
        for _ in range(self.sinkhorn_iters):
            # Normalise rows
            M = M / (M.sum(dim=1, keepdim=True) + 1e-12)
            # Normalise columns
            M = M / (M.sum(dim=0, keepdim=True) + 1e-12)
        return M

    def mixing_matrix(self):
        """Return the projected doubly stochastic matrix."""
        raw = torch.exp(self.W)              # ensure positivity
        return self.sinkhorn(raw)            # (n_streams, n_streams)

    def forward(self, streams):
        """
        Args:
            streams: list of tensors [x1, x2, ...] each (B, N, dim)
        Returns:
            list of updated tensors [x1', x2', ...]
        """
        # Apply attention + MLP to each stream independently
        processed = []
        for i in range(self.n_streams):
            stream_norm = self.norms1[i](streams[i])
            a = streams[i] + self.attns[i](
                stream_norm,
                stream_norm,
                stream_norm,
                need_weights=False,
            )[0]
            m = a + self.mlps[i](self.norms2[i](a))
            processed.append(m)
        
        # Mix streams
        M = self.mixing_matrix()  # (n, n)
        new_streams = []
        for i in range(self.n_streams):
            # M[i,0]*processed[0] + M[i,1]*processed[1] + ...
            mixed = sum(M[i, j] * processed[j] for j in range(self.n_streams))
            new_streams.append(mixed)
        return new_streams


class HyperloopBlock(nn.Module):
    """Loops a stack of mHC layers K times (parameter sharing).
    Args:
        num_layers: number of unique mHC layers in this block
        dim: hidden dimension
        num_heads: attention heads
        K: number of loops (weight sharing across depth)
    """
    def __init__(self, num_layers, dim, num_heads, K=3, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.K = K
        self.layers = nn.ModuleList([
            mHCTransformerLayer(
                dim,
                num_heads,
                n_streams=2,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, streams):
        """
        Args:
            streams: list of tensors [x1, x2] each (B, N, dim)
        Returns:
            list of tensors [x1, x2] after K loops through all layers.
        """
        for _ in range(self.K):
            for layer in self.layers:
                streams = layer(streams)  # layer expects list, returns list
        return streams


class HyperloopViT(nn.Module):
    """Hyperloop-mHC Vision Transformer.
    Args:
        in_channels: input feature map channels (e.g., 192 for temporal FCK output)
        img_size: (H, W) of input feature map
        embed_dim: hidden dimension
        num_heads: attention heads
        begin_depth: number of standard layers in Begin block
        middle_depth: number of unique mHC layers in Middle block
        end_depth: number of standard layers in End block
        K: number of loops for Middle block
        mlp_ratio: MLP expansion ratio
        dropout: dropout rate
    """
    def __init__(self, in_channels, img_size=(32, 64), embed_dim=128, num_heads=4,
                 begin_depth=2, middle_depth=4, end_depth=2, K=3,
                 mlp_ratio=4, dropout=0.0):
        super().__init__()
        H, W = img_size
        # Patch embedding (patch_size=1 => token per pixel)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.img_size = img_size
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, H, W) * 0.02)
        # Begin block: standard Transformer layers
        self.begin = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(begin_depth)
        ])
        # Middle block: mHC layers, looped K times
        self.middle = HyperloopBlock(middle_depth, embed_dim, num_heads, K, mlp_ratio, dropout)
        # End block: standard Transformer layers
        self.end = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(end_depth)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map from FCK
        Returns:
            (B, embed_dim, H, W) feature map
        """
        B, C, H, W = x.shape
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H, W)
        # Add positional embedding
        if self.pos_embed.shape[-2:] == (H, W):
            pos = self.pos_embed
        else:
            pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        x = x + pos
        # Flatten to tokens
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        # Begin block
        for blk in self.begin:
            x = blk(x)
        # Middle block: split into 2 identical streams, process, merge
        streams = [x, x]               # both streams start from the same representation
        streams = self.middle(streams)  # returns list of 2 tensors
        # Merge streams (simple average)
        x = 0.5 * (streams[0] + streams[1])
        # End block
        for blk in self.end:
            x = blk(x)
        # Final norm
        x = self.norm_out(x)
        # Reshape back to spatial feature map
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


def hyperloop_vit_tiny(in_channels, img_size=(32, 64)):
    """Tiny Hyperloop-mHC ViT (matches GeoFAR ViT depth but with mHC)."""
    return HyperloopViT(
        in_channels=in_channels,
        img_size=img_size,
        embed_dim=128,
        num_heads=4,
        begin_depth=2,
        middle_depth=4,
        end_depth=2,
        K=3,
        mlp_ratio=4,
        dropout=0.0,
    )
