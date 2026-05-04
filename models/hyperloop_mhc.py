"""Hyperloop-mHC Vision Transformer backbone.

This implementation follows the Hyperloop Transformer design more closely:

    Begin:  standard Transformer layers
    Middle: standard Transformer layers reused K times
    End:    standard Transformer layers

The mHC-style part is applied at the loop boundary, not inside every
Transformer sub-layer. This keeps the model parameter-efficient while still
allowing multiple residual streams to interact through a constrained
doubly-stochastic mixing matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )[0]
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class LoopLevelMHC(nn.Module):
    """Lightweight loop-level mHC-style hyper-connection."""

    def __init__(
        self,
        dim,
        n_streams=2,
        sinkhorn_iters=20,
        write_init=-1.0,
    ):
        super().__init__()

        self.dim = dim
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        self.read_logits = nn.Parameter(torch.zeros(n_streams))
        self.write_logits = nn.Parameter(torch.full((n_streams,), write_init))

        eye = torch.eye(n_streams)
        self.residual_logits = nn.Parameter(
            eye + 0.01 * torch.randn(n_streams, n_streams)
        )

    def sinkhorn(self, matrix):
        """Project positive matrix to approximately doubly-stochastic matrix."""
        for _ in range(self.sinkhorn_iters):
            matrix = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-12)
            matrix = matrix / (matrix.sum(dim=0, keepdim=True) + 1e-12)
        return matrix

    def residual_mixing_matrix(self):
        positive = torch.exp(self.residual_logits)
        return self.sinkhorn(positive)

    def read(self, streams):
        weights = torch.softmax(self.read_logits, dim=0)

        x = 0.0
        for i in range(self.n_streams):
            x = x + weights[i] * streams[i]
        return x

    def write(self, streams, x):
        M = self.residual_mixing_matrix()
        write_gates = torch.sigmoid(self.write_logits)

        new_streams = []
        for i in range(self.n_streams):
            mixed_old = 0.0
            for j in range(self.n_streams):
                mixed_old = mixed_old + M[i, j] * streams[j]

            gate = write_gates[i]
            new_stream = (1.0 - gate) * mixed_old + gate * x
            new_streams.append(new_stream)

        return new_streams

    def forward(self, streams, middle_block, loop_pos=None):
        x = self.read(streams)

        if loop_pos is not None:
            x = x + loop_pos

        for block in middle_block:
            x = block(x)

        return self.write(streams, x)


class HyperloopBlock(nn.Module):
    """Hyperloop middle block with loop-level mHC."""

    def __init__(
        self,
        num_layers,
        dim,
        num_heads,
        K=3,
        mlp_ratio=4,
        dropout=0.0,
        n_streams=2,
        sinkhorn_iters=20,
    ):
        super().__init__()

        self.K = K
        self.dim = dim
        self.n_streams = n_streams

        self.middle = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.hyper = LoopLevelMHC(
            dim=dim,
            n_streams=n_streams,
            sinkhorn_iters=sinkhorn_iters,
        )

        self.loop_pos_embed = nn.Parameter(torch.zeros(K, 1, dim))
        nn.init.normal_(self.loop_pos_embed, std=0.02)

    def forward(self, streams):
        for k in range(self.K):
            loop_pos = self.loop_pos_embed[k].unsqueeze(0)
            streams = self.hyper(
                streams=streams,
                middle_block=self.middle,
                loop_pos=loop_pos,
            )
        return streams


class HyperloopViT(nn.Module):
    """Hyperloop-mHC Vision Transformer backbone."""

    def __init__(
        self,
        in_channels,
        img_size=(32, 64),
        embed_dim=128,
        num_heads=4,
        begin_depth=2,
        middle_depth=4,
        end_depth=2,
        K=3,
        n_streams=2,
        mlp_ratio=4,
        dropout=0.0,
        sinkhorn_iters=20,
    ):
        super().__init__()

        H, W = img_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.n_streams = n_streams

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=1,
        )

        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, H, W) * 0.02)

        self.begin = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(begin_depth)
            ]
        )

        self.middle = HyperloopBlock(
            num_layers=middle_depth,
            dim=embed_dim,
            num_heads=num_heads,
            K=K,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            n_streams=n_streams,
            sinkhorn_iters=sinkhorn_iters,
        )

        self.end = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(end_depth)
            ]
        )

        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.patch_embed(x)

        if self.pos_embed.shape[-2:] == (H, W):
            pos = self.pos_embed
        else:
            pos = F.interpolate(
                self.pos_embed,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        x = x + pos
        x = x.flatten(2).transpose(1, 2)

        for block in self.begin:
            x = block(x)

        streams = [x.clone() for _ in range(self.n_streams)]
        streams = self.middle(streams)
        x = sum(streams) / len(streams)

        for block in self.end:
            x = block(x)

        x = self.norm_out(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        return x


def hyperloop_vit_tiny(in_channels, img_size=(32, 64)):
    """Tiny Hyperloop-mHC ViT with 8 unique layers and 16 effective passes."""
    return HyperloopViT(
        in_channels=in_channels,
        img_size=img_size,
        embed_dim=128,
        num_heads=4,
        begin_depth=2,
        middle_depth=4,
        end_depth=2,
        K=3,
        n_streams=2,
        mlp_ratio=4,
        dropout=0.0,
        sinkhorn_iters=20,
    )
